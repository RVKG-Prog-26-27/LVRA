"""
The latvian_htr CRNN model, run with numpy instead of PyTorch.

Why: PyTorch is larger than the free PythonAnywhere disk allows. The
network itself is small, so its forward pass is rewritten here layer by
layer with numpy. It reads the same checkpoint files (.pt) that
latvian_htr/train.py saves and gives the same output as latvian_htr/infer.py.

Nothing in latvian_htr is changed. The server only reads the trained
weights (latvian_htr/checkpoints_v2/*.pt). The character table is in
alphabet_table.py, a copy without PyTorch.

KEEP IN SYNC with latvian_htr:
  model.py          the layers in forward() below, including the pooling sizes
                    (they are not stored in the checkpoint, except x_stride,
                    which save_checkpoint() puts into the checkpoint package)
  synthetic_data.py TARGET_HEIGHT
  dataset.py / infer.py  how a line image is prepared (prepare_line below)
  alphabet.py       alphabet_table.py
If model.py changes, loading the checkpoint fails with a clear message in
the error log instead of returning wrong text.
"""

import collections
import pickle
import threading
import zipfile

import numpy as np
from PIL import Image, ImageOps

from alphabet_table import NUM_CLASSES, decode_beam_search

TARGET_HEIGHT = 64  # same as latvian_htr/synthetic_data.py
BEAM_WIDTH = 5  # same default as latvian_htr/infer.py

_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "ByteStorage": np.uint8,
    "CharStorage": np.int8,
    "BoolStorage": np.bool_,
}


# ---------------------------------------------------------------- loading

def _rebuild_tensor(storage, offset, size, stride, *_):
    if len(size) == 0:
        return np.array(storage[offset])
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * storage.itemsize for s in stride)
    )
    return np.array(view)


def load_checkpoint(path):
    """
    Reads a PyTorch checkpoint without PyTorch and returns (weights, x_stride),
    like load_checkpoint() in latvian_htr/model.py. A checkpoint that holds
    only the weights (older format) gets x_stride 2.
    Only plain tensors, numbers and dictionaries are allowed; any other object
    in the file is refused, so a tampered checkpoint cannot run code.
    """
    archive = zipfile.ZipFile(path)
    pkl_name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
    prefix = pkl_name[: -len("data.pkl")]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor
            if module == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, requires_grad, hooks: data
            if module == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if module == "torch" and name in _DTYPES:
                return name
            raise pickle.UnpicklingError(f"Not allowed in a checkpoint: {module}.{name}")

        def persistent_load(self, pid):
            _, storage_type, key, _location, _numel = pid
            dtype = _DTYPES[storage_type]
            return np.frombuffer(archive.read(f"{prefix}data/{key}"), dtype=dtype)

    with archive.open(pkl_name) as f:
        content = Unpickler(f).load()
    state = _find_weights(content)
    if state is None:
        raise RuntimeError(f"No model weights (rnn.fc.weight) found in {path}")
    weights = {
        k: (v if v.dtype == np.int64 else np.asarray(v, dtype=np.float32))
        for k, v in state.items()
        if isinstance(v, np.ndarray)
    }
    x_stride = content.get("x_stride", 2) if isinstance(content, dict) and "rnn.fc.weight" not in content else 2
    if x_stride not in (2, 4):
        raise RuntimeError(f"Checkpoint has x_stride={x_stride!r}; the server knows only 2 and 4 (see latvian_htr/model.py)")
    return weights, int(x_stride)


_WEIGHT_KEY = "rnn.fc.weight"


def _find_weights(obj):
    """
    Finds the model weights in a checkpoint.

    A checkpoint can be saved in two ways:
      torch.save(model.state_dict(), ...)          the file is the weights
      torch.save({"model": model.state_dict(),     the weights are inside a
                  "step": ..., "optimizer": ...})   package with other things
    The package keys differ between projects ("model", "state_dict",
    "model_state_dict", ...), so the weights are found by looking for the
    dictionary that holds the layer "rnn.fc.weight". Prefixes that PyTorch
    adds in some setups ("module.", "_orig_mod.") are removed.
    """
    if not isinstance(obj, dict):
        return None
    for key in obj:
        if isinstance(key, str) and key.endswith(_WEIGHT_KEY):
            prefix = key[: -len(_WEIGHT_KEY)]
            return {k[len(prefix):]: v for k, v in obj.items() if isinstance(k, str) and k.startswith(prefix)}
    for value in obj.values():
        found = _find_weights(value)
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------- layers

def _conv2d(x, w, b, pad):
    """x [C, H, W], w [O, C, kh, kw] -> [O, H', W'] (stride 1)."""
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    x = np.ascontiguousarray(x)
    c, h, wd = x.shape
    _, _, kh, kw = w.shape
    ho, wo = h - kh + 1, wd - kw + 1
    s = x.strides
    cols = np.lib.stride_tricks.as_strided(x, (c, kh, kw, ho, wo), (s[0], s[1], s[2], s[1], s[2]))
    return np.tensordot(w, cols, axes=([1, 2, 3], [0, 1, 2])) + b[:, None, None]


def _maxpool(x, kh, kw):
    c, h, w = x.shape
    h2, w2 = h // kh, w // kw
    return x[:, : h2 * kh, : w2 * kw].reshape(c, h2, kh, w2, kw).max(axis=(2, 4))


def _batchnorm(x, sd, p):
    mean = sd[p + "running_mean"][:, None, None]
    var = sd[p + "running_var"][:, None, None]
    return (x - mean) / np.sqrt(var + 1e-5) * sd[p + "weight"][:, None, None] + sd[p + "bias"][:, None, None]


def _relu(x):
    return np.maximum(x, 0)


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable


def _lstm_direction(x, w_ih, w_hh, b_ih, b_hh, reverse):
    steps = x.shape[0]
    hidden = w_hh.shape[1]
    h = np.zeros(hidden, np.float32)
    c = np.zeros(hidden, np.float32)
    out = np.zeros((steps, hidden), np.float32)
    gates_x = x @ w_ih.T + b_ih + b_hh
    order = range(steps - 1, -1, -1) if reverse else range(steps)
    for t in order:
        g = gates_x[t] + w_hh @ h
        i = _sigmoid(g[:hidden])
        f = _sigmoid(g[hidden : 2 * hidden])
        cell = np.tanh(g[2 * hidden : 3 * hidden])
        o = _sigmoid(g[3 * hidden :])
        c = f * c + i * cell
        h = o * np.tanh(c)
        out[t] = h
    return out


def check_architecture(sd):
    """Fails with a clear message if the checkpoint does not fit the layers below."""
    expected = {
        "cnn.net.0.weight": (64, 1, 3, 3),
        "cnn.net.3.weight": (128, 64, 3, 3),
        "cnn.net.6.weight": (256, 128, 3, 3),
        "cnn.net.8.weight": (256, 256, 3, 3),
        "cnn.net.11.weight": (512, 256, 3, 3),
        "cnn.net.14.weight": (512, 512, 3, 3),
        "cnn.net.18.weight": (512, 512, 4, 1),
        "rnn.lstm.weight_ih_l0": (1024, 512),
        "rnn.fc.weight": (NUM_CLASSES, 512),
    }
    for key, shape in expected.items():
        if key not in sd or tuple(sd[key].shape) != shape:
            found = tuple(sd[key].shape) if key in sd else "missing"
            raise RuntimeError(
                f"Checkpoint does not match the server's copy of the model: {key} is {found}, expected {shape}. "
                "Update server/htr_numpy.py (and alphabet_table.py) to match latvian_htr."
            )


def forward(sd, image, x_stride=2):
    """
    image: float32 [64, W], ink close to 1, paper close to 0 (as in
    latvian_htr/dataset.py image_to_tensor). Returns log probabilities [T, num_classes].
    Layer numbers are the indexes inside CNNBackbone.net in latvian_htr/model.py:
      0 conv 3x3, 2 maxpool 2x2, 3 conv, 5 maxpool (2,1) or 2x2 when x_stride is 4,
      6 conv, 8 conv, 10 maxpool (2,1), 11 conv + 12 batchnorm, 14 conv + 15 batchnorm,
      17 maxpool (2,1), 18 conv (4,1) no padding.
    Height 64 -> 32 -> 16 -> 8 -> 4 -> 1, width W -> W/2 (x_stride 2) or W/4 (x_stride 4).
    """
    p = "cnn.net."
    x = image[None].astype(np.float32)
    x = _maxpool(_relu(_conv2d(x, sd[p + "0.weight"], sd[p + "0.bias"], 1)), 2, 2)
    x = _maxpool(_relu(_conv2d(x, sd[p + "3.weight"], sd[p + "3.bias"], 1)), 2, 2 if x_stride == 4 else 1)
    x = _relu(_conv2d(x, sd[p + "6.weight"], sd[p + "6.bias"], 1))
    x = _maxpool(_relu(_conv2d(x, sd[p + "8.weight"], sd[p + "8.bias"], 1)), 2, 1)
    x = _relu(_batchnorm(_conv2d(x, sd[p + "11.weight"], sd[p + "11.bias"], 1), sd, p + "12."))
    x = _maxpool(_relu(_batchnorm(_conv2d(x, sd[p + "14.weight"], sd[p + "14.bias"], 1), sd, p + "15.")), 2, 1)
    x = _relu(_conv2d(x, sd[p + "18.weight"], sd[p + "18.bias"], 0))
    seq = x[:, 0, :].T  # [T, 512]
    for layer in range(2):
        k = f"rnn.lstm.%s_l{layer}"
        fwd = _lstm_direction(seq, sd[k % "weight_ih"], sd[k % "weight_hh"], sd[k % "bias_ih"], sd[k % "bias_hh"], False)
        k = f"rnn.lstm.%s_l{layer}_reverse"
        bwd = _lstm_direction(seq, sd[k % "weight_ih"], sd[k % "weight_hh"], sd[k % "bias_ih"], sd[k % "bias_hh"], True)
        seq = np.concatenate([fwd, bwd], axis=1)
    logits = seq @ sd["rnn.fc.weight"].T + sd["rnn.fc.bias"]
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))  # log_softmax


# ---------------------------------------------------------------- recogniser

def prepare_line(img):
    """
    Same steps as latvian_htr/infer.py load_image + dataset.py image_to_tensor:
    greyscale, scale to 64 px height, auto contrast, invert (ink = 1).
    The browser already crops each line with margins like the training images.
    """
    gray = img.convert("L")
    w, h = gray.size
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    gray = gray.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    return 1.0 - np.asarray(gray, dtype=np.float32) / 255.0


class Recognizer:
    """Loads the model once (on first use) and reads one line image at a time."""

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self._weights = None
        self._x_stride = 2
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        with self._lock:
            if self._weights is None:
                weights, x_stride = load_checkpoint(self.checkpoint_path)
                check_architecture(weights)
                self._x_stride = x_stride
                self._weights = weights

    @property
    def loaded(self):
        return self._weights is not None

    def recognize(self, img):
        """
        Returns the text exactly as the model wrote it, including the mark for
        a crossed-out word (alphabet_table.STRIKE_MARK). app.py removes the
        mark from the text the user sees and keeps it in the training data.
        """
        self._ensure_loaded()
        log_probs = forward(self._weights, prepare_line(img), self._x_stride)
        return decode_beam_search(log_probs, beam_width=BEAM_WIDTH).strip()

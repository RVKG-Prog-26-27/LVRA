"""
The latvian_htr CRNN model, run with numpy instead of PyTorch.

Why: PyTorch is larger than the free PythonAnywhere disk allows. The
network itself is small, so its forward pass is rewritten here layer by
layer with numpy. It reads the same checkpoint files (.pt) that
latvian_htr/train.py saves and gives the same output as latvian_htr/infer.py.

Nothing in latvian_htr is changed. This file only reads:
  latvian_htr/alphabet.py        the character table and CTC decoding
  latvian_htr/checkpoints/*.pt   the trained weights

IMPORTANT: the layers below must match latvian_htr/model.py. If the AI side
changes the network (layer sizes, number of layers, order), this file has to
be updated too, otherwise loading the checkpoint fails with a clear error.
"""

import collections
import importlib.util
import os
import pickle
import threading
import zipfile

import numpy as np
from PIL import Image, ImageOps

TARGET_HEIGHT = 32  # same as latvian_htr/synthetic_data.py

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

def load_alphabet(htr_dir):
    """Imports latvian_htr/alphabet.py by file path (it has no dependencies)."""
    path = os.path.join(htr_dir, "alphabet.py")
    spec = importlib.util.spec_from_file_location("lvra_alphabet", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rebuild_tensor(storage, offset, size, stride, *_):
    if len(size) == 0:
        return np.array(storage[offset])
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * storage.itemsize for s in stride)
    )
    return np.array(view)


def load_state_dict(path):
    """
    Reads a PyTorch checkpoint (a state_dict saved with torch.save) without
    PyTorch. Only plain tensors are allowed; any other object in the file is
    refused, so a tampered checkpoint cannot run code.
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
        state = Unpickler(f).load()
    return {k: np.asarray(v, dtype=np.float32) if v.dtype != np.int64 else v for k, v in state.items()}


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


def forward(sd, image):
    """
    image: float32 [32, W], ink close to 1, paper close to 0 (as in
    latvian_htr/dataset.py image_to_tensor). Returns logits [T, num_classes].
    Layer numbers are the indexes inside CNNBackbone.net in latvian_htr/model.py.
    """
    p = "cnn.net."
    x = image[None].astype(np.float32)
    x = _maxpool(_relu(_conv2d(x, sd[p + "0.weight"], sd[p + "0.bias"], 1)), 2, 2)
    x = _maxpool(_relu(_conv2d(x, sd[p + "3.weight"], sd[p + "3.bias"], 1)), 2, 2)
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
    return seq @ sd["rnn.fc.weight"].T + sd["rnn.fc.bias"]


# ---------------------------------------------------------------- recogniser

def prepare_line(img):
    """
    Makes a line image from the browser look like the training images:
    greyscale, full contrast, cropped to the writing with white margins of
    about a quarter of the text height (synthetic_data.render_text_line uses
    10 px around 30 to 50 px text), then scaled to 32 px height.
    """
    gray = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    arr = np.asarray(gray, dtype=np.uint8)
    ink = arr < 140
    if ink.any():
        rows = np.where(ink.any(axis=1))[0]
        cols = np.where(ink.any(axis=0))[0]
        arr = arr[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        pad = max(2, int(round(arr.shape[0] * 0.27)))
        arr = np.pad(arr, pad, constant_values=255)
    gray = Image.fromarray(arr)
    w, h = gray.size
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    gray = gray.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
    return 1.0 - np.asarray(gray, dtype=np.float32) / 255.0


class Recognizer:
    """Loads the model once (on first use) and reads one line image at a time."""

    def __init__(self, htr_dir, checkpoint_path):
        self.htr_dir = htr_dir
        self.checkpoint_path = checkpoint_path
        self._weights = None
        self._alphabet = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        with self._lock:
            if self._weights is None:
                self._alphabet = load_alphabet(self.htr_dir)
                weights = load_state_dict(self.checkpoint_path)
                classes = weights["rnn.fc.weight"].shape[0]
                if classes != self._alphabet.NUM_CLASSES:
                    raise RuntimeError(
                        f"Checkpoint has {classes} classes, alphabet.py has {self._alphabet.NUM_CLASSES}"
                    )
                self._weights = weights

    @property
    def loaded(self):
        return self._weights is not None

    def recognize(self, img):
        self._ensure_loaded()
        logits = forward(self._weights, prepare_line(img))
        return self._alphabet.decode_greedy(logits.argmax(axis=1).tolist()).strip()

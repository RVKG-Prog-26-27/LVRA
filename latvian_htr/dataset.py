"""
Data providers for training and evaluation:
- SyntheticHTRDataset: generates fresh synthetic examples on the fly (used by evaluate.py).
- LineFolderDataset: reads ready-made lines from a folder (labels.csv + images): the real scans or a synthetic
  set generated in advance by make_synth_dataset.py.
- split_real(): splits the real lines into "train" and "validation" parts (the same split in train.py and
  evaluate.py, stable when new lines are added).
- BucketBatchSampler: builds batches from lines of SIMILAR width (bucketing) so the GPU does not compute padding.
- SyntheticBucketStream: alternative mode - endless generation of synthetic lines right inside the workers.
- collate_batch: pads the images of a batch to a common width and prepares the CTC targets.

Speed:
  * images travel as uint8 (4x smaller than float32) and are converted to float on the GPU;
  * the batch width is rounded up to a multiple of PAD_TO - cuDNN then sees few different shapes and
    cudnn.benchmark can cache the fastest algorithm for each of them.
"""
import csv
import os
import random
import re
import zlib

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, IterableDataset, Sampler

from alphabet import CHAR_TO_IDX, encode
from synthetic_data import TARGET_HEIGHT, load_words, make_sample

PAD_TO = 32  # the batch width is rounded up to a multiple of this number

# --- real data: train / validation split ---
DEFAULT_REAL_VAL_FRACTION = 0.1   # share of real lines held out for validation
REAL_SPLIT_BLOCK = 50             # consecutive file numbers are assigned to one part together (see is_real_val)


def image_to_uint8(img: Image.Image) -> torch.Tensor:
    """Pillow image -> uint8 tensor [1, H, W] where background = 0 and ink ~ 255.
    autocontrast: a gray/dark background (photos, scans) becomes close to white, the ink close to black.
    Then the brightness is inverted (255 - x), so that "there is ink" = a big number for the network."""
    img = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    arr = 255 - np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Pillow image -> float tensor [1, H, W] with values 0..1 (background 0, ink ~1).
    The same step is used in training and in inference (infer.py, evaluate.py)."""
    return image_to_uint8(img).float() / 255.0


def resize_to_height(img: Image.Image) -> Image.Image:
    """Scale to TARGET_HEIGHT keeping the proportions."""
    w, h = img.size
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    return img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)


def open_line_image(path: str) -> Image.Image:
    """Opens an image file as a grayscale line of height TARGET_HEIGHT (the single place where this is done:
    training, evaluation and infer.py all use it; the web server mirrors the same steps)."""
    img = Image.open(path).convert("L")
    return img if img.size[1] == TARGET_HEIGHT else resize_to_height(img)


class SyntheticHTRDataset(Dataset):
    """A "warehouse" of artificial examples: every access draws a new one instead of storing old ones."""

    def __init__(self, length: int = 2000, corpus: list[str] | None = None):
        self.length = length
        self.corpus = corpus or load_words()   # a word list (not ready-made lines)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, text = make_sample(self.corpus)
        return image_to_tensor(img), text   # a pair (tensor, text)


class LineFolderDataset(Dataset):
    """A folder with ready-made lines: labels.csv (filename,text[,width]) + images.
    The images may be in the subfolder images/ or right next to labels.csv.
    Fits both the real scans and a synthetic set generated in advance (make_synth_dataset.py).

    as_uint8=True  - for training: returns (uint8 tensor, text), as collate_batch expects.
    as_uint8=False - for evaluate.py / infer: (float tensor 0..1, text)."""

    def __init__(self, root: str, as_uint8: bool = False):
        self.root = root
        self.as_uint8 = as_uint8
        self.samples: list[tuple[str, str]] = []
        self._csv_widths: list[int] | None = []
        labels_path = os.path.join(root, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"File not found: {labels_path} (it must have the columns 'filename,text').")
        with open(labels_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append((row["filename"], row["text"]))
                if self._csv_widths is not None:
                    w = row.get("width")
                    if w:
                        self._csv_widths.append(int(w))
                    else:
                        self._csv_widths = None      # no width column - the widths are read from the files
        self._drop_unknown_chars()
        img_dir = os.path.join(root, "images")
        self.img_dir = img_dir if os.path.isdir(img_dir) else root

    def _drop_unknown_chars(self):
        """Lines with characters that are not in alphabet.py are skipped (otherwise training would crash midway)."""
        keep, bad = [], []
        for k, (fn, text) in enumerate(self.samples):
            unknown = {c for c in text if c not in CHAR_TO_IDX}
            (bad if unknown else keep).append((k, fn, unknown))
        if bad:
            print(f"WARNING: {self.root}: skipped {len(bad)} lines with characters that are not in the alphabet "
                  f"(add them to alphabet.py or fix the labels):")
            for _, fn, unknown in bad[:10]:
                print(f"    {fn}: {''.join(sorted(unknown))!r}")
            ids = [k for k, _, _ in keep]
            self.samples = [self.samples[k] for k in ids]
            if self._csv_widths is not None and self._csv_widths:
                self._csv_widths = [self._csv_widths[k] for k in ids]

    def __len__(self):
        return len(self.samples)

    def load(self, idx) -> tuple[Image.Image, str]:
        """The image, already scaled to the height TARGET_HEIGHT, and its text."""
        filename, text = self.samples[idx]
        return open_line_image(os.path.join(self.img_dir, filename)), text

    def widths(self) -> list[int]:
        """Width of every image after scaling to the height TARGET_HEIGHT (needed for bucketing).
        Taken from the width column, otherwise only the file header is read (fast, no decoding)."""
        if self._csv_widths is not None and len(self._csv_widths) == len(self.samples):
            return list(self._csv_widths)
        out = []
        for filename, _ in self.samples:
            with Image.open(os.path.join(self.img_dir, filename)) as im:
                w, h = im.size
            out.append(w if h == TARGET_HEIGHT else max(1, int(w * (TARGET_HEIGHT / h))))
        return out

    def __getitem__(self, idx):
        img, text = self.load(idx)
        return (image_to_uint8(img) if self.as_uint8 else image_to_tensor(img)), text


def is_real_val(filename: str, fraction: float) -> bool:
    """Does this real line belong to the validation part?

    The decision depends only on the file name, so it does not change when new lines are added to the data set
    (a random shuffle by index would move lines between the parts every time the data set grows).
    Files whose names end with a number (0001.png, 0002.png, ...) are grouped in blocks of REAL_SPLIT_BLOCK
    consecutive numbers and a block goes to one part as a whole: neighbouring lines usually come from the same
    page and the same writer, and splitting them would make the validation result too optimistic."""
    if fraction <= 0:
        return False
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(\d+)$", stem)
    key = f"block{int(m.group(1)) // REAL_SPLIT_BLOCK}" if m else stem
    return zlib.crc32(f"real-split:{key}".encode("utf-8")) % 10000 < fraction * 10000


def split_real(dataset: LineFolderDataset, fraction: float) -> tuple[list[int], list[int]]:
    """Indices of (train, validation) lines of a real data set. fraction=0 -> everything is for training."""
    train, val = [], []
    for i, (filename, _) in enumerate(dataset.samples):
        (val if is_real_val(filename, fraction) else train).append(i)
    return train, val


class BucketBatchSampler(Sampler):
    """Yields batches of example indices ENDLESSLY, epoch after epoch.

    Every epoch: shuffle all examples, cut them into chunks of batch_size * pool_batches, sort every chunk by
    width and cut it into batches, then shuffle the order of the batches.
    Result: every example exactly once per epoch, but a batch holds lines of almost the same width."""

    def __init__(self, widths: list[int], batch_size: int, pool_batches: int = 50, seed: int = 0):
        self.widths = widths
        self.batch_size = batch_size
        self.pool = batch_size * pool_batches
        self.seed = seed

    def __iter__(self):
        n, bs = len(self.widths), self.batch_size
        epoch = 0
        while True:
            rng = random.Random(self.seed * 100003 + epoch)
            idx = list(range(n))
            rng.shuffle(idx)
            batches = []
            for s in range(0, n, self.pool):
                chunk = sorted(idx[s:s + self.pool], key=self.widths.__getitem__)
                batches += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
            rng.shuffle(batches)
            yield from batches
            epoch += 1


def collate_batch(batch):
    """
    Merges several examples into one batch. batch - a list of (uint8 image [1, H, W], text).

    Returns:
      images         uint8 [B, 1, H, W_pad]  (background and padding = 0)
      targets        long  [sum of lengths]  - all texts one after another as character numbers (for CTC)
      target_lengths long  [B]
      widths         long  [B]               - the real width of every image BEFORE padding
      texts          list[str]
    """
    imgs, texts = zip(*batch)
    max_w = max(img.shape[-1] for img in imgs)
    max_w = ((max_w + PAD_TO - 1) // PAD_TO) * PAD_TO
    padded = torch.zeros(len(imgs), 1, TARGET_HEIGHT, max_w, dtype=torch.uint8)
    widths = torch.zeros(len(imgs), dtype=torch.long)
    for i, img in enumerate(imgs):
        w = img.shape[-1]
        padded[i, :, :, :w] = img
        widths[i] = w

    # The CTC "correct answers" are prepared here (in the worker), not in the main loop.
    encoded = [encode(t) for t in texts]
    targets = torch.tensor([c for e in encoded for c in e], dtype=torch.long)
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    return padded, targets, target_lengths, widths, list(texts)


class SyntheticBucketStream(IterableDataset):
    """An endless stream of READY batches for training.

    Every worker keeps a "pool" of pool_size ready examples. To produce a batch it:
      1) tops the pool up to the full size (exactly batch_size new examples per batch - the work is even,
         without pauses);
      2) sorts the pool by width, takes a random example and batch_size neighbours by width;
      3) yields them as one batch.
    A batch holds lines of almost the same width -> almost no padding -> the GPU computes only useful things.

    real: a LineFolderDataset or None; real_indices: which of its lines may be used (None = all of them);
    real_fraction: the share of real examples in the stream.
    """

    def __init__(self, batch_size: int, real: LineFolderDataset | None = None,
                 real_fraction: float = 0.0, pool_batches: int = 16, real_indices: list[int] | None = None):
        self.batch_size = batch_size
        self.real = real
        self.real_indices = list(real_indices) if real_indices is not None else (
            list(range(len(real))) if real is not None else [])
        self.real_fraction = real_fraction if self.real_indices else 0.0
        self.pool_size = batch_size * pool_batches

    def _make(self, words):
        if self.real_fraction and random.random() < self.real_fraction:
            img, text = self.real.load(random.choice(self.real_indices))
        else:
            img, text = make_sample(words)
        return image_to_uint8(img), text

    def __iter__(self):
        # `random` in every DataLoader worker is already seeded differently (base_seed + worker_id), so the
        # workers do not repeat each other, and a run with the same seed is reproducible.
        words = load_words()
        bs = self.batch_size
        pool: list = []
        while True:
            while len(pool) < self.pool_size:
                pool.append(self._make(words))
            pool.sort(key=lambda s: s[0].shape[-1])
            anchor = random.randrange(len(pool))
            start = min(max(0, anchor - bs // 2), len(pool) - bs)
            batch = pool[start:start + bs]
            del pool[start:start + bs]
            yield collate_batch(batch)

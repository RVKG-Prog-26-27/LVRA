"""
packed_lines.py - keeps a ready-made set of lines (labels.csv + images/) in RAM.

Why: when training from a folder, tens of thousands of PNG files are opened and decoded every epoch.
This script runs the set through LineFolderDataset (dataset.py) ONCE and saves the ready uint8 images as one
file in the subfolder <set>/packed/. During training that file is read through a memory map: Windows keeps it
in RAM once, shared by all workers.

How to use:
    # 1) pack (once after every generation of the set):
    python packed_lines.py data/synth_100k
    python packed_lines.py data/synth_100k data/synth_val      # several folders at once

    # 2) train as usual - give THE SAME folder:
    python train.py --synth-data data/synth_100k

train.py sees data/synth_100k/packed/ by itself and takes the data from there. The pack is NOT used (the PNG
files are read as before) if:
    * labels.csv changed after packing (for example, the set was generated again);
    * dataset.py, the alphabet (alphabet.py) or TARGET_HEIGHT changed - even a comment edit in dataset.py counts;
    * the check after packing found a difference.
In these cases train.py prints the reason and the command to re-pack.
To stop using packs completely - just delete the packed/ folder.
"""

import argparse
import hashlib
import inspect
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PACK_SUBDIR = "packed"
FORMAT_VERSION = 1


# ------------------------------------------------------------------ helpers

def _identity(x):
    return x


def _dataset_source_hash() -> str:
    """Fingerprint of everything the packed images depend on: the code of dataset.py, TARGET_HEIGHT and the set
    of alphabet characters (LineFolderDataset skips lines with unknown characters).
    If any of these changes, the old pack becomes invalid."""
    import dataset as ds_mod
    h = hashlib.sha1()
    with open(inspect.getsourcefile(ds_mod), "rb") as f:
        h.update(f.read())
    h.update(repr(getattr(ds_mod, "TARGET_HEIGHT", None)).encode())
    h.update(repr(sorted(getattr(ds_mod, "CHAR_TO_IDX", {}).keys())).encode("utf-8"))
    return h.hexdigest()


def _labels_stamp(folder: str) -> dict:
    st = os.stat(os.path.join(folder, "labels.csv"))
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def _pack_dir(folder: str) -> str:
    return os.path.join(folder, PACK_SUBDIR)


# ------------------------------------------------------------------ dataset

class PackedLineDataset(Dataset):
    """Returns EXACTLY what LineFolderDataset(folder, as_uint8=True) returns, but from one file in RAM."""

    def __init__(self, folder: str, preload: bool = False):
        d = _pack_dir(folder)
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        idx = np.load(os.path.join(d, "index.npz"))
        self.offsets, self.shapes, self._widths = idx["offsets"], idx["shapes"], idx["widths"]
        with open(os.path.join(d, "extras.pkl"), "rb") as f:
            self.extras = pickle.load(f)
        self.kind = meta["kind"]                 # "tensor" or "numpy"
        self.as_list = meta["item_type"] == "list"
        self.bin_path = os.path.join(d, "pixels.bin")
        # preload=True: the whole file is copied into the memory of THIS process. Only with --num-workers 0,
        # otherwise on Windows every worker would get its own full copy.
        self.preload = preload
        self._data = np.fromfile(self.bin_path, dtype=np.uint8) if preload else None

    def __len__(self):
        return len(self.extras)

    def widths(self):
        return [int(w) for w in self._widths]

    def _buf(self):
        if self._data is None:                   # opened lazily, separately in every worker
            self._data = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
        return self._data

    def __getitem__(self, i):
        o = int(self.offsets[i])
        shape = tuple(int(s) for s in self.shapes[i])
        count = int(np.prod(shape))
        a = np.array(self._buf()[o:o + count]).reshape(shape)    # a copy: the image may be modified
        img = torch.from_numpy(a) if self.kind == "tensor" else a
        item = (img,) + tuple(self.extras[i])
        return list(item) if self.as_list else item

    def __getstate__(self):
        # the memmap is NOT passed to workers (pickle would copy gigabytes); every worker opens the file itself
        state = self.__dict__.copy()
        if not self.preload:
            state["_data"] = None
        return state


# ------------------------------------------------------------------ status / open

def pack_status(folder: str):
    """(True, n) if the pack is up to date, otherwise (False, reason)."""
    meta_path = os.path.join(_pack_dir(folder), "meta.json")
    if not os.path.isfile(meta_path):
        return False, "no pack"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False, "meta.json is damaged"
    if meta.get("version") != FORMAT_VERSION:
        return False, "old pack format"
    if meta.get("labels") != _labels_stamp(folder):
        return False, "labels.csv changed after packing (new dataset?)"
    if meta.get("dataset_py") != _dataset_source_hash():
        return False, "dataset.py / alphabet / TARGET_HEIGHT changed after packing"
    return True, meta["n"]


def open_line_dataset(folder: str, as_uint8: bool = False, **kwargs):
    """Instead of LineFolderDataset(folder, as_uint8=True): uses the pack if it is up to date, otherwise the PNG files."""
    from dataset import LineFolderDataset
    if as_uint8 and not kwargs:
        ok, info = pack_status(folder)
        if ok:
            print(f"[packed] {folder}: {info} lines from RAM pack ({_pack_dir(folder)})")
            return PackedLineDataset(folder)
        if info != "no pack":
            print(f"[packed] {folder}: pack NOT used - {info}. Reading PNG files.")
            print(f"         To re-pack:  python packed_lines.py {folder}")
    return LineFolderDataset(folder, as_uint8=as_uint8, **kwargs)


# ------------------------------------------------------------------ packing

def pack(folder: str, workers: int, verify: int = 300):
    from dataset import LineFolderDataset

    d = _pack_dir(folder)
    os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(d, "meta.json")
    if os.path.exists(meta_path):
        os.remove(meta_path)                     # while packing, the old pack is invalid

    stamp_before = _labels_stamp(folder)
    src_hash = _dataset_source_hash()
    src = LineFolderDataset(folder, as_uint8=True)
    n = len(src)
    widths = np.asarray([int(w) for w in src.widths()], dtype=np.int64)
    print(f"Packing {folder}: {n} lines, {workers} workers")

    loader = DataLoader(src, batch_size=None, shuffle=False, num_workers=workers, collate_fn=_identity)
    offsets = np.zeros(n, dtype=np.int64)
    shapes = None
    extras, kind, item_type = [], None, None
    bin_path = os.path.join(d, "pixels.bin")
    tmp_path = bin_path + ".part"
    pos, t0, i = 0, time.time(), -1
    with open(tmp_path, "wb") as f:
        for i, item in enumerate(loader):
            if not isinstance(item, (tuple, list)):
                sys.exit(f"LineFolderDataset returns {type(item).__name__}, expected tuple/list - check dataset.py")
            img, rest = item[0], tuple(item[1:])
            if isinstance(img, torch.Tensor):
                k, a = "tensor", img.numpy()
            elif isinstance(img, np.ndarray):
                k, a = "numpy", img
            else:
                sys.exit(f"First element is {type(img).__name__}, expected tensor/array - check dataset.py")
            if a.dtype != np.uint8:
                sys.exit(f"Image dtype is {a.dtype}, expected uint8 (as_uint8=True) - check dataset.py")
            if kind is None:
                kind, item_type = k, ("list" if isinstance(item, list) else "tuple")
                shapes = np.zeros((n, a.ndim), dtype=np.int32)
            a = np.ascontiguousarray(a)
            offsets[i], shapes[i] = pos, a.shape
            f.write(a.tobytes())
            pos += a.nbytes
            extras.append(rest)
            if (i + 1) % 10000 == 0 or i + 1 == n:
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"  {i + 1:,}/{n:,}  ({rate:,.0f} lines/s, {pos / 1e9:.2f} GB)", flush=True)
    if i + 1 != n:
        sys.exit(f"Got {i + 1} items instead of {n} - packing aborted")
    os.replace(tmp_path, bin_path)
    np.savez(os.path.join(d, "index.npz"), offsets=offsets, shapes=shapes, widths=widths)
    with open(os.path.join(d, "extras.pkl"), "wb") as f:
        pickle.dump(extras, f, protocol=pickle.HIGHEST_PROTOCOL)

    if _labels_stamp(folder) != stamp_before:
        sys.exit("labels.csv changed while packing - run again when generation is finished")

    # check: the packed data == what LineFolderDataset returns now
    packed = PackedLineDataset.__new__(PackedLineDataset)
    packed.offsets, packed.shapes, packed._widths = offsets, shapes, widths
    packed.extras, packed.kind, packed.as_list = extras, kind, item_type == "list"
    packed.bin_path, packed.preload, packed._data = bin_path, False, None
    fresh = LineFolderDataset(folder, as_uint8=True)
    check = random.Random(0).sample(range(n), min(verify, n))
    bad = [j for j in check if not _same(packed[j], fresh[j])]
    if bad or packed.widths() != [int(w) for w in fresh.widths()]:
        print(f"!! CHECK FAILED: {len(bad)}/{len(check)} lines differ from LineFolderDataset (e.g. #{bad[:5]}).")
        print("!! Probably dataset.py adds random augmentation - packing would freeze it.")
        print("!! The pack will NOT be used. Move the augmentation out of dataset.py (or apply it after packing).")
        sys.exit(1)

    meta = {"version": FORMAT_VERSION, "n": n, "kind": kind, "item_type": item_type,
            "labels": stamp_before, "dataset_py": src_hash, "bytes": pos,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(meta_path, "w", encoding="utf-8") as f:     # meta.json is written LAST = the pack is ready
        json.dump(meta, f, indent=1)
    print(f"OK: {n:,} lines, {pos / 1e9:.2f} GB, checked {len(check)} random lines - identical. "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pack dataset folders (labels.csv + images/) for RAM training.")
    ap.add_argument("folders", nargs="+", help="folders with labels.csv + images/")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--verify", type=int, default=300, help="how many random lines to compare after packing")
    a = ap.parse_args()
    for folder in a.folders:
        pack(folder, a.workers, a.verify)
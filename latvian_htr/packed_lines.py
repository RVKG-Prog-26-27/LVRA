"""
packed_lines.py - держать готовый набор строк (labels.csv + images/) в оперативной памяти.

Зачем: при обучении из папки каждую эпоху открываются и декодируются десятки тысяч PNG.
Этот скрипт ОДИН РАЗ прогоняет набор через ваш же LineFolderDataset (dataset.py) и сохраняет
готовые uint8-картинки одним файлом в подпапку <набор>/packed/. При обучении этот файл
читается через memory-map: Windows держит его в RAM один раз, общим для всех воркеров.

Как пользоваться:
    # 1) упаковать (один раз после каждой генерации набора):
    python packed_lines.py data/synth_100k
    python packed_lines.py data/synth_100k data/synth_val      # можно несколько папок сразу

    # 2) обучать как раньше - указываете ТУ ЖЕ папку:
    python train.py --synth-data data/synth_100k

train.py сам видит data/synth_100k/packed/ и берёт данные оттуда. Упаковка НЕ используется
(и читаются PNG, как раньше), если:
    * labels.csv изменился после упаковки (например, набор сгенерирован заново);
    * изменился dataset.py, алфавит (alphabet.py) или TARGET_HEIGHT;
    * проверка после упаковки показала расхождение.
В этих случаях train.py печатает, почему, и подсказывает команду для переупаковки.
Чтобы отказаться от упаковки совсем - просто удалите папку packed/.
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
    """Отпечаток всего, от чего зависят упакованные картинки: код dataset.py, TARGET_HEIGHT
    и набор символов алфавита (строки с неизвестными символами LineFolderDataset пропускает).
    Если что-то из этого поменяли, старая упаковка становится недействительной."""
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
    """Отдаёт РОВНО то же, что LineFolderDataset(folder, as_uint8=True), но из одного файла в RAM."""

    def __init__(self, folder: str, preload: bool = False):
        d = _pack_dir(folder)
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        idx = np.load(os.path.join(d, "index.npz"))
        self.offsets, self.shapes, self._widths = idx["offsets"], idx["shapes"], idx["widths"]
        with open(os.path.join(d, "extras.pkl"), "rb") as f:
            self.extras = pickle.load(f)
        self.kind = meta["kind"]                 # "tensor" или "numpy"
        self.as_list = meta["item_type"] == "list"
        self.bin_path = os.path.join(d, "pixels.bin")
        # preload=True: весь файл копируется в память ЭТОГО процесса. Только при --num-workers 0,
        # иначе на Windows каждый воркер получит свою полную копию.
        self.preload = preload
        self._data = np.fromfile(self.bin_path, dtype=np.uint8) if preload else None

    def __len__(self):
        return len(self.extras)

    def widths(self):
        return [int(w) for w in self._widths]

    def _buf(self):
        if self._data is None:                   # открывается лениво, отдельно в каждом воркере
            self._data = np.memmap(self.bin_path, dtype=np.uint8, mode="r")
        return self._data

    def __getitem__(self, i):
        o = int(self.offsets[i])
        shape = tuple(int(s) for s in self.shapes[i])
        count = int(np.prod(shape))
        a = np.array(self._buf()[o:o + count]).reshape(shape)    # копия: картинку можно менять
        img = torch.from_numpy(a) if self.kind == "tensor" else a
        item = (img,) + tuple(self.extras[i])
        return list(item) if self.as_list else item

    def __getstate__(self):
        # memmap НЕ передаём в воркеры (pickle скопировал бы гигабайты); каждый воркер откроет файл сам
        state = self.__dict__.copy()
        if not self.preload:
            state["_data"] = None
        return state


# ------------------------------------------------------------------ status / open

def pack_status(folder: str):
    """(True, n) если упаковка актуальна, иначе (False, причина)."""
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
    """Вместо LineFolderDataset(folder, as_uint8=True): берёт упаковку, если она актуальна, иначе PNG."""
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
        os.remove(meta_path)                     # пока пакуем, старая упаковка недействительна

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
                sys.exit(f"LineFolderDataset returns {type(item).__name__}, expected tuple/list - send dataset.py")
            img, rest = item[0], tuple(item[1:])
            if isinstance(img, torch.Tensor):
                k, a = "tensor", img.numpy()
            elif isinstance(img, np.ndarray):
                k, a = "numpy", img
            else:
                sys.exit(f"First element is {type(img).__name__}, expected tensor/array - send dataset.py")
            if a.dtype != np.uint8:
                sys.exit(f"Image dtype is {a.dtype}, expected uint8 (as_uint8=True) - send dataset.py")
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

    # проверка: упакованное == то, что сейчас отдаёт LineFolderDataset
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
        print("!! The pack will NOT be used. Send dataset.py so the augmentation can be moved after packing.")
        sys.exit(1)

    meta = {"version": FORMAT_VERSION, "n": n, "kind": kind, "item_type": item_type,
            "labels": stamp_before, "dataset_py": src_hash, "bytes": pos,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(meta_path, "w", encoding="utf-8") as f:     # meta.json пишется ПОСЛЕДНИМ = упаковка готова
        json.dump(meta, f, indent=1)
    print(f"OK: {n:,} lines, {pos / 1e9:.2f} GB, checked {len(check)} random lines - identical. "
          f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pack dataset folders (labels.csv + images/) for RAM training.")
    ap.add_argument("folders", nargs="+", help="папки с labels.csv + images/")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--verify", type=int, default=300, help="сколько случайных строк сверить после упаковки")
    a = ap.parse_args()
    for folder in a.folders:
        pack(folder, a.workers, a.verify)
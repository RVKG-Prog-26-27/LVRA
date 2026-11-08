"""
"Datu nodrošinātāja" klases apmācībai:
- SyntheticHTRDataset: nepārtraukti un dinamiski ģenerē sintētiskus piemērus (izmanto evaluate.py).
- LineFolderDataset (=RealHTRDataset): nolasa gatavas rindas no mapes (labels.csv + attēli):
  реальные сканы или заранее сгенерированная синтетика (make_synth_dataset.py).
- BucketBatchSampler: собирает батчи из строк ПОХОЖЕЙ ширины (bucketing), чтобы GPU не считал паддинг.
- SyntheticBucketStream: альтернативный режим - бесконечная генерация синтетики прямо в воркерах.
Funkcija `collate_batch` saskaņo attēlus pēc platuma.

Скорость:
  * картинки передаются как uint8 (в 4 раза меньше, чем float32), в float переводятся уже на GPU;
  * ширина батча округляется вверх до кратного PAD_TO - так у cuDNN мало разных форм и
    cudnn.benchmark может закэшировать самый быстрый алгоритм для каждой.
"""
import csv
import os
import random

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, IterableDataset, Sampler

from alphabet import CHAR_TO_IDX, encode
from synthetic_data import TARGET_HEIGHT, load_words, make_sample

PAD_TO = 32  # ширина батча округляется вверх до кратного этому числу


def image_to_uint8(img: Image.Image) -> torch.Tensor:
    """Pillow-картинка -> uint8 тензор [1, H, W], где фон = 0, чернила ~ 255.
    autocontrast: pelēks/tumšs fons (foto, skenējumi) -> tuvu baltam, tinte -> tuvu melnai.
    Apgriežam spilgtumu (255 - x): «fons = 0, tinte ~ 255» - tīklam «ir tinte» = liels skaitlis."""
    img = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    arr = 255 - np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    """Neironu tīklam pillow attēlu pārvērš skaitļu tenzorā no 0 līdz 1 (fons 0, tinte ~1), forma [1, H, W].
    Tas pats solis darbojas gan apmācībā, gan secināšanā (infer.py, evaluate.py)."""
    return image_to_uint8(img).float() / 255.0


def resize_to_height(img: Image.Image) -> Image.Image:
    """Mērogojam līdz TARGET_HEIGHT, saglabājot proporcijas."""
    w, h = img.size
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    return img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)


class SyntheticHTRDataset(Dataset):
    """Mākslīgu piemēru noliktava: katru reizi tā zīmē jaunus, nevis glabā vecos."""
    def __init__(self, length: int = 2000, corpus: list[str] | None = None):
        self.length = length
        self.corpus = corpus or load_words()   # vārdu saraksts (nevis gatavas rindas)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, text = make_sample(self.corpus)
        return image_to_tensor(img), text   # atgriežam pāri (tenzors, teksts)


class LineFolderDataset(Dataset):
    """Mape ar gatavām rindām: labels.csv (filename,text[,width]) + attēli.
    Attēli var būt apakšmapē images/ vai turpat blakus labels.csv.
    Подходит и для реальных сканов, и для заранее сгенерированной синтетики (make_synth_dataset.py).

    as_uint8=True - для обучения: отдаёт (uint8 тензор, текст), как ждёт collate_batch.
    as_uint8=False - для evaluate.py/infer: (float тензор 0..1, текст)."""

    def __init__(self, root: str, as_uint8: bool = False):
        self.root = root
        self.as_uint8 = as_uint8
        self.samples: list[tuple[str, str]] = []
        self._csv_widths: list[int] | None = []
        labels_path = os.path.join(root, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"file not found {labels_path} with columns 'filename,text'. "
            )
        with open(labels_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append((row["filename"], row["text"]))
                if self._csv_widths is not None:
                    w = row.get("width")
                    if w:
                        self._csv_widths.append(int(w))
                    else:
                        self._csv_widths = None      # нет колонки width - узнаем ширину из файлов
        self._drop_unknown_chars()
        img_dir = os.path.join(root, "images")
        self.img_dir = img_dir if os.path.isdir(img_dir) else root

    def _drop_unknown_chars(self):
        """Строки с символами, которых нет в alphabet.py, пропускаем (иначе обучение упадёт посередине)."""
        keep, bad = [], []
        for k, (fn, text) in enumerate(self.samples):
            unknown = {c for c in text if c not in CHAR_TO_IDX}
            (bad if unknown else keep).append((k, fn, unknown))
        if bad:
            print(f"WARNING: {self.root}: пропущено {len(bad)} строк с символами не из алфавита "
                  f"(добавьте их в alphabet.py или исправьте подписи):")
            for _, fn, unknown in bad[:10]:
                print(f"    {fn}: {''.join(sorted(unknown))!r}")
            ids = [k for k, _, _ in keep]
            self.samples = [self.samples[k] for k in ids]
            if self._csv_widths is not None and self._csv_widths:
                self._csv_widths = [self._csv_widths[k] for k in ids]

    def __len__(self):
        return len(self.samples)

    def load(self, idx) -> tuple[Image.Image, str]:
        """Картинка, уже приведённая к высоте TARGET_HEIGHT, и её текст."""
        filename, text = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, filename)).convert("L")
        if img.size[1] != TARGET_HEIGHT:
            img = resize_to_height(img)
        return img, text

    def widths(self) -> list[int]:
        """Ширина каждой картинки после приведения к высоте TARGET_HEIGHT (нужна для bucketing).
        Берётся из колонки width, иначе читается только заголовок файла (быстро, без декодирования)."""
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


# Vecais nosaukums - lai evaluate.py un citi skripti strādātu kā agrāk.
RealHTRDataset = LineFolderDataset


class BucketBatchSampler(Sampler):
    """Выдаёт номера примеров батчами БЕСКОНЕЧНО, эпоха за эпохой.

    Каждая эпоха: перемешиваем все примеры, режем на куски по batch_size*pool_batches,
    внутри куска сортируем по ширине и режем на батчи, потом перемешиваем порядок батчей.
    Итог: каждый пример ровно один раз за эпоху, но в батче - строки почти одинаковой ширины."""

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
    Apvieno vairākus piemērus vienā "batch". batch - saraksts no (uint8 attēls [1,H,W], teksts).

    Возвращает:
      images         uint8 [B, 1, H, W_pad] (фон и паддинг = 0)
      targets        long  [сумма длин]  - все тексты подряд в виде номеров символов (для CTC)
      target_lengths long  [B]
      widths         long  [B]           - настоящая ширина каждой картинки ДО паддинга
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

    # CTC «pareizās atbildes» gatavojam jau šeit (воркер), а не в главном цикле.
    encoded = [encode(t) for t in texts]
    targets = torch.tensor([c for e in encoded for c in e], dtype=torch.long)
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    return padded, targets, target_lengths, widths, list(texts)


class SyntheticBucketStream(IterableDataset):
    """Бесконечный поток ГОТОВЫХ батчей для обучения.

    Каждый воркер держит «пул» из pool_size готовых примеров. Чтобы выдать батч, он:
      1) дорисовывает пул до полного размера (ровно batch_size новых примеров на каждый батч -
         работа идёт равномерно, без пауз);
      2) сортирует пул по ширине, берёт случайный пример и batch_size соседей по ширине;
      3) отдаёт их одним батчем.
    В батч попадают строки почти одинаковой ширины -> паддинга почти нет -> GPU считает только полезное.

    real: RealHTRDataset или None; real_fraction - доля реальных примеров в потоке.
    """

    def __init__(self, batch_size: int, real: RealHTRDataset | None = None,
                 real_fraction: float = 0.0, pool_batches: int = 16):
        self.batch_size = batch_size
        self.real = real
        self.real_fraction = real_fraction if real is not None and len(real) > 0 else 0.0
        self.pool_size = batch_size * pool_batches

    def _make(self, words):
        if self.real_fraction and random.random() < self.real_fraction:
            img, text = self.real.load(random.randrange(len(self.real)))
        else:
            img, text = make_sample(words)
        return image_to_uint8(img), text

    def __iter__(self):
        # Random в каждом воркере DataLoader уже засеян по-разному (base_seed + worker_id),
        # поэтому воркеры не повторяют друг друга, а запуск с тем же seed воспроизводим.
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

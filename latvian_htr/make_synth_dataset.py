"""
Заранее генерирует синтетический набор на диск, в том же формате, что и реальные данные:

    <out>/images/000001.png ...
    <out>/labels.csv          (filename,text,width)

Kā skriet:
    python make_synth_dataset.py --out data/synth_30k --n 30000
    python make_synth_dataset.py --out data/synth_val --n 1000 --seed 1   # отдельный набор для проверки

Генерация идёт параллельно на всех ядрах процессора. Один и тот же --seed даёт один и тот же набор.
Колонка width (ширина картинки) нужна train.py, чтобы собирать батчи из строк похожей ширины
без открытия всех файлов при старте.
"""

import argparse
import csv
import os
import random
import time
from multiprocessing import Pool

CHUNK = 250  # сколько картинок генерирует один воркер за одно задание


def _work(job):
    """Генерирует картинки с номерами [first, first+count) и сохраняет их. Возвращает строки для labels.csv."""
    first, count, out_dir, seed = job
    from synthetic_data import load_words, make_sample   # импорт внутри воркера (Windows/spawn)

    random.seed(seed * 1_000_003 + first)     # у каждого куска своё зерно -> результат не зависит от числа воркеров
    words = _work.words if hasattr(_work, "words") else load_words()
    _work.words = words
    rows = []
    for i in range(first, first + count):
        img, text = make_sample(words)
        name = f"{i + 1:06d}.png"
        img.save(os.path.join(out_dir, "images", name), compress_level=1)   # быстрое сжатие
        rows.append((name, text, img.size[0]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="папка для набора")
    parser.add_argument("--n", type=int, default=30000, help="сколько картинок сгенерировать")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    jobs = [(s, min(CHUNK, args.n - s), args.out, args.seed) for s in range(0, args.n, CHUNK)]
    rows = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, part in enumerate(pool.imap(_work, jobs), 1):     # imap сохраняет порядок
            rows += part
            if k % 10 == 0 or k == len(jobs):
                done = len(rows)
                el = time.time() - t0
                print(f"{done}/{args.n}  {el:.0f}s  (ещё ~{el / done * (args.n - done):.0f}s)")

    with open(os.path.join(args.out, "labels.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text", "width"])
        w.writerows(rows)
    print(f"Готово: {len(rows)} картинок в {args.out}")


if __name__ == "__main__":
    main()

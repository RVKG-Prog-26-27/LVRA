"""
Generates a synthetic set in advance and saves it to disk in the same format as the real data:

    <out>/images/000001.png ...
    <out>/labels.csv          (filename,text,width)

How to run:
    python make_synth_dataset.py --out data/synth_30k --n 30000
    python make_synth_dataset.py --out data/synth_val --n 1000 --seed 1   # a separate validation set

Generation runs in parallel on all CPU cores. The same --seed gives the same set.
The width column (image width) lets train.py build batches of lines of similar width without opening all the
files at start-up.
"""

import argparse
import csv
import os
import random
import time
from multiprocessing import Pool

CHUNK = 250  # how many images one worker generates per job


def _work(job):
    """Generates the images numbered [first, first+count) and saves them. Returns the rows for labels.csv."""
    first, count, out_dir, seed = job
    from synthetic_data import load_words, make_sample   # imported inside the worker (Windows / spawn)

    random.seed(seed * 1_000_003 + first)     # every chunk has its own seed -> the result does not depend on the number of workers
    words = _work.words if hasattr(_work, "words") else load_words()
    _work.words = words
    rows = []
    for i in range(first, first + count):
        img, text = make_sample(words)
        name = f"{i + 1:06d}.png"
        img.save(os.path.join(out_dir, "images", name), compress_level=1)   # fast compression
        rows.append((name, text, img.size[0]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="output folder of the set")
    parser.add_argument("--n", type=int, default=30000, help="how many images to generate")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    jobs = [(s, min(CHUNK, args.n - s), args.out, args.seed) for s in range(0, args.n, CHUNK)]
    rows = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for k, part in enumerate(pool.imap(_work, jobs), 1):     # imap keeps the order
            rows += part
            if k % 10 == 0 or k == len(jobs):
                done = len(rows)
                el = time.time() - t0
                print(f"{done}/{args.n}  {el:.0f}s  (~{el / done * (args.n - done):.0f}s left)")

    with open(os.path.join(args.out, "labels.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "text", "width"])
        w.writerows(rows)
    print(f"Done: {len(rows)} images in {args.out}")


if __name__ == "__main__":
    main()

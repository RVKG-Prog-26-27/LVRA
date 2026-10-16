"""
Novērtēšana: CER (Character Error Rate) = (aizvietojumi + izlaidumi + ielikumi) / pareizo simbolu skaits.
CER 0.05 = 5% simbolu kļūdu. Mazāk = labāk.

Kā skriet:
    python evaluate.py --checkpoint checkpoints/crnn_step20000.pt
    python evaluate.py --checkpoint checkpoints/crnn_step20000.pt --real-data data/real_dataset --synthetic 300

Svarīgi: reālos paraugus (data/real_dataset) labāk NEIEKĻAUT apmācībā, bet izmantot tikai šeit -
tā var godīgi redzēt, kā modelis strādā ar īstu rokrakstu.
"""

import argparse
import os
import random

import torch

from alphabet import NUM_CLASSES, decode_greedy
from dataset import RealHTRDataset, SyntheticHTRDataset
from model import CRNN


def edit_distance(a: str, b: str) -> int:
    """Levenšteina attālums: cik minimāli rakstzīmju labojumu vajag, lai a pārvērstu par b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    total_edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total_chars = sum(len(r) for r in refs)
    return total_edits / max(total_chars, 1)


def evaluate(model, dataset, device, n: int | None = None, show: int = 5, title: str = ""):
    model.eval()
    n = min(n or len(dataset), len(dataset))
    refs, hyps = [], []
    with torch.no_grad():
        for i in range(n):
            img, text = dataset[i]
            log_probs = model(img.unsqueeze(0).to(device))          # [T, 1, C]
            idx = log_probs.argmax(dim=2).squeeze(1).tolist()
            refs.append(text)
            hyps.append(decode_greedy(idx))
    print(f"\n=== {title}: {n} paraugi, CER = {cer(refs, hyps):.3f}  "
          f"(precīzi sakrīt: {sum(r == h for r, h in zip(refs, hyps))}/{n}) ===")
    for r, h in list(zip(refs, hyps))[:show]:
        print(f"  pareizi : {r}\n  modelis : {h}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--real-data", type=str, default=os.path.join("data", "real_dataset"))
    parser.add_argument("--synthetic", type=int, default=200, help="cik svaigu sintētisko paraugu pārbaudīt (0 = nepārbaudīt)")
    args = parser.parse_args()

    random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if os.path.exists(os.path.join(args.real_data, "labels.csv")):
        evaluate(model, RealHTRDataset(args.real_data), device, title="REĀLIE paraugi")
    if args.synthetic > 0:
        evaluate(model, SyntheticHTRDataset(length=args.synthetic), device, n=args.synthetic, title="SINTĒTISKIE (svaigi)")

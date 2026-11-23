"""
Evaluation: CER (Character Error Rate) and WER (Word Error Rate), see metrics.py. CER 0.05 = 5% wrong characters.

How to run:
    python evaluate.py --checkpoint checkpoints/crnn_best.pt
    python evaluate.py --checkpoint checkpoints/crnn_best.pt --real-data data/real_dataset --synthetic 300
    python evaluate.py --checkpoint checkpoints/crnn_best.pt --beam-width 5      # also CTC beam search (slow)

Real lines: by default only the part held out from training is used (--real-split val; the same split as in
train.py, controlled by --real-val-fraction), because that is the only honest measurement. Use
--real-split all to measure on everything (the training part is then "known" to the model and the result is
too optimistic).
"""

import argparse
import os
import random

import torch
from torch.utils.data import Subset

from alphabet import NUM_CLASSES, decode_beam_search, decode_greedy
from dataset import DEFAULT_REAL_VAL_FRACTION, LineFolderDataset, SyntheticHTRDataset, split_real
from metrics import cer, wer
from model import load_model


def evaluate(model, dataset, device, n: int | None = None, show: int = 5, title: str = "", beam_width: int = 0):
    """Prints CER / WER / exact matches (greedy decoding; with beam_width > 0 also beam search) and a few examples."""
    model.eval()
    n = min(n or len(dataset), len(dataset))
    refs, greedy, beam = [], [], []
    with torch.no_grad():
        for i in range(n):
            img, text = dataset[i]
            log_probs = model(img.unsqueeze(0).to(device))          # [T, 1, C]
            idx = log_probs.argmax(dim=2).squeeze(1).tolist()
            refs.append(text)
            greedy.append(decode_greedy(idx))
            if beam_width > 0:
                beam.append(decode_beam_search(log_probs.squeeze(1), beam_width=beam_width))

    def line(name, hyps):
        return (f"{name}: CER = {cer(refs, hyps):.3f}  WER = {wer(refs, hyps):.3f}  "
                f"(exact matches: {sum(r == h for r, h in zip(refs, hyps))}/{n})")

    print(f"\n=== {title}: {n} samples ===")
    print("  " + line("greedy", greedy))
    if beam_width > 0:
        print("  " + line(f"beam{beam_width}", beam))
    best = beam if beam_width > 0 else greedy
    for r, h in list(zip(refs, best))[:show]:
        print(f"  correct: {r}\n  model  : {h}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--real-data", type=str, default=os.path.join("data", "real_dataset"))
    parser.add_argument("--real-split", type=str, default="val", choices=["val", "train", "all"],
                        help="which real lines to measure on: the held-out validation part (default), "
                             "the part used in training, or all of them")
    parser.add_argument("--real-val-fraction", type=float, default=DEFAULT_REAL_VAL_FRACTION,
                        help="must be the same value as the one used in train.py")
    parser.add_argument("--synthetic", type=int, default=200,
                        help="how many fresh synthetic samples to check (0 = skip)")
    parser.add_argument("--beam-width", type=int, default=0,
                        help="also measure CTC beam search with this width (0 = greedy only; beam search is slow)")
    parser.add_argument("--show", type=int, default=5, help="how many examples to print per set")
    args = parser.parse_args()

    random.seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load_model understands both checkpoint formats (and x_stride) and checks the alphabet size.
    model = load_model(args.checkpoint, NUM_CLASSES, device)

    if os.path.exists(os.path.join(args.real_data, "labels.csv")):
        real = LineFolderDataset(args.real_data)
        train_idx, val_idx = split_real(real, args.real_val_fraction)
        chosen = {"val": val_idx, "train": train_idx, "all": list(range(len(real)))}[args.real_split]
        if chosen:
            evaluate(model, Subset(real, chosen), device, show=args.show, beam_width=args.beam_width,
                     title=f"REAL ({args.real_split}, {len(chosen)} of {len(real)} lines)")
        else:
            print(f"No real lines in the '{args.real_split}' part (fraction {args.real_val_fraction}).")
    if args.synthetic > 0:
        evaluate(model, SyntheticHTRDataset(length=args.synthetic), device, n=args.synthetic, show=args.show,
                 beam_width=args.beam_width, title="SYNTHETIC (fresh)")

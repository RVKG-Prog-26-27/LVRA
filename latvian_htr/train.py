"""
Training script.

Usage:
    python train.py --steps 2000                     # synthetic data only
    python train.py --real-data path/to/real_dataset  # mix in real scans once you have them

Maps to the flowchart's "Projektu struktūra -> AI" step, trained against data
prepared in "Rokrakstu bāzes sagatavošana".
"""

import argparse
import os
import random

import torch
from torch.utils.data import ConcatDataset, DataLoader

from alphabet import NUM_CLASSES, encode
from dataset import RealHTRDataset, SyntheticHTRDataset, collate_batch
from model import CRNN


def build_dataset(real_data_path: str | None, synth_length: int):
    datasets = [SyntheticHTRDataset(length=synth_length)]
    if real_data_path:
        datasets.append(RealHTRDataset(real_data_path))
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(args.real_data, args.synth_per_epoch)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=args.num_workers,
    )

    model = CRNN(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    step = 0
    model.train()
    while step < args.steps:
        for images, texts, widths in loader:
            images = images.to(device)

            targets, target_lengths = [], []
            for t in texts:
                enc = encode(t)
                targets.extend(enc)
                target_lengths.append(len(enc))
            targets = torch.tensor(targets, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)

            log_probs = model(images)  # [T, B, C]
            T = log_probs.size(0)
            input_lengths = torch.full((images.size(0),), T, dtype=torch.long)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            step += 1
            if step % args.log_every == 0:
                print(f"step {step}/{args.steps}  loss={loss.item():.4f}")
            if step % args.checkpoint_every == 0 or step == args.steps:
                ckpt_path = os.path.join(args.checkpoint_dir, f"crnn_step{step}.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"saved checkpoint -> {ckpt_path}")
            if step >= args.steps:
                break

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synth-per-epoch", type=int, default=5000, help="synthetic samples generated per 'epoch' pass")
    parser.add_argument("--real-data", type=str, default=None, help="path to a RealHTRDataset root, once available")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    train(args)

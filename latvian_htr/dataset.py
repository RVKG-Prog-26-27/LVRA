"""
Dataset classes.

- SyntheticHTRDataset: infinite on-the-fly synthetic samples (current stand-in,
  "Sintētiskie dati" on the flowchart).
- RealHTRDataset: loads scanned line images once you have them
  ("Reāli dati" on the flowchart). Expects a folder with images plus a
  labels.csv of `filename,text` pairs - this is the "Rokrakstu bāzes
  sagatavošana" step from the diagram, done offline before training.

Both return (image_tensor[1,H,W], text_string). Collation pads variable-width
images to the batch max width, since line images differ in length.
"""

import csv
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from synthetic_data import TARGET_HEIGHT, generate_batch, load_corpus, render_text_line

import numpy as np


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    arr = 1.0 - arr  # invert: background 0, ink ~1 (easier for the CNN)
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]


class SyntheticHTRDataset(Dataset):
    """Generates `length` synthetic (image, text) pairs per epoch, freshly each time."""

    def __init__(self, length: int = 2000, corpus: list[str] | None = None):
        self.length = length
        self.corpus = corpus or load_corpus()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        import random

        text = random.choice(self.corpus)
        img = render_text_line(text)
        return image_to_tensor(img), text


class RealHTRDataset(Dataset):
    """
    Loads real scanned handwriting samples.

    Expected layout:
        root/
          labels.csv        # header: filename,text
          images/
            0001.png
            0002.png
            ...
    """

    def __init__(self, root: str):
        self.root = root
        self.samples: list[tuple[str, str]] = []
        labels_path = os.path.join(root, "labels.csv")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Expected {labels_path} with columns 'filename,text'. "
                "This is the 'Rokrakstu bāzes sagatavošana' step - prepare it offline."
            )
        with open(labels_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], row["text"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        img = Image.open(os.path.join(self.root, "images", filename)).convert("L")
        w, h = img.size
        new_w = max(1, int(w * (TARGET_HEIGHT / h)))
        img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
        return image_to_tensor(img), text


def collate_batch(batch):
    """Pad images to max width in batch; return images, texts, and original widths."""
    imgs, texts = zip(*batch)
    max_w = max(img.shape[-1] for img in imgs)
    padded = torch.zeros(len(imgs), 1, TARGET_HEIGHT, max_w)
    widths = torch.zeros(len(imgs), dtype=torch.long)
    for i, img in enumerate(imgs):
        w = img.shape[-1]
        padded[i, :, :, :w] = img
        widths[i] = w
    return padded, list(texts), widths

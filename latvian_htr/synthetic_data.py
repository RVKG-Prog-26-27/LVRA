"""
Synthetic data generator.

Maps to the flowchart's "Sintētiskie dati" branch: since we don't have scanned
handwriting yet, we render Latvian text as images and distort them (rotation,
shear, noise, blur, varying stroke width) to roughly approximate handwriting
variability. This is a placeholder, not a replacement for real data - swap in
scanned samples via `RealDataset` in dataset.py as soon as you have them.

For much better realism, drop a handwriting-style .ttf (e.g. any free cursive/
print handwriting font that supports Latvian diacritics: ā č ē ģ ī ķ ļ ņ š ū ž)
into FONT_DIR and it will be picked up automatically.
"""

import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

TARGET_HEIGHT = 32  # fixed line height fed into the CNN


def _available_fonts() -> list[str]:
    fonts = glob.glob(os.path.join(FONT_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_DIR, "*.otf"))
    return fonts if fonts else [FALLBACK_FONT]


def _default_corpus() -> list[str]:
    """Small built-in Latvian word/phrase list, used if data/corpus.txt is absent."""
    return [
        "Labdien", "paldies", "lūdzu", "sveiki", "atā", "jā", "nē",
        "Rīga", "Latvija", "valoda", "grāmata", "skola", "pilsēta",
        "saule", "lietus", "sniegs", "vējš", "koks", "upe", "jūra",
        "draugs", "ģimene", "māja", "ceļš", "darbs", "laiks",
        "Es mācos latviešu valodu.", "Šodien ir skaista diena.",
        "Viņš dzīvo Rīgā.", "Mums ir daudz darba.", "Kur ir tuvākā aptieka?",
        "Cik tas maksā?", "Es gribu kafiju.", "Rīt būs saulains laiks.",
        "Bērni spēlējas parkā.", "Šī grāmata ir ļoti interesanta.",
    ]


def load_corpus() -> list[str]:
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if lines:
            return lines
    return _default_corpus()


def render_text_line(text: str, font_path: str | None = None, augment: bool = True) -> Image.Image:
    """Render `text` as a single grayscale line image of fixed height."""
    font_path = font_path or random.choice(_available_fonts())
    font_size = random.randint(26, 40)
    font = ImageFont.truetype(font_path, font_size)

    # Measure text to size the canvas, with padding
    dummy = Image.new("L", (10, 10), color=255)
    bbox = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font)
    w = max(1, bbox[2] - bbox[0]) + 20
    h = max(1, bbox[3] - bbox[1]) + 20

    img = Image.new("L", (w, h), color=255)
    draw = ImageDraw.Draw(img)
    draw.text((10 - bbox[0], 10 - bbox[1]), text, font=font, fill=0)

    if augment:
        img = _augment(img)

    img = _resize_keep_ratio(img, TARGET_HEIGHT)
    return img


def _augment(img: Image.Image) -> Image.Image:
    # Slight random rotation
    angle = random.uniform(-3, 3)
    img = img.rotate(angle, expand=True, fillcolor=255)

    # Slight shear via affine transform, mimics slanted handwriting
    shear = random.uniform(-0.25, 0.25)
    w, h = img.size
    img = img.transform(
        (w + int(abs(shear) * h), h),
        Image.AFFINE,
        (1, shear, -shear * h if shear < 0 else 0, 0, 1, 0),
        fillcolor=255,
    )

    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    if random.random() < 0.5:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(3, 10), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def _resize_keep_ratio(img: Image.Image, target_height: int) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * (target_height / h)))
    return img.resize((new_w, target_height), Image.BILINEAR)


def generate_batch(n: int, corpus: list[str] | None = None) -> list[tuple[Image.Image, str]]:
    corpus = corpus or load_corpus()
    samples = []
    for _ in range(n):
        text = random.choice(corpus)
        samples.append((render_text_line(text), text))
    return samples


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    for i, (img, text) in enumerate(generate_batch(500)):
        img.save(f"samples/sample_{i}.png")
        print(f"sample_{i}.png -> {text!r}  size={img.size}")

"""
Synthetic data generator (v2).

What changed vs. v1 (see notes in the chat):
  * TARGET_HEIGHT 32 -> 64 and the text is rendered BIG and downscaled, so letters
    (and diacritics) stay readable. v1 gave ~5 px per character.
  * The canvas height comes from font metrics, not from the text's own bounding box,
    so letter size no longer changes from line to line.
  * Texts are random word sequences (random_line), not a few fixed lines.
  * Augmentation mimics photos of notebook pages: grid paper, grey/uneven light,
    wavy baseline, thicker/thinner ink, noise, blur.
"""

import functools
import glob
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from alphabet import ALPHABET

FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")

# Candidate paths for a default font that ships with the OS, checked in order.
# Must support Latvian diacritics (ā č ē ģ ī ķ ļ ņ š ū ž).
_FALLBACK_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
    "C:\\Windows\\Fonts\\arial.ttf",                      # Windows
    "C:\\Windows\\Fonts\\segoeui.ttf",                    # Windows
    "/System/Library/Fonts/Supplemental/Arial.ttf",       # macOS
    "/Library/Fonts/Arial.ttf",                           # macOS
]


def _find_fallback_font() -> str:
    for path in _FALLBACK_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No usable default font found and fonts/ is empty. "
        "Put at least one .ttf/.otf file (with Latvian diacritics support) into the fonts/ folder."
    )


FALLBACK_FONT = _find_fallback_font()

TARGET_HEIGHT = 64  # fixed line height fed into the CNN (model.py expects exactly 64)


def _available_fonts() -> list[str]:
    fonts = glob.glob(os.path.join(FONT_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_DIR, "*.otf"))
    return fonts if fonts else [FALLBACK_FONT]


def _glyph_bytes(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    img = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(img).text((10, 10), ch, font=font, fill=255)
    return img.tobytes()


@functools.lru_cache(maxsize=None)
def font_charset(font_path: str) -> frozenset:
    """Which characters of our alphabet this font can really draw.
    A missing glyph is drawn as an empty box (.notdef) while the label keeps the real letter -
    the network would then learn 'box = a / ā / š / ž ...', which is impossible. So we never
    put such characters into texts rendered with this font."""
    font = ImageFont.truetype(font_path, 48)
    notdef = _glyph_bytes(font, "\uFFFF")
    ok = {" "}
    for ch in ALPHABET[1:]:  # [0] is the CTC blank
        if ch != " " and _glyph_bytes(font, ch) != notdef:
            ok.add(ch)
    return frozenset(ok)


# --------------------------------------------------------------------------- texts

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


def load_words() -> list[str]:
    """All words (punctuation kept attached) from data/corpus.txt; fallback to the built-in list."""
    corpus_path = os.path.join(os.path.dirname(__file__), "data", "corpus.txt")
    words: list[str] = []
    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            words = f.read().split()
    if not words:
        words = [w for line in _default_corpus() for w in line.split()]
    return words


def load_corpus() -> list[str]:
    """Kept for backward compatibility - returns the word pool."""
    return load_words()


def _random_number() -> str:
    a, b = random.randint(0, 9999), random.randint(0, 99)
    return random.choice([str(a), f"{a},{b}", f"({a})", f"{a}.", f"{a}%", f"{b})", f"{a} - {b}"])


def random_line(words: list[str], max_chars: int = 60, allowed=None) -> str:
    """A random 'handwritten line': random words glued together, 6..max_chars characters.
    `allowed` - set of characters the font can draw; words/numbers with other characters are skipped."""
    if allowed is not None:
        words = [w for w in words if set(w) <= allowed] or [w for w in words if set(w) <= allowed | set("0123456789")]
        if not words:
            words = ["a"]
    target = random.randint(6, max_chars)
    parts: list[str] = []
    length = 0
    guard = 0
    while length < target and guard < 200:
        guard += 1
        w = _random_number() if random.random() < 0.07 else random.choice(words)
        if allowed is not None and not set(w) <= allowed:
            continue
        extra = len(w) + (1 if parts else 0)
        if length + extra > max_chars:
            if parts:
                break
            continue
        parts.append(w)
        length += extra
    return " ".join(parts) if parts else random.choice(words)[:max_chars]


def make_sample(words: list[str], max_chars: int = 60, augment: bool = True) -> tuple[Image.Image, str]:
    """Pick a random font, build a text this font can fully draw, render it. Returns (image, text)."""
    fonts = _available_fonts()
    font_path = random.choice(fonts)
    allowed = font_charset(font_path)
    text = random_line(words, max_chars, allowed)
    return render_text_line(text, font_path=font_path, augment=augment), text


# --------------------------------------------------------------------------- rendering

def render_text_line(text: str, font_path: str | None = None, augment: bool = True) -> Image.Image:
    """Render `text` as a grayscale line image of height TARGET_HEIGHT."""
    font_path = font_path or random.choice(_available_fonts())
    font_size = random.randint(56, 88)  # big on purpose; downscaled at the end
    font = ImageFont.truetype(font_path, font_size)
    ascent, descent = font.getmetrics()

    dummy = ImageDraw.Draw(Image.new("L", (10, 10), color=255))
    bbox = dummy.textbbox((0, 0), text, font=font)

    pad_x = int(font_size * random.uniform(0.15, 0.5))
    pad_y = int(font_size * random.uniform(0.15, 0.3))
    top = min(0, bbox[1])
    bottom = max(ascent + descent, bbox[3])
    w = max(1, bbox[2] - bbox[0]) + 2 * pad_x
    h = (bottom - top) + 2 * pad_y

    ink = random.randint(0, 90)
    img = Image.new("L", (w, h), color=255)
    ImageDraw.Draw(img).text((pad_x - bbox[0], pad_y - top), text, font=font, fill=ink)

    if augment:
        img = _augment(img)

    w, h = img.size
    new_w = max(1, int(w * (TARGET_HEIGHT / h)))
    return img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)


def _paper_background(w: int, h: int) -> np.ndarray:
    """Grey-ish paper, optionally with notebook grid lines (like the real scans)."""
    level = random.randint(185, 255)
    bg = np.full((h, w), level, dtype=np.float32)

    if random.random() < 0.55:  # grid paper
        cell = max(8, int(h * random.uniform(0.18, 0.4)))
        thick = max(1, int(round(h / 70)))
        dark = random.randint(25, 80)
        ox, oy = random.randint(0, cell - 1), random.randint(0, cell - 1)
        for x in range(ox, w, cell):
            bg[:, x:x + thick] -= dark
        for y in range(oy, h, cell):
            bg[y:y + thick, :] -= dark
    return np.clip(bg, 0, 255)


def _augment(img: Image.Image) -> Image.Image:
    # thicker ink (MinFilter grows dark strokes on a light background)
    if random.random() < 0.3:
        img = img.filter(ImageFilter.MinFilter(3))

    # rotation + shear (slanted handwriting)
    img = img.rotate(random.uniform(-2, 2), expand=True, fillcolor=255)
    shear = random.uniform(-0.3, 0.3)
    w, h = img.size
    img = img.transform(
        (w + int(abs(shear) * h), h),
        Image.AFFINE,
        (1, shear, -shear * h if shear < 0 else 0, 0, 1, 0),
        fillcolor=255,
    )

    # wavy baseline: shift every column up/down by a slow sine wave
    w, h = img.size
    amp = int(random.uniform(0, 0.05) * h)
    if amp > 0:
        img = ImageOps.expand(img, border=(0, amp, 0, amp), fill=255)
        arr = np.array(img)
        wavelength = random.uniform(300, 1200)
        phase = random.uniform(0, 2 * np.pi)
        shifts = np.round(amp * np.sin(2 * np.pi * np.arange(arr.shape[1]) / wavelength + phase)).astype(int)
        for x in range(arr.shape[1]):
            arr[:, x] = np.roll(arr[:, x], shifts[x])
        img = Image.fromarray(arr)

    # paper + uneven lighting, ink stays darker than paper
    w, h = img.size
    text = np.array(img).astype(np.float32)
    paper = _paper_background(w, h)
    out = np.minimum(text, paper)
    light = np.linspace(random.uniform(0.75, 1.0), random.uniform(0.75, 1.0), w, dtype=np.float32)
    out = out * light[None, :]
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    if random.random() < 0.6:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.8)))

    if random.random() < 0.6:
        arr = np.array(img).astype(np.float32)
        arr += np.random.normal(0, random.uniform(2, 12), arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    return img


def generate_batch(n: int, words: list[str] | None = None) -> list[tuple[Image.Image, str]]:
    words = words or load_words()
    samples = []
    for _ in range(n):
        samples.append(make_sample(words))
    return samples


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    for i, (img, text) in enumerate(generate_batch(10000)):
        img.save(f"samples/sample_{i}.png")
        print(f"sample_{i}.png -> {text!r}  size={img.size}")
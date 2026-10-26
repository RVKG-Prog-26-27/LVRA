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
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from alphabet import ALPHABET, STRIKE_MARK

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


def random_tokens(words: list[str], max_chars: int = 60, allowed=None) -> list[str]:
    """A random 'handwritten line' as a list of tokens (words/numbers), 6..max_chars characters long.
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
    return parts if parts else [random.choice(words)[:max_chars]]


def random_line(words: list[str], max_chars: int = 60, allowed=None) -> str:
    return " ".join(random_tokens(words, max_chars, allowed))


def _pick_strike() -> str:
    return random.choices(list(STRIKE_WEIGHTS), weights=list(STRIKE_WEIGHTS.values()))[0]


def make_sample(words: list[str], max_chars: int = 60, augment: bool = True) -> tuple[Image.Image, str]:
    """Pick a random font, build a text this font can fully draw, render it. Returns (image, label).
    Some lines get crossed-out words (STRIKE_LINE_PROB). Label: STRIKE_MODE 'mark' -> the struck word is
    replaced by STRIKE_MARK (¤); 'skip' -> the struck word is simply missing from the label."""
    fonts = _available_fonts()
    font_path = random.choice(fonts)
    allowed = font_charset(font_path)
    tokens = random_tokens(words, max_chars, allowed)

    kinds: list = [None] * len(tokens)
    if len(tokens) >= 2 and random.random() < STRIKE_LINE_PROB:
        k = min(len(tokens) - 1, random.choice([1, 1, 1, 2, 2, 3]))      # always leave some real text
        for i in random.sample(range(len(tokens)), k):
            if " " not in tokens[i]:                                      # not the multi-word numbers
                kinds[i] = _pick_strike()

    text = " ".join(tokens)
    word_kinds = [kd for tok, kd in zip(tokens, kinds) for _ in tok.split()]   # one entry per drawn word
    if STRIKE_MODE == "mark":
        label = " ".join(STRIKE_MARK if kd else tok for tok, kd in zip(tokens, kinds))
    else:
        label = " ".join(tok for tok, kd in zip(tokens, kinds) if not kd)
    img = render_text_line(text, font_path=font_path, augment=augment, struck=word_kinds)
    return img, label


# --------------------------------------------------------------------------- rendering

# --------------------------------------------------------------------------- handwritten look
# Letter-by-letter rendering that imitates a real hand:
#   * every letter gets its own size / vertical shift / tilt, and these drift smoothly from letter to
#     letter (neighbours look alike, like in real writing, instead of being independent noise);
#   * spacing depends on the SHAPES of the two neighbours (silhouette 'kerning'): round letters nest closer,
#     tall / overhanging ones keep more distance; every word also gets its own spacing style;
#   * neighbouring letters of a word can be joined with a pen stroke (cursive);
#   * punctuation is much sloppier than letters: bigger size / position / tilt scatter, wilder gaps.

HANDWRITTEN_PROB = 0.8   # share of lines drawn letter-by-letter (the rest = plain font rendering)
CONNECT_PROB = 0.5       # of those, share of lines whose letters are joined by strokes
JOIN_PROB = 0.85         # in a joined line: chance that two neighbouring letters get joined

# crossed-out words (people strike words out when they write)
STRIKE_LINE_PROB = 0.2    # share of lines that contain at least one crossed-out word
STRIKE_MODE = "mark"      # "mark": label gets STRIKE_MARK (¤) in place of the word; "skip": word is left out
STRIKE_WEIGHTS = {"line": 0.35,       # one line through the word
                  "scribble": 0.25,   # word smeared over with dense zigzag / coil strokes
                  "cross": 0.20,      # crosses (X), several for a long word
                  "multi": 0.20}      # several roughly parallel lines

# scatter strengths. scale/dy are relative (dy in font sizes), rot in degrees
LETTER_JITTER = {"scale": 0.07, "dy": 0.03, "rot": 4.0, "gap": 0.03}
PUNCT_JITTER = {"scale": 0.25, "dy": 0.12, "rot": 20.0, "gap": (-0.05, 0.28)}
WORD_GAP_RANGE = (0.03, 0.15)      # base distance between letters of a word (in font sizes), per word


@functools.lru_cache(maxsize=512)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


def _glyph_mask(font: ImageFont.FreeTypeFont, ch: str, fs: int):
    """Ink of one glyph: (mask[h,w] uint8, left, top); left/top are relative to the glyph origin
    (x = pen position, y = baseline), so top < 0 means 'above the baseline'."""
    ox, oy = int(fs * 1.0), int(fs * 2.2)
    img = Image.new("L", (int(fs * 3), int(fs * 3.2)), 0)
    ImageDraw.Draw(img).text((ox, oy), ch, font=font, fill=255, anchor="ls")
    arr = np.array(img)
    ys, xs = np.where(arr > 30)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return arr[y0:y1, x0:x1], int(x0 - ox), int(y0 - oy)


def _stem_width(font: ImageFont.FreeTypeFont, fs: int) -> int:
    """Pen thickness of the font = width of the vertical stem of 'l'."""
    g = _glyph_mask(font, "l", fs)
    if g is None:
        return max(2, int(fs * 0.06))
    mask = g[0]
    row = mask[mask.shape[0] // 2] > 128
    return int(min(max(2, row.sum()), fs * 0.12))


def _transform(mask: np.ndarray, top: int, scale: float, rot: float):
    """Scale the glyph around the pen origin and tilt it around its centre. Returns (mask, top)."""
    img = Image.fromarray(mask)
    h, w = mask.shape
    if abs(scale - 1) > 0.01:
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        top, w, h = round(top * scale), nw, nh
    if abs(rot) > 0.3:
        img = img.rotate(rot, expand=True, resample=Image.BILINEAR)
        nw, nh = img.size
        top += (h - nh) / 2
    return np.array(img), int(round(top))


def _profiles(mask: np.ndarray):
    """Per row: x of the leftmost / rightmost ink pixel (-1 where the row is empty)."""
    ink = mask > 100
    has = ink.any(axis=1)
    left = np.where(has, ink.argmax(axis=1), -1)
    right = np.where(has, mask.shape[1] - 1 - ink[:, ::-1].argmax(axis=1), -1)
    return left, right


def _place_x(prev: dict, cur: dict, gap_px: float) -> float:
    """Silhouette spacing: slide `cur` towards `prev` until their outlines are `gap_px` apart."""
    lo = max(prev["top"], cur["top"])
    hi = min(prev["top"] + prev["h"], cur["top"] + cur["h"])
    X = prev["right"] + gap_px
    if hi > lo:
        ys = np.arange(lo, hi)
        pr = prev["rprof"][ys - prev["top"]]
        cl = cur["lprof"][ys - cur["top"]]
        ok = (pr >= 0) & (cl >= 0)
        if ok.any():
            X = prev["X"] + float((pr[ok] - cl[ok]).max()) + gap_px
    return max(X, prev["X"] + 0.3 * prev["w"])      # never let letters sink into each other


def _edge_point(mask: np.ndarray, top: int, side: str):
    """Where a stroke leaves (right) / enters (left) a glyph: (col, row) inside the mask, or None.
    We take the lowest ink point (not below the baseline) in the outermost columns - for 'n' that is
    the foot of the stem, for 'o' / 'v' the point where real handwriting continues.
    None = the edge of the glyph is only a descender (g, j, y ...) -> no join there."""
    h, w = mask.shape
    s = max(2, int(0.1 * w))
    cols = slice(0, s) if side == "left" else slice(w - s, w)
    rows = np.where((mask[:, cols] > 128).any(axis=1))[0]
    above = rows[(top + rows) <= 2]           # not below the baseline
    if len(above) == 0:
        return None
    return (0 if side == "left" else w - 1), int(above.max())


def _bezier(p0, p1, ctrl, n=14):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p1[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p1[1])
            for t in np.linspace(0, 1, n)]


def _x_height(font: ImageFont.FreeTypeFont, fs: int) -> int:
    g = _glyph_mask(font, "x", fs)
    return g[0].shape[0] if g is not None else int(fs * 0.5)


def _wobbly(p0, p1, amp: float, step: float):
    """A hand-drawn straight-ish stroke: points on the segment p0-p1 with small sideways noise."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    n = max(2, int(length / step))
    nx, ny = -dy / length, dx / length
    pts = []
    for k in range(n + 1):
        off = 0.0 if k in (0, n) else random.gauss(0, amp)
        pts.append((p0[0] + dx * k / n + nx * off, p0[1] + dy * k / n + ny * off))
    return pts


def _dab_path(draw: ImageDraw.ImageDraw, pts, width: float, ox: float, oy: float):
    """Draw a pen stroke along pts as dense round dabs (smooth, no jagged polyline)."""
    r = width / 2
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        steps = max(1, int(max(abs(xb - xa), abs(yb - ya)) / max(1.0, r / 2)))
        for k in range(steps + 1):
            px = ox + xa + (xb - xa) * k / steps
            py = oy + ya + (yb - ya) * k / steps
            draw.ellipse((px - r, py - r, px + r, py + r), fill=255)


STRIKE_WIDTH = {"line": (0.8, 1.3), "cross": (0.7, 1.1), "multi": (0.55, 0.9), "scribble": (0.5, 0.8)}  # x pen stem


def _strike_paths(kind: str, x0: float, x1: float, xh: float, fs: int) -> list:
    """Strokes that cross out one word spanning x0..x1 (coordinates: x from the left, y from the baseline)."""
    x0 -= fs * random.uniform(0.03, 0.12)
    x1 += fs * random.uniform(0.03, 0.12)
    w = x1 - x0
    y_hi, y_lo = -xh * 1.15, xh * 0.2          # band that covers the body of the letters
    step, amp = 0.25 * fs, 0.02 * fs
    paths = []
    if kind == "line":                          # one line through the word
        ym = -xh * random.uniform(0.35, 0.6)
        tilt = random.uniform(-0.15, 0.15) * xh
        paths.append(_wobbly((x0, ym - tilt), (x1, ym + tilt), amp, step))
    elif kind == "multi":                       # 2-4 roughly parallel lines
        k = random.randint(2, 4)
        slope = random.uniform(-0.2, 0.2) * xh
        for y in np.linspace(-xh * 0.95, -xh * 0.05, k):
            y += random.gauss(0, 0.07 * xh)
            xa, xb = x0 + random.uniform(0, 0.06) * fs, x1 - random.uniform(0, 0.06) * fs
            paths.append(_wobbly((xa, y - slope), (xb, y + slope), amp, step))
    elif kind == "cross":                       # X, several in a chain for a long word
        n = int(np.clip(round(w / (1.1 * fs)), 1, 6))
        cw = w / n
        for i in range(n):
            cx0 = x0 + i * cw
            cx1 = cx0 + cw * random.uniform(0.92, 1.05)
            j = lambda: random.uniform(-0.1, 0.1) * xh
            paths.append(_wobbly((cx0, y_hi + j()), (cx1, y_lo + j()), amp, step))
            paths.append(_wobbly((cx0, y_lo + j()), (cx1, y_hi + j()), amp, step))
    else:                                       # "scribble": the word is smeared over
        if random.random() < 0.5:               # dense zigzag while moving right
            n = max(4, int(w / (random.uniform(0.07, 0.13) * fs)))
            paths.append([(x0 + w * i / n, (y_hi if i % 2 == 0 else y_lo) + random.gauss(0, 0.1 * xh))
                          for i in range(n + 1)])
        else:                                   # coil: horizontal passes back and forth, going down
            passes = random.randint(4, 7)
            pts = []
            for i in range(passes):
                y = y_hi + (y_lo - y_hi) * i / (passes - 1) + random.gauss(0, 0.05 * xh)
                xa, xb = (x0, x1) if i % 2 == 0 else (x1, x0)
                pts += [(xa + random.gauss(0, 0.03 * fs), y), (xb + random.gauss(0, 0.03 * fs), y + random.gauss(0, 0.1 * xh))]
            paths.append(pts)
    return paths


@functools.lru_cache(maxsize=1024)
def _vertical_extent(font_path: str, fs: int) -> tuple:
    """(top, bottom) of the ink of tall and descending letters of this font, relative to the baseline
    (top < 0). Used instead of font.getmetrics(): script fonts often reserve 40-90% more height than
    their letters need, which made the text tiny inside the 64 px line (letters looked small & narrow)."""
    font = _load_font(font_path, fs)
    cs = font_charset(font_path)
    top, bottom = 0, 0
    for ch in "bdfhklpqgjyBDFHKLĀŠŽ":
        if ch in cs:
            g = _glyph_mask(font, ch, fs)
            if g is not None:
                top = min(top, g[2])
                bottom = max(bottom, g[2] + g[0].shape[0])
    if bottom - top < fs * 0.5:                       # font without those letters: fall back to metrics
        a, d = font.getmetrics()
        return -a, d
    return top, bottom


def _render_handwritten(text: str, font: ImageFont.FreeTypeFont, fs: int, ink: int,
                        pad_x: int, pad_y: int, join: bool, extent: tuple, struck: list | None = None) -> Image.Image:
    ref_top, ref_bottom = extent
    stem = _stem_width(font, fs)
    items, links = [], []          # coordinates: x from the left, y relative to the baseline
    x, prev, prev_ch = 0.0, None, ""
    size_state = dy_state = 0.0    # slowly drifting 'hand state' shared by neighbouring letters
    word_gap = random.uniform(*WORD_GAP_RANGE)
    widx, in_word = 0, False        # index of the current word (for crossing out)
    for ch in text:
        if ch == " ":
            if in_word:
                widx, in_word = widx + 1, False
            x = (prev["right"] if prev else x) + fs * random.uniform(0.2, 0.6)
            prev = None
            size_state *= 0.5
            word_gap = random.uniform(*WORD_GAP_RANGE)      # every word has its own spacing style
            continue
        g = _glyph_mask(font, ch, fs)
        if g is None:
            x += fs * 0.3
            prev = None
            continue
        mask, _, top = g
        in_word = True
        is_punct = not ch.isalnum()

        if is_punct:       # punctuation: written carelessly - big scatter in size, height and tilt
            J = PUNCT_JITTER
            scale = float(np.clip(1 + random.gauss(0, J["scale"]), 0.55, 1.7))
            dy = dy_state * fs + random.gauss(0, J["dy"] * fs)
            rot = float(np.clip(random.gauss(0, J["rot"]), -40, 40))
            gap = random.uniform(*J["gap"]) * fs
        else:              # letters: neighbours keep similar size / height (smooth drift)
            J = LETTER_JITTER
            size_state = 0.6 * size_state + 0.4 * random.gauss(0, J["scale"])
            dy_state = 0.5 * dy_state + 0.5 * random.gauss(0, J["dy"])
            scale = float(np.clip(1 + size_state, 0.8, 1.25))
            dy = dy_state * fs
            rot = random.gauss(0, J["rot"])
            gap = max(0.02, word_gap + random.gauss(0, J["gap"])) * fs     # never glued together
        mask, top = _transform(mask, top, scale, rot)
        top += int(round(dy))
        h, w = mask.shape
        lprof, rprof = _profiles(mask)
        cur = {"top": top, "h": h, "w": w, "mask": mask, "lprof": lprof, "rprof": rprof,
               "exit": _edge_point(mask, top, "right"), "entry": _edge_point(mask, top, "left"), "scale": scale, "wi": widx}
        if prev is None:
            cur["X"] = x
        else:       # distance also follows the size of the two neighbours
            cur["X"] = _place_x(prev, cur, gap * (prev["scale"] + scale) / 2)
        cur["right"] = cur["X"] + w

        if (join and prev is not None and prev["exit"] is not None and cur["entry"] is not None
                and ch.isalpha() and prev_ch.isalpha() and random.random() < JOIN_PROB):
            p0 = (prev["X"] + prev["exit"][0], prev["top"] + prev["exit"][1])
            p1 = (cur["X"] + cur["entry"][0], cur["top"] + cur["entry"][1])
            if p1[0] - p0[0] > 1:     # a real gap to bridge (touching letters are already connected)
                ctrl = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + random.uniform(-0.15, 0.15) * fs)
                width = max(2, int(stem * (prev["scale"] + scale) / 2 * random.uniform(0.6, 1.0)))
                links.append((_bezier(p0, p1, ctrl), width))
        items.append(cur)
        prev, prev_ch = cur, ch
    if not items:
        return Image.new("L", (10, 10), 255)

    y_min = min(min(i["top"] for i in items), ref_top)
    y_max = max(max(i["top"] + i["h"] for i in items), ref_bottom)
    W = int(max(i["right"] for i in items)) + 2 * pad_x
    H = int(y_max - y_min) + 2 * pad_y
    ox, oy = pad_x, pad_y - y_min       # canvas position of (x=0, baseline)

    canvas = np.zeros((H, W), dtype=np.uint8)
    for it in items:
        m = it["mask"]
        x0, y0 = int(ox + it["X"]), int(oy + it["top"])
        region = canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        np.maximum(region, m[:region.shape[0], :region.shape[1]], out=region)
    pen = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pen)
    for pts, width in links:
        _dab_path(draw, pts, width, ox, oy)
    if struck:                                           # cross out the chosen words
        xh = _x_height(font, fs)
        for wi, kind in enumerate(struck):
            word = [i for i in items if i["wi"] == wi]
            if not kind or not word:
                continue
            x0, x1 = min(i["X"] for i in word), max(i["right"] for i in word)
            width = max(2, int(stem * random.uniform(*STRIKE_WIDTH[kind])))
            for pts in _strike_paths(kind, x0, x1, xh, fs):
                _dab_path(draw, pts, width, ox, oy)
    canvas = np.array(pen).astype(np.float32)
    return Image.fromarray((255 - canvas * (255 - ink) / 255).astype(np.uint8))


def render_text_line(text: str, font_path: str | None = None, augment: bool = True,
                     hand: bool | None = None, connect: bool | None = None,
                     struck: list | None = None) -> Image.Image:
    """Render `text` as a grayscale line image of height TARGET_HEIGHT."""
    font_path = font_path or random.choice(_available_fonts())
    font_size = random.randint(56, 88)  # big on purpose; downscaled at the end
    font = _load_font(font_path, font_size)
    ref_top, ref_bottom = _vertical_extent(font_path, font_size)

    dummy = ImageDraw.Draw(Image.new("L", (10, 10), color=255))
    bbox = dummy.textbbox((0, 0), text, font=font, anchor="ls")   # relative to the baseline

    pad_x = int(font_size * random.uniform(0.15, 0.5))
    pad_y = int((ref_bottom - ref_top) * random.uniform(0.05, 0.14))
    top = min(ref_top, bbox[1])
    bottom = max(ref_bottom, bbox[3])
    w = max(1, bbox[2] - bbox[0]) + 2 * pad_x
    h = (bottom - top) + 2 * pad_y

    ink = random.randint(0, 75)
    if struck and any(struck):
        hand = True                                    # crossing out needs letter positions
    if hand is None:
        hand = random.random() < HANDWRITTEN_PROB      # letter-by-letter, hand-like variation
    if connect is None:
        connect = random.random() < CONNECT_PROB       # ... with letters joined by strokes
    if hand and any(c.isalnum() for c in text):
        img = _render_handwritten(text, font, font_size, ink, pad_x, pad_y, join=connect,
                                  extent=(ref_top, ref_bottom), struck=struck)
    else:
        img = Image.new("L", (w, h), color=255)
        ImageDraw.Draw(img).text((pad_x - bbox[0], pad_y - top), text, font=font, fill=ink, anchor="ls")

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
        dark = random.randint(15, 55)
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
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))

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
    for i, (img, text) in enumerate(generate_batch(20)):
        img.save(f"samples/sample_{i}.png")
        print(f"sample_{i}.png -> {text!r}  size={img.size}")
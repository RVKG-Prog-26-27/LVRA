"""
Synthetic data generator (v3.1).

v3.1 - speed only, the data looks the same (labels identical for the same seed, image statistics unchanged):
  * Glyphs are rendered once per font at REF_FS and scaled inside the per-letter transform, instead of being
    rendered and cached for every font size 56..88. The cache is ~35x smaller, warms up after a few hundred
    lines and no longer overflows with many fonts (before: 60 000-entry cache, 500+ MB per worker).
  * Per-letter scale + tilt in one affine resampling (before: resize + rotate = two passes).
  * Join points of letters are only computed for joined (cursive) lines.
  * Ink thickening: separable numpy 3x3 minimum instead of ImageFilter.MinFilter (bit-identical, ~25x faster).
  * Wavy baseline moves runs of columns at once instead of a per-pixel gather (bit-identical).
  * Scalar np.clip -> plain min/max; float32 noise; sized font loaded only for the plain-font path.
  Net: ~1.5x faster per line with a warm cache, ~2x with many fonts / fresh workers.

What changed in v3 vs. v2:
  * CORRECTIONS ABOVE CROSSED-OUT WORDS, like in real handwriting. A struck word can now be a
    "mistake" (typo, missing / wrong diacritic, swapped or doubled letter, wrong digit, another word):
    it is drawn and crossed out, and the RIGHT word is written smaller above it. The label holds the
    right word at that position (the model must learn: struck word + text above -> read the text above).
    Struck words without a correction keep the old behaviour (STRIKE_MODE 'mark' / 'skip').
    Punctuation stays outside the strike: "valodu." -> "valudu." with only "valudu" struck, "valodu" above.
  * Speed: glyph masks, stem width and x-height are cached (before: every letter of every line was
    rendered twice from scratch); the wavy-baseline warp is vectorised; the corpus filter per font is cached.
  * Noise / blur / JPEG artefacts are applied AFTER the downscale to 64 px. Before, the 3-4x downscale
    averaged most of the noise and blur away, so the model rarely saw a really noisy line.
  * All randomness goes through `random` (numpy noise uses a generator seeded from it): one random.seed()
    makes a dataset reproducible, and DataLoader workers no longer repeat the same numpy noise.
  * Ruled (lined) notebook paper in addition to grid paper.
  * Fixes: random_tokens could fall back to a word the font cannot draw; font_charset now also rejects
    glyphs that render empty; a missing glyph in the middle of a word no longer makes the next letter
    jump back onto the word start; word indices stay in sync even if a word has no drawable glyphs.

v2 (kept): TARGET_HEIGHT 64, big rendering + downscale, metric-based line height, random word lines,
notebook-photo augmentation, letter-by-letter "handwritten" rendering, roman numerals, strike-outs.
"""

import functools
import glob
import io
import math
import os
import random
import re

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


@functools.lru_cache(maxsize=1)
def _available_fonts() -> list[str]:
    """Fonts in fonts/ (scanned once per process, not for every sample)."""
    fonts = glob.glob(os.path.join(FONT_DIR, "*.ttf")) + glob.glob(os.path.join(FONT_DIR, "*.otf"))
    return sorted(fonts) if fonts else [FALLBACK_FONT]


def _glyph_bytes(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    img = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(img).text((10, 10), ch, font=font, fill=255)
    return img.tobytes()


_EMPTY_GLYPH = bytes(100 * 100)


@functools.lru_cache(maxsize=None)
def font_charset(font_path: str) -> frozenset:
    """Which characters of our alphabet this font can really draw.
    A missing glyph is drawn as an empty box (.notdef) or as nothing at all, while the label keeps the real
    letter - the network would then learn 'box = a / ā / š / ž ...', which is impossible. So we never
    put such characters into texts rendered with this font."""
    font = ImageFont.truetype(font_path, 48)
    notdef = _glyph_bytes(font, "\uFFFF")
    ok = {" "}
    for ch in ALPHABET[1:]:  # [0] is the CTC blank
        if ch == " ":
            continue
        b = _glyph_bytes(font, ch)
        if b != notdef and b != _EMPTY_GLYPH:
            ok.add(ch)
    return frozenset(ok)


# --------------------------------------------------------------------------- roman numerals

ROMAN_PROB = 0.006        # share of tokens that are roman numerals (I, II, XIV., (XX), iv, MCMXCIV ...)
# How the letter I of a roman numeral is drawn: the font's own glyph / a plain stick / a stick with
# small bars on top and bottom (serifs). People write roman numerals in all these ways.
ROMAN_I_STYLES = {"font": 0.4, "stick": 0.35, "serif": 0.25}

_ROMAN_PART = r"(?:[IVXLCDM]+|[ivxlcdm]+)"
_ROMAN_RE = re.compile(r"^[\(\[]?" + _ROMAN_PART + r"(?:-" + _ROMAN_PART + r")?[.,;:\)\]]*$")


def _to_roman(n: int, additive: bool = False) -> str:
    """1..3999 -> roman numeral. additive=True gives the 'clock' forms IIII, VIIII, XXXX, LXXXX ..."""
    table = ([(1000, "M"), (500, "D"), (100, "C"), (50, "L"), (10, "X"), (5, "V"), (1, "I")] if additive else
             [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
              (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")])
    out = ""
    for value, sym in table:
        while n >= value:
            out += sym
            n -= value
    return out


def _roman_token() -> str:
    """A random roman numeral as people really write it: small numbers most often, sometimes big ones,
    sometimes lowercase, with a dot / bracket, or a range like XIV-XVI."""
    lo, hi = random.choices([(1, 20), (21, 100), (101, 1000), (1001, 3999)], weights=[0.55, 0.25, 0.15, 0.05])[0]
    n = random.randint(lo, hi)
    add = random.random() < 0.08
    r = _to_roman(n, add)
    if random.random() < 0.15:
        r = r.lower()
    form = random.choices(["{}", "{}.", "{})", "({})", "range"], weights=[0.55, 0.2, 0.1, 0.1, 0.05])[0]
    if form == "range":
        return r + "-" + _to_roman(min(3999, n + random.randint(1, 10)), add)
    return form.format(r)


def _roman_i(glyph_h: float, top: float, stem: int, style: str):
    """The letter I of a roman numeral drawn by us: a plain stick or a stick with top/bottom bars.
    glyph_h / top: height and top of the font's own I at the line's font size."""
    h = max(int(round(glyph_h)), 8)
    top = int(round(top))
    sw = max(2, int(stem * random.uniform(0.9, 1.3)))
    ww = sw + 4 if style == "stick" else int(h * random.uniform(0.4, 0.6)) + sw
    img = Image.new("L", (ww, h), 0)
    d = ImageDraw.Draw(img)
    r, cx, lean = sw / 2, ww / 2, random.uniform(-0.06, 0.06) * h
    _dab_path(d, _wobbly((cx + lean, r), (cx - lean, h - r - 1), 0.01 * h, 0.3 * h), sw, 0, 0)
    if style == "serif":
        half = (ww - sw) / 2
        for y in (r, h - r - 1):
            tilt = random.uniform(-0.04, 0.04) * h
            _dab_path(d, _wobbly((cx - half, y - tilt), (cx + half, y + tilt), 0.01 * h, 0.3 * h), max(2, int(sw * 0.8)), 0, 0)
    return np.array(img), top


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


# Filtering a big corpus for every sample is O(corpus); the result only depends on (word list, font charset).
_WORDS_CACHE: dict = {}


def _words_for(words: list[str], allowed: frozenset) -> list[str]:
    key = (id(words), allowed)
    hit = _WORDS_CACHE.get(key)
    if hit is None or hit[0] is not words or hit[1] != len(words):
        hit = (words, len(words), [w for w in words if set(w) <= allowed])
        _WORDS_CACHE[key] = hit
    return hit[2]


def _safe_token(allowed) -> str:
    """Last-resort token that the font can surely draw."""
    if allowed is None:
        return "a"
    pool = sorted(c for c in allowed if c.isalnum())
    return "".join(random.choice(pool) for _ in range(random.randint(1, 4))) if pool else "0"


def random_tokens(words: list[str], max_chars: int = 60, allowed=None) -> list[str]:
    """A random 'handwritten line' as a list of tokens (words/numbers), 6..max_chars characters long.
    `allowed` - set of characters the font can draw; words/numbers with other characters are skipped."""
    if allowed is not None:
        words = _words_for(words, frozenset(allowed))
    target = random.randint(6, max_chars)
    parts: list[str] = []
    length = 0
    guard = 0
    while length < target and guard < 200:
        guard += 1
        r = random.random()
        if r < 0.07 or not words:
            w = _random_number()
        elif r < 0.07 + ROMAN_PROB:
            w = _roman_token()
        else:
            w = random.choice(words)
        if allowed is not None and not set(w) <= allowed:
            continue
        extra = len(w) + (1 if parts else 0)
        if length + extra > max_chars:
            if parts:
                break
            continue
        parts.append(w)
        length += extra
    if parts:
        return parts
    return [random.choice(words)[:max_chars]] if words else [_safe_token(allowed)]


def random_line(words: list[str], max_chars: int = 60, allowed=None) -> str:
    return " ".join(random_tokens(words, max_chars, allowed))


# --------------------------------------------------------------------------- mistakes for corrections

CORRECTION_PROB = 0.6        # share of crossed-out words that get the right word written above them
CORRECTION_SCALE = (0.55, 0.8)   # size of the correction relative to the line's font size
# kind of mistake that was crossed out
MISTAKE_WEIGHTS = {"diacritic": 0.28,  # rīga -> riga, skola -> skoļa
                   "confuse": 0.22,    # a<->o, e<->i, n<->m, 3<->8 ...
                   "drop": 0.14,       # missing letter
                   "double": 0.12,     # doubled letter
                   "swap": 0.12,       # two neighbours swapped
                   "word": 0.12}       # a different word altogether
# strike style used for a word that is corrected (people mostly just draw a line through a mistake)
CORRECTION_STRIKE_WEIGHTS = {"line": 0.5, "multi": 0.25, "scribble": 0.2, "cross": 0.05}

_DIACRITIC_PAIRS = ["aā", "cč", "eē", "gģ", "iī", "kķ", "lļ", "nņ", "sš", "uū", "zž"]
_DIACRITIC = {}
for _p in _DIACRITIC_PAIRS:
    for _a, _b in ((_p[0], _p[1]), (_p[1], _p[0]), (_p[0].upper(), _p[1].upper()), (_p[1].upper(), _p[0].upper())):
        _DIACRITIC[_a] = _b
_CONFUSABLE = ["ao", "ei", "uv", "nm", "sz", "td", "kg", "bp", "lt", "rn", "ce", "hn", "uo",
               "17", "38", "56", "09", "24", "35"]
_CONFUSE = {}
for _g in _CONFUSABLE:
    for _c in _g:
        _CONFUSE.setdefault(_c, set()).update(set(_g) - {_c})


def _split_token(tok: str) -> tuple[str, str, str]:
    """'(valodu.)' -> ('(', 'valodu', '.)'): leading punctuation, core, trailing punctuation."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", tok)
    return m.group(1), m.group(2), m.group(3)


def _misspell(core: str, allowed, words: list[str]) -> str | None:
    """A plausible wrong version of `core` that the font can draw, or None."""
    is_roman = bool(_ROMAN_RE.match(core))
    for _ in range(12):
        if is_roman:      # roman numerals: wrong / missing / extra / swapped symbol, never diacritics
            op = random.choice(["roman", "roman", "drop", "double", "swap"])
        else:
            op = _pick(MISTAKE_WEIGHTS)
        ch = list(core)
        idx = [i for i, c in enumerate(ch) if c.isalnum()]
        if not idx:
            return None
        if op == "roman":
            i = random.choice(idx)
            new = random.choice([c for c in "IVXLC" if c != ch[i].upper()])
            ch[i] = new.lower() if ch[i].islower() else new
        elif op == "drop" and is_roman:
            if len(idx) < 2:
                continue
            del ch[random.choice(idx)]
        elif op == "diacritic":
            cand = [i for i in idx if ch[i] in _DIACRITIC]
            if not cand:
                continue
            i = random.choice(cand)
            ch[i] = _DIACRITIC[ch[i]]
        elif op == "confuse":
            cand = [i for i in idx if ch[i].lower() in _CONFUSE]
            if not cand:
                continue
            i = random.choice(cand)
            new = random.choice(sorted(_CONFUSE[ch[i].lower()]))
            ch[i] = new.upper() if ch[i].isupper() else new
        elif op == "drop":
            if len(idx) < 3:
                continue
            del ch[random.choice(idx)]
        elif op == "double":
            i = random.choice(idx)
            ch.insert(i, ch[i])
        elif op == "swap":
            pairs = [i for i in idx if i + 1 < len(ch) and ch[i + 1].isalnum() and ch[i] != ch[i + 1]]
            if not pairs:
                continue
            i = random.choice(pairs)
            ch[i], ch[i + 1] = ch[i + 1], ch[i]
        else:  # "word": the writer started a different word
            if not words or not core.isalpha():
                continue
            other = _split_token(random.choice(words))[1]
            if not other.isalpha() or abs(len(other) - len(core)) > 3:
                continue
            ch = list(other.capitalize() if core[0].isupper() else other.lower())
        wrong = "".join(ch)
        if wrong and wrong != core and (allowed is None or set(wrong) <= allowed):
            return wrong
    return None


def _make_fix(tok: str, allowed, words: list[str]):
    """(drawn_token, correction, (a, b)) - the mistake to draw, the right core written above it and the
    character span of the mistake inside the drawn token (only that span is crossed out). None if impossible."""
    lead, core, trail = _split_token(tok)
    if not core or not any(c.isalnum() for c in core):
        return None
    wrong = _misspell(core, allowed, words)
    if wrong is None:
        return None
    a = len(lead)
    return lead + wrong + trail, core, (a, a + len(wrong))


def _pick(weights: dict) -> str:
    return random.choices(list(weights), weights=list(weights.values()))[0]


def _clip(v: float, lo: float, hi: float) -> float:
    """float(np.clip(...)) for a single number, without numpy's ~5 us per call overhead."""
    return lo if v < lo else hi if v > hi else float(v)


def _pick_strike() -> str:
    return _pick(STRIKE_WEIGHTS)


def make_sample(words: list[str], max_chars: int = 60, augment: bool = True) -> tuple[Image.Image, str]:
    """Pick a random font, build a text this font can fully draw, render it. Returns (image, label).
    Some lines get crossed-out words (STRIKE_LINE_PROB). A crossed-out word is either
      * a mistake with the right word written above it (CORRECTION_PROB) -> label has the right word, or
      * just struck -> STRIKE_MODE 'mark': label has STRIKE_MARK (¤); 'skip': the word is missing from the label."""
    fonts = _available_fonts()
    font_path = random.choice(fonts)
    allowed = font_charset(font_path)
    pool = _words_for(words, allowed)
    tokens = random_tokens(words, max_chars, allowed)

    n = len(tokens)
    kinds: list = [None] * n
    fixes: list = [None] * n
    if n >= 2 and random.random() < STRIKE_LINE_PROB:
        k = min(n - 1, random.choice([1, 1, 1, 2, 2, 3]))      # always leave some real text
        for i in random.sample(range(n), k):
            if " " in tokens[i]:                                 # not the multi-word numbers
                continue
            if random.random() < CORRECTION_PROB:
                fixes[i] = _make_fix(tokens[i], allowed, pool)
            kinds[i] = _pick(CORRECTION_STRIKE_WEIGHTS) if fixes[i] else _pick_strike()

    drawn = [fx[0] if fx else tok for tok, fx in zip(tokens, fixes)]
    text = " ".join(drawn)

    word_kinds, word_fixes, word_roman = [], [], []          # one entry per drawn word
    for tok, dtok, kd, fx in zip(tokens, drawn, kinds, fixes):
        is_roman = bool(_ROMAN_RE.match(tok))
        for _ in dtok.split():
            word_kinds.append(kd)
            word_fixes.append((fx[1], fx[2]) if fx else None)
            word_roman.append(is_roman)

    label_parts = []
    for tok, kd, fx in zip(tokens, kinds, fixes):
        if fx or not kd:
            label_parts.append(tok)                 # corrected word -> the right word; normal word -> itself
        elif STRIKE_MODE == "mark":
            label_parts.append(STRIKE_MARK)
    label = " ".join(label_parts)

    img = render_text_line(text, font_path=font_path, augment=augment, struck=word_kinds,
                           roman=word_roman, fixes=word_fixes)
    return img, label


# --------------------------------------------------------------------------- handwritten look
# Letter-by-letter rendering that imitates a real hand:
#   * every letter gets its own size / vertical shift / tilt, and these drift smoothly from letter to
#     letter (neighbours look alike, like in real writing, instead of being independent noise);
#   * spacing depends on the SHAPES of the two neighbours (silhouette 'kerning'): round letters nest closer,
#     tall / overhanging ones keep more distance; every word also gets its own spacing style;
#   * neighbouring letters of a word can be joined with a pen stroke (cursive);
#   * punctuation is much sloppier than letters: bigger size / position / tilt scatter, wilder gaps;
#   * crossed-out mistakes get the right word written smaller above them.

HANDWRITTEN_PROB = 0.8   # share of lines drawn letter-by-letter (the rest = plain font rendering)
CONNECT_PROB = 0.5       # of those, share of lines whose letters are joined by strokes
JOIN_PROB = 0.85         # in a joined line: chance that two neighbouring letters get joined

# crossed-out words (people strike words out when they write)
STRIKE_LINE_PROB = 0.2    # share of lines that contain at least one crossed-out word
STRIKE_MODE = "mark"      # struck word WITHOUT correction: "mark" -> STRIKE_MARK (¤) in the label; "skip" -> left out
STRIKE_WEIGHTS = {"line": 0.35,       # one line through the word
                  "scribble": 0.25,   # word smeared over with dense zigzag / coil strokes
                  "cross": 0.20,      # one cross (X) fitted to the word
                  "multi": 0.20}      # several roughly parallel lines

# scatter strengths. scale/dy are relative (dy in font sizes), rot in degrees
LETTER_JITTER = {"scale": 0.07, "dy": 0.03, "rot": 4.0, "gap": 0.03}
PUNCT_JITTER = {"scale": 0.25, "dy": 0.12, "rot": 20.0, "gap": (-0.05, 0.28)}
WORD_GAP_RANGE = (0.03, 0.15)      # base distance between letters of a word (in font sizes), per word


@functools.lru_cache(maxsize=1024)
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


# Glyphs are rendered ONCE per font at REF_FS and scaled to the real size inside _transform (every letter is
# resampled there anyway for its own size/tilt). Before, every (font, size 56..88, char) was rendered and
# cached separately: with many fonts the cache never warmed up, and every DataLoader worker kept its own
# hundreds of MB of masks.
REF_FS = 80


@functools.lru_cache(maxsize=None)
def _ref_glyph(font_path: str, ch: str):
    """_glyph_mask at REF_FS, cached. The mask is read-only: callers must copy before changing it."""
    g = _glyph_mask(_load_font(font_path, REF_FS), ch, REF_FS)
    if g is not None:
        g = (np.ascontiguousarray(g[0]), g[1], g[2])
        g[0].setflags(write=False)
    return g


@functools.lru_cache(maxsize=60000)
def _glyph(font_path: str, fs: int, ch: str):
    """Glyph at size fs (resized from the reference render). Only used where a few exact-size glyphs are
    needed (font metrics); the letters of a line go through _ref_glyph + _transform directly."""
    g = _ref_glyph(font_path, ch)
    if g is None or fs == REF_FS:
        return g
    mask, top = _transform(np.asarray(g[0]), g[2], fs / REF_FS, 0.0)
    mask.setflags(write=False)
    return mask, int(round(g[1] * fs / REF_FS)), top


@functools.lru_cache(maxsize=None)
def _ref_stem(font_path: str) -> float:
    g = _ref_glyph(font_path, "l")
    if g is None:
        return -1.0
    mask = g[0]
    return float((mask[mask.shape[0] // 2] > 128).sum())


@functools.lru_cache(maxsize=4096)
def _stem_width(font_path: str, fs: int) -> int:
    """Pen thickness of the font = width of the vertical stem of 'l'."""
    s = _ref_stem(font_path)
    if s < 0:
        return max(2, int(fs * 0.06))
    return int(min(max(2, round(s * fs / REF_FS)), fs * 0.12))


@functools.lru_cache(maxsize=4096)
def _x_height(font_path: str, fs: int) -> int:
    g = _ref_glyph(font_path, "x")
    return int(round(g[0].shape[0] * fs / REF_FS)) if g is not None else int(fs * 0.5)


AFFINE_MIN_SCALE = 0.6   # below this a glyph is first shrunk with an anti-aliased resize (thin strokes stay clean)


def _transform(mask: np.ndarray, top: int, scale: float, rot: float, around_centre: bool = False):
    """Scale the glyph around the pen origin (or around its own centre) and tilt it around its centre.
    Scale + tilt are done in ONE affine resampling (before: resize, then rotate = two passes). Returns (mask, top)."""
    img = Image.fromarray(mask)
    h, w = mask.shape
    if abs(scale - 1) <= 0.01:
        scale = 1.0
    elif scale < AFFINE_MIN_SCALE:
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.BILINEAR)
        top = top + (h - nh) / 2 if around_centre else top * scale
        w, h, scale = nw, nh, 1.0
    else:
        top = top + (h - h * scale) / 2 if around_centre else top * scale
    if scale == 1.0 and abs(rot) <= 0.3:
        return (mask if img.size == mask.shape[::-1] else np.array(img)), int(round(top))
    sw, sh = w * scale, h * scale                         # size after scaling (float)
    a = math.radians(rot)
    c, s = math.cos(a), math.sin(a)
    if abs(rot) <= 0.3:
        c, s = 1.0, 0.0
    W = max(1, int(math.ceil(abs(sw * c) + abs(sh * s) - 1e-6)))
    H = max(1, int(math.ceil(abs(sw * s) + abs(sh * c) - 1e-6)))
    # inverse map: output pixel -> input pixel (rotate back around the centre, then undo the scale)
    inv = (c / scale, -s / scale, (-c * W / 2 + s * H / 2 + sw / 2) / scale,
           s / scale, c / scale, (-s * W / 2 - c * H / 2 + sh / 2) / scale)
    img = img.transform((W, H), Image.AFFINE, inv, resample=Image.BILINEAR)
    top += (sh - H) / 2
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


SMALL_PUNCT = set(",.;:'\"\u201e\u201c\u201d\u2018\u2019")   # marks that fonts draw as tiny specks
PEN_PUNCT = set(".,;:")      # these four are drawn with our own pen (size follows the pen / font size)
PUNCT_MIN_SIZE = 0.16        # quotes: smallest size, in font sizes
# scatter of the small marks: clearly looser than letters, but a dot must never fly away or turn into a blot
SMALL_JITTER = {"scale": 0.15, "dy": 0.035, "rot": 12.0, "gap": (-0.02, 0.16)}


def _boost_small(mask: np.ndarray, top: int, stem: int, fs: int):
    """Quotes in a font are tiny and vanish under blur/noise: enlarge them a little (around their OWN centre,
    so a raised quote stays raised) and thicken thin pens. Size is capped relative to the font size."""
    h, w = mask.shape
    target = min(max(stem * 1.8, PUNCT_MIN_SIZE * fs), 0.22 * fs) * random.uniform(1.0, 1.25)
    cur = max(h, w)
    if cur < target:
        k = min(target / cur, 2.0)
        nh, nw = max(1, round(h * k)), max(1, round(w * k))
        mask = np.array(Image.fromarray(mask).resize((nw, nh), Image.BILINEAR))
        top = int(round(top + (h - nh) / 2))
    ratio = stem / fs
    n = 0 if ratio >= 0.07 else (1 if ratio >= 0.04 else 2)
    img = Image.fromarray(np.pad(mask, n))
    for _ in range(n):
        img = img.filter(ImageFilter.MaxFilter(3))
    return np.array(img), top - n


def _pen_punct(ch: str, stem: int, fs: int, xh: int):
    """Draw . , ; : with a pen. The dot is as thick as the pen of this font (stem), clamped to a sensible share
    of the x-height, so every font gets dots that fit it: not specks, not blots. Returns (mask, top)."""
    d = _clip(stem * random.uniform(1.2, 1.7), 0.18 * xh, 0.30 * xh)     # follows the pen, bounded by x-height
    d = max(d, 3.0)
    low = -d / 2 + random.gauss(0, 0.012 * fs)                    # lower dot sits on the baseline
    high = -random.uniform(0.45, 0.7) * xh                        # upper dot of : and ; at x-height
    dots = {".": [("dot", low)], ",": [("comma", low)],
            ":": [("dot", high), ("dot", low)], ";": [("dot", high), ("comma", low)]}[ch]
    tail = d * random.uniform(1.4, 2.3)
    top = min(cy for _, cy in dots) - d / 2 - 1
    bottom = max(cy for _, cy in dots) + d / 2 + (tail if any(k == "comma" for k, _ in dots) else 0) + 1
    ss = 3                                                        # supersampling = smooth edges
    W, H = int(d * 2 + 4), int(bottom - top) + 2
    img = Image.new("L", (W * ss, H * ss), 0)
    draw = ImageDraw.Draw(img)
    cx = W * ss / 2
    for kind, cy in dots:
        y = (cy - top) * ss
        rx, ry = d / 2 * ss * random.uniform(0.9, 1.1), d / 2 * ss * random.uniform(0.9, 1.1)
        draw.ellipse((cx - rx, y - ry, cx + rx, y + ry), fill=255)
        if kind == "comma":                                       # tapering tail, down and to the left
            end = (cx - random.uniform(0.2, 0.9) * d * ss, y + tail * ss)
            ctrl = (cx + random.uniform(0.0, 0.3) * d * ss, y + tail * 0.5 * ss)
            pts = _bezier((cx, y), end, ctrl, 10)
            for k, (px, py) in enumerate(pts):
                r = d / 2 * ss * (0.85 - 0.55 * k / (len(pts) - 1))
                draw.ellipse((px - r, py - r, px + r, py + r), fill=255)
    return np.array(img.resize((W, H), Image.LANCZOS)), int(round(top))


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


def _strike_paths(kind: str, x0: float, x1: float, xh: float, fs: int, y_word: tuple = (0, 0),
                  y_limits: tuple = (-1e9, 1e9)) -> list:
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
    elif kind == "cross":                       # ONE X that adapts to the word: as wide as the word, as tall as its letters
        top = max(y_word[0], -xh * 1.9) - random.uniform(0, 0.1) * xh
        bot = min(y_word[1], xh * 0.65) + random.uniform(0, 0.1) * xh
        need = min(0.3 * w, 2.4 * xh)               # a long word still gets a real X, not two flat lines
        if bot - top < need:
            mid = (top + bot) / 2
            top, bot = mid - need / 2, mid + need / 2
        top, bot = max(top, y_limits[0] * 0.9), min(bot, y_limits[1] * 0.9)     # stay inside the line image
        jx = lambda: random.uniform(-0.04, 0.04) * w
        jy = lambda: random.uniform(-0.08, 0.08) * (bot - top)
        paths.append(_wobbly((x0 + jx(), top + jy()), (x1 + jx(), bot + jy()), amp, step))
        paths.append(_wobbly((x0 + jx(), bot + jy()), (x1 + jx(), top + jy()), amp, step))
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
    top, bottom = _ref_extent(font_path)
    k = fs / REF_FS
    top, bottom = int(math.floor(top * k)), int(math.ceil(bottom * k))
    if bottom - top < fs * 0.5:                       # font without those letters: fall back to metrics
        a, d = _load_font(font_path, fs).getmetrics()
        return -a, d
    return top, bottom


@functools.lru_cache(maxsize=None)
def _ref_extent(font_path: str) -> tuple:
    cs = font_charset(font_path)
    top, bottom = 0, 0
    for ch in "bdfhklpqgjyBDFHKLĀŠŽ":
        if ch in cs:
            g = _ref_glyph(font_path, ch)
            if g is not None:
                top = min(top, g[2])
                bottom = max(bottom, g[2] + g[0].shape[0])
    return top, bottom


def _layout(text: str, font_path: str, fs: int, join: bool, roman: list | None = None):
    """Place the letters of `text` like a hand would. Returns (items, links, stem, xh).
    items: dicts with X / right (x from the left), top (relative to the baseline), mask, wi (word index),
    ci (character index inside the word). links: [(points, width)] - joining strokes between letters."""
    stem = _stem_width(font_path, fs)
    xh = _x_height(font_path, fs)
    k_ref = fs / REF_FS
    items, links = [], []
    x, prev, prev_ch = 0.0, None, ""
    size_state = dy_state = 0.0    # slowly drifting 'hand state' shared by neighbouring letters
    word_gap = random.uniform(*WORD_GAP_RANGE)
    widx, ci, in_word = 0, 0, False
    for ch in text:
        if ch == " ":
            if in_word:
                widx, in_word = widx + 1, False
            ci = 0
            x = (prev["right"] if prev else x) + fs * random.uniform(0.2, 0.6)
            prev = None
            size_state *= 0.5
            word_gap = random.uniform(*WORD_GAP_RANGE)      # every word has its own spacing style
            continue
        in_word = True                                       # also for undrawable chars: keeps word indices in sync
        g = _ref_glyph(font_path, ch)
        if g is None:
            x = (prev["right"] if prev else x) + fs * 0.3
            prev = None
            ci += 1
            continue
        mask, _, top = g
        k = k_ref               # size factor still to apply to `mask` (1.0 once it is already at size fs)
        if ch == "I" and roman and widx < len(roman) and roman[widx]:       # I inside a roman numeral
            style = _pick(ROMAN_I_STYLES)
            if style != "font":
                mask, top = _roman_i(mask.shape[0] * k, top * k, stem, style)
                k = 1.0
        is_punct = not ch.isalnum()
        if ch in PEN_PUNCT:
            mask, top = _pen_punct(ch, stem, fs, xh)
            k = 1.0
        elif ch in SMALL_PUNCT:
            mask, top = _transform(mask, top, k, 0.0)
            mask, top = _boost_small(mask, top, stem, fs)
            k = 1.0

        if is_punct:       # punctuation: written carelessly - looser than letters (more for ! ? ( ) -, less for dots)
            small = ch in SMALL_PUNCT
            J = SMALL_JITTER if small else PUNCT_JITTER
            scale = _clip(1 + random.gauss(0, J["scale"]), 0.8, 1.25 if small else 1.7)
            dy_noise = random.gauss(0, J["dy"] * fs)
            if small:
                dy_noise = _clip(dy_noise, -0.08 * fs, 0.08 * fs)     # a dot never flies away
            dy = dy_state * fs + dy_noise
            rot = _clip(random.gauss(0, J["rot"]), -40, 40)
            gap = random.uniform(*J["gap"]) * fs
        else:              # letters: neighbours keep similar size / height (smooth drift)
            J = LETTER_JITTER
            size_state = 0.6 * size_state + 0.4 * random.gauss(0, J["scale"])
            dy_state = 0.5 * dy_state + 0.5 * random.gauss(0, J["dy"])
            scale = _clip(1 + size_state, 0.8, 1.25)
            dy = dy_state * fs
            rot = random.gauss(0, J["rot"])
            gap = max(0.02, word_gap + random.gauss(0, J["gap"])) * fs     # never glued together
        mask, top = _transform(mask, top, scale * k, rot, around_centre=ch in SMALL_PUNCT)
        top += int(round(dy))
        h, w = mask.shape
        lprof, rprof = _profiles(mask)
        cur = {"top": top, "h": h, "w": w, "mask": mask, "lprof": lprof, "rprof": rprof,
               "exit": _edge_point(mask, top, "right") if join else None,     # only joined lines need these
               "entry": _edge_point(mask, top, "left") if join else None,
               "scale": scale, "wi": widx, "ci": ci}
        ci += 1
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
    return items, links, stem, xh


def _place_correction(fix: str, font_path: str, fs: int, join: bool, x0: float, x1: float,
                      top: float, xh: float, min_x: float):
    """Lay out the correction `fix` smaller and put it above a struck span x0..x1 whose highest ink
    (letters + strike strokes) is at `top`. Returns (items, links, right edge) or None."""
    fs_c = max(16, int(round(fs * random.uniform(*CORRECTION_SCALE))))
    c_items, c_links, _, _ = _layout(fix, font_path, fs_c, join, [bool(_ROMAN_RE.match(fix))])
    if not c_items:
        return None
    left = min(i["X"] for i in c_items)
    cw = max(i["right"] for i in c_items) - left
    if random.random() < 0.6:                                   # centred over the mistake ...
        cx = (x0 + x1) / 2 - cw / 2 + random.gauss(0, 0.05 * fs)
    else:                                                       # ... or starting where it starts
        cx = x0 + random.uniform(-0.1, 0.25) * fs
    cx = max(cx, min_x + 0.25 * fs)                             # don't run into the previous correction
    dx = cx - left
    # baseline of the correction: just above the struck word; descenders may touch it, like on paper
    base = top - random.uniform(-0.1, 0.3) * xh
    tilt = random.uniform(-0.04, 0.04)                          # the correction's own baseline slope
    for it in c_items:
        it["X"] += dx
        it["right"] += dx
        it["top"] += int(round(base + tilt * (it["X"] - cx)))
        it["wi"] = -1
    links = [([(px + dx, py + base + tilt * (px + dx - cx)) for px, py in pts], w) for pts, w in c_links]
    return c_items, links, max(i["right"] for i in c_items)


def _render_handwritten(text: str, font_path: str, fs: int, ink: int, pad_x: int, pad_y: int, join: bool,
                        extent: tuple, struck: list | None = None, roman: list | None = None,
                        fixes: list | None = None) -> Image.Image:
    ref_top, ref_bottom = extent
    items, strokes, stem, xh = _layout(text, font_path, fs, join, roman)
    if not items:
        return Image.new("L", (10, 10), 255)

    words: dict = {}
    for it in items:
        words.setdefault(it["wi"], []).append(it)
    corr_right = -1e9
    for wi, kind in enumerate(struck or []):                 # cross out the chosen words
        word = words.get(wi)
        if not kind or not word:
            continue
        fix = fixes[wi] if fixes and wi < len(fixes) else None
        part = word
        if fix:                                              # only the mistake, not its punctuation
            a, b = fix[1]
            part = [i for i in word if a <= i["ci"] < b] or word
        x0, x1 = min(i["X"] for i in part), max(i["right"] for i in part)
        width = max(2, int(stem * random.uniform(*STRIKE_WIDTH[kind])))
        y_word = (min(i["top"] for i in part), max(i["top"] + i["h"] for i in part))
        paths = _strike_paths(kind, x0, x1, xh, fs, y_word, (ref_top, ref_bottom))
        strokes += [(pts, width) for pts in paths]
        if fix:                                              # write the right word above the mistake
            strike_top = min(y for pts in paths for _, y in pts) - width / 2
            placed = _place_correction(fix[0], font_path, fs, join, x0, x1,
                                       min(y_word[0], strike_top), xh, corr_right)
            if placed:
                c_items, c_links, corr_right = placed
                items += c_items
                strokes += c_links

    x_min = min(0.0, min(i["X"] for i in items))
    y_min = min(min(i["top"] for i in items), ref_top)
    y_max = max(max(i["top"] + i["h"] for i in items), ref_bottom)
    W = int(math.ceil(max(i["right"] for i in items) - x_min)) + 2 * pad_x
    H = int(math.ceil(y_max - y_min)) + 2 * pad_y
    ox, oy = pad_x - x_min, pad_y - y_min       # canvas position of (x=0, baseline)

    canvas = np.zeros((H, W), dtype=np.uint8)
    for it in items:
        m = it["mask"]
        x0, y0 = int(round(ox + it["X"])), int(round(oy + it["top"]))
        if x0 < 0 or y0 < 0:
            m = m[max(0, -y0):, max(0, -x0):]
            x0, y0 = max(0, x0), max(0, y0)
        region = canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        np.maximum(region, m[:region.shape[0], :region.shape[1]], out=region)
    pen = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pen)
    for pts, width in strokes:
        _dab_path(draw, pts, width, ox, oy)
    canvas = np.array(pen).astype(np.float32)
    return Image.fromarray((255 - canvas * (255 - ink) / 255).astype(np.uint8))


def render_text_line(text: str, font_path: str | None = None, augment: bool = True,
                     hand: bool | None = None, connect: bool | None = None,
                     struck: list | None = None, roman: list | None = None,
                     fixes: list | None = None) -> Image.Image:
    """Render `text` as a grayscale line image of height TARGET_HEIGHT.
    struck: per word None / strike kind; fixes: per word None / (correction, (a, b)) - the correction is
    written above characters a..b of that word (those characters are the ones crossed out)."""
    font_path = font_path or random.choice(_available_fonts())
    font_size = random.randint(56, 88)  # big on purpose; downscaled at the end
    ref_top, ref_bottom = _vertical_extent(font_path, font_size)

    pad_x = int(font_size * random.uniform(0.15, 0.5))
    pad_y = int((ref_bottom - ref_top) * random.uniform(0.05, 0.14))

    ink = random.randint(0, 75)
    if (struck and any(struck)) or (hand is None and any(c in SMALL_PUNCT for c in text)):
        hand = True        # crossing out needs letter positions; lines with commas/dots must keep them visible
    if hand is None:
        hand = random.random() < HANDWRITTEN_PROB      # letter-by-letter, hand-like variation
    if connect is None:
        connect = random.random() < CONNECT_PROB       # ... with letters joined by strokes
    if hand and any(c.isalnum() for c in text):
        img = _render_handwritten(text, font_path, font_size, ink, pad_x, pad_y, join=connect,
                                  extent=(ref_top, ref_bottom), struck=struck, roman=roman, fixes=fixes)
    else:
        font = _load_font(font_path, font_size)      # only the plain path needs the sized font
        bbox = ImageDraw.Draw(Image.new("L", (10, 10))).textbbox((0, 0), text, font=font, anchor="ls")
        top = min(ref_top, bbox[1])
        bottom = max(ref_bottom, bbox[3])
        w = max(1, bbox[2] - bbox[0]) + 2 * pad_x
        h = (bottom - top) + 2 * pad_y
        img = Image.new("L", (w, h), color=255)
        ImageDraw.Draw(img).text((pad_x - bbox[0], pad_y - top), text, font=font, fill=ink, anchor="ls")

    if augment:
        img = _augment(img)

    w, h = img.size
    new_w = max(1, int(round(w * (TARGET_HEIGHT / h))))
    img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

    if augment:
        img = _degrade(img)
    return img


# --------------------------------------------------------------------------- augmentation

def _min3x3(a: np.ndarray) -> np.ndarray:
    """3x3 minimum filter (= ImageFilter.MinFilter(3)); outside the image counts as white paper."""
    p = np.pad(a, 1, mode="constant", constant_values=255)
    r = np.minimum(np.minimum(p[:-2], p[1:-1]), p[2:])
    return np.minimum(np.minimum(r[:, :-2], r[:, 1:-1]), r[:, 2:])


def _rng() -> np.random.Generator:
    """numpy generator driven by `random`, so random.seed() controls everything."""
    return np.random.default_rng(random.getrandbits(64))


def _paper_background(w: int, h: int) -> np.ndarray:
    """Grey-ish paper: plain, notebook grid or ruled lines (like the real scans)."""
    level = random.randint(185, 255)
    bg = np.full((h, w), level, dtype=np.float32)
    style = random.choices(["plain", "grid", "ruled"], weights=[0.3, 0.5, 0.2])[0]
    thick = max(1, int(round(h / 70)))
    dark = random.randint(15, 55)
    if style == "grid":
        cell = max(8, int(h * random.uniform(0.18, 0.4)))
        ox, oy = random.randint(0, cell - 1), random.randint(0, cell - 1)
        for x in range(ox, w, cell):
            bg[:, x:x + thick] -= dark
        for y in range(oy, h, cell):
            bg[y:y + thick, :] -= dark
    elif style == "ruled":
        step = max(10, int(h * random.uniform(0.45, 0.9)))
        for y in range(random.randint(0, step - 1), h, step):
            bg[y:y + thick, :] -= dark
    return np.clip(bg, 0, 255)


def _augment(img: Image.Image) -> Image.Image:
    """High-resolution part: ink thickness, slant, wavy baseline, paper and light."""
    # thicker ink (3x3 minimum grows dark strokes on a light background). Same result as
    # ImageFilter.MinFilter(3), but separable in numpy: ~25x faster on a big line.
    if random.random() < 0.3:
        img = Image.fromarray(_min3x3(np.asarray(img)))

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

    # wavy baseline: shift every column up/down by a slow sine wave. The wave is slow, so neighbouring
    # columns share the same integer shift: move whole runs of columns at once (instead of a per-pixel gather).
    w, h = img.size
    amp = int(random.uniform(0, 0.05) * h)
    if amp > 0:
        img = ImageOps.expand(img, border=(0, amp, 0, amp), fill=255)
        arr = np.asarray(img)
        H, W = arr.shape
        wavelength = random.uniform(300, 1200)
        phase = random.uniform(0, 2 * np.pi)
        shifts = np.round(amp * np.sin(2 * np.pi * np.arange(W) / wavelength + phase)).astype(int)
        out = np.empty_like(arr)
        cuts = np.flatnonzero(np.diff(shifts)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, W]):
            out[:, a:b] = np.roll(arr[:, a:b], shifts[a], axis=0)
        img = Image.fromarray(out)

    # paper + uneven lighting, ink stays darker than paper
    w, h = img.size
    text = np.array(img).astype(np.float32)
    out = np.minimum(text, _paper_background(w, h))
    light = np.linspace(random.uniform(0.75, 1.0), random.uniform(0.75, 1.0), w, dtype=np.float32)
    out = out * light[None, :]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _degrade(img: Image.Image) -> Image.Image:
    """Final-resolution part (64 px): blur, sensor noise, JPEG artefacts - what the camera does."""
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))
    if random.random() < 0.6:
        arr = np.array(img).astype(np.float32)
        arr += _rng().standard_normal(arr.shape, dtype=np.float32) * random.uniform(2, 10)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if random.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(35, 85))
        buf.seek(0)
        img = Image.open(buf).convert("L")
    return img


def generate_batch(n: int, words: list[str] | None = None) -> list[tuple[Image.Image, str]]:
    words = words or load_words()
    return [make_sample(words) for _ in range(n)]


if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    for i, (img, text) in enumerate(generate_batch(20)):
        img.save(f"samples/sample_{i}.png")
        print(f"sample_{i}.png -> {text!r}  size={img.size}")
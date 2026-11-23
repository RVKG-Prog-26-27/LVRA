"""
Alphabet (character vocabulary).

Every character gets a unique number: "a" -> 1, "ā" -> 2 and so on. This file holds that
"character <-> number" table and converts in both directions.

Index 0 is the CTC "blank" symbol. It is not a real character; CTC needs it to separate
repeated letters and to mark "nothing here".

IMPORTANT: web-app/server/alphabet_table.py keeps a hand-made copy of this table (the server runs
without PyTorch). Whenever the character set below changes, a model has to be retrained AND that copy
has to be updated, otherwise the server will print wrong letters.
"""
import math

LATVIAN_LETTERS = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž"
BASE_LATIN = "qwxy"
DIGITS = "0123456789"
EXTRA_LETTERS = "öüÇĆ"          # rare letters that appear in names and loan words

# --- punctuation and symbols (grouped only for readability; CHARS is built from all groups) ---

# Basic punctuation. The space is a character too (CTC has to be able to output it).
PUNCT_BASIC = " .,!?-:;()\"'/•"
# Quotes, dashes, ellipsis and other typographic marks.
PUNCT_TYPOGRAPHIC = (
    "\u201e\u201c\u201d\u201a\u2018\u2019"    # „ “ ” ‚ ‘ ’  (Latvian and curved quotes)
    "\u2039\u203a«»"                          # ‹ › « »
    "\u2013\u2014\u2212"                      # – — −  (en dash, em dash, minus sign)
    "\u2026"                                  # …
    "\u00b7"                                  # ·  (middle dot)
    "\u2032\u2033"                            # ′ ″  (prime, double prime)
    "`\u00b4"                                 # ` ´
    "\u00a1\u00bf"                            # ¡ ¿
)
BRACKETS = "[]{}"
# Keyboard symbols.
SYMBOLS_KEYBOARD = "*_+=<>@#%&^~|\\"
# Math, arrows, signs.
SYMBOLS_MATH = (
    "\u00b1\u00d7\u00f7"                      # ± × ÷
    "\u2260\u2264\u2265\u2248"                # ≠ ≤ ≥ ≈
    "\u221e\u221a"                            # ∞ √
    "\u00b0\u2030"                            # ° ‰
    "\u2190\u2192\u2191\u2193\u21d2"          # ← → ↑ ↓ ⇒
)
# Superscripts and fractions.
SYMBOLS_NUMERIC = "\u00b2\u00b3\u00b9\u00bc\u00bd\u00be"   # ² ³ ¹ ¼ ½ ¾
# Other signs and currencies.
SYMBOLS_OTHER = (
    "\u00a7\u00b6"                            # § ¶
    "\u00a9\u00ae\u2122"                      # © ® ™
    "\u2116"                                  # №
    "\u20ac$\u00a3\u00a5\u00a2"               # € $ £ ¥ ¢
)

# Special symbol: "this word is crossed out". One symbol = one crossed-out word. No font has it and it is
# never drawn; it only appears in labels, and the network "writes" it in place of a struck word.
STRIKE_MARK = "\u00a4"   # ¤

ALL_PUNCT = (PUNCT_BASIC + PUNCT_TYPOGRAPHIC + BRACKETS + SYMBOLS_KEYBOARD + SYMBOLS_MATH
             + SYMBOLS_NUMERIC + SYMBOLS_OTHER)

# - LATVIAN_LETTERS.upper() adds the capital letters (A, Ā, B, ...);
# - set(...) removes duplicates;
# - sorted(...) makes the order identical on every run (it defines the class numbers!).
CHARS = sorted(set(
    LATVIAN_LETTERS + LATVIAN_LETTERS.upper()
    + BASE_LATIN + BASE_LATIN.upper()
    + DIGITS + EXTRA_LETTERS + ALL_PUNCT + STRIKE_MARK
))

BLANK = "<blank>"

ALPHABET = [BLANK] + CHARS

# Two dictionaries for translating in both directions.
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

# Number of classes the network chooses from at every step: every character + blank.
NUM_CLASSES = len(ALPHABET)


def encode(text: str) -> list[int]:
    """Convert text to a list of character codes (blank is never produced)."""
    unknown = set(ch for ch in text if ch not in CHAR_TO_IDX)
    if unknown:
        raise ValueError(f"The text contains characters that are not in the alphabet: {unknown!r}.")
    return [CHAR_TO_IDX[ch] for ch in text]


def decode_greedy(indices: list[int]) -> str:
    """Convert the network output back to text.

    For the word "sveiki" the network might output: s s <blank> v e e i <blank> k k i.
    Rules:
      1) merge consecutive equal symbols into one (e e -> e);
      2) drop blank (it means "nothing here"; it also lets a double letter survive: l <blank> l -> ll).
    """
    out = []
    prev = None   # index of the previous step (none at the start)
    for idx in indices:
        if idx != prev:           # the index changed -> this is a new symbol (rule 1)
            if idx != 0:          # rule 2
                out.append(IDX_TO_CHAR[idx])
        prev = idx
    return "".join(out)


def decode_beam_search(log_probs, beam_width: int = 5) -> str:
    """CTC prefix beam search: keeps several best text variants at the same time.

    For every prefix two probabilities are stored separately: that it ends with a CTC blank, and that it
    ends with a normal symbol. log_probs: tensor/array of shape [T, C] (log-probabilities).
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    if log_probs.ndim != 2:
        raise ValueError(f"decode_beam_search expects a [T, C] array, got shape {tuple(log_probs.shape)}")
    if log_probs.size(1) > NUM_CLASSES:
        raise ValueError(
            f"The model has {log_probs.size(1)} classes, but alphabet.py defines only {NUM_CLASSES}. "
            "The checkpoint was trained with a different alphabet."
        )

    def log_add(a: float, b: float) -> float:
        """log(exp(a) + exp(b))"""
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        if a < b:
            a, b = b, a
        return a + math.log1p(math.exp(b - a))

    # The empty prefix starts with probability 1 via blank => log(1) = 0.
    neg_inf = -math.inf
    beams = {(): (0.0, neg_inf)}
    probs = log_probs.detach().float().cpu()

    for t in range(probs.size(0)):
        step = probs[t].tolist()
        next_beams = {}

        def add(prefix, p_blank=neg_inf, p_nonblank=neg_inf):
            old_blank, old_nonblank = next_beams.get(prefix, (neg_inf, neg_inf))
            next_beams[prefix] = (
                log_add(old_blank, p_blank),
                log_add(old_nonblank, p_nonblank),
            )

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = log_add(p_blank, p_nonblank)

            # CTC blank: the text prefix does not change.
            add(prefix, p_blank=total + step[0])

            last = prefix[-1] if prefix else None

            for c in range(1, log_probs.size(1)):
                p = step[c]

                if c == last:
                    # The same symbol without a blank in between stays the same prefix:
                    add(prefix, p_nonblank=p_nonblank + p)

                    # If a blank came before the second c, it is a new letter: ... c blank c -> ... cc.
                    add(prefix + (c,), p_blank=p_blank + p)
                else:
                    # A new symbol can follow both a blank state and a non-blank state.
                    add(prefix + (c,), p_nonblank=total + p)

        # Sort by the total probability of the prefix and keep the best beam_width.
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda item: log_add(item[1][0], item[1][1]),
                reverse=True,
            )[:beam_width]
        )

    best_prefix = max(
        beams,
        key=lambda prefix: log_add(beams[prefix][0], beams[prefix][1]),
    )
    return "".join(IDX_TO_CHAR[i] for i in best_prefix)


def strip_struck(text: str) -> str:
    """Remove the crossed-out-word marks (¤) from recognised text and normalise the spaces.
    "Ekonomika ¤ tirgus" -> "Ekonomika tirgus" """
    return " ".join(text.replace(STRIKE_MARK, " ").split())

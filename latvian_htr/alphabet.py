"""
Charset / vocabulary for Latvian handwriting recognition (CTC-based).

Index 0 is reserved for the CTC "blank" symbol - do not remove it.
"""

LATVIAN_LETTERS = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž"
BASE_LATIN = "qwxy"  # appear in loanwords / foreign names, kept for robustness
DIGITS = "0123456789"
PUNCT = " .,!?-:;()\"'/"

CHARS = sorted(set(LATVIAN_LETTERS + LATVIAN_LETTERS.upper() + BASE_LATIN + BASE_LATIN.upper() + DIGITS + PUNCT))

BLANK = "<blank>"
ALPHABET = [BLANK] + CHARS

CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

NUM_CLASSES = len(ALPHABET)


def encode(text: str) -> list[int]:
    """Text -> list of class indices (no blanks inserted, CTCLoss wants targets without blanks)."""
    unknown = set(ch for ch in text if ch not in CHAR_TO_IDX)
    if unknown:
        raise ValueError(f"Unsupported characters in text: {unknown!r}. Extend alphabet.py CHARS.")
    return [CHAR_TO_IDX[ch] for ch in text]


def decode_greedy(indices: list[int]) -> str:
    """Collapse repeats and drop blanks - standard CTC greedy decoding."""
    out = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != 0:  # 0 == blank
                out.append(IDX_TO_CHAR[idx])
        prev = idx
    return "".join(out)

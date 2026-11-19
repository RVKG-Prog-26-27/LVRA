"""
The character tables and CTC decoding, copied from latvian_htr/alphabet.py.

Why a copy: latvian_htr/alphabet.py imports PyTorch, and PyTorch does not
fit on the free PythonAnywhere plan. This file has the same tables and the
same decoding, without PyTorch.

Two tables are kept, because checkpoints exist for both:
  137 classes  the current latvian_htr/alphabet.py (crnn_best_migrated.pt and newer)
  131 classes  the earlier alphabet, latvian_htr/old_alphabet.py (crnn_best.pt)
The server counts the rows of the model's last layer (rnn.fc.weight) and
picks the table with that many classes, so either kind of checkpoint reads
the right letters. A checkpoint with any other count is refused with a clear
message in the error log.

KEEP IN SYNC: if the AI side changes LATVIAN_LETTERS, BASE_LATIN, DIGITS,
PUNCT or STRIKE_MARK in latvian_htr/alphabet.py, update CURRENT_PUNCT (or add
a new table) here. Otherwise the numbers no longer mean the same letters.

The en and em dash are written as \\u escapes so the source file does not contain those characters.
"""

import math

LATVIAN_LETTERS = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž"
BASE_LATIN = "qwxy"
DIGITS = "0123456789"

_PUNCT_TAIL = (
    "\u201e\u201c\u201d\u2018\u2019"         # „ " " ' ' (latviešu/izliektas pēdiņas)
    + "\u2013\u2014"                         # defise en un em
    + "[]{}"                                 # iekavas
    + "«»"                                   # eglītes
    + "*_+=<>@#%&^~|\\"                      # simboli
    + "\u2026"                               # …
    + "\u2116"                               # №
    + "€$"                                   # valūta
)

# latvian_htr/alphabet.py now (137 classes): the basic punctuation plus ° ≈ ö Ç ü Ć
CURRENT_PUNCT = " .,!?-:;()\"'/•°≈öÇöüĆ" + _PUNCT_TAIL
# latvian_htr/old_alphabet.py (131 classes), the table crnn_best.pt was trained with
OLD_PUNCT = " .,!?-:;()\"'/•" + _PUNCT_TAIL

# The model writes this mark in place of one crossed-out word (same as alphabet.STRIKE_MARK).
STRIKE_MARK = "\u00a4"
BLANK = "<blank>"


def _build(punct):
    chars = sorted(
        set(LATVIAN_LETTERS + LATVIAN_LETTERS.upper() + BASE_LATIN + BASE_LATIN.upper() + DIGITS + punct + STRIKE_MARK)
    )
    return [BLANK] + chars


ALPHABETS = {len(a): a for a in (_build(CURRENT_PUNCT), _build(OLD_PUNCT))}

# The current table (same as latvian_htr/alphabet.py).
ALPHABET = _build(CURRENT_PUNCT)
NUM_CLASSES = len(ALPHABET)


def alphabet_for(num_classes):
    """The table with num_classes entries, or a clear error if the server has none."""
    if num_classes not in ALPHABETS:
        known = ", ".join(str(n) for n in sorted(ALPHABETS))
        raise RuntimeError(
            f"The checkpoint has {num_classes} classes, but the server only knows tables with {known}. "
            "Update server/alphabet_table.py to match latvian_htr/alphabet.py."
        )
    return ALPHABETS[num_classes]


def decode_greedy(indices, alphabet=ALPHABET):
    """Same as alphabet.decode_greedy: merge repeats, drop blanks (index 0)."""
    out = []
    prev = None
    for idx in indices:
        if idx != prev and idx != 0:
            out.append(alphabet[idx])
        prev = idx
    return "".join(out)


def _log_add(a, b):
    """log(exp(a) + exp(b))"""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def decode_beam_search(log_probs, beam_width=5, prune=12.0, alphabet=ALPHABET):
    """
    Same algorithm as alphabet.decode_beam_search (CTC prefix beam search),
    for a numpy array log_probs [T, C]. alphabet must have C entries.

    One addition for speed: at each step only characters whose log
    probability is within `prune` of that step's best are tried (12 means
    about 160 000 times less likely than the best). Such characters cannot
    reach the 5 best guesses, so the result is the same while recognition is
    many times faster on the server.
    """
    if log_probs.shape[1] != len(alphabet):
        raise ValueError(f"Model gives {log_probs.shape[1]} classes, the table has {len(alphabet)}")
    neg_inf = -math.inf
    beams = {(): (0.0, neg_inf)}  # prefix -> (log p ending in blank, log p ending in a character)

    for step in log_probs:
        step = step.tolist()
        best = max(step)
        candidates = [c for c in range(1, len(step)) if step[c] >= best - prune]
        next_beams = {}

        def add(prefix, p_blank=neg_inf, p_nonblank=neg_inf):
            old_blank, old_nonblank = next_beams.get(prefix, (neg_inf, neg_inf))
            next_beams[prefix] = (_log_add(old_blank, p_blank), _log_add(old_nonblank, p_nonblank))

        for prefix, (p_blank, p_nonblank) in beams.items():
            total = _log_add(p_blank, p_nonblank)
            add(prefix, p_blank=total + step[0])
            last = prefix[-1] if prefix else None
            for c in candidates:
                p = step[c]
                if c == last:
                    add(prefix, p_nonblank=p_nonblank + p)
                    add(prefix + (c,), p_blank=p_blank + p)
                else:
                    add(prefix + (c,), p_nonblank=total + p)

        beams = dict(
            sorted(next_beams.items(), key=lambda item: _log_add(item[1][0], item[1][1]), reverse=True)[:beam_width]
        )

    best_prefix = max(beams, key=lambda prefix: _log_add(beams[prefix][0], beams[prefix][1]))
    return "".join(alphabet[i] for i in best_prefix)


def strip_struck(text):
    """Same as alphabet.strip_struck: removes the crossed-out word marks and tidies the spaces.
    "Ekonomika ¤ tirgus" -> "Ekonomika tirgus" """
    return " ".join(text.replace(STRIKE_MARK, " ").split())

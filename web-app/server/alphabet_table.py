"""
The character table and CTC decoding, copied from latvian_htr/alphabet.py.

Why a copy: latvian_htr/alphabet.py imports PyTorch, and PyTorch does not
fit on the free PythonAnywhere plan. This file has the same table and the
same decoding, without PyTorch.

KEEP IN SYNC: if the AI side changes LATVIAN_LETTERS, BASE_LATIN, DIGITS or
PUNCT or STRIKE_MARK in latvian_htr/alphabet.py, make the same change here. Otherwise the
numbers no longer mean the same letters. If the number of characters does
not match the model file, the server refuses to start recognising and
writes the reason to the error log.

The en and em dash are written as \\u escapes so the source file does not contain those characters.
"""

import math

LATVIAN_LETTERS = "aābcčdeēfgģhiījkķlļmnņoprsštuūvzž"
BASE_LATIN = "qwxy"
DIGITS = "0123456789"
PUNCT = (
    " .,!?-:;()\"'/•"                        # pamata pieturzīmes
    + "\u201e\u201c\u201d\u2018\u2019"       # „ " " ' ' (latviešu/izliektas pēdiņas)
    + "\u2013\u2014"                         # defise en un em
    + "[]{}"                                 # iekavas 
    + "«»"                                   # eglītes
    + "*_+=<>@#%&^~|\\"                      # simboli
    + "\u2026"                               # … 
    + "\u2116"                               # № 
    + "€$"                                   # valūta 
)

# The model writes this mark in place of one crossed-out word (same as alphabet.STRIKE_MARK).
STRIKE_MARK = "\u00a4"

CHARS = sorted(set(LATVIAN_LETTERS + LATVIAN_LETTERS.upper() + BASE_LATIN + BASE_LATIN.upper() + DIGITS + PUNCT + STRIKE_MARK))
BLANK = "<blank>"
ALPHABET = [BLANK] + CHARS
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}
NUM_CLASSES = len(ALPHABET)


def decode_greedy(indices):
    """Same as alphabet.decode_greedy: merge repeats, drop blanks (index 0)."""
    out = []
    prev = None
    for idx in indices:
        if idx != prev and idx != 0:
            out.append(IDX_TO_CHAR[idx])
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


def decode_beam_search(log_probs, beam_width=5, prune=12.0):
    """
    Same algorithm as alphabet.decode_beam_search (CTC prefix beam search),
    for a numpy array log_probs [T, C].

    One addition for speed: at each step only characters whose log
    probability is within `prune` of that step's best are tried (12 means
    about 160 000 times less likely than the best). Such characters cannot
    reach the 5 best guesses, so the result is the same while recognition is
    many times faster on the server.
    """
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
    return "".join(IDX_TO_CHAR[i] for i in best_prefix)


def strip_struck(text):
    """Same as alphabet.strip_struck: removes the crossed-out word marks and tidies the spaces.
    "Ekonomika \u00a4 tirgus" -> "Ekonomika tirgus" """
    return " ".join(text.replace(STRIKE_MARK, " ").split())

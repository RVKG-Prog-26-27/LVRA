"""
Word confidence and the three categories of a recognised word.

Every word of a recognised line gets a confidence (0..1) and falls into one of three categories:

    1 пункт - could not recognise it   -> the site gets NO value for this word (site_value = None)
    2 пункт - not sure                 -> the site gets ANOTHER value (see _alternative: the weakest letter is
                                          replaced by the network's second choice for that place)
    3 пункт - sure                     -> everything is fine, nothing changes, the word goes to the site as read

Only numpy is used (no PyTorch), so this file can be copied to the web server next to alphabet.py.

Tune the two thresholds below after looking at `python evaluate.py ...` (it prints how many words fall into
every category and how many of them are really correct).
"""
from dataclasses import dataclass

import numpy as np

from alphabet import CHAR_TO_IDX, IDX_TO_CHAR

# confidence below this value        -> category 1 (could not recognise)
CONF_UNREADABLE = 0.40
# confidence from this value and up  -> category 3 (sure); everything in between -> category 2 (not sure)
CONF_SURE = 0.85

CAT_UNREADABLE, CAT_UNSURE, CAT_SURE = 1, 2, 3

_SPACE = CHAR_TO_IDX[" "]


@dataclass
class WordResult:
    raw: str                  # the word exactly as the network read it
    confidence: float         # 0..1
    category: int             # 1, 2 or 3
    site_value: "str | None"  # what goes to the site: None for category 1, another value for 2, raw for 3


def classify(confidence: float) -> int:
    if confidence < CONF_UNREADABLE:
        return CAT_UNREADABLE
    if confidence < CONF_SURE:
        return CAT_UNSURE
    return CAT_SURE


def _alternative(word_chars, log_probs) -> str:
    """Another value for an unsure word: the weakest letter is replaced by the second best symbol the network
    saw at the same place (blank and space are not allowed as a replacement)."""
    weakest = min(word_chars, key=lambda c: c[1])
    ch, _prob, frame, idx = weakest
    scores = log_probs[frame].copy()
    scores[0] = -np.inf
    scores[_SPACE] = -np.inf
    scores[idx] = -np.inf
    alt = IDX_TO_CHAR[int(scores.argmax())]
    return "".join(alt if c is weakest else c[0] for c in word_chars)


def read_words(log_probs) -> list[WordResult]:
    """log_probs: [T, C] log-probabilities of ONE line (numpy array or torch tensor). Greedy CTC reading;
    returns the words of the line with confidence, category and the value for the site."""
    if hasattr(log_probs, "detach"):
        log_probs = log_probs.detach().float().cpu().numpy()
    lp = np.asarray(log_probs, dtype=np.float32)
    best = lp.argmax(axis=1)

    # 1) collapse CTC: one entry per emitted symbol = (char, probability, frame of the best step, class index)
    emitted = []
    t = 0
    while t < len(best):
        idx = int(best[t])
        end = t
        while end + 1 < len(best) and best[end + 1] == idx:
            end += 1
        if idx != 0:
            frame = t + int(lp[t:end + 1, idx].argmax())
            emitted.append((IDX_TO_CHAR[idx], float(np.exp(lp[frame, idx])), frame, idx))
        t = end + 1

    # 2) cut the symbols into words by spaces
    words, current = [], []
    for item in emitted + [(" ", 1.0, -1, _SPACE)]:
        if item[3] == _SPACE:
            if current:
                words.append(current)
            current = []
        else:
            current.append(item)

    # 3) confidence of a word = geometric mean of its letters' probabilities
    results = []
    for chars in words:
        raw = "".join(c[0] for c in chars)
        conf = float(np.exp(np.mean([np.log(max(c[1], 1e-9)) for c in chars])))
        cat = classify(conf)
        if cat == CAT_UNREADABLE:
            site = None
        elif cat == CAT_UNSURE:
            site = _alternative(chars, lp)
        else:
            site = raw
        results.append(WordResult(raw, conf, cat, site))
    return results


def site_text(words: list[WordResult]) -> str:
    """The line as it goes to the site: category 1 words are left out (no value), category 2 words are replaced."""
    return " ".join(w.site_value for w in words if w.site_value is not None)


def category_percent(words: list[WordResult]) -> dict[int, float]:
    total = max(len(words), 1)
    return {c: 100.0 * sum(w.category == c for w in words) / total
            for c in (CAT_UNREADABLE, CAT_UNSURE, CAT_SURE)}


def format_categories(words: list[WordResult]) -> str:
    """Console form:   1 пункт(0%)  2 пункт(50%)  3 пункт(50%)   (one per line)"""
    pct = category_percent(words)
    return "\n".join(f"{c} пункт({pct[c]:.0f}%)" for c in (CAT_UNREADABLE, CAT_UNSURE, CAT_SURE))

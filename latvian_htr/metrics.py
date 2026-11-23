"""
Recognition quality metrics (no PyTorch needed).

CER (Character Error Rate) = (substitutions + deletions + insertions) / number of characters in the reference.
CER 0.05 = 5% of characters are wrong. WER is the same on the level of words. Lower is better.
"""


def edit_distance(a, b) -> int:
    """Levenshtein distance: the minimal number of edits that turn sequence a into sequence b
    (works for strings and for lists of words)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(refs: list[str], hyps: list[str]) -> float:
    """Character error rate of a whole set (sum of all edits / sum of all reference lengths)."""
    total_edits = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    total_chars = sum(len(r) for r in refs)
    return total_edits / max(total_chars, 1)


def wer(refs: list[str], hyps: list[str]) -> float:
    """Word error rate of a whole set; words are separated by whitespace."""
    total_edits = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    total_words = sum(len(r.split()) for r in refs)
    return total_edits / max(total_words, 1)

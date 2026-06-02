from __future__ import annotations

import re

try:
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein
except Exception:  # pragma: no cover - optional acceleration dependency
    _RapidLevenshtein = None


MAX_CER_CELLS = 5_000_000
MAX_WER_CELLS = 2_000_000


def normalize_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    if _RapidLevenshtein is not None:
        return int(_RapidLevenshtein.distance(a, b))

    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def cer(pred: str, truth: str) -> float | None:
    truth_n = normalize_text(truth)
    pred_n = normalize_text(pred)
    if not truth_n:
        return None
    if _RapidLevenshtein is None and len(pred_n) * len(truth_n) > MAX_CER_CELLS:
        return None
    return edit_distance(pred_n, truth_n) / max(1, len(truth_n))


def wer(pred: str, truth: str) -> float | None:
    truth_words = normalize_text(truth).split()
    pred_words = normalize_text(pred).split()
    if not truth_words:
        return None
    if _RapidLevenshtein is None and len(pred_words) * len(truth_words) > MAX_WER_CELLS:
        return None
    return edit_distance(pred_words, truth_words) / max(1, len(truth_words))

"""Closed title list for the phase-1 recognizer.

The synthetic generator only emits these strings, so decoding searches the
list instead of trusting a raw length. Production models should keep the
lexicon as a post-process on top of CTC, not as the model itself.
"""

from __future__ import annotations

import numpy as np

from rapidstamp.charset import CHARSET

CITIES = "京沪粤浙苏川渝津冀鲁湘鄂闽皖赣陕晋桂云黔琼辽"
BRANDS = ["中天", "明天", "新华", "新盛", "达通", "信源", "华信", "明达", "天盛", "中信"]
SUFFIXES = ["有限公司", "股份公司", "集团"]
FIXED = ["发票专用章", "合同专用章", "财务专用章", "集团专用章"]


def seal_titles() -> list[str]:
    titles = list(FIXED)
    for city in CITIES:
        for brand in BRANDS:
            for suffix in SUFFIXES:
                titles.append(city + brand + suffix)
    missing = sorted({ch for title in titles for ch in title if ch not in CHARSET})
    if missing:
        raise RuntimeError(f"title characters outside CHARSET: {missing}")
    return titles


def decode_titles(log_probs: dict[int, np.ndarray], titles: list[str], charset: str) -> tuple[str, float]:
    """Pick the title with the highest sum of per-character log-probabilities.

    ``log_probs[n]`` is ``(n, C)`` for a strip cut into ``n`` slots. The score
    returned with the string is the geometric mean of those probabilities.
    """
    index = {ch: i for i, ch in enumerate(charset)}
    best_text = ""
    best = -1.0e300
    grouped: dict[int, list[str]] = {}
    for title in titles:
        grouped.setdefault(len(title), []).append(title)
    for n in sorted(grouped):
        table = log_probs.get(n)
        if table is None:
            continue
        for title in grouped[n]:
            score = 0.0
            ok = True
            for i, ch in enumerate(title):
                cls = index.get(ch)
                if cls is None or cls >= table.shape[1]:
                    ok = False
                    break
                score += float(table[i, cls])
            if ok and score > best:
                best = score
                best_text = title
    if not best_text:
        return "", 0.0
    return best_text, float(np.exp(best / len(best_text)))

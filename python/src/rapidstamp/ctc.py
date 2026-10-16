"""Greedy CTC decode for the seal recognizer."""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-6)


def ctc_greedy(logits: np.ndarray, charset: list[str], blank: int = 0) -> tuple[str, float]:
    """Decode (T, C) or (1, T, C) logits. ``charset`` does not include blank.

    The score is the sum of emitted-step probabilities divided by the fixed
    time length. Averaging only the emitted characters lets a short confident
    fragment beat a longer title during the angle search.
    """
    scores = np.asarray(logits, dtype=np.float32)
    if scores.ndim == 3:
        scores = scores[0]
    probs = softmax(scores)
    indices = probs.argmax(axis=-1)
    chars: list[str] = []
    chosen: list[float] = []
    previous = blank
    for step, index in enumerate(indices.tolist()):
        if index != blank and index != previous:
            char_index = index - 1
            if 0 <= char_index < len(charset):
                chars.append(charset[char_index])
                chosen.append(float(probs[step, index]))
        previous = index
    steps = int(scores.shape[0])
    if not chosen or steps <= 0:
        return "", 0.0
    return "".join(chars), float(sum(chosen) / steps)

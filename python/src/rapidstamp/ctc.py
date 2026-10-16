"""Greedy CTC decode shared by the numpy engine and training evaluation."""

from __future__ import annotations

import numpy as np

from rapidstamp.charset import CHARSET


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def ctc_greedy(logits: np.ndarray, charset: str = CHARSET) -> tuple[str, float]:
    """Decode ``(T, C)`` logits. Class 0 is the blank.

    The reported score is the mean softmax probability of the emitted
    (non-blank, collapsed) steps. Empty predictions score 0.
    """
    if logits.ndim != 2:
        raise ValueError(f"expected (T, C) logits, got {logits.shape}")
    probs = softmax(logits.astype(np.float32))
    indices = np.argmax(logits, axis=-1)
    chars: list[str] = []
    confs: list[float] = []
    prev = -1
    for t, idx in enumerate(indices.tolist()):
        if idx != 0 and idx != prev:
            if idx - 1 >= len(charset):
                raise ValueError(f"class index {idx} exceeds charset")
            chars.append(charset[idx - 1])
            confs.append(float(probs[t, idx]))
        prev = idx
    text = "".join(chars)
    score = float(np.mean(confs)) if confs else 0.0
    return text, score

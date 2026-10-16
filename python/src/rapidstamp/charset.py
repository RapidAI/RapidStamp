"""Seal-title character set used by the phase-1 recognizer.

Index 0 is the CTC blank and is not stored here. Character ``CHARSET[i]`` has
class index ``i + 1``.
"""

from __future__ import annotations

CHARSET = (
    "京沪粤浙苏川渝津冀鲁湘鄂闽皖赣陕晋桂云黔琼辽"
    "公司有限责任股份"
    "发票专用章合同财务集团"
    "中天明华新盛达通信源"
)

if len(CHARSET) != len(set(CHARSET)):
    raise RuntimeError("CHARSET contains duplicate characters")


def encode(text: str) -> list[int]:
    index = {ch: i + 1 for i, ch in enumerate(CHARSET)}
    try:
        return [index[ch] for ch in text]
    except KeyError as exc:
        raise ValueError(f"character {exc} is outside the phase-1 charset") from exc


def decode_indices(indices: list[int]) -> str:
    chars = []
    for idx in indices:
        if idx <= 0:
            continue
        chars.append(CHARSET[idx - 1])
    return "".join(chars)

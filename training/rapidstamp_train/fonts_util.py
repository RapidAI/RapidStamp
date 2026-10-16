"""Locate the CJK and Latin fonts used by the synthesizer."""

from __future__ import annotations

import os
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(candidates: list[str | None]) -> str | None:
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    return None


def find_cjk_font() -> str:
    found = _first_existing(
        [
            os.environ.get("RAPIDSTAMP_CJK_FONT"),
            str(_TRAINING_ROOT / "fonts" / "RapidStampCJK-ext.ttf"),
            str(_TRAINING_ROOT / "fonts" / "RapidStampCJK.ttf"),
            str(_TRAINING_ROOT / "fonts" / "RapidStampCJK.otf"),
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
    )
    if found is None:
        raise FileNotFoundError(
            "No CJK font found. Set RAPIDSTAMP_CJK_FONT or add training/fonts/RapidStampCJK-ext.ttf."
        )
    return found


def find_latin_font() -> str:
    found = _first_existing(
        [
            os.environ.get("RAPIDSTAMP_LATIN_FONT"),
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            str(_TRAINING_ROOT / "fonts" / "DejaVuSans-Bold.ttf"),
        ]
    )
    if found is None:
        raise FileNotFoundError(
            "No Latin font found. Install DejaVu Sans or set RAPIDSTAMP_LATIN_FONT."
        )
    return found

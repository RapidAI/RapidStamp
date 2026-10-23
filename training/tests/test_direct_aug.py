import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rapidstamp_train.direct_aug import (
    Charset,
    augment_seal,
    match_gold_image,
    pad_square,
    perspective_coeffs,
)


def test_charset_roundtrip_keeps_spaces_and_stops_at_eos():
    charset = Charset(["甲", "乙", " "])
    ids = charset.encode("甲 乙")
    assert ids is not None
    assert charset.decode(ids + [9, 8]) == "甲 乙"
    assert charset.encode("甲" * 26) is None
    assert charset.encode("丙") is None


def test_gold_name_matches_the_numbered_crop_only():
    paths = [
        Path("008_1b15ccfe098911ed9a5b6c4b90409ede_0.png"),
        Path("009_1b15ccfe098911ed9a5b6c4b90409ede_1.png"),
        Path("001_322b29cc4a9b11edacf3fa163d64817d.png"),
    ]
    assert match_gold_image("images/1b15ccfe098911ed9a5b6c4b90409ede_0.png", paths) == paths[0]
    assert match_gold_image("images/1b15ccfe098911ed9a5b6c4b90409ede_1.png", paths) == paths[1]
    assert match_gold_image("compound-img/322b29cc4a9b11edacf3fa163d64817d.png", paths) == paths[2]


def test_identity_perspective_is_a_copy():
    corners = [(0.0, 0.0), (9.0, 0.0), (9.0, 9.0), (0.0, 9.0)]
    coeffs = perspective_coeffs(corners, corners)
    assert coeffs[0] == pytest.approx(1.0, abs=1e-5)
    assert coeffs[4] == pytest.approx(1.0, abs=1e-5)
    assert coeffs[2] == pytest.approx(0.0, abs=1e-5)
    assert coeffs[5] == pytest.approx(0.0, abs=1e-5)


def test_augment_keeps_the_square_and_rotation_moves_ink():
    image = Image.new("RGB", (48, 32), (255, 255, 255))
    image.putpixel((0, 0), (200, 20, 20))
    square = pad_square(image)
    assert square.size == (48, 48)
    spun = square.rotate(180, fillcolor=(255, 255, 255))
    assert spun.getpixel((47, 47))[0] > 150
    out = augment_seal(square, random.Random(0), font_path=None)
    assert out.size == square.size
    assert out.mode == "RGB"
    assert np.asarray(out).shape == (48, 48, 3)

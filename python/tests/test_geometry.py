import numpy as np

from rapidstamp.geometry import polar_unwrap
from rapidstamp.labels import (
    ANGLE_STEPS,
    DET_SIZE,
    GRID,
    REC_H,
    REC_W,
    UNWRAP_R_IN,
    UNWRAP_R_OUT,
    UNWRAP_SPAN_DEG,
    UNWRAP_START_DEG,
)


def test_constants_match_cpp_header():
    text = open("cpp/src/constants.h", encoding="utf-8").read()
    assert f"kDetSize = {DET_SIZE}" in text
    assert f"kGrid = {GRID}" in text
    assert f"kRecH = {REC_H}" in text
    assert f"kRecW = {REC_W}" in text
    assert f"kAngleSteps = {ANGLE_STEPS}" in text
    assert f"kUnwrapStartDeg = {int(UNWRAP_START_DEG)}" in text
    assert str(UNWRAP_R_IN) in text
    assert str(UNWRAP_R_OUT) in text
    assert str(int(UNWRAP_SPAN_DEG)) in text


def test_top_mark_lands_on_expected_column():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[4, 16] = (255, 0, 0)
    strip = polar_unwrap(image, out_w=64, out_h=8, r_in=0.2, r_out=1.0, start_deg=140.0, span_deg=360.0)
    column = int(strip[..., 0].sum(axis=0).argmax())
    expected = int(round((270.0 - 140.0) / 360.0 * 64.0))
    assert abs(column - expected) <= 2


def test_unwrap_matches_golden_file():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image[..., 1] = 255 - image[..., 0]
    image[..., 2] = 128
    strip = polar_unwrap(image, out_w=8, out_h=4, r_in=0.5, r_out=1.0, start_deg=10.0, span_deg=180.0)
    tokens = open("cpp/testdata/unwrap_golden.txt", encoding="utf-8").read().split()
    header = list(map(int, tokens[:5]))
    assert header == [8, 8, 3, 8, 4]
    assert [float(token) for token in tokens[5:9]] == [0.5, 1.0, 10.0, 180.0]
    pixels = list(map(int, tokens[9 : 9 + 8 * 8 * 3]))
    expected = list(map(int, tokens[9 + 8 * 8 * 3 :]))
    assert pixels == image.reshape(-1).tolist()
    assert expected == strip.reshape(-1).tolist()

import numpy as np

from rapidstamp.geometry import estimate_ring_start, polar_unwrap
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


def test_ppocr_tensor_is_bgr_centered():
    from rapidstamp.rec_preproc import ppocr_nchw, ppocr_width

    image = np.zeros((32, 64, 3), dtype=np.uint8)
    image[..., 0] = 200
    image[..., 1] = 10
    image[..., 2] = 10
    tensor = ppocr_nchw(image, height=48)
    assert tensor.shape[1] == 48
    assert tensor.shape[2] == ppocr_width(32, 64, 48)
    # Channel 0 is blue (the small channel). Channel 2 is the original red.
    assert tensor[0, 0, 0] < 0
    assert tensor[2, 0, 0] > 0


def test_rec_route_parser_matches_the_cpp_fixture():
    from rapidstamp.labels import load_rec_route, routed_reader

    route = load_rec_route(text="square_personal ppocr\nround_official ctc\n# comment\noval\nbad nope\n")
    assert route == {"square_personal": "ppocr", "round_official": "ctc", "oval": "ppocr"}
    assert routed_reader(route, "square_personal", True) == "ppocr"
    assert routed_reader(route, "round_official", True) == "ctc"
    assert routed_reader(route, "oval", False) == "ctc"
    assert routed_reader(route, "finance", True) == "ctc"
    published = load_rec_route("models/rec_route.txt")
    assert published.get("square_personal") == "ppocr"
    assert published.get("round_official") == "ctc"
    assert published.get("company_latin") == "ctc"


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


def test_decoded_head_is_already_xyxy_and_probabilities():
    from rapidstamp.geometry import decode_detections

    row = np.zeros(5 + 8, dtype=np.float32)
    row[:4] = (0.1, 0.2, 0.4, 0.55)
    row[4] = 0.8
    row[7] = 0.5
    boxes, obj, classes = decode_detections(row.reshape(1, 1, -1))
    assert boxes.shape == (1, 4)
    assert np.allclose(boxes[0], [0.1, 0.2, 0.4, 0.55])
    assert obj[0] == np.float32(0.8)
    assert int(classes[0].argmax()) == 2


def test_ring_gap_starts_at_the_clockwise_end():
    image = np.full((180, 180, 3), 245, dtype=np.uint8)
    cx = cy = 89.5
    radius = 80.0
    for deg in range(180, 350):
        rad = np.deg2rad(deg)
        x = int(round(cx + radius * np.cos(rad)))
        y = int(round(cy + radius * np.sin(rad)))
        image[y - 1 : y + 2, x - 1 : x + 2] = (190, 30, 30)
    start, gap = estimate_ring_start(image)
    assert start is not None
    assert gap >= 40.0
    assert abs(start - 180.0) <= 10.0


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

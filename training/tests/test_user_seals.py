import numpy as np

from rapidstamp_train.data import crop_seal_large
from rapidstamp_train.user_seals import PASTE_SCALES, annotate, assign_splits, pasted_page, seal_class, ink_mask, seal_box


def _ring() -> np.ndarray:
    size = 180
    image = np.full((size, size, 3), 255, np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    radius = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    image[(radius > 64) & (radius < 72)] = (180, 30, 30)
    return image


def _square_frame() -> np.ndarray:
    image = np.full((180, 180, 3), 255, np.uint8)
    image[20:160, 20:28] = (180, 30, 30)
    image[20:160, 152:160] = (180, 30, 30)
    image[20:28, 20:160] = (180, 30, 30)
    image[152:160, 20:160] = (180, 30, 30)
    return image


def test_ring_box_is_round_and_text_stays_empty():
    seals = annotate(_ring())
    assert len(seals) == 1
    assert seals[0]["name"] == "round_official"
    assert seals[0]["text"] == ""
    x1, y1, x2, y2 = seals[0]["bbox"]
    assert x1 < 40 and y1 < 40
    assert x2 > 140 and y2 > 140


def test_square_frame_is_not_called_round():
    image = _square_frame()
    box = seal_box(ink_mask(image))
    assert box is not None
    assert seal_class(ink_mask(image), box) == "square_personal"


def test_large_crop_makes_the_seal_fill_the_frame():
    image = np.full((400, 400, 3), 240, np.uint8)
    boxes = np.array([[160, 150, 230, 220]], dtype=np.float32)
    classes = np.array([0], dtype=np.int64)
    cropped, new_boxes, new_classes = crop_seal_large(image, boxes, classes, np.random.default_rng(0), frac=0.9)
    assert new_classes.tolist() == [0]
    box = new_boxes[0]
    long = max(box[2] - box[0], box[3] - box[1])
    assert long / max(cropped.shape[:2]) > 0.7
    assert 0 <= box[0] < box[2] <= cropped.shape[1]
    assert 0 <= box[1] < box[3] <= cropped.shape[0]


def test_pasted_page_scales_the_seal():
    crop = np.full((100, 80, 3), 200, np.uint8)
    page, box = pasted_page(crop, [0, 0, 80, 100], PASTE_SCALES[-1], page=200, origin=(0, 0))
    assert page.shape == (200, 200, 3)
    long = max(box[2] - box[0], box[3] - box[1])
    assert long >= int(0.9 * 200)
    assert box[2] <= 200 and box[3] <= 200


def test_split_keeps_captures_together_and_drops_copies():
    rows = [
        {"file": "images/bbbb_1.png", "capture": "bbbb", "pixel_md5": "p1", "seals": [{"name": "round_official"}]},
        {"file": "images/bbbb_0.png", "capture": "bbbb", "pixel_md5": "p2", "seals": [{"name": "round_official"}]},
        {"file": "images/aaaa.png", "capture": "aaaa", "pixel_md5": "p1", "seals": [{"name": "round_official"}]},
        {"file": "images/cccc.png", "capture": "cccc", "pixel_md5": "p3", "seals": []},
    ]
    assign_splits(rows)
    by_file = {row["file"]: row for row in rows}
    assert by_file["images/bbbb_0.png"]["split"] == by_file["images/aaaa.png"]["split"]
    assert by_file["images/bbbb_0.png"]["split"] in {"train", "test"}
    assert by_file["images/bbbb_1.png"]["split"] == "duplicate"
    assert by_file["images/bbbb_1.png"]["duplicate_of"] == "images/aaaa.png"
    assert by_file["images/cccc.png"]["split"] == "skipped"
    again = [dict(row) for row in rows]
    for row in again:
        row.pop("split", None)
        row.pop("group", None)
        row.pop("duplicate_of", None)
    assign_splits(again)
    assert [row["split"] for row in again] == [row["split"] for row in rows]

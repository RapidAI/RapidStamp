import json
import random
from pathlib import Path

import numpy as np
import pytest

from rapidstamp_train.ring_text import (
    dedupe_by_md5,
    edit_distance,
    is_gold,
    label_reason,
    load_gold,
    one_ned,
    summarize_pairs,
    unwrap_ring,
    val_split,
)
from rapidstamp_train.seal_aug import SealStripAug, roll_strip


def _row(**kwargs):
    base = {
        "status": 200,
        "ring_text": "新都区石板滩镇兴太种子经营部",
        "probs": {"ring": 0.97},
        "det_class": "round_official",
    }
    base.update(kwargs)
    return base


def test_label_filter_keeps_a_real_ring_and_drops_the_rest():
    assert label_reason(_row()) is None
    assert label_reason(_row(status=464)) == "status"
    assert label_reason(_row(ring_text="", probs={"ring": 1.0})) == "empty"
    assert label_reason(_row(probs={"ring": 0.89})) == "prob"
    assert label_reason(_row(probs={"ring": 0.90})) is None
    assert label_reason(_row(ring_text="某某某某有限公司")) == "template"
    assert label_reason(_row(ring_text="XX公司专用章")) == "template"
    assert label_reason(_row(ring_text="演示章")) == "template"
    assert label_reason(_row(ring_text="Sample Seal")) == "template"
    assert label_reason(_row(det_class="square_personal")) == "class"
    assert label_reason(_row(det_class="rect_stamp")) == "class"
    assert label_reason(_row(ring_text="司")) == "length"
    assert label_reason(_row(ring_text="司" * 26)) == "length"
    assert label_reason(_row(ring_text="司" * 25)) is None
    assert label_reason(_row(det_class="oval")) is None


def test_val_split_uses_the_file_name_only():
    assert val_split("00007_3c973fa9.png") == val_split("/tmp/crops/00007_3c973fa9.png")
    assert val_split("00007_3c973fa9.png") in {"train", "val"}


def test_dedupe_keeps_the_higher_probability():
    rows = [
        {"file": "b.png", "file_md5": "aa", "prob": 0.91},
        {"file": "a.png", "file_md5": "aa", "prob": 0.95},
        {"file": "c.png", "file_md5": "bb", "prob": 0.99},
        {"file": "d.png", "file_md5": "cc", "prob": 0.93},
        {"file": "e.png", "file_md5": "cc", "prob": 0.93},
    ]
    kept = dedupe_by_md5(rows)
    by_md5 = {row["file_md5"]: row["file"] for row in kept}
    assert by_md5 == {"aa": "a.png", "bb": "c.png", "cc": "d.png"}


def test_gold_identity_blocks_the_test_crop(tmp_path: Path):
    manifest = tmp_path / "split.jsonl"
    row = {
        "split": "test",
        "file": "compound-img/322b29cc4a9b11edacf3fa163d64817d.png",
        "capture": "322b29cc4a9b11edacf3fa163d64817d",
        "sha256": "ab" * 32,
        "pixel_md5": "cd" * 16,
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    gold = load_gold(manifest)
    assert is_gold(row["file"], "0" * 64, "0" * 32, gold)
    assert is_gold("other.png", row["sha256"], "0" * 32, gold)
    assert is_gold("other.png", "0" * 64, row["pixel_md5"], gold)
    assert not is_gold("00007_3c973fa9.png", "0" * 64, "0" * 32, gold)


def test_one_ned_keeps_spaces_and_averages_per_row():
    assert edit_distance("宁 回族", "宁回族") == 1
    assert one_ned("abc", "abc") == 1.0
    assert one_ned("", "") == 1.0
    assert one_ned("ab", "a") == pytest.approx(0.5)
    summary = summarize_pairs([("甲乙", "甲乙"), ("甲", "甲乙")])
    assert summary["exact"] == 1
    assert summary["n"] == 2
    assert summary["one_ned"] == pytest.approx((1.0 + 0.5) / 2)


def _offset_ring() -> np.ndarray:
    size = 220
    image = np.full((size, size, 3), 248, np.uint8)
    cy, cx = 90.0, 140.0
    yy, xx = np.mgrid[0:size, 0:size]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360.0
    ink = (rad >= 70) & (rad <= 86) & ~((ang >= 20) & (ang < 160))
    image[ink] = (190, 40, 40)
    return image


def test_unwrap_recenters_an_offset_ring_and_uses_one_start():
    image = _offset_ring()
    unwound = unwrap_ring(image, out_w=96, out_h=16)
    strip = unwound["strip"]
    assert strip.shape == (16, 96, 3)
    assert unwound["wide_gap"]
    assert 140 <= unwound["start_deg"] <= 185
    assert 0.65 <= unwound["r_in"] / unwound["r_out"] <= 0.92
    columns = strip[..., 0].mean(axis=0)
    assert (columns > 230).sum() >= 8
    assert (columns < 210).sum() >= 8
    from rapidstamp.geometry import polar_unwrap

    naive = polar_unwrap(
        image,
        out_w=96,
        out_h=16,
        r_in=unwound["r_in"],
        r_out=unwound["r_out"],
        start_deg=unwound["start_deg"],
    )
    fitted_red = float(np.mean(strip[..., 0] < 210))
    naive_red = float(np.mean(naive[..., 0] < 210))
    assert fitted_red > naive_red


def test_blank_crop_still_unwraps():
    image = np.full((40, 60, 3), 255, np.uint8)
    strip = unwrap_ring(image, out_w=32, out_h=8)["strip"]
    assert strip.shape == (8, 32, 3)


def test_roll_moves_only_a_wide_gap_strip(monkeypatch):
    image = np.zeros((8, 36, 3), np.uint8)
    image[:, 0] = 255
    assert not np.array_equal(roll_strip(image, 90), image)
    monkeypatch.setattr(random, "uniform", lambda _low, _high: 90.0)
    monkeypatch.setattr(random, "random", lambda: 0.5)
    op = SealStripAug(jpeg_prob=0.0, roll_prob=1.0, roll_deg=90.0)
    moved = op({"image": image.copy(), "img_path": "/tmp/a__gap.png"})["image"]
    assert not np.array_equal(moved, image)
    same = op({"image": image.copy(), "img_path": "/tmp/a.png"})["image"]
    assert np.array_equal(same, image)


def test_jpeg_aug_changes_a_noisy_strip():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(24, 48, 3), dtype=np.uint8)
    op = SealStripAug(jpeg_prob=1.0, roll_prob=0.0, jpeg_quality=(30, 30))
    out = op({"image": image, "img_path": "plain.png"})["image"]
    assert out.shape == image.shape
    assert not np.array_equal(out, image)

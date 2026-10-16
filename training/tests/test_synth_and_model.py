import numpy as np
import torch

from rapidstamp.ctc import ctc_greedy, prefer_reading
from rapidstamp.labels import CLASS_NAMES, DET_SIZE
from rapidstamp_train.corpus import MIDDLES, all_chars
from rapidstamp_train.fonts_util import find_cjk_font, find_latin_font
from rapidstamp_train.free_rings import render_free_seal, sample_ring_text
from rapidstamp_train.models import SealDet, SealRec, count_params
from rapidstamp_train.losses import detection_loss
from rapidstamp_train.synthesize import FontBank, make_seal


def test_free_ring_text_leaves_the_template_list():
    rng = np.random.default_rng(0)
    cjk = list("甲乙丙丁戊己庚辛壬癸测章")
    latin = list("ABCDEFGH")
    digits = list("0123456789")
    texts = [sample_ring_text(rng, cjk, latin, digits)[0] for _ in range(40)]
    assert any(text not in set(MIDDLES) for text in texts)
    assert any(any(ch in "甲乙丙丁戊己庚辛壬癸" for ch in text) for text in texts)
    fonts = FontBank(find_cjk_font(), find_latin_font())
    rgba = render_free_seal(rng, fonts, "甲乙丙丁公司", "", elliptic=False)
    assert rgba.shape[-1] == 4
    assert rgba[..., 3].max() > 0


def test_charset_covers_rendered_text():
    fonts = FontBank(find_cjk_font(), find_latin_font())
    rng = np.random.default_rng(3)
    charset = set(all_chars())
    for name in CLASS_NAMES:
        seal = make_seal(rng, fonts, name)
        assert set(seal["text"]) <= charset
        assert set(seal["center_text"]) <= charset
        assert seal["rgba"].shape[-1] == 4


def test_detector_loss_and_recognizer_shape():
    det = SealDet()
    assert count_params(det) < 5_000_000
    images = torch.rand(2, 3, DET_SIZE, DET_SIZE)
    det.train()
    raw = det(images)
    anchors = (DET_SIZE // 8) ** 2 + (DET_SIZE // 16) ** 2 + (DET_SIZE // 32) ** 2
    assert raw.shape == (2, anchors, 5 + len(CLASS_NAMES))
    targets = [
        torch.tensor([[2.0, 0.2, 0.2, 0.45, 0.45]]),
        torch.zeros(0, 5),
    ]
    loss, _parts = detection_loss(raw, targets)
    loss.backward()
    det.eval()
    decoded = det(images)
    assert decoded.shape == raw.shape
    assert float(decoded[..., :4].detach().min()) >= 0.0
    assert float(decoded[..., :4].detach().max()) <= 1.0

    charset = all_chars()
    rec = SealRec(1 + len(charset))
    assert count_params(rec) < 20_000_000
    strips = torch.rand(2, 1, 32, 384)
    logits = rec(strips)
    assert logits.shape == (2, 96, 1 + len(charset))
    text, score = ctc_greedy(logits[0].detach().numpy(), charset)
    assert isinstance(text, str)
    assert 0.0 <= score <= 1.0


def test_prefer_reading_keeps_the_full_title_on_a_near_tie():
    short = ("厦门国泰", 0.99)
    full = ("厦门国泰有限公司", 0.985)
    rotated = ("0932958166224厦门盛达有限公司", 0.991)
    upright = ("厦门盛达有限公司0932958166224", 0.989)
    assert prefer_reading(short, full) == full
    assert prefer_reading(rotated, upright)[0].startswith("厦")
    assert prefer_reading(("ACME", 0.2), ("GLOBAL", 0.05))[0] == "ACME"


def test_ctc_score_depends_on_script():
    steps = np.array(
        [
            [0.0, 8.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    longer = np.array(
        [
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Latin uses probability mass over the fixed length, so the longer reading wins.
    short_text, short_score = ctc_greedy(steps, ["A", "B"])
    long_text, long_score = ctc_greedy(longer, ["A", "B"])
    assert short_text == "A" and long_text == "AB"
    assert long_score > short_score
    # CJK uses the mean of emitted glyphs, so one extra weak character loses.
    cjk_short, cjk_short_score = ctc_greedy(steps, ["甲", "乙"])
    cjk_long, cjk_long_score = ctc_greedy(longer, ["甲", "乙"])
    assert cjk_short == "甲" and cjk_long == "甲乙"
    assert cjk_short_score > cjk_long_score
    for score in (short_score, long_score, cjk_short_score, cjk_long_score):
        assert 0.0 <= score <= 1.0

import numpy as np
import torch

from rapidstamp.ctc import ctc_greedy
from rapidstamp.labels import CLASS_NAMES, DET_SIZE, GRID
from rapidstamp_train.corpus import all_chars
from rapidstamp_train.fonts_util import find_cjk_font, find_latin_font
from rapidstamp_train.models import SealDet, SealRec, count_params
from rapidstamp_train.losses import detection_loss
from rapidstamp_train.synthesize import FontBank, make_seal


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
    raw = det(images)
    assert raw.shape[0] == 2
    assert raw.shape[1] == GRID * GRID
    assert raw.shape[2] == 5 + len(CLASS_NAMES)
    obj = torch.zeros(2, GRID, GRID)
    obj[:, 4, 5] = 1
    box = torch.zeros(2, GRID, GRID, 4)
    box[:, 4, 5] = torch.tensor([0.2, 0.2, 0.5, 0.5])
    cls = torch.zeros(2, GRID, GRID, dtype=torch.long)
    loss, _parts = detection_loss(raw, obj, box, cls)
    loss.backward()

    charset = all_chars()
    rec = SealRec(1 + len(charset))
    assert count_params(rec) < 20_000_000
    strips = torch.rand(2, 1, 32, 384)
    logits = rec(strips)
    assert logits.shape[0] == 2
    assert logits.shape[2] == 1 + len(charset)
    text, score = ctc_greedy(logits[0].detach().numpy(), charset)
    assert isinstance(text, str)
    assert 0.0 <= score <= 1.0

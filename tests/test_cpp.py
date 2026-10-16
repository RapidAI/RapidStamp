"""Numpy reference versus the C API on the same synthetic page."""

import numpy as np
import pytest

from rapidstamp.engine import SealEngine
from rapidstamp.native import find_library
from rapidstamp_train.synth import render_page


def test_native_text_matches_numpy():
    if find_library() is None:
        pytest.skip("librapidstamp is not built")
    numpy_engine = SealEngine.from_pretrained("numpy")
    native_engine = SealEngine.from_pretrained("native")
    rng = np.random.default_rng(7)
    for _ in range(4):
        page = render_page(rng, n_seals=1, augment=False)
        a = numpy_engine.infer(page.image)
        b = native_engine.infer(page.image)
        assert len(a) == len(b)
        assert a, "expected a detection on a clean seal"
        assert a[0].text == b[0].text
        # One step of the shared ring search is a few pixels. Float accumulation
        # in the two samplers can land on neighboring steps.
        assert abs((a[0].x + a[0].w / 2) - (b[0].x + b[0].w / 2)) < 8.0
        assert abs((a[0].y + a[0].h / 2) - (b[0].y + b[0].h / 2)) < 8.0
        assert abs(a[0].w - b[0].w) < 12.0
        assert abs(a[0].h - b[0].h) < 12.0

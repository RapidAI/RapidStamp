import math

import numpy as np

from rapidstamp.geometry import (
    N_ANGLES,
    angular_ink,
    box_to_ring,
    circular_distance,
    estimate_text_start,
    refine_ring,
    unwrap_seal,
)
from rapidstamp_train.synth import render_page


def test_gap_alignment_on_clean_seals():
    errors = []
    for i, rotation in enumerate(np.linspace(0.0, 2.0 * math.pi, 8, endpoint=False)):
        page = render_page(np.random.default_rng(i + 10), n_seals=1, augment=False, rotation=float(rotation))
        assert page.texts and all(ch for ch in page.texts[0])
        cx, cy, radius = box_to_ring(page.boxes[0])
        from rapidstamp.geometry import TEXT_INNER_FRAC, TEXT_OUTER_FRAC

        ink = angular_ink(page.image, cx, cy, radius * TEXT_INNER_FRAC, radius * TEXT_OUTER_FRAC)
        estimate = estimate_text_start(ink) * (2.0 * math.pi / N_ANGLES)
        errors.append(circular_distance(estimate, page.starts[0]))
    assert max(errors) < math.radians(20)


def test_unwrap_shape_and_contrast():
    page = render_page(np.random.default_rng(4), n_seals=1, augment=False, rotation=0.3)
    strip = unwrap_seal(page.image, page.boxes[0])
    assert strip.shape == (32, 160)
    assert strip.min() < 0.7
    assert strip.max() > 0.9


def test_refine_pulls_a_shifted_circle_back():
    page = render_page(np.random.default_rng(12), n_seals=1, augment=False, rotation=0.4)
    cx, cy, radius = box_to_ring(page.boxes[0])
    shifted = [cx + 0.22 * radius - radius, cy - 0.18 * radius - radius, radius * 2.15, radius * 2.15]
    refined = refine_ring(page.image, shifted)
    rcx, rcy, rr = box_to_ring(refined)
    assert abs(rcx - cx) / radius < 0.08
    assert abs(rcy - cy) / radius < 0.08
    assert abs(rr - radius) / radius < 0.08


def test_empty_page_has_no_box():
    page = render_page(np.random.default_rng(5), n_seals=0, augment=False)
    assert page.boxes == []
    assert page.image.shape == (256, 256, 3)

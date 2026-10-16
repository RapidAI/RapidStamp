"""CPU end-to-end checks against the committed synthetic demo weights."""

import numpy as np

from rapidstamp.engine import SealEngine
from rapidstamp.postprocess import iou
from rapidstamp_train.synth import render_page


def test_clean_single_seal_is_read():
    engine = SealEngine.from_pretrained("numpy")
    rng = np.random.default_rng(100)
    exact = 0
    detected = 0
    n = 16
    for _ in range(n):
        page = render_page(rng, n_seals=1, augment=False)
        preds = engine.infer(page.image)
        if not preds or not page.boxes:
            continue
        if iou([preds[0].x, preds[0].y, preds[0].w, preds[0].h], page.boxes[0]) >= 0.5:
            detected += 1
            if preds[0].text == page.texts[0]:
                exact += 1
    assert detected >= 12, detected
    assert exact >= 8, exact


def test_blank_page_has_no_seal():
    engine = SealEngine.from_pretrained("numpy")
    page = render_page(np.random.default_rng(5), n_seals=0, augment=False)
    assert engine.infer(page.image) == []


def test_onnx_backend_matches_numpy_text():
    onnxruntime = __import__("importlib").import_module("onnxruntime")
    assert onnxruntime is not None
    numpy_engine = SealEngine.from_pretrained("numpy")
    onnx_engine = SealEngine.from_pretrained("onnx")
    page = render_page(np.random.default_rng(21), n_seals=1, augment=False)
    a = numpy_engine.infer(page.image)
    b = onnx_engine.infer(page.image)
    assert len(a) == len(b) >= 1
    assert a[0].text == b[0].text

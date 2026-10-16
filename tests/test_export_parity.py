import numpy as np
import torch

from rapidstamp.geometry import CHAR_SIZE, DET_SIZE
from rapidstamp.runtime import load_rsp, run_model, save_rsp
from rapidstamp_train.models import SealDetTiny, SealRecTiny, collect_ops


def _compare(model, shape, kind, meta=None, classes=None):
    model.eval()
    torch.manual_seed(0)
    sample = torch.randn(*shape)
    with torch.no_grad():
        expected = model(sample).numpy()
    ops = collect_ops(model)
    path = f"/tmp/rapidstamp_{kind}"
    save_rsp(path, kind, shape, ops, meta=meta, classes=classes)
    loaded = load_rsp(path + ".manifest")
    actual = run_model(loaded, sample.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    return loaded


def test_detector_matches_folded_numpy():
    loaded = _compare(
        SealDetTiny(),
        (2, 3, DET_SIZE, DET_SIZE),
        "detector",
        meta={"grid": 8, "det_thresh": 0.5},
    )
    assert loaded.kind == "detector"
    assert loaded.ops[-1].kind == "conv"


def test_recognizer_matches_folded_numpy():
    loaded = _compare(
        SealRecTiny(),
        (2, 1, CHAR_SIZE, CHAR_SIZE),
        "recognizer",
        classes=list("甲乙"),
    )
    assert loaded.classes == ["甲", "乙"]
    assert loaded.ops[-1].kind == "linear"


def test_onnx_matches_torch_when_runtime_present():
    ort = __import__("onnxruntime")
    model = SealDetTiny().eval()
    shape = (1, 3, DET_SIZE, DET_SIZE)
    sample = torch.randn(*shape)
    with torch.no_grad():
        expected = model(sample).numpy()
    path = "/tmp/rapidstamp_det.onnx"
    try:
        torch.onnx.export(model, sample, path, input_names=["input"], output_names=["output"], opset_version=17, dynamo=False)
    except TypeError:
        torch.onnx.export(model, sample, path, input_names=["input"], output_names=["output"], opset_version=17)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": sample.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

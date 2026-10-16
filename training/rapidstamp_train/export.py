"""Export a trained checkpoint to the RSP weight pack and to ONNX."""

from __future__ import annotations

from pathlib import Path

import torch

from rapidstamp.charset import CHARSET
from rapidstamp.geometry import CHAR_SIZE, DET_SIZE, GRID
from rapidstamp.runtime import save_rsp
from rapidstamp.titles import seal_titles
from rapidstamp_train.models import SealDetTiny, SealRecTiny, collect_ops


def load_for_export(checkpoint: Path, model: torch.nn.Module) -> torch.nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def export_detector(checkpoint: Path, out_prefix: Path, det_thresh: float, nms_thresh: float = 0.45) -> None:
    model = load_for_export(checkpoint, SealDetTiny())
    ops = collect_ops(model)
    save_rsp(
        out_prefix,
        "detector",
        (1, 3, DET_SIZE, DET_SIZE),
        ops,
        meta={"grid": GRID, "cell": DET_SIZE // GRID, "det_size": DET_SIZE, "det_thresh": det_thresh, "nms_thresh": nms_thresh},
    )
    _export_onnx(model, (1, 3, DET_SIZE, DET_SIZE), out_prefix.with_suffix(".onnx"))


def export_recognizer(checkpoint: Path, out_prefix: Path) -> None:
    model = load_for_export(checkpoint, SealRecTiny())
    ops = collect_ops(model)
    save_rsp(
        out_prefix,
        "recognizer",
        (1, 1, CHAR_SIZE, CHAR_SIZE),
        ops,
        meta={"decode": "classify", "char_size": CHAR_SIZE},
        classes=list(CHARSET),
        words=seal_titles(),
    )
    _export_onnx(model, (1, 1, CHAR_SIZE, CHAR_SIZE), out_prefix.with_suffix(".onnx"))


def _export_onnx(model: torch.nn.Module, shape: tuple[int, int, int, int], path: Path) -> None:
    dummy = torch.zeros(*shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy,
            str(path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

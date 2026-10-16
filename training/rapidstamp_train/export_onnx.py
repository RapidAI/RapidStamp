"""Export the trained detector and recognizer to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rapidstamp.labels import CLASS_NAMES, DET_SIZE, REC_H, REC_W, write_preprocess
from rapidstamp_train.corpus import all_chars
from rapidstamp_train.models import SealDet, SealRec


def _export(module: torch.nn.Module, dummy: torch.Tensor, path: Path, output: str) -> None:
    module.eval()
    kwargs = dict(
        input_names=["image"],
        output_names=[output],
        opset_version=17,
        dynamic_axes={"image": {0: "batch"}, output: {0: "batch"}},
    )
    try:
        torch.onnx.export(module, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(module, dummy, str(path), **kwargs)


def export(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    det_ckpt = torch.load(args.det, map_location="cpu", weights_only=False)
    rec_ckpt = torch.load(args.rec, map_location="cpu", weights_only=False)
    det = SealDet(len(CLASS_NAMES))
    det.load_state_dict(det_ckpt["model"])
    charset = rec_ckpt.get("charset") or all_chars()
    rec = SealRec(1 + len(charset))
    rec.load_state_dict(rec_ckpt["model"])
    _export(det, torch.zeros(1, 3, DET_SIZE, DET_SIZE), out / "seal_det.onnx", "pred")
    _export(rec, torch.zeros(1, 1, REC_H, REC_W), out / "seal_rec.onnx", "logits")
    (out / "charset.txt").write_text("\n".join(charset) + "\n", encoding="utf-8")
    (out / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    write_preprocess(str(out / "preprocess.txt"))
    print(f"wrote {out / 'seal_det.onnx'} and {out / 'seal_rec.onnx'}", flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export RapidStamp ONNX models")
    parser.add_argument("--det", type=Path, required=True)
    parser.add_argument("--rec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    export(build_argparser().parse_args())


if __name__ == "__main__":
    main()

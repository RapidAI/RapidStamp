"""Synthesize data, train, export ONNX, and evaluate.

Run this on the GPU training machine. Example:

    python training/scripts/train_all.py --out /data/rapidstamp --models models
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.eval_onnx import evaluate as eval_onnx  # noqa: E402
from rapidstamp_train.export_onnx import export as export_onnx  # noqa: E402
from rapidstamp_train.synthesize import generate  # noqa: E402
from rapidstamp_train.train_det import train as train_det  # noqa: E402
from rapidstamp_train.train_rec import train as train_rec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Full RapidStamp training pipeline")
    parser.add_argument("--out", type=Path, required=True, help="Generated data and checkpoints")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--det-train", type=int, default=10000)
    parser.add_argument("--det-val", type=int, default=800)
    parser.add_argument("--det-epochs", type=int, default=36)
    parser.add_argument("--rec-epochs", type=int, default=24)
    parser.add_argument("--det-batch", type=int, default=64)
    parser.add_argument("--rec-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-synth", action="store_true")
    parser.add_argument("--fast", action="store_true", help="Tiny run used to check the pipeline")
    args = parser.parse_args()
    if args.fast:
        args.det_train = 24
        args.det_val = 8
        args.det_epochs = 1
        args.rec_epochs = 1
        args.det_batch = 4
        args.rec_batch = 8
        args.workers = 0

    if not args.skip_synth:
        generate(
            argparse.Namespace(
                out=args.out,
                det_train=args.det_train,
                det_val=args.det_val,
                seed=args.seed,
                workers=args.workers,
                cjk_font="",
                latin_font="",
            )
        )

    ckpt = args.out / "checkpoints"
    det_report = train_det(
        argparse.Namespace(
            data=args.out,
            out=ckpt,
            epochs=args.det_epochs,
            batch=args.det_batch,
            lr=2e-3,
            workers=args.workers,
            device=args.device,
        )
    )
    rec_report = train_rec(
        argparse.Namespace(
            data=args.out,
            out=ckpt,
            epochs=args.rec_epochs,
            batch=args.rec_batch,
            lr=1.5e-3,
            workers=args.workers,
            device=args.device,
        )
    )
    export_onnx(
        argparse.Namespace(
            det=ckpt / "best_det.pt",
            rec=ckpt / "best_rec.pt",
            out=args.models,
        )
    )
    metrics = eval_onnx(
        argparse.Namespace(
            data=args.out,
            models=args.models,
            out=args.models / "metrics.json",
            limit=0,
            skip_rec_strips=False,
        )
    )
    metrics["train_det"] = {"params": det_report["params"], "best_map50_letterbox": det_report["best_map50"]}
    metrics["train_rec"] = {"params": rec_report["params"], "best_exact_torch": rec_report["best_exact"]}
    (args.models / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundled = ROOT / "python" / "src" / "rapidstamp" / "models"
    bundled.mkdir(parents=True, exist_ok=True)
    for name in ("seal_det.onnx", "seal_rec.onnx", "charset.txt", "classes.txt", "preprocess.txt", "metrics.json"):
        source = args.models / name
        if source.is_file():
            shutil.copy2(source, bundled / name)
    print(json.dumps({"det_params": det_report["params"], "rec_params": rec_report["params"], "e2e": metrics["e2e_title_recall"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

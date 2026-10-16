"""Fine-tune YOLOX-Nano on seal pages plus red/blue non-seal ink.

The COCO checkpoint is Apache-2.0 (Megvii YOLOX). It is downloaded on the
training machine and is not committed.

    python training/scripts/train_yolox_det.py \
        --coco /home/mayong/rapidstamp/weights/yolox_nano.pth
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.export_onnx import export as export_onnx  # noqa: E402
from rapidstamp_train.realdata import link_mix  # noqa: E402
from rapidstamp_train.train_det import train as train_det  # noqa: E402
from rapidstamp_train.yolox_nano import _COCO_URL  # noqa: E402


def _ensure(path: Path) -> None:
    if path.is_file() and path.stat().st_size > 1_000_000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {_COCO_URL}", flush=True)
    urllib.request.urlretrieve(_COCO_URL, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", type=Path, default=Path("/home/mayong/rapidstamp/synth"))
    parser.add_argument("--rest", type=Path, default=Path("/home/mayong/rapidstamp/real/rest"))
    parser.add_argument("--paddle", type=Path, default=Path("/home/mayong/rapidstamp/real/paddle"))
    parser.add_argument("--web", type=Path, default=Path("/home/mayong/rapidstamp/real/web"))
    parser.add_argument("--mix", type=Path, default=Path("/home/mayong/rapidstamp/mix_yolox"))
    parser.add_argument("--out", type=Path, default=Path("/home/mayong/rapidstamp/checkpoints/yolox"))
    parser.add_argument("--models", type=Path, default=Path("/home/mayong/rapidstamp/models_yolox"))
    parser.add_argument("--coco", type=Path, default=Path("/home/mayong/rapidstamp/weights/yolox_nano.pth"))
    parser.add_argument("--rec", type=Path, default=Path("/home/mayong/rapidstamp/checkpoints/ringrec_paddle/best_rec.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--web-repeat", type=int, default=4)
    parser.add_argument("--distractors", type=int, default=4000)
    args = parser.parse_args()
    _ensure(args.coco)
    sources = [("synth_", args.synth), ("rest_", args.rest)]
    if args.paddle.is_dir():
        sources.append(("paddle_", args.paddle))
    if args.web.is_dir():
        sources.append(("web_", args.web))
    link_mix(sources, args.mix, det_repeat={"web_": args.web_repeat})
    train_det(
        argparse.Namespace(
            data=args.mix,
            out=args.out,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            workers=args.workers,
            device="cuda",
            init=args.coco,
            distractors=args.distractors,
        )
    )
    export_onnx(
        argparse.Namespace(
            det=args.out / "best_det.pt",
            rec=args.rec,
            out=args.models,
        )
    )
    print("yolox-det exit 0", flush=True)


if __name__ == "__main__":
    main()

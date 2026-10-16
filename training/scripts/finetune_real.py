"""Fine-tune SealDet and SealRec on synthetic pages plus real seal crops.

Checkpoints, the mixed tree, and exports go under the large disk by default.
The interpreter is the existing PyTorch env; this script does not install one.

    python training/scripts/finetune_real.py \
        --synth /home/mayong/rapidstamp/synth \
        --rest /home/mayong/rapidstamp/real/rest \
        --paddle /home/mayong/rapidstamp/real/paddle \
        --init-det /home/znsoft/rapidstamp-data/checkpoints/best_det.pt \
        --init-rec /home/znsoft/rapidstamp-data/checkpoints/best_rec.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.export_onnx import export as export_onnx  # noqa: E402
from rapidstamp_train.realdata import link_mix  # noqa: E402
from rapidstamp_train.train_det import train as train_det  # noqa: E402
from rapidstamp_train.train_rec import train as train_rec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", type=Path, required=True)
    parser.add_argument("--rest", type=Path, default=None)
    parser.add_argument("--paddle", type=Path, default=None)
    parser.add_argument("--mix", type=Path, default=Path("/home/mayong/rapidstamp/mix"))
    parser.add_argument("--out", type=Path, default=Path("/home/mayong/rapidstamp/checkpoints/finetune"))
    parser.add_argument("--models", type=Path, default=Path("/home/mayong/rapidstamp/models"))
    parser.add_argument("--init-det", type=Path, required=True)
    parser.add_argument("--init-rec", type=Path, required=True)
    parser.add_argument("--det-epochs", type=int, default=10)
    parser.add_argument("--rec-epochs", type=int, default=12)
    parser.add_argument("--det-batch", type=int, default=64)
    parser.add_argument("--rec-batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-det", action="store_true")
    parser.add_argument("--skip-rec", action="store_true")
    args = parser.parse_args()

    sources: list[tuple[str, Path]] = [("synth_", args.synth)]
    if args.rest:
        if not args.rest.is_dir():
            raise SystemExit(f"missing prepared ReST tree: {args.rest}")
        sources.append(("rest_", args.rest))
    if args.paddle:
        if not args.paddle.is_dir():
            raise SystemExit(f"missing prepared Paddle tree: {args.paddle}")
        sources.append(("paddle_", args.paddle))
    link_mix(sources, args.mix)

    if not args.skip_det:
        train_det(
            argparse.Namespace(
                data=args.mix,
                out=args.out,
                epochs=args.det_epochs,
                batch=args.det_batch,
                lr=5e-4,
                workers=args.workers,
                device=args.device,
                init=args.init_det,
            )
        )
    if not args.skip_rec:
        train_rec(
            argparse.Namespace(
                data=args.mix,
                out=args.out,
                epochs=args.rec_epochs,
                batch=args.rec_batch,
                lr=4e-4,
                workers=args.workers,
                device=args.device,
                init=args.init_rec,
            )
        )
    export_onnx(
        argparse.Namespace(
            det=args.out / "best_det.pt",
            rec=args.out / "best_rec.pt",
            out=args.models,
        )
    )


if __name__ == "__main__":
    main()

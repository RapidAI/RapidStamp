"""Train a recognizer from scratch on template strips, ReST, and free rings.

The detector checkpoint is only used at export time. Recognition does not
load the previous backbone: the channel width changed.

    python training/scripts/train_ring_rec.py \
        --synth /home/mayong/rapidstamp/synth \
        --rest /home/mayong/rapidstamp/real/rest \
        --free /home/mayong/rapidstamp/free_rings \
        --det /home/mayong/rapidstamp/checkpoints/finetune/best_det.pt \
        --charset models/charset.txt
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
from rapidstamp_train.train_rec import train as train_rec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synth", type=Path, required=True)
    parser.add_argument("--rest", type=Path, default=None)
    parser.add_argument("--free", type=Path, default=None)
    parser.add_argument("--paddle", type=Path, default=None)
    parser.add_argument("--charset", type=Path, default=ROOT / "models" / "charset.txt")
    parser.add_argument("--mix", type=Path, default=Path("/home/mayong/rapidstamp/mix_ring"))
    parser.add_argument("--out", type=Path, default=Path("/home/mayong/rapidstamp/checkpoints/ringrec"))
    parser.add_argument("--models", type=Path, default=Path("/home/mayong/rapidstamp/models"))
    parser.add_argument("--det", type=Path, required=True, help="Existing detector checkpoint to export beside the new recognizer")
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rest-repeat", type=int, default=4)
    args = parser.parse_args()

    sources: list[tuple[str, Path]] = [("synth_", args.synth)]
    repeats: dict[str, int] = {}
    if args.rest:
        if not args.rest.is_dir():
            raise SystemExit(f"missing prepared ReST tree: {args.rest}")
        sources.append(("rest_", args.rest))
        repeats["rest_"] = args.rest_repeat
    if args.free:
        if not args.free.is_dir():
            raise SystemExit(f"missing free-ring tree: {args.free}")
        sources.append(("free_", args.free))
    if args.paddle:
        if not args.paddle.is_dir():
            raise SystemExit(f"missing prepared Paddle tree: {args.paddle}")
        sources.append(("paddle_", args.paddle))
    link_mix(sources, args.mix, rec_repeat=repeats, base_charset=args.charset)
    train_rec(
        argparse.Namespace(
            data=args.mix,
            out=args.out,
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            workers=args.workers,
            device=args.device,
            init=None,
        )
    )
    export_onnx(
        argparse.Namespace(
            det=args.det,
            rec=args.out / "best_rec.pt",
            out=args.models,
        )
    )


if __name__ == "__main__":
    main()

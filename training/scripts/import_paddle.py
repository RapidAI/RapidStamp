"""Convert a PaddleOCR seal archive (dataset/train/train.txt) into pages.

The zip stays on the training machine. This script does not download it.

    python training/scripts/import_paddle.py \
        --src /home/mayong/datasets/paddle_seal/paddle_seal.zip \
        --dst /home/mayong/rapidstamp/real/paddle
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.realdata import convert_paddle_zip  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    if not args.src.is_file():
        raise SystemExit(f"Paddle seal zip is not on disk yet: {args.src}")
    convert_paddle_zip(args.src, args.dst, limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()

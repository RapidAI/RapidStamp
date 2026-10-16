"""Convert an ICDAR 2023 ReST archive into RapidStamp pages.

The competition zip is not downloaded by this script and must not be committed.
Point ``--src`` at ``rest_icdar2023.zip`` (outer zip containing
``train/train_gts.json`` and ``train/train_images.zip``). The unlabeled
``test_images.zip`` is ignored.

    python training/scripts/import_rest.py \
        --src /home/mayong/datasets/rest/rest_icdar2023.zip \
        --dst /home/mayong/rapidstamp/real/rest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.realdata import convert_rest_zip  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not args.src.is_file():
        raise SystemExit(f"missing ReST zip: {args.src}")
    convert_rest_zip(args.src, args.dst, limit=args.limit, seed=args.seed)


if __name__ == "__main__":
    main()

"""``python -m rapidstamp.cli image.png``."""

from __future__ import annotations

import argparse
import json

import numpy as np

from rapidstamp.engine import SealEngine


def _read_image(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Read seals in an image")
    parser.add_argument("image")
    parser.add_argument("--det", default=None)
    parser.add_argument("--rec", default=None)
    parser.add_argument("--backend", default="auto", choices=["auto", "numpy", "native", "onnx"])
    args = parser.parse_args()
    if args.det and args.rec:
        engine = SealEngine(args.det, args.rec, backend=args.backend)
    else:
        engine = SealEngine.from_pretrained(backend=args.backend)
    seals = engine.infer(_read_image(args.image))
    print(json.dumps([seal.as_dict() for seal in seals], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

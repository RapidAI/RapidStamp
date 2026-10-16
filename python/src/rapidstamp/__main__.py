"""``python -m rapidstamp image.jpg``"""

from __future__ import annotations

import argparse
import json

from rapidstamp.pipeline import RapidStamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RapidStamp on an image")
    parser.add_argument("image")
    parser.add_argument("--model-dir", default="")
    args = parser.parse_args()
    engine = RapidStamp(args.model_dir or None)
    seals = engine(args.image)
    payload = [
        {
            "bbox": [seal.x1, seal.y1, seal.x2, seal.y2],
            "score": seal.score,
            "class_name": seal.class_name,
            "text": seal.text,
            "text_score": seal.text_score,
            "center_text": seal.center_text,
        }
        for seal in seals
    ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

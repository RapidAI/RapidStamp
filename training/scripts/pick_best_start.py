"""Keep the highest CTC score among start-angle copies of one strip.

infer_rec writes ``path\\tpred\\tscore``. The kept name matches the training
strip (``foo__gap.png``), so score_ring_preds.py can join on the file name.

    python training/scripts/pick_best_start.py \
        --pred /home/rapidstamp_data/logs/ring_v2_multistart.txt \
        --out /home/rapidstamp_data/logs/ring_v2_multistart_best.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.ring_text import best_ctc_rows  # noqa: E402


def read_scored(path: Path) -> list[tuple[str, str, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        parts = line.split("\t")
        pred = parts[1] if len(parts) > 1 else ""
        try:
            score = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
        except ValueError:
            score = 0.0
        rows.append((parts[0], pred, score))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    picked = best_ctc_rows(read_scored(args.pred))
    text = "\n".join(f"{name}\t{pred}\t{score:.6f}" for name, pred, score in picked)
    args.out.write_text(text + ("\n" if text else ""), encoding="utf-8")
    print(f"kept {len(picked)}", flush=True)


if __name__ == "__main__":
    main()

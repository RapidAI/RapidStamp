"""Write free-ring recognition strips under a data root. Images stay out of git.

    python training/scripts/synth_free_rings.py \
        --charset models/charset.txt \
        --out /home/mayong/rapidstamp/free_rings
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.free_rings import main  # noqa: E402

if __name__ == "__main__":
    main()

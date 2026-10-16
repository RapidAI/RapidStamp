"""Build a PaddleOCR recognition list from seal strips already on the training host.

Images stay where they are. The list stores absolute paths. Characters outside
the official PP-OCRv5 dictionary are dropped from that label; a row with no
remaining characters is skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_dict(path: Path) -> set[str]:
    chars: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.endswith("\r"):
            line = line[:-1]
        if line == "":
            continue
        chars.add(line)
    return chars


def keep_text(text: str, charset: set[str]) -> str:
    return "".join(ch for ch in text if ch in charset or ch == " ")


def add_tsv(
    rows: list[str],
    tsv: Path,
    image_dir: Path,
    charset: set[str],
    repeat: int,
    stats: dict,
) -> None:
    kept = skipped = 0
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        name, text = line.split("\t", 1)
        image = image_dir / name
        if not image.is_file():
            skipped += 1
            continue
        cleaned = keep_text(text, charset)
        if not cleaned:
            skipped += 1
            continue
        entry = f"{image}\t{cleaned}"
        for _ in range(max(1, repeat)):
            rows.append(entry)
        kept += 1
    stats[str(tsv)] = {"kept": kept, "skipped": skipped, "repeat": repeat}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", action="append", default=[], help="tsv,image_dir,repeat")
    parser.add_argument("--val", action="append", default=[], help="tsv,image_dir,repeat")
    args = parser.parse_args()
    charset = load_dict(args.dict)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"dict": len(charset)}
    for split, specs in (("train", args.train), ("val", args.val)):
        rows: list[str] = []
        for spec in specs:
            tsv_s, image_s, repeat_s = spec.split(",", 2)
            add_tsv(rows, Path(tsv_s), Path(image_s), charset, int(repeat_s), stats)
        (out / f"{split}_list.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        stats[split] = len(rows)
    print(stats, flush=True)


if __name__ == "__main__":
    main()

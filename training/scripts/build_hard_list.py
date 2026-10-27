#!/usr/bin/env python3
"""Train rows that share a verified gold-error character or a 4-gram.

Gold crops are not copied. Pending-recheck strings are not used: several of
those gold labels are themselves incomplete.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

COMMON = set("有限公司省市县区镇乡村街道路号的和与")


def grams(text: str, size: int = 4) -> set[str]:
    if len(text) < size:
        return set()
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-char", type=int, default=25)
    parser.add_argument("--per-gold", type=int, default=15)
    args = parser.parse_args()
    payload = json.loads(args.errors.read_text(encoding="utf-8"))
    verified = [row for row in payload["failures"] if row.get("status") == "user_verified"]
    chars: set[str] = set()
    gold_grams: list[set[str]] = []
    blocked = set()
    for row in verified:
        blocked.add(row["gold"])
        gold_grams.append(grams(row["gold"]))
        for op in row.get("ops") or []:
            if op.get("op") == "sub":
                gold_char = op.get("gold") or ""
                pred_char = op.get("pred") or ""
                if gold_char and gold_char not in COMMON:
                    chars.add(gold_char)
                if pred_char and pred_char not in COMMON:
                    chars.add(pred_char)
    by_char: dict[str, list[dict]] = defaultdict(list)
    by_gram: dict[str, list[dict]] = defaultdict(list)
    train_rows = []
    with args.kept.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            if row.get("split") == "val" or text in blocked or not text:
                continue
            item = {"file": row["file"], "text": text}
            train_rows.append(item)
            for char in set(text):
                if char in chars and len(by_char[char]) < args.per_char:
                    by_char[char].append(item)
            for gram in grams(text):
                if len(by_gram[gram]) < 8:
                    by_gram[gram].append(item)
    chosen: dict[str, dict] = {}
    for char, rows in by_char.items():
        for item in rows:
            chosen.setdefault(item["file"], {**item, "why": f"char:{char}"})
    for gram_set in gold_grams:
        added = 0
        for gram in gram_set:
            for item in by_gram.get(gram, []):
                if item["file"] in chosen:
                    continue
                chosen[item["file"]] = {**item, "why": f"gram:{gram}"}
                added += 1
                if added >= args.per_gold:
                    break
            if added >= args.per_gold:
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for item in chosen.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"chars": len(chars), "hard": len(chosen), "train": len(train_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

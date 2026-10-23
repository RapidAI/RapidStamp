"""Exact match and 1-NED for the direct reader on val crops and the 63 gold seals.

    python training/scripts/eval_direct_reader.py \
        --ckpt /home/rapidstamp_data/checkpoints/seal_reader/best.pt \
        --dict /home/rapidstamp_data/pretrained/ppocr_keys_v1.txt \
        --kept /home/rapidstamp_data/ring_v2/kept.jsonl \
        --crops /home/rapidstamp_data/web_seals_cn_all/crops \
        --manifest training/manifests/user_seals_split.jsonl \
        --gold /home/rapidstamp_data/gold \
        --out /home/rapidstamp_data/checkpoints/seal_reader/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.direct_aug import list_images, load_charset, match_gold_image  # noqa: E402
from rapidstamp_train.direct_reader import DirectReader  # noqa: E402
from rapidstamp_train.ring_text import summarize_pairs  # noqa: E402
from train_direct_reader import SealCrops, evaluate, load_rows  # noqa: E402


def gold_rows(manifest: Path, image_root: Path, charset) -> tuple[list[dict], int]:
    paths = list_images(image_root)
    rows = []
    missing = 0
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            seals = row.get("seals") or [{}]
            text = str(seals[0].get("ring") or seals[0].get("text") or "")
            image = match_gold_image(str(row.get("file") or ""), paths)
            if image is None:
                missing += 1
                continue
            ids = charset.encode(text) or [2]
            rows.append(
                {
                    "file": image.name,
                    "abs": str(image),
                    "text": text,
                    "ids": ids,
                    "text_status": row.get("text_status") or "",
                }
            )
    return rows, missing


class GoldSet(SealCrops):
    def __getitem__(self, index: int):
        row = self.rows[index]
        from PIL import Image

        from rapidstamp_train.direct_aug import pad_square
        from train_direct_reader import _tensor

        with Image.open(row["abs"]) as image:
            canvas = pad_square(image)
        target = torch.zeros(self.charset.max_len, dtype=torch.long)
        return _tensor(canvas), target, row["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    charset = load_charset(args.dict)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectReader(ckpt["vocab_size"], ckpt["max_len"], pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    _train_rows, val_rows, _skipped = load_rows(args.kept, args.crops, charset)
    val_loader = DataLoader(
        SealCrops(val_rows, args.crops, charset, train=False, font=None),
        batch_size=args.batch,
        shuffle=False,
    )
    val = evaluate(model, val_loader, charset, device)
    gold, missing = gold_rows(args.manifest, args.gold, charset)
    gold_loader = DataLoader(GoldSet(gold, args.crops, charset, train=False, font=None), batch_size=args.batch)
    # GoldSet ignores crops; evaluate only needs the loader.
    pairs_status = {row["text_status"]: [] for row in gold}
    # Re-run gold to keep status. evaluate() drops status, so score groups here.
    model.eval()
    offset = 0
    all_pairs = []
    with torch.inference_mode():
        for images, _target, texts in gold_loader:
            images = images.to(device)
            preds = [charset.decode(row.tolist()) for row in model.greedy(images)]
            for pred, text in zip(preds, texts):
                status = gold[offset]["text_status"]
                pairs_status.setdefault(status, []).append((pred, text))
                all_pairs.append((pred, text))
                offset += 1
    report = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "val": val,
        "gold_missing_images": missing,
        "gold_all": summarize_pairs(all_pairs),
        "gold_user_verified": summarize_pairs(pairs_status.get("user_verified", [])),
        "gold_user_verified_pending_recheck": summarize_pairs(
            pairs_status.get("user_verified_pending_recheck", [])
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

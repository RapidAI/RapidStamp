"""Train the direct seal reader and score val. Gold is scored by the eval script.

    python training/scripts/train_direct_reader.py \
        --kept /home/rapidstamp_data/ring_v2/kept.jsonl \
        --crops /home/rapidstamp_data/web_seals_cn_all/crops \
        --dict /home/rapidstamp_data/pretrained/ppocr_keys_v1.txt \
        --out /home/rapidstamp_data/checkpoints/seal_reader
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.direct_aug import (  # noqa: E402
    augment_seal,
    load_charset,
    pad_square,
)
from rapidstamp_train.direct_reader import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    DirectReader,
    ids_to_text,
    teacher_tokens,
)
from rapidstamp_train.ring_text import one_ned, summarize_pairs  # noqa: E402


def _tensor(image: Image.Image) -> torch.Tensor:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class SealCrops(Dataset):
    def __init__(self, rows: list[dict], crops: Path, charset, train: bool, font: Path | None):
        self.rows = rows
        self.crops = crops
        self.charset = charset
        self.train = train
        self.font = font

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.crops / row["file"]) as image:
            canvas = pad_square(image)
        if self.train:
            canvas = augment_seal(canvas, random.Random(), self.font)
        target = torch.full((self.charset.max_len,), 0, dtype=torch.long)
        ids = row["ids"]
        target[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return _tensor(canvas), target, row["text"]


def load_rows(path: Path, crops: Path, charset) -> tuple[list[dict], list[dict], int]:
    train, val = [], []
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            ids = charset.encode(text)
            if ids is None or not (crops / row["file"]).is_file():
                skipped += 1
                continue
            item = {"file": row["file"], "text": text, "ids": ids, "split": row.get("split")}
            if row.get("split") == "val":
                val.append(item)
            else:
                train.append(item)
    return train, val, skipped


@torch.inference_mode()
def evaluate(model, loader, charset, device) -> dict:
    model.eval()
    pairs = []
    for images, _target, texts in loader:
        images = images.to(device, non_blocking=True)
        pred_ids = model.greedy(images)
        preds = ids_to_text(pred_ids, charset)
        pairs.extend(zip(preds, texts))
    summary = summarize_pairs(pairs)
    summary["empty_pred"] = sum(pred == "" for pred, _gold in pairs)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    charset = load_charset(args.dict)
    train_rows, val_rows, skipped = load_rows(args.kept, args.crops, charset)
    print(
        json.dumps(
            {"train": len(train_rows), "val": len(val_rows), "skipped": skipped, "vocab": len(charset)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    train_set = SealCrops(train_rows, args.crops, charset, train=True, font=args.font)
    val_set = SealCrops(val_rows, args.crops, charset, train=False, font=args.font)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectReader(len(charset), charset.max_len, pretrained=args.pretrained).to(device)
    others = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": others, "lr": args.lr},
        ],
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "train_log.jsonl"
    best_score: tuple[float, float] | None = None
    steps_per_epoch = max(len(train_loader), 1)
    warmup_steps = max(int(args.warmup_epochs * steps_per_epoch), 1)
    total_steps = steps_per_epoch * args.epochs
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for images, target, _texts in train_loader:
            global_step += 1
            scale = global_step / warmup_steps if global_step < warmup_steps else 0.5 * (
                1 + np.cos(np.pi * (global_step - warmup_steps) / max(total_steps - warmup_steps, 1))
            )
            optimizer.param_groups[0]["lr"] = args.encoder_lr * scale
            optimizer.param_groups[1]["lr"] = args.lr * scale
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images, teacher_tokens(target))
                loss = loss_fn(logits.transpose(1, 2), target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach()) * images.shape[0]
            seen += images.shape[0]
            if global_step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "loss": running / max(seen, 1),
                            "lr": optimizer.param_groups[1]["lr"],
                        }
                    ),
                    flush=True,
                )
        val = evaluate(model, val_loader, charset, device)
        record = {"epoch": epoch, "train_loss": running / max(seen, 1), "val": val}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val": val,
            "vocab_size": len(charset),
            "max_len": charset.max_len,
            "pretrained": args.pretrained,
        }
        torch.save(state, args.out / "last.pt")
        score = (float(val["exact"] or 0), float(val["one_ned"] or 0.0))
        if best_score is None or score > best_score:
            best_score = score
            torch.save(state, args.out / "best.pt")
            print(json.dumps({"saved_best": epoch, "exact": score[0], "one_ned": score[1]}), flush=True)


if __name__ == "__main__":
    main()

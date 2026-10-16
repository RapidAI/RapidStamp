"""Train the seal-text recognizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rapidstamp.ctc import ctc_greedy
from rapidstamp_train.corpus import all_chars
from rapidstamp_train.data import RecDataset, rec_collate
from rapidstamp_train.models import SealRec, adapt_recognizer, count_params
from rapidstamp_train.realdata import read_charset_file


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    charset_path = Path(args.data) / "charset.txt"
    charset = read_charset_file(charset_path) if charset_path.is_file() else all_chars()
    train_set = RecDataset(Path(args.data) / "rec" / "train", charset)
    val_set = RecDataset(Path(args.data) / "rec" / "val", charset)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=rec_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model = SealRec(num_classes=1 + len(charset))
    init = getattr(args, "init", None)
    if init:
        state = torch.load(init, map_location="cpu", weights_only=False)
        model = adapt_recognizer(state, charset)
        print(f"loaded recognizer {init} classes {1+len(charset)}", flush=True)
    model = model.to(device)
    params = count_params(model)
    print(f"recognizer params {params} classes {1+len(charset)} device {device} train {len(train_set)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    use_amp = device.type == "cuda"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, (images, flat, lengths) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            flat = flat.to(device, non_blocking=True)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images)
            log_probs = logits.float().log_softmax(dim=-1).permute(1, 0, 2)
            time_steps = torch.full((images.shape[0],), log_probs.shape[0], dtype=torch.long, device=device)
            loss = ctc(log_probs, flat, time_steps, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} ctc {losses[-1]:.4f}", flush=True)
        scheduler.step()
        # Validate on a loader-free slice so filenames stay aligned. Full val is small.
        accuracy = _accuracy(model, val_set, charset, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses) if losses else 0.0), "exact": accuracy}
        history.append(row)
        print(f"epoch {epoch} loss {row['loss']:.4f} exact {accuracy:.4f}", flush=True)
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "exact": accuracy,
            "charset": charset,
            "params": params,
        }
        torch.save(state, out / "last_rec.pt")
        if accuracy >= best:
            best = accuracy
            torch.save(state, out / "best_rec.pt")
    report = {"params": params, "best_exact": best, "history": history, "device": str(device)}
    (out / "rec_history.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def _accuracy(model, dataset, charset, device) -> float:
    model.eval()
    correct = 0
    batch = 128
    with torch.no_grad():
        for start in range(0, len(dataset), batch):
            samples = [dataset[i] for i in range(start, min(len(dataset), start + batch))]
            images = torch.stack([item[0] for item in samples]).to(device)
            logits = model(images).float().cpu().numpy()
            for row, sample in enumerate(samples):
                text, _score = ctc_greedy(logits[row], charset)
                truth = "".join(charset[i - 1] for i in sample[1].tolist())
                correct += int(text == truth)
    model.train()
    return correct / max(1, len(dataset))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SealRec")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--init", type=Path, default=None, help="Checkpoint to fine-tune")
    return parser


def main() -> None:
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()

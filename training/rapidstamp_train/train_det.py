"""Train the seal detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rapidstamp.labels import CLASS_NAMES
from rapidstamp_train.data import DetDataset, det_collate
from rapidstamp_train.losses import detection_loss
from rapidstamp_train.metrics_det import average_precision, predictions_from_raw
from rapidstamp_train.models import SealDet, count_params


def evaluate(model, loader, device) -> dict:
    model.eval()
    predictions = []
    ground_truth = []
    with torch.no_grad():
        for images, obj, box, cls in loader:
            raw = model(images.to(device)).float().cpu().numpy()
            obj_np = obj.numpy()
            box_np = box.numpy()
            cls_np = cls.numpy()
            for index in range(raw.shape[0]):
                predictions.append(predictions_from_raw(raw[index]))
                gt = []
                positive = np.argwhere(obj_np[index] > 0.5)
                for gy, gx in positive:
                    gt.append((int(cls_np[index, gy, gx]), box_np[index, gy, gx]))
                ground_truth.append(gt)
    model.train()
    return average_precision(predictions, ground_truth)


def train(args: argparse.Namespace) -> dict:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_set = DetDataset(Path(args.data) / "det" / "train", train=True)
    val_set = DetDataset(Path(args.data) / "det" / "val", train=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=det_collate,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=det_collate,
        pin_memory=device.type == "cuda",
    )
    model = SealDet(len(CLASS_NAMES)).to(device)
    init = getattr(args, "init", None)
    if init:
        state = torch.load(init, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        print(f"loaded detector {init}", flush=True)
    params = count_params(model)
    print(f"detector params {params} device {device}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []
    use_amp = device.type == "cuda"
    for epoch in range(args.epochs):
        model.train()
        totals = []
        for step, (images, obj, box, cls) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            obj = obj.to(device, non_blocking=True)
            box = box.to(device, non_blocking=True)
            cls = cls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                raw = model(images)
            loss, parts = detection_loss(raw.float(), obj, box, cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals.append(float(loss.detach()))
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} loss {totals[-1]:.4f} {parts}", flush=True)
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics["loss"] = float(np.mean(totals) if totals else 0.0)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"epoch {epoch} loss {metrics['loss']:.4f} map50 {metrics['map50']:.4f}",
            flush=True,
        )
        state = {"model": model.state_dict(), "epoch": epoch, "map50": metrics["map50"], "params": params}
        torch.save(state, out / "last_det.pt")
        if metrics["map50"] >= best:
            best = metrics["map50"]
            torch.save(state, out / "best_det.pt")
    report = {"params": params, "best_map50": best, "history": history, "device": str(device)}
    (out / "det_history.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SealDet")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--init", type=Path, default=None, help="Checkpoint to fine-tune")
    return parser


def main() -> None:
    train(build_argparser().parse_args())


if __name__ == "__main__":
    main()

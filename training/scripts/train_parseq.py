"""Pretrain on synthetic seals, then fine-tune on the real train split.

The process caps its own GPU allocation so the resident VLLM server keeps
the rest of the card. Val chooses the checkpoint. Gold is reported and is
not used for selection.

    python training/scripts/train_parseq.py \
        --names /home/rapidstamp_data/synth/names.txt \
        --kept /home/rapidstamp_data/ring_v2/kept.jsonl \
        --crops /home/rapidstamp_data/web_seals_cn_all/crops \
        --dict /home/rapidstamp_data/pretrained/ppocr_keys_v1.txt \
        --fonts /home/rapidstamp_data/pretrained/fonts \
        --manifest training/manifests/user_seals_split.jsonl \
        --gold /home/rapidstamp_data/gold/seals_to_label \
        --out /home/rapidstamp_data/checkpoints/parseq_reader
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.direct_aug import augment_seal, list_images, load_charset, match_gold_image, pad_square  # noqa: E402
from rapidstamp_train.hard_aug import augment_hard  # noqa: E402
from rapidstamp_train.name_corpus import load_blocked, sample_uniform  # noqa: E402
from rapidstamp_train.parseq_reader import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGE_SIZE,
    ParseqReader,
    ids_to_text,
    load_reader_weights,
)
from rapidstamp_train.ring_text import summarize_pairs  # noqa: E402
from rapidstamp_train.seal_render import FontBank, list_fonts, render_seal  # noqa: E402

_FONTS: FontBank | None = None


def _bank(paths: list[str]) -> FontBank:
    global _FONTS
    if _FONTS is None:
        _FONTS = FontBank([Path(path) for path in paths])
    return _FONTS


def to_tensor(image: Image.Image, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    image = pad_square(image).resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image).astype(np.float32) / 255.0
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def load_real(path: Path, crops: Path, charset) -> tuple[list[dict], list[dict]]:
    train, val = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or "")
            ids = charset.encode(text)
            if ids is None or not (crops / row["file"]).is_file():
                continue
            item = {"file": row["file"], "text": text, "ids": ids}
            if row.get("split") == "val":
                val.append(item)
            else:
                train.append(item)
    return train, val


def load_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
            rows.append(
                {
                    "abs": str(image),
                    "text": text,
                    "text_status": row.get("text_status") or "",
                }
            )
    return rows, missing


class MixedSeals(Dataset):
    """Synthetic names, a uniform-character draw, and real train crops."""

    def __init__(
        self,
        name_path: Path,
        real_rows: list[dict],
        crops: Path,
        font_paths: list[str],
        charset,
        blocked: set[str],
        real_prob: float,
        uniform_prob: float,
        length: int,
        augment_real: bool,
        hard_rows: list[dict] | None = None,
        hard_prob: float = 0.0,
        image_size: int = IMAGE_SIZE,
    ):
        self.name_path = str(name_path)
        self._names: list[str] | None = None
        self.real_rows = real_rows
        self.hard_rows = hard_rows or []
        self.hard_prob = hard_prob
        self.image_size = int(image_size)
        self.crops = crops
        self.font_paths = font_paths
        self.charset = charset
        self.blocked = blocked
        self.real_prob = real_prob
        self.uniform_prob = uniform_prob
        self.length = length
        self.augment_real = augment_real

    def __len__(self) -> int:
        return self.length

    def names(self) -> list[str]:
        # Loaded in the worker so the 1.5M-name list is not pickled to every process.
        if self._names is None:
            self._names = load_lines(Path(self.name_path))
            if not self._names:
                raise RuntimeError(f"no names in {self.name_path}")
        return self._names

    def _target(self, text: str) -> torch.Tensor | None:
        ids = self.charset.encode(text)
        if not ids:
            return None
        target = torch.zeros(self.charset.max_len, dtype=torch.long)
        target[: len(ids)] = torch.tensor(ids, dtype=torch.long)
        return target

    def _sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random()
        roll = rng.random()
        if self.real_rows and roll < self.real_prob:
            use_hard = self.hard_rows and rng.random() < self.hard_prob
            pool = self.hard_rows if use_hard else self.real_rows
            row = pool[rng.randrange(len(pool))]
            with Image.open(self.crops / row["file"]) as image:
                canvas = pad_square(image)
            if self.augment_real:
                font = Path(self.font_paths[0]) if self.font_paths else None
                canvas = augment_hard(canvas, rng, font) if use_hard else augment_seal(canvas, rng, font)
            target = self._target(row["text"])
            if target is None:
                raise RuntimeError("real label does not encode")
            return to_tensor(canvas, self.image_size), target
        if roll < self.real_prob + self.uniform_prob:
            text = sample_uniform(rng, self.charset, self.blocked)
        else:
            names = self.names()
            text = names[rng.randrange(len(names))]
        image = render_seal(text, rng, _bank(self.font_paths))
        target = self._target(text)
        if target is None:
            raise RuntimeError("synthetic label does not encode")
        return to_tensor(image, self.image_size), target

    def __getitem__(self, index: int):
        last = None
        for _ in range(4):
            try:
                return self._sample()
            except Exception as exc:  # noqa: BLE001 - a bad glyph should not kill the epoch
                last = exc
        raise RuntimeError(f"sample failed: {last}")


class EvalCrops(Dataset):
    def __init__(self, rows: list[dict], crops: Path | None, charset, image_size: int = IMAGE_SIZE):
        self.rows = rows
        self.crops = crops
        self.charset = charset
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["abs"]) if row.get("abs") else self.crops / row["file"]
        with Image.open(path) as image:
            canvas = pad_square(image)
        return to_tensor(canvas, self.image_size), row["text"]


@torch.inference_mode()
def evaluate(model, loader, charset, device) -> dict:
    model.eval()
    pairs = []
    for images, texts in loader:
        images = images.to(device, non_blocking=True)
        preds = ids_to_text(model.tokens(images), charset)
        pairs.extend(zip(preds, texts))
    summary = summarize_pairs(pairs)
    summary["empty_pred"] = sum(pred == "" for pred, _gold in pairs)
    summary["examples"] = [{"pred": pred, "gold": gold} for pred, gold in pairs[:4]]
    return summary


def _score_group(pairs: list[tuple[str, str]]) -> dict:
    summary = summarize_pairs(pairs)
    summary["empty_pred"] = sum(pred == "" for pred, _gold in pairs)
    return summary


@torch.inference_mode()
def evaluate_gold(model, rows: list[dict], charset, device, batch: int, image_size: int = IMAGE_SIZE) -> dict:
    model.eval()
    loader = DataLoader(
        EvalCrops(rows, None, charset, image_size=image_size),
        batch_size=batch,
        shuffle=False,
        num_workers=0,
    )
    grouped: dict[str, list[tuple[str, str]]] = {}
    offset = 0
    all_pairs = []
    for images, texts in loader:
        images = images.to(device, non_blocking=True)
        preds = ids_to_text(model.tokens(images), charset)
        for pred, text in zip(preds, texts):
            status = rows[offset]["text_status"]
            grouped.setdefault(status, []).append((pred, text))
            all_pairs.append((pred, text))
            offset += 1
    return {
        "missing": 0,
        "all": _score_group(all_pairs),
        "user_verified": _score_group(grouped.get("user_verified", [])),
        "user_verified_pending_recheck": _score_group(grouped.get("user_verified_pending_recheck", [])),
        "examples": [{"pred": pred, "gold": gold} for pred, gold in all_pairs[:6]],
    }


def refinement_loss(model, images, target, loss_fn):
    """Blind pass plus a second pass that has to correct a noisy first guess."""
    memory = model.encode(images)
    blank = torch.zeros_like(target)
    logits1 = model.decode(memory, blank)
    loss1 = loss_fn(logits1.float().transpose(1, 2), target)
    context = logits1.argmax(dim=-1).detach()
    content = target.ne(0)
    flip = (torch.rand(target.shape, device=target.device) < 0.2) & content
    random_ids = torch.randint(1, model.vocab_size, target.shape, device=target.device)
    context = torch.where(flip, random_ids, context)
    logits2 = model.decode(memory, context)
    loss2 = loss_fn(logits2.float().transpose(1, 2), target)
    return loss1 + 0.5 * loss2


def save_checkpoint(path: Path, model, step: int, args, val: dict | None, gold: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "val": val,
            "gold": gold,
            "vocab_size": len(args.charset) if hasattr(args, "charset") else None,
            "max_len": args.max_len,
            "backbone": args.backbone,
            "image_size": args.image_size,
            "pretrained": args.pretrained,
        },
        path,
    )


def run_stage(args, model, data, loader, val_loader, gold, device, stage: str, steps: int, optimizer) -> dict:
    out = args.out / stage
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best: tuple[float, float] | None = None
    seen = 0
    running = 0.0
    started = time.time()
    report = {}
    warmup = max(args.warmup, 1)
    while seen < steps:
        try:
            images, target = next(data)
        except StopIteration:
            data = iter(loader)
            images, target = next(data)
        seen += images.shape[0]
        progress = min(seen, steps)
        scale = progress / warmup if progress < warmup else 0.5 * (
            1 + np.cos(np.pi * (progress - warmup) / max(steps - warmup, 1))
        )
        optimizer.param_groups[0]["lr"] = args.encoder_lr * scale
        optimizer.param_groups[1]["lr"] = args.lr * scale
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = refinement_loss(model, images, target, loss_fn)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach()) * images.shape[0]
        if seen % args.log_every < images.shape[0]:
            elapsed = max(time.time() - started, 1e-3)
            row = {
                "stage": stage,
                "seen": seen,
                "loss": running / seen,
                "img_per_s": seen / elapsed,
                "lr": optimizer.param_groups[1]["lr"],
            }
            print(json.dumps(row), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        if seen % args.save_every < images.shape[0]:
            save_checkpoint(out / "last.pt", model, seen, args, None, None)
        if seen % args.eval_every < images.shape[0] or seen >= steps:
            val = evaluate(model, val_loader, args.charset, device)
            gold_report = evaluate_gold(model, gold, args.charset, device, args.batch, args.image_size)
            report = {"stage": stage, "seen": seen, "val": val, "gold": gold_report}
            print(json.dumps(report, ensure_ascii=False), flush=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report, ensure_ascii=False) + "\n")
            (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            score = (val["exact"], val["one_ned"] or 0.0)
            if best is None or score > best:
                best = score
                save_checkpoint(out / "best.pt", model, seen, args, val, gold_report)
            save_checkpoint(out / "last.pt", model, seen, args, val, gold_report)
            model.train()
    return report


def build_model(args, device) -> ParseqReader:
    model = ParseqReader(
        len(args.charset),
        args.charset.max_len,
        backbone=args.backbone,
        pretrained=args.pretrained,
        checkpointing=args.checkpointing,
        img_size=args.image_size,
    )
    if args.init:
        ckpt = torch.load(args.init, map_location="cpu")
        load_reader_weights(model, ckpt["model"])
    return model.to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocked", type=Path, default=None)
    parser.add_argument("--backbone", default="vit_small_patch16_384")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=2_000_000)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-encoder-lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--real-prob", type=float, default=0.12)
    parser.add_argument("--uniform-prob", type=float, default=0.10)
    parser.add_argument("--finetune-real-prob", type=float, default=0.60)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--save-every", type=int, default=20_000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--memory-fraction", type=float, default=0.105)
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--finetune-only", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=0)
    parser.add_argument("--hard-list", type=Path, default=None)
    parser.add_argument("--hard-prob", type=float, default=0.0)
    args = parser.parse_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if args.image_size % 16 != 0:
        raise SystemExit(f"image size {args.image_size} is not a multiple of the patch size 16")
    charset = load_charset(args.dict)
    args.charset = charset
    args.max_len = charset.max_len
    font_paths = [str(path) for path in list_fonts(args.fonts)]
    if not font_paths:
        raise SystemExit(f"no fonts in {args.fonts}")
    name_count = sum(1 for line in args.names.open(encoding="utf-8") if line.strip())
    if name_count < 1:
        raise SystemExit(f"no names in {args.names}")
    blocked = set(load_lines(args.blocked)) if args.blocked else load_blocked(args.kept, args.manifest)
    real_train, real_val = load_real(args.kept, args.crops, charset)
    gold, missing = gold_rows(args.manifest, args.gold, charset)
    print(
        json.dumps(
            {
                "names": name_count,
                "blocked": len(blocked),
                "train": len(real_train),
                "val": len(real_val),
                "gold": len(gold),
                "gold_missing": missing,
                "fonts": len(font_paths),
                "vocab": len(charset),
                "image_size": args.image_size,
                "backbone": args.backbone,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if missing:
        raise SystemExit(f"{missing} gold images missing")

    pretrain_set = MixedSeals(
        args.names,
        real_train,
        args.crops,
        font_paths,
        charset,
        blocked,
        args.real_prob,
        args.uniform_prob,
        length=max(args.samples, args.batch),
        augment_real=True,
        image_size=args.image_size,
    )
    val_loader = DataLoader(
        EvalCrops(real_val, args.crops, charset, image_size=args.image_size),
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args, device)
    if args.probe_steps:
        probe = _loader(pretrain_set, args, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        started = time.time()
        seen = 0
        model.train()
        for images, target in probe:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = refinement_loss(model, images, target, loss_fn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            seen += images.shape[0]
            if seen >= args.probe_steps * args.batch:
                break
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "probe_images": seen,
                    "seconds": time.time() - started,
                    "img_per_s": seen / max(time.time() - started, 1e-3),
                    "mem_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
                }
            ),
            flush=True,
        )
        return

    if not args.finetune_only:
        others = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder.parameters(), "lr": args.encoder_lr},
                {"params": others, "lr": args.lr},
            ],
            weight_decay=0.01,
        )
        pretrain_loader = _loader(pretrain_set, args, device)
        run_stage(
            args,
            model,
            iter(pretrain_loader),
            pretrain_loader,
            val_loader,
            gold,
            device,
            "pretrain",
            args.samples,
            optimizer,
        )
        best_path = args.out / "pretrain" / "best.pt"
        if best_path.is_file():
            model.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
        _export(model, args.out / "pretrain" / "parseq_reader.onnx", args)

    if args.skip_finetune:
        return
    if args.finetune_only and args.init is None:
        best_path = args.out / "pretrain" / "best.pt"
        if best_path.is_file():
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    finetune_steps = int(args.finetune_epochs * max(len(real_train), 1) / max(args.finetune_real_prob, 0.1))
    hard_rows = []
    if args.hard_list is not None and args.hard_prob > 0:
        train_files = {row["file"] for row in real_train}
        with args.hard_list.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("file") in train_files and charset.encode(str(item.get("text") or "")):
                    hard_rows.append({"file": item["file"], "text": item["text"]})
        print(json.dumps({"hard_rows": len(hard_rows), "hard_prob": args.hard_prob}), flush=True)
    finetune_set = MixedSeals(
        args.names,
        real_train,
        args.crops,
        font_paths,
        charset,
        blocked,
        args.finetune_real_prob,
        args.uniform_prob,
        length=max(finetune_steps, args.batch),
        augment_real=True,
        hard_rows=hard_rows,
        hard_prob=args.hard_prob,
        image_size=args.image_size,
    )
    args.encoder_lr = args.finetune_encoder_lr
    args.lr = args.finetune_lr
    args.warmup = min(args.warmup, max(finetune_steps // 20, 1))
    others = [param for name, param in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": others, "lr": args.lr},
        ],
        weight_decay=0.01,
    )
    finetune_loader = _loader(finetune_set, args, device)
    run_stage(
        args,
        model,
        iter(finetune_loader),
        finetune_loader,
        val_loader,
        gold,
        device,
        "finetune",
        finetune_steps,
        optimizer,
    )
    best_path = args.out / "finetune" / "best.pt"
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location="cpu")["model"])
    _export(model, args.out / "finetune" / "parseq_reader.onnx", args)


def _loader(dataset, args, device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
        multiprocessing_context="spawn" if args.workers else None,
    )


def _export(model, path: Path, args) -> None:
    """Export a CPU copy. The training module stays on the GPU."""
    try:
        replica = ParseqReader(
            len(args.charset),
            args.charset.max_len,
            backbone=args.backbone,
            pretrained=False,
            checkpointing=False,
            img_size=args.image_size,
        )
        load_reader_weights(replica, model.state_dict())
        replica.eval()

        class _Greedy(torch.nn.Module):
            def __init__(self, reader):
                super().__init__()
                self.reader = reader

            def forward(self, image: torch.Tensor) -> torch.Tensor:
                return self.reader.tokens(image)

        path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, 3, args.image_size, args.image_size)
        torch.onnx.export(
            _Greedy(replica),
            (dummy,),
            str(path),
            input_names=["image"],
            output_names=["tokens"],
            dynamic_axes={"image": {0: "batch"}, "tokens": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        print(json.dumps({"onnx": str(path), "bytes": path.stat().st_size}), flush=True)
    except Exception:
        print(json.dumps({"onnx_error": traceback.format_exc()}), flush=True)


if __name__ == "__main__":
    main()

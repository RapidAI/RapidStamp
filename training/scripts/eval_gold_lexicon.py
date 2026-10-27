#!/usr/bin/env python3
"""Gold exact and 1-NED with and without the name-lexicon rescore.

The lexicon is a character trigram over synth/names.txt. Gold strings are
not added. alpha 0 is the greedy decode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "training" / "scripts"))

from rapidstamp_train.direct_aug import load_charset  # noqa: E402
from rapidstamp_train.lexicon_decode import NameLM, constrained_text  # noqa: E402
from rapidstamp_train.parseq_reader import ParseqReader, load_reader_weights  # noqa: E402
from rapidstamp_train.ring_text import one_ned, summarize_pairs  # noqa: E402
from train_parseq import EvalCrops, gold_rows  # noqa: E402


def load_names(path: Path, blocked: set[str]) -> list[str]:
    names = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and text not in blocked:
                names.append(text)
    return names


def score_split(pairs: list[tuple[str, str, str]], status: str) -> dict:
    chosen = [(pred, gold) for pred, gold, row_status in pairs if row_status == status]
    summary = summarize_pairs(chosen)
    summary["n"] = len(chosen)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()

    charset = load_charset(args.dict)
    rows, missing = gold_rows(args.manifest, args.gold, charset)
    blocked = {row["text"] for row in rows}
    names = load_names(args.names, blocked)
    lm = NameLM(names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    backbone = ckpt.get("backbone") or "vit_small_patch16_384"
    image_size = int(ckpt.get("image_size") or 384)
    model = ParseqReader(
        len(charset),
        charset.max_len,
        backbone=backbone,
        pretrained=False,
        checkpointing=False,
        img_size=image_size,
    )
    load_reader_weights(model, ckpt["model"])
    model.to(device).eval()
    loader = DataLoader(
        EvalCrops(rows, None, charset, image_size=image_size),
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    stored = []
    with torch.inference_mode():
        for images, _texts in loader:
            images = images.to(device)
            memory = model.encode(images)
            blank = torch.zeros(images.shape[0], model.max_len, dtype=torch.long, device=device)
            logits1 = model.decode(memory, blank)
            context = logits1.argmax(dim=-1)
            logits2 = model.decode(memory, context)
            stored.append(F.log_softmax(logits2.float(), dim=-1).cpu())
    logp = torch.cat(stored, dim=0)

    def run(alpha: float, tau: float) -> dict:
        pairs = []
        detailed = []
        for index, row in enumerate(rows):
            pred = constrained_text(logp[index], charset, lm, alpha=alpha, tau=tau)
            pairs.append((pred, row["text"]))
            detailed.append({"file": Path(row["abs"]).name, "status": row["text_status"], "gold": row["text"], "pred": pred, "one_ned": one_ned(pred, row["text"])})
        tagged = [(pred, gold, row["text_status"]) for (pred, gold), row in zip(pairs, rows)]
        return {
            "alpha": alpha,
            "tau": tau,
            "all": summarize_pairs(pairs),
            "user_verified": score_split(tagged, "user_verified"),
            "user_verified_pending_recheck": score_split(tagged, "user_verified_pending_recheck"),
            "failures": [item for item in detailed if item["pred"] != item["gold"]],
        }

    greedy = run(0.0, args.tau)
    if args.grid:
        reports = [greedy]
        for alpha in (0.4, 0.8, 1.5, 3.0):
            for tau in (1.5, 4.0, 8.0):
                reports.append(run(alpha, tau))
        best = max(reports, key=lambda item: (item["all"]["exact"], item["all"]["one_ned"] or 0.0))
        payload = {"ckpt": str(args.ckpt), "step": ckpt.get("step"), "names": len(names), "missing": missing, "greedy": greedy, "grid_best": best, "grid": [{k: v for k, v in item.items() if k != "failures"} for item in reports]}
    else:
        constrained = run(args.alpha, args.tau)
        payload = {"ckpt": str(args.ckpt), "step": ckpt.get("step"), "names": len(names), "missing": missing, "greedy": greedy, "constrained": constrained}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shown = payload.get("grid_best", payload.get("constrained", greedy))
    print(json.dumps({"greedy": greedy["all"], "chosen": {k: shown[k] for k in ("alpha", "tau", "all", "user_verified", "user_verified_pending_recheck")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()

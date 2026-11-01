#!/usr/bin/env python3
"""Export a PARSeq ring reader for deployment (log-probability output).

The training-time ONNX (``parseq_reader.onnx``) returns argmax ``tokens`` only,
so the lexicon rescoring used in evaluation cannot run on it. This export
returns ``logprobs`` float32 (N, max_len, vocab): log_softmax of the second
decode pass, the same tensor ``eval_gold_v2_cli.py`` feeds to
``constrained_text``. Input ``image`` float32 (N, 3, S, S), ImageNet-normalized,
from a white-padded square crop resized with PIL bilinear.

    python export_ring_reader.py --ckpt .../finetune/best.pt --out .../reader.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rapidstamp_train.direct_aug import load_charset  # noqa: E402
from rapidstamp_train.parseq_reader import ParseqReader, load_reader_weights  # noqa: E402


class LogProbHead(torch.nn.Module):
    def __init__(self, model: ParseqReader):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        memory = self.model.encode(image)
        context = torch.zeros(image.shape[0], self.model.max_len, dtype=torch.long, device=image.device)
        context = self.model.decode(memory, context).argmax(dim=-1)
        return F.log_softmax(self.model.decode(memory, context).float(), dim=-1)


def load(ckpt_path: Path, charset_path: Path) -> tuple[ParseqReader, int]:
    charset = load_charset(charset_path)
    ck = torch.load(ckpt_path, map_location="cpu")
    size = int(ck.get("image_size") or 384)
    model = ParseqReader(len(charset), charset.max_len, backbone=ck.get("backbone") or "vit_small_patch16_384",
                         pretrained=False, checkpointing=False, img_size=size)
    load_reader_weights(model, ck["model"])
    return model.eval(), size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--charset", type=Path, default=Path("/home/rapidstamp_data/pretrained/ppocr_keys_v1.txt"))
    args = ap.parse_args()
    model, size = load(args.ckpt, args.charset)
    head = LogProbHead(model).eval()
    dummy = torch.zeros(1, 3, size, size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(head, dummy, str(args.out), input_names=["image"], output_names=["logprobs"],
                      dynamic_axes={"image": {0: "n"}, "logprobs": {0: "n"}}, opset_version=17, dynamo=False)
    print({"out": str(args.out), "image_size": size, "max_len": model.max_len})


if __name__ == "__main__":
    main()

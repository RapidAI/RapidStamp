"""Export the direct reader's greedy decode to ONNX.

Input ``image`` is NCHW float, ImageNet-normalized, 224 square.
Output ``tokens`` is int64 (N, max_len), ending at the first eos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.direct_reader import IMAGE_SIZE, DirectReader  # noqa: E402


class _Greedy(nn.Module):
    def __init__(self, model: DirectReader):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.tokens(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = DirectReader(ckpt["vocab_size"], ckpt["max_len"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    greedy = _Greedy(model)
    dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        greedy,
        (dummy,),
        str(args.out),
        input_names=["image"],
        output_names=["tokens"],
        dynamic_axes={"image": {0: "batch"}, "tokens": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {args.out} bytes {args.out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()

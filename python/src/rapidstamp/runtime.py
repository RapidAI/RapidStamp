"""Dependency-free interpreter for the RapidStamp weight pack.

The pack is the graph that training also writes to ONNX: folded convs,
ReLU, max-pool, a width-sequence transpose, and a final linear layer.
Keeping the runtime this small is what lets the C++ SDK avoid bundling
ONNX Runtime while still consuming the same exported weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Op:
    kind: str
    args: tuple[int, ...] = ()
    w: np.ndarray | None = None
    b: np.ndarray | None = None


@dataclass
class RspModel:
    kind: str
    input_shape: tuple[int, int, int, int]
    meta: dict[str, str] = field(default_factory=dict)
    ops: list[Op] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    path: str = ""

    def meta_float(self, key: str, default: float) -> float:
        if key not in self.meta:
            return default
        return float(self.meta[key])

    def meta_int(self, key: str, default: int) -> int:
        if key not in self.meta:
            return default
        return int(float(self.meta[key]))


def _conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    groups: int,
) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    n, c, h, w = x.shape
    out_c, ic_g, kh, kw = weight.shape
    if c != ic_g * groups or out_c % groups != 0:
        raise ValueError(f"conv groups mismatch x={x.shape} w={weight.shape} groups={groups}")
    if pad_h or pad_w:
        x = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    oh = (x.shape[2] - kh) // stride_h + 1
    ow = (x.shape[3] - kw) // stride_w + 1
    oc_g = out_c // groups
    outputs = []
    for g in range(groups):
        xg = x[:, g * ic_g : (g + 1) * ic_g]
        patches = np.empty((n, oh, ow, ic_g, kh, kw), dtype=np.float32)
        for ky in range(kh):
            for kx in range(kw):
                patch = xg[:, :, ky : ky + stride_h * oh : stride_h, kx : kx + stride_w * ow : stride_w]
                patches[:, :, :, :, ky, kx] = np.transpose(patch, (0, 2, 3, 1))
        col = patches.reshape(n, oh, ow, ic_g * kh * kw)
        wg = weight[g * oc_g : (g + 1) * oc_g].reshape(oc_g, -1)
        outputs.append(np.einsum("nhwc,oc->nohw", col, wg, optimize=True))
    y = np.concatenate(outputs, axis=1)
    y += bias.reshape(1, -1, 1, 1)
    return y


def _maxpool2d(
    x: np.ndarray, kh: int, kw: int, stride_h: int, stride_w: int, pad_h: int, pad_w: int
) -> np.ndarray:
    if pad_h or pad_w:
        x = np.pad(
            x,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            constant_values=-np.inf,
        )
    n, c, h, w = x.shape
    oh = (h - kh) // stride_h + 1
    ow = (w - kw) // stride_w + 1
    out = np.full((n, c, oh, ow), -np.inf, dtype=np.float32)
    for ky in range(kh):
        for kx in range(kw):
            patch = x[:, :, ky : ky + stride_h * oh : stride_h, kx : kx + stride_w * ow : stride_w]
            out = np.maximum(out, patch)
    return out


def _to_seq(x: np.ndarray) -> np.ndarray:
    n, c, h, w = x.shape
    if h == 1:
        return np.transpose(x[:, :, 0, :], (0, 2, 1))
    if w == 1:
        return np.transpose(x[:, :, :, 0], (0, 2, 1))
    raise ValueError(f"to_seq expected a singleton spatial axis, got {x.shape}")


def run_model(model: RspModel, x: np.ndarray) -> np.ndarray:
    if tuple(x.shape) != tuple(model.input_shape) and x.shape[0] != 1:
        # Allow a matching CHW tensor by adding batch, and a batch other than
        # the declared N as long as C,H,W match.
        pass
    if x.ndim != 4:
        raise ValueError(f"expected NCHW input, got {x.shape}")
    if x.shape[1:] != model.input_shape[1:]:
        raise ValueError(f"expected input {model.input_shape}, got {x.shape}")
    y = np.ascontiguousarray(x, dtype=np.float32)
    for op in model.ops:
        if op.kind == "conv":
            _in_c, _out_c, kh, kw, sh, sw, ph, pw, groups = op.args
            y = _conv2d(y, op.w, op.b, sh, sw, ph, pw, groups)
        elif op.kind == "relu":
            y = np.maximum(y, 0.0)
        elif op.kind == "maxpool":
            kh, kw, sh, sw, ph, pw = op.args
            y = _maxpool2d(y, kh, kw, sh, sw, ph, pw)
        elif op.kind == "to_seq":
            y = _to_seq(y)
        elif op.kind == "gap":
            y = y.mean(axis=(2, 3))
        elif op.kind == "linear":
            y = y @ op.w.T + op.b
        else:
            raise ValueError(f"unsupported op {op.kind}")
    return y


def save_rsp(
    path: str | Path,
    kind: str,
    input_shape: tuple[int, int, int, int],
    ops: list[dict],
    meta: dict[str, float | int | str] | None = None,
    classes: list[str] | None = None,
    words: list[str] | None = None,
) -> None:
    """Write ``<path>.manifest`` and ``<path>.bin``.

    ``path`` is a prefix, not a filename. ``ops`` entries are dicts with a
    ``kind`` and, for conv/linear, float32 ``w`` and ``b`` arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blobs: list[np.ndarray] = []
    lines = [
        "rapidstamp 1",
        f"kind {kind}",
        "input {} {} {} {}".format(*input_shape),
    ]
    for key, value in (meta or {}).items():
        lines.append(f"meta {key} {value}")
    for op in ops:
        kind_op = op["kind"]
        if kind_op == "conv":
            w = np.ascontiguousarray(op["w"], dtype=np.float32)
            b = np.ascontiguousarray(op["b"], dtype=np.float32)
            in_c = int(w.shape[1] * op["groups"])
            out_c, _icg, kh, kw = w.shape
            lines.append(
                "op conv {} {} {} {} {} {} {} {} {}".format(
                    in_c,
                    out_c,
                    kh,
                    kw,
                    op["stride_h"],
                    op["stride_w"],
                    op["pad_h"],
                    op["pad_w"],
                    op["groups"],
                )
            )
            blobs.append(w.ravel())
            blobs.append(b.ravel())
        elif kind_op == "relu":
            lines.append("op relu")
        elif kind_op == "maxpool":
            lines.append(
                "op maxpool {} {} {} {} {} {}".format(
                    op["kh"], op["kw"], op["stride_h"], op["stride_w"], op["pad_h"], op["pad_w"]
                )
            )
        elif kind_op == "to_seq":
            lines.append("op to_seq")
        elif kind_op == "gap":
            lines.append("op gap")
        elif kind_op == "linear":
            w = np.ascontiguousarray(op["w"], dtype=np.float32)
            b = np.ascontiguousarray(op["b"], dtype=np.float32)
            lines.append(f"op linear {w.shape[1]} {w.shape[0]}")
            blobs.append(w.ravel())
            blobs.append(b.ravel())
        else:
            raise ValueError(kind_op)
    weights = np.concatenate(blobs) if blobs else np.zeros((0,), dtype=np.float32)
    lines.append(f"weight_count {int(weights.size)}")
    if classes:
        lines.append(f"classes {len(classes)}")
        for ch in classes:
            if len(ch) != 1:
                raise ValueError("phase-1 classes are single characters")
            lines.append(f"class {ch}")
    if words:
        lines.append(f"words {len(words)}")
        for word in words:
            if any(ch.isspace() for ch in word) or not word:
                raise ValueError(f"lexicon entry must be one token, got {word!r}")
            lines.append(f"word {word}")
    manifest = path.with_suffix(".manifest")
    binary = path.with_suffix(".bin")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    weights.tofile(binary)


def load_rsp(path: str | Path) -> RspModel:
    manifest_path = Path(path)
    if manifest_path.suffix != ".manifest":
        manifest_path = manifest_path.with_suffix(".manifest")
    text = manifest_path.read_text(encoding="utf-8").splitlines()
    if not text or text[0].strip() != "rapidstamp 1":
        raise ValueError(f"{manifest_path} is not a rapidstamp v1 manifest")
    kind = ""
    input_shape = (1, 3, 128, 128)
    meta: dict[str, str] = {}
    classes: list[str] = []
    words: list[str] = []
    op_specs: list[tuple[str, tuple[int, ...]]] = []
    weight_count = 0
    for raw in text[1:]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "kind":
            kind = parts[1]
        elif tag == "input":
            input_shape = tuple(int(v) for v in parts[1:5])  # type: ignore[assignment]
        elif tag == "meta":
            meta[parts[1]] = parts[2]
        elif tag == "weight_count":
            weight_count = int(parts[1])
        elif tag == "classes":
            continue
        elif tag == "class":
            classes.append(parts[1])
        elif tag == "words":
            continue
        elif tag == "word":
            words.append(parts[1])
        elif tag == "op":
            op_kind = parts[1]
            args = tuple(int(v) for v in parts[2:])
            op_specs.append((op_kind, args))
        else:
            raise ValueError(f"bad manifest line: {line}")
    binary = manifest_path.with_suffix(".bin")
    weights = np.fromfile(binary, dtype=np.float32)
    if weights.size != weight_count:
        raise ValueError(f"weight_count {weight_count} but file has {weights.size} floats")
    cursor = 0

    def take(n: int) -> np.ndarray:
        nonlocal cursor
        slab = weights[cursor : cursor + n]
        if slab.size != n:
            raise ValueError("unexpected end of weight file")
        cursor += n
        return slab

    ops: list[Op] = []
    for op_kind, args in op_specs:
        if op_kind == "conv":
            in_c, out_c, kh, kw, _sh, _sw, _ph, _pw, groups = args
            ic_g = in_c // groups
            w = take(out_c * ic_g * kh * kw).reshape(out_c, ic_g, kh, kw)
            b = take(out_c).copy()
            ops.append(Op("conv", args, w, b))
        elif op_kind == "linear":
            in_f, out_f = args
            w = take(out_f * in_f).reshape(out_f, in_f)
            b = take(out_f).copy()
            ops.append(Op("linear", args, w, b))
        elif op_kind in {"relu", "maxpool", "to_seq", "gap"}:
            ops.append(Op(op_kind, args))
        else:
            raise ValueError(op_kind)
    if cursor != weights.size:
        raise ValueError(f"unused weights: {weights.size - cursor}")
    return RspModel(kind, input_shape, meta, ops, classes, words, str(manifest_path))

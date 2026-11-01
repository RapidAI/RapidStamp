"""PARSeq whole-crop ring reader with lexicon rescoring and length routing.

A release directory holds:

    route.json          {"main": "reader_main.onnx", "long": "reader_long.onnx",
                         "charset": "charset.txt", "names": "names_lm.txt",
                         "long_threshold": 16, "alpha": 0.4, "tau": 4.0,
                         "max_flips": 3, "image_size": 512}
    reader_*.onnx       input ``image`` (N,3,S,S) float32, output ``logprobs``
    charset.txt         one character per line (ids start at 3; 0/1/2 = pad/bos/eos)
    names_lm.txt        seal names for the character trigram (not in git)

Routing: read with ``main``; if its lexicon text has ``long_threshold`` or
more characters, read again with ``long`` and return that text. Drop the
``long`` key (or set the threshold to 0) to use ``main`` alone, which is how
the rollback release is laid out. Keep NameLM / constrained_text in step with
``training/rapidstamp_train/lexicon_decode.py`` and ``cpp/src/ring_reader.cpp``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

SUFFIXES = (
    "有限责任公司", "股份有限公司", "集团有限公司", "有限公司", "发票监制章", "人民政府", "监督管理局",
    "市场监督管理局", "委员会", "办事处", "办公室", "管理局", "公安局", "检察院", "法院", "卫生院", "事务所",
    "合作社", "研究院", "研究所", "服务中心", "交易中心", "分公司", "支行", "分行", "大学", "学院", "学校",
    "医院", "银行", "中心", "集团", "政府", "幼儿园", "经营部", "管理处", "信用社", "公司", "局", "厅", "院",
    "所", "站", "队", "部", "社", "厂", "店", "馆", "会", "署", "处", "自治区", "自治州", "街道", "省", "市",
    "区", "县", "镇", "乡", "村",
)
_SUFFIX_ORDER = sorted(set(SUFFIXES), key=len, reverse=True)


def matching_suffix(text: str) -> str:
    for suffix in _SUFFIX_ORDER:
        if len(suffix) < len(text) and text.endswith(suffix):
            return suffix
    return ""


class NameLM:
    """Interpolated character trigram over seal names (same maths as training)."""

    def __init__(self, names: list[str]):
        self.uni: Counter = Counter()
        self.bi: Counter = Counter()
        self.tri: Counter = Counter()
        self.total = 0
        for name in names:
            if not name:
                continue
            self.total += len(name)
            self.bi[("^", "^")] += 1
            prev = ("^", "^")
            for char in name:
                self.uni[char] += 1
                self.bi[(prev[1], char)] += 1
                self.tri[(prev[0], prev[1], char)] += 1
                prev = (prev[1], char)

    def logp(self, left: str, mid: str, char: str) -> float:
        n_ab = self.bi[(left, mid)] if left != "" else 0
        n_abc = self.tri[(left, mid, char)]
        n_b = self.uni[mid] if mid != "^" else self.total
        n_bc = self.bi[(mid, char)]
        p_tri = (n_abc + 0.05) / (n_ab + 0.05 * 40.0) if n_ab or n_abc else 0.0
        p_bi = (n_bc + 0.05) / (n_b + 0.05 * 40.0) if n_b or n_bc else 0.0
        p_uni = (self.uni[char] + 0.05) / (self.total + 0.05 * 40.0)
        if p_tri == 0.0 and p_bi == 0.0:
            mixed = p_uni
        elif p_tri == 0.0:
            mixed = 0.75 * p_bi + 0.25 * p_uni
        else:
            mixed = 0.65 * p_tri + 0.25 * p_bi + 0.10 * p_uni
        return math.log(max(mixed, 1e-8))


def decode_ids(ids, chars: list[str]) -> str:
    out = []
    for raw in ids:
        index = int(raw)
        if index in (EOS_ID, PAD_ID):
            break
        if index == BOS_ID:
            continue
        j = index - 3
        if 0 <= j < len(chars):
            out.append(chars[j])
    return "".join(out)


def constrained_text(logp: np.ndarray, chars: list[str], lm: NameLM | None, alpha: float, tau: float,
                     max_flips: int = 3) -> str:
    """``logp`` (max_len, vocab). alpha 0 or no LM returns the greedy string."""
    greedy_ids, gaps, runners = [], [], []
    for row in logp:
        # Stable descending order, ties broken by lower index (matches sorted(..., reverse=True)).
        order = np.argsort(-row.astype(np.float64), kind="stable")
        top, second = int(order[0]), int(order[1])
        greedy_ids.append(top)
        gaps.append(float(row[top]) - float(row[second]))
        runners.append(second)
        if top in (PAD_ID, EOS_ID):
            break
    if greedy_ids and greedy_ids[-1] in (PAD_ID, EOS_ID):
        char_ids, char_gaps, char_runners = greedy_ids[:-1], gaps[:-1], runners[:-1]
    else:
        char_ids, char_gaps, char_runners = list(greedy_ids), list(gaps), list(runners)
    greedy = decode_ids(char_ids + [EOS_ID], chars)
    if alpha == 0.0 or lm is None or not greedy:
        return greedy
    stem_end = len(greedy) - len(matching_suffix(greedy))
    uncertain = sorted((i for i in range(stem_end) if char_gaps[i] < tau), key=lambda i: char_gaps[i])[:max_flips]
    if not uncertain:
        return greedy

    def score(ids: list[int]) -> float:
        text = decode_ids(ids + [EOS_ID], chars)
        if len(text) != len(ids):
            return -1e9
        model = 0.0
        language = 0.0
        left, mid = "^", "^"
        for index, token in enumerate(ids):
            model += float(logp[index][token])
            char = text[index]
            language += lm.logp(left, mid, char)
            left, mid = mid, char
        count = max(len(ids), 1)
        return model / count + alpha * (language / count)

    best_ids, best = list(char_ids), score(list(char_ids))
    for mask in range(1 << len(uncertain)):
        ids = list(char_ids)
        for bit, index in enumerate(uncertain):
            if mask & (1 << bit):
                ids[index] = char_runners[index]
        value = score(ids)
        if value > best:
            best, best_ids = value, ids
    return decode_ids(best_ids + [EOS_ID], chars)


def preprocess(image, size: int) -> np.ndarray:
    """White-pad to a square, PIL bilinear resize, ImageNet normalize -> (3,S,S)."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as im:
            image = im.convert("RGB")
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., :3].astype(np.uint8), "RGB")
    else:
        image = image.convert("RGB")
    w, h = image.size
    side = max(w, h, 1)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    arr = np.asarray(canvas.resize((size, size), Image.BILINEAR)).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


@dataclass
class RingResult:
    text: str
    main_text: str
    routed: bool


class RingReader:
    def __init__(self, release_dir: str | Path, providers: list[str] | None = None, names: str | Path | None = None,
                 long_threshold: int | None = None, intra_threads: int = 0):
        import onnxruntime as ort

        self.dir = Path(release_dir)
        cfg = json.loads((self.dir / "route.json").read_text(encoding="utf-8"))
        self.cfg = cfg
        self.size = int(cfg.get("image_size", 512))
        self.alpha = float(cfg.get("alpha", 0.4))
        self.tau = float(cfg.get("tau", 4.0))
        self.max_flips = int(cfg.get("max_flips", 3))
        self.threshold = int(cfg.get("long_threshold", 0) if long_threshold is None else long_threshold)
        self.chars = [line[:-1] if line.endswith("\r") else line
                      for line in (self.dir / cfg.get("charset", "charset.txt")).read_text(encoding="utf-8").split("\n")
                      if line not in ("", "\r")]
        names_path = Path(names) if names else (self.dir / cfg["names"] if cfg.get("names") else None)
        self.lm = None
        if names_path is not None:
            self.lm = NameLM([t.strip() for t in names_path.read_text(encoding="utf-8").splitlines() if t.strip()])
        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        providers = providers or ["CPUExecutionProvider"]
        self.main = ort.InferenceSession(str(self.dir / cfg["main"]), sess_options=opts, providers=providers)
        self.long = None
        if cfg.get("long") and self.threshold > 0:
            self.long = ort.InferenceSession(str(self.dir / cfg["long"]), sess_options=opts, providers=providers)

    def logprobs(self, session, batch: np.ndarray) -> np.ndarray:
        return session.run(["logprobs"], {"image": batch.astype(np.float32)})[0]

    def decode(self, lp: np.ndarray) -> str:
        return constrained_text(lp, self.chars, self.lm, self.alpha, self.tau, self.max_flips)

    def read_batch(self, images: list) -> list[RingResult]:
        if not images:
            return []
        batch = np.stack([preprocess(im, self.size) for im in images])
        main = [self.decode(lp) for lp in self.logprobs(self.main, batch)]
        out = [RingResult(t, t, False) for t in main]
        if self.long is not None:
            idx = [i for i, t in enumerate(main) if len(t) >= self.threshold]
            if idx:
                long_lp = self.logprobs(self.long, batch[idx])
                for i, lp in zip(idx, long_lp):
                    out[i] = RingResult(self.decode(lp), main[i], True)
        return out

    def __call__(self, image) -> RingResult:
        return self.read_batch([image])[0]

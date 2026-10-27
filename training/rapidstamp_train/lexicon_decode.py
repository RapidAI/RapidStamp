"""Rescore a greedy ring string with a character model built from seal names.

The synthetic name list does not contain the user gold seals, so this is not
a nearest-name snap. It is a trigram model of those names, plus the
requirement that an organization suffix stays put. Positions inside the
suffix are frozen. Only a few low-margin stem characters are allowed to
flip to the runner-up.
"""

from __future__ import annotations

import math
from collections import Counter

SUFFIXES = (
    "有限责任公司",
    "股份有限公司",
    "集团有限公司",
    "有限公司",
    "发票监制章",
    "人民政府",
    "监督管理局",
    "市场监督管理局",
    "委员会",
    "办事处",
    "办公室",
    "管理局",
    "公安局",
    "检察院",
    "法院",
    "卫生院",
    "事务所",
    "合作社",
    "研究院",
    "研究所",
    "服务中心",
    "交易中心",
    "分公司",
    "支行",
    "分行",
    "大学",
    "学院",
    "学校",
    "医院",
    "银行",
    "中心",
    "集团",
    "政府",
    "幼儿园",
    "经营部",
    "管理处",
    "信用社",
    "公司",
    "局",
    "厅",
    "院",
    "所",
    "站",
    "队",
    "部",
    "社",
    "厂",
    "店",
    "馆",
    "会",
    "署",
    "处",
    "自治区",
    "自治州",
    "街道",
    "省",
    "市",
    "区",
    "县",
    "镇",
    "乡",
    "村",
)


def matching_suffix(text: str) -> str:
    for suffix in sorted(set(SUFFIXES), key=len, reverse=True):
        if len(suffix) < len(text) and text.endswith(suffix):
            return suffix
    return ""


class NameLM:
    """Interpolated character trigram over seal names. Unknown names still share local stats."""

    def __init__(self, names: list[str]):
        self.uni: Counter[str] = Counter()
        self.bi: Counter[tuple[str, str]] = Counter()
        self.tri: Counter[tuple[str, str, str]] = Counter()
        self.total = 0
        self.names = 0
        for name in names:
            if not name:
                continue
            self.names += 1
            self.total += len(name)
            self.bi[("^", "^")] += 1
            previous = ("^", "^")
            for char in name:
                self.uni[char] += 1
                self.bi[(previous[1], char)] += 1
                self.tri[(previous[0], previous[1], char)] += 1
                previous = (previous[1], char)

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


def _decode_ids(ids: list[int], charset) -> str:
    return charset.decode(ids)


def constrained_text(
    logp,
    charset,
    lm: NameLM,
    alpha: float,
    tau: float,
    max_flips: int = 3,
) -> str:
    """`logp` is a sequence of log-probability vectors, one per position.

    alpha 0 leaves the greedy string unchanged, because the positions are
    independent given the first-pass context.
    """
    length = logp.shape[0] if hasattr(logp, "shape") else len(logp)
    greedy_ids = []
    gaps = []
    runners = []
    for index in range(length):
        row = logp[index]
        if hasattr(row, "tolist"):
            # torch or numpy
            order = sorted(range(len(row)), key=lambda item: float(row[item]), reverse=True)
        else:
            order = sorted(range(len(row)), key=lambda item: row[item], reverse=True)
        top = order[0]
        second = order[1]
        greedy_ids.append(top)
        gaps.append(float(row[top]) - float(row[second]))
        runners.append(second)
        if top in (0, 2):  # pad or eos
            break
    # Drop the eos position from the character string. decode() stops at eos.
    if greedy_ids and greedy_ids[-1] in (0, 2):
        char_ids = greedy_ids[:-1]
        char_gaps = gaps[:-1]
        char_runners = runners[:-1]
    else:
        char_ids = list(greedy_ids)
        char_gaps = list(gaps)
        char_runners = list(runners)
    greedy = _decode_ids(char_ids + [2], charset)
    if alpha == 0.0 or not greedy:
        return greedy
    suffix = matching_suffix(greedy)
    stem_end = len(greedy) - len(suffix)
    uncertain = [index for index in range(stem_end) if char_gaps[index] < tau]
    uncertain.sort(key=lambda index: char_gaps[index])
    uncertain = uncertain[:max_flips]
    if not uncertain:
        return greedy

    def score(ids: list[int]) -> float:
        text = _decode_ids(ids + [2], charset)
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

    base_ids = list(char_ids)
    best_ids = base_ids
    best = score(base_ids)
    flips = 1
    for _ in uncertain:
        flips *= 2
    for mask in range(flips):
        ids = list(base_ids)
        for bit, index in enumerate(uncertain):
            if mask & (1 << bit):
                ids[index] = char_runners[index]
        value = score(ids)
        if value > best:
            best = value
            best_ids = ids
    return _decode_ids(best_ids + [2], charset)

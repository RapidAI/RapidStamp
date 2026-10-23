"""Real and generated Chinese seal names, with val and gold texts kept out.

Public company and organization lists supply the frequent names. Generated
place + brand + industry + suffix strings, government agencies, and a
uniform-character pass fill the gaps, including rare charset characters.
Ring text from the train split is added. Val labels and gold ring, center,
and code strings are rejected.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from rapidstamp_train.direct_aug import Charset
from rapidstamp_train.ring_text import is_template

MIN_CHARS = 2
MAX_CHARS = 25

BRAND_CHARS = (
    "华中国泰盛达博雅明德天成瑞丰恒通嘉和正大新远景宏图信安科创联海东方"
    "金源鑫诚业兴隆昌顺康宁宇航鹏龙祥福智云数伟光电力工贸实业"
)
INDUSTRIES = (
    "科技",
    "贸易",
    "建设",
    "工程",
    "信息技术",
    "网络",
    "电子",
    "机械",
    "医药",
    "食品",
    "农业",
    "物流",
    "咨询",
    "文化",
    "教育",
    "环保",
    "能源",
    "装饰",
    "软件",
    "生物",
    "材料",
    "电力",
    "水务",
    "广告",
    "传媒",
    "投资",
    "种子",
    "测绘",
    "勘察",
    "设计",
    "保安",
)
SUFFIXES = (
    "有限公司",
    "有限责任公司",
    "股份有限公司",
    "集团有限公司",
    "事务所",
    "中心",
    "研究院",
    "合作社",
)
AGENCIES = (
    "人民政府",
    "市场监督管理局",
    "公安局",
    "人力资源和社会保障局",
    "卫生健康委员会",
    "应急管理局",
    "住房和城乡建设局",
    "自然资源局",
    "教育局",
    "财政局",
    "农业农村局",
    "发展和改革委员会",
    "生态环境局",
    "交通运输局",
    "民政局",
    "司法局",
    "审计局",
    "统计局",
    "医疗保障局",
    "文化和旅游局",
    "退役军人事务局",
    "水利局",
    "科学技术局",
    "商务局",
    "税务局",
    "消防救援大队",
    "公证处",
)
FALLBACK_PLACES = (
    "北京市",
    "上海市",
    "杭州市",
    "广州市",
    "成都市",
    "武汉市",
    "南京市",
    "西安市",
    "重庆市",
    "天津市",
    "苏州市",
    "青岛市",
    "厦门市",
    "长沙市",
    "郑州市",
    "合肥市",
    "福州市",
    "济南市",
    "大连市",
    "东城区",
    "西湖区",
    "武侯区",
)


def reject_reason(text: str, charset: Charset, blocked: set[str]) -> str | None:
    """Why ``text`` cannot be a training ring string. None means keep it."""
    if not text or len(text) < MIN_CHARS:
        return "short"
    if len(text) > MAX_CHARS:
        return "long"
    if text in blocked:
        return "blocked"
    if is_template(text):
        return "template"
    cjk = 0
    for char in text:
        if char not in charset.stoi:
            return "charset"
        if "\u4e00" <= char <= "\u9fff":
            cjk += 1
    if cjk < 2 or cjk * 2 < len(text):
        return "not_cjk"
    return None


def accept_name(text: str, charset: Charset, blocked: set[str]) -> bool:
    return reject_reason(text, charset, blocked) is None


def clean_line(line: str) -> str:
    text = line.strip().strip("\ufeff")
    if not text or text.startswith("By@") or "NameMoe" in text:
        return ""
    if text[0].isdigit():
        return ""
    return text


def load_blocked(kept: Path | None, manifest: Path | None) -> set[str]:
    """Val ring text, plus every gold ring, center, and code string."""
    blocked: set[str] = set()
    if kept is not None and kept.is_file():
        with kept.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != "val":
                    continue
                text = str(row.get("text") or "").strip()
                if text:
                    blocked.add(text)
    if manifest is not None and manifest.is_file():
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") != "test":
                    continue
                for seal in row.get("seals") or []:
                    for key in ("ring", "text", "center", "code"):
                        text = str(seal.get(key) or "").strip()
                        if text:
                            blocked.add(text)
    blocked.discard("")
    return blocked


def train_ring_texts(kept: Path, charset: Charset, blocked: set[str]) -> list[str]:
    """Train-split ring strings. They are real labels, so the CJK-ratio rule does not apply."""
    texts: list[str] = []
    seen: set[str] = set()
    with kept.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "val":
                continue
            text = str(row.get("text") or "").strip()
            if not text or text in seen or text in blocked or is_template(text):
                continue
            if charset.encode(text) is None:
                continue
            seen.add(text)
            texts.append(text)
    return texts


def flatten_places(tree: list[dict]) -> list[str]:
    """Province, city, and district forms short enough to sit on a seal."""
    places: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if 2 <= len(name) <= 12 and name not in seen and "直辖" not in name and name != "市辖区":
            seen.add(name)
            places.append(name)

    for province in tree:
        pname = str(province.get("name") or "")
        add(pname)
        for city in province.get("children") or []:
            cname = str(city.get("name") or "")
            if cname in {"市辖区", "县", "省直辖县级行政区划", "自治区直辖县级行政区划"}:
                cname = ""
            else:
                add(cname)
                if pname.endswith("市"):
                    add(pname + cname)
            for area in city.get("children") or []:
                aname = str(area.get("name") or "")
                add(aname)
                if cname:
                    add(cname + aname)
                if pname.endswith("市"):
                    add(pname + aname)
    return places


def load_places(path: Path | None) -> list[str]:
    if path is None or not path.is_file():
        return list(FALLBACK_PLACES)
    tree = json.loads(path.read_text(encoding="utf-8"))
    places = flatten_places(tree)
    return places or list(FALLBACK_PLACES)


def _compose_company(rng: random.Random, places: list[str]) -> str:
    parts: list[str] = []
    if rng.random() < 0.75:
        parts.append(rng.choice(places))
    brand_len = 2 if rng.random() < 0.7 else 3
    parts.append("".join(rng.choice(BRAND_CHARS) for _ in range(brand_len)))
    if rng.random() < 0.8:
        parts.append(rng.choice(INDUSTRIES))
    parts.append(rng.choice(SUFFIXES))
    return "".join(parts)


def _compose_agency(rng: random.Random, places: list[str]) -> str:
    return rng.choice(places) + rng.choice(AGENCIES)


def generate_names(
    rng: random.Random,
    places: list[str],
    charset: Charset,
    blocked: set[str],
    companies: int,
    agencies: int,
) -> tuple[list[str], list[str]]:
    """Return ``(company strings, agency strings)`` that pass ``accept_name``."""
    company_rows: list[str] = []
    agency_rows: list[str] = []
    seen: set[str] = set()

    def take(bucket: list[str], text: str, limit: int) -> bool:
        if len(bucket) >= limit or text in seen or not accept_name(text, charset, blocked):
            return False
        seen.add(text)
        bucket.append(text)
        return True

    tries = 0
    limit_tries = (companies + agencies) * 30 + 100
    while (len(company_rows) < companies or len(agency_rows) < agencies) and tries < limit_tries:
        tries += 1
        if len(company_rows) < companies:
            take(company_rows, _compose_company(rng, places), companies)
        if len(agency_rows) < agencies:
            take(agency_rows, _compose_agency(rng, places), agencies)
    return company_rows, agency_rows


def uniform_strings(charset: Charset, blocked: set[str], per_char: int, rng: random.Random) -> list[str]:
    """Several strings per charset character, so rare glyphs are not absent."""
    chars = [char for char in charset.chars if char != " "]
    rows: list[str] = []
    seen: set[str] = set()
    for char in chars:
        made = 0
        tries = 0
        while made < per_char and tries < per_char * 30:
            tries += 1
            length = rng.randint(4, 12)
            pieces = [chars[rng.randrange(len(chars))] for _ in range(length)]
            pieces[rng.randrange(length)] = char
            text = "".join(pieces)
            if text in seen or text in blocked:
                continue
            seen.add(text)
            rows.append(text)
            made += 1
    return rows


def sample_uniform(rng: random.Random, charset: Charset, blocked: set[str]) -> str:
    """A fresh uniform string. Retries if it lands on a blocked label."""
    chars = [char for char in charset.chars if char != " "]
    for _ in range(8):
        length = rng.randint(4, 14)
        text = "".join(chars[rng.randrange(len(chars))] for _ in range(length))
        if text not in blocked:
            return text
    return "公司"


def reservoir(stream, limit: int, rng: random.Random) -> tuple[list[str], int]:
    """Uniform sample of at most ``limit`` accepted strings."""
    chosen: list[str] = []
    seen = 0
    for text in stream:
        seen += 1
        if seen % 200_000 == 0:
            print(f"reservoir {seen} kept {min(len(chosen), limit)}", flush=True)
        if len(chosen) < limit:
            chosen.append(text)
            continue
        slot = rng.randrange(seen)
        if slot < limit:
            chosen[slot] = text
    return chosen, seen


def iter_accepted(path: Path, charset: Charset, blocked: set[str], stats: Counter):
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = clean_line(line)
            if not text:
                stats["blank"] += 1
                continue
            reason = reject_reason(text, charset, blocked)
            if reason is None:
                stats["ok"] += 1
                yield text
            else:
                stats[reason] += 1


def dedupe(rows: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for text in rows:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out

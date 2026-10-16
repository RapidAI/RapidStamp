"""Closed vocabulary for synthetic Chinese and foreign seals.

Every character the renderer can emit is listed here, so the recognizer's
charset stays aligned with the data. This is an original component list, not
a copy of an external name corpus.
"""

from __future__ import annotations

import numpy as np

DIGITS = "0123456789"
LATIN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# Latin-1 letters used by a few European stamp phrases. DejaVu covers these.
LATIN_EXTRA = "ÁÉÍÓÚÄÖÜÑÀÈÙÂÊÔÃÕÇáéíóúäöüñàèùâêôãõçÜü"
PUNCT = " -./&'"

CITIES = [
    "北京",
    "上海",
    "广州",
    "深圳",
    "杭州",
    "成都",
    "武汉",
    "南京",
    "西安",
    "重庆",
    "天津",
    "苏州",
    "青岛",
    "厦门",
    "长沙",
    "郑州",
    "合肥",
    "福州",
    "济南",
    "大连",
]
MIDDLES = [
    "华远",
    "中信",
    "国泰",
    "盛达",
    "博雅",
    "明德",
    "天成",
    "瑞丰",
    "恒通",
    "嘉和",
    "正大",
    "新华",
    "远景",
    "宏图",
    "德信",
    "安泰",
    "科创",
    "联创",
    "海川",
    "东方",
]
COMPANY_SUFFIX = ["有限公司", "股份有限公司", "集团有限公司"]
BOTTOM_PHRASES = ["财务专用章", "发票专用章", "合同专用章", "业务专用章", "报关专用章"]
SURNAMES = list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜")
GIVEN = list("伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀英")
NOTARY_MARK = "公证处"

EN_FIRST = [
    "ACME",
    "GLOBAL",
    "NORTH",
    "PACIFIC",
    "UNION",
    "APEX",
    "ORION",
    "HELIX",
    "SUMMIT",
    "BRIDGE",
    "ATLAS",
    "NOVA",
    "HARBOR",
    "SILVER",
]
EN_SECOND = ["TRADE", "WORKS", "GROUP", "LOGIC", "METAL", "FOODS", "MEDIA", "LABS"]
EN_SUFFIX = ["CORP", "LTD", "LLC", "INC", "GMBH", "SAS", "BV", "PLC"]
RECT_WORDS = [
    "APPROVED",
    "PAID",
    "RECEIVED",
    "CONFIDENTIAL",
    "CERTIFIED COPY",
    "ORIGINAL",
    "URGENT",
    "DRAFT",
    "SAMPLE",
    "GEPRÜFT",
    "ESPAÑA",
    "MÜNCHEN",
]
NOTARY_EN = ["NOTARY PUBLIC", "NOTARY", "OFFICIAL SEAL", "STATE NOTARY"]


def _pick(rng: np.random.Generator, items: list[str]) -> str:
    return items[int(rng.integers(0, len(items)))]


def chinese_company(rng: np.random.Generator) -> str:
    return _pick(rng, CITIES) + _pick(rng, MIDDLES) + _pick(rng, COMPANY_SUFFIX)


def latin_company(rng: np.random.Generator) -> str:
    name = _pick(rng, EN_FIRST)
    if rng.random() < 0.65:
        name += " " + _pick(rng, EN_SECOND)
    name += " " + _pick(rng, EN_SUFFIX)
    return name


def personal_name(rng: np.random.Generator) -> tuple[str, int, int]:
    """Return text, rows, cols for a square personal seal."""
    kind = int(rng.integers(0, 3))
    if kind == 0:
        text = _pick(rng, SURNAMES) + _pick(rng, GIVEN)
        return text, 2, 1
    if kind == 1:
        text = _pick(rng, SURNAMES) + _pick(rng, GIVEN) + _pick(rng, GIVEN)
        return text, 3, 1
    text = "".join(_pick(rng, SURNAMES) + _pick(rng, GIVEN) for _ in range(2))
    return text[:4], 2, 2


def tax_id(rng: np.random.Generator) -> str:
    body = "".join(str(int(rng.integers(0, 10))) for _ in range(16))
    return "91" + body


def seal_code(rng: np.random.Generator) -> str:
    return "".join(str(int(rng.integers(0, 10))) for _ in range(13))


def chinese_notary(rng: np.random.Generator) -> str:
    return _pick(rng, CITIES) + "市" + NOTARY_MARK


def all_chars() -> list[str]:
    """Stable charset, blank excluded. Index 0 is reserved for CTC blank."""
    blobs = [
        DIGITS,
        LATIN,
        LATIN_EXTRA,
        PUNCT,
        "".join(CITIES),
        "".join(MIDDLES),
        "".join(COMPANY_SUFFIX),
        "".join(BOTTOM_PHRASES),
        "".join(SURNAMES),
        "".join(GIVEN),
        NOTARY_MARK,
        "市",
        "".join(EN_FIRST),
        "".join(EN_SECOND),
        "".join(EN_SUFFIX),
        "".join(RECT_WORDS),
        "".join(NOTARY_EN),
    ]
    seen: set[str] = set()
    chars: list[str] = []
    for blob in blobs:
        for ch in blob:
            if ch not in seen:
                seen.add(ch)
                chars.append(ch)
    return chars


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)

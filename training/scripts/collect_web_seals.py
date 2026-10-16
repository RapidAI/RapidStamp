"""Collect real seal photos from image search for train/eval.

Images stay on the training machine. This script writes a URL manifest and
the picture files under ``--out``. Do not commit the pictures.

Bing, Baidu, and 360 image search are all queried. Google image search
redirects and is not scraped. From this network Baidu rejects spiders.
Bing returns hits for some English queries and an empty page for the
Chinese ones. 360 (image.so.com) is the source of most kept photos.
Each kept row records the engine, the query, the image URL, and the page URL.

    python training/scripts/collect_web_seals.py \
        --out /home/mayong/datasets/web_seals
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

QUERIES = [
    "公章 合同 扫描件",
    "合同上的公章 红章",
    "财务专用章 文件",
    "发票专用章 扫描件",
    "合同专用章 盖章",
    "私章 人名章 文件",
    "法人章 合同",
    "公司公章 红章 文档",
    "company stamp on contract",
    "corporate seal stamped document",
    "notary seal on certificate",
    "company chop on invoice",
    "official round stamp on document",
]

# Tutorial, clip-art, and logo pages are not stamps on paper.
_SKIP_TITLE = (
    "photoshop",
    " ps",
    "ps ",
    "抠图",
    "抠出",
    "教程",
    "如何",
    "怎么",
    "模板",
    "免费下载",
    "素材",
    "图标",
    "logo",
    "矢量",
    "png免抠",
    "刷单",
    "被骗",
    "兼职",
)


def title_ok(title: str) -> bool:
    folded = f" {title.casefold()} "
    if any(token in folded for token in _SKIP_TITLE):
        return False
    return re.search(r"(^|[^a-z])ps([^a-z]|$)", folded) is None


def _get(url: str, referer: str, timeout: float = 20.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": referer})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def search_so(query: str, start: int, count: int) -> list[dict]:
    url = "https://image.so.com/j?" + urllib.parse.urlencode({"q": query, "sn": start, "pn": count})
    payload = json.loads(_get(url, "https://image.so.com/").decode("utf-8", "replace"))
    rows = []
    for item in payload.get("list") or []:
        if not isinstance(item, dict):
            continue
        image = item.get("img") or item.get("thumb") or ""
        if image.startswith("//"):
            image = "https:" + image
        if not image or image.lower().endswith(".gif") or str(item.get("imgtype") or "").lower() == "gif":
            continue
        rows.append(
            {
                "engine": "360",
                "query": query,
                "title": str(item.get("title") or ""),
                "image_url": image,
                "page_url": str(item.get("link") or ""),
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
                "group": str(item.get("grpmd5") or ""),
            }
        )
    return rows


def search_bing(query: str) -> list[dict]:
    url = "https://www.bing.com/images/search?" + urllib.parse.urlencode(
        {"q": query, "form": "HDRSC2", "first": 1}
    )
    text = _get(url, "https://www.bing.com/").decode("utf-8", "replace")
    rows = []
    for raw in re.findall(r'\sm="(\{.*?\})"', text):
        try:
            item = json.loads(html.unescape(raw))
        except json.JSONDecodeError:
            continue
        image = item.get("murl") or ""
        if not image or image.lower().endswith(".gif"):
            continue
        rows.append(
            {
                "engine": "bing",
                "query": query,
                "title": str(item.get("t") or ""),
                "image_url": image,
                "page_url": str(item.get("purl") or ""),
                "width": int(item.get("mw") or 0),
                "height": int(item.get("mh") or 0),
                "group": str(item.get("md5") or ""),
            }
        )
    # A blocked Bing page returns one unrelated hit. Treat that as no results.
    if len(rows) < 4:
        return []
    return rows


def search_baidu(query: str) -> list[dict]:
    url = "https://image.baidu.com/search/acjson?" + urllib.parse.urlencode(
        {
            "tn": "resultjson_com",
            "ipn": "rj",
            "ct": "201326592",
            "fp": "result",
            "ie": "utf-8",
            "word": query,
            "queryWord": query,
            "pn": 0,
            "rn": 30,
        }
    )
    payload = json.loads(_get(url, "https://image.baidu.com/").decode("utf-8", "replace"))
    if payload.get("antiFlag") or payload.get("message"):
        return []
    rows = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        image = item.get("hoverURL") or item.get("middleURL") or item.get("thumbURL") or ""
        if not image:
            continue
        rows.append(
            {
                "engine": "baidu",
                "query": query,
                "title": str(item.get("fromPageTitle") or ""),
                "image_url": image,
                "page_url": str(item.get("fromURL") or ""),
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
                "group": "",
            }
        )
    return rows


def _ext_for(blob: bytes) -> str | None:
    if blob.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if blob.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if blob.startswith(b"RIFF") and blob[8:12] == b"WEBP":
        return ".webp"
    return None


def download_image(url: str) -> bytes | None:
    try:
        blob = _get(url, "https://image.so.com/", timeout=25)
    except (urllib.error.URLError, TimeoutError, ValueError, http.client.HTTPException, OSError):
        return None
    if len(blob) < 12_000 or len(blob) > 8_000_000:
        return None
    if _ext_for(blob) is None:
        return None
    return blob


def collect(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    seen_hash: set[str] = set()
    seen_group: set[str] = set()
    existing = out / "sources.jsonl"
    if args.fresh and existing.is_file():
        existing.unlink()
    if existing.is_file() and not args.fresh:
        for line in existing.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen_hash.add(row["sha256"])
            if row.get("group"):
                seen_group.add(row["group"])
    kept = 0
    engines = {"360": search_so, "bing": search_bing, "baidu": search_baidu}
    new_rows: list[dict] = []
    queries = args.only or QUERIES
    for query in queries:
        if kept >= args.limit:
            break
        for engine, search in engines.items():
            if kept >= args.limit:
                break
            try:
                if engine == "360":
                    hits = []
                    for start in (0, args.per_query):
                        hits.extend(search_so(query, start, args.per_query))
                        time.sleep(args.pause)
                else:
                    hits = search(query)
                    time.sleep(args.pause)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, http.client.HTTPException, OSError) as exc:
                print(f"{engine} {query!r} failed: {exc}", flush=True)
                continue
            print(f"{engine} {query!r} hits {len(hits)}", flush=True)
            for hit in hits:
                if kept >= args.limit:
                    break
                if not title_ok(hit["title"]):
                    continue
                if hit["width"] and hit["height"] and min(hit["width"], hit["height"]) < 160:
                    continue
                if hit["group"] and hit["group"] in seen_group:
                    continue
                blob = download_image(hit["image_url"])
                if blob is None:
                    continue
                digest = hashlib.sha256(blob).hexdigest()
                if digest in seen_hash:
                    continue
                seen_hash.add(digest)
                if hit["group"]:
                    seen_group.add(hit["group"])
                name = digest[:20] + _ext_for(blob)
                (image_dir / name).write_bytes(blob)
                row = {
                    "sha256": digest,
                    "file": f"images/{name}",
                    "engine": hit["engine"],
                    "query": hit["query"],
                    "title": hit["title"],
                    "image_url": hit["image_url"],
                    "page_url": hit["page_url"],
                    "width": hit["width"],
                    "height": hit["height"],
                    "group": hit["group"],
                    "bytes": len(blob),
                    "use": "train_eval_only",
                }
                new_rows.append(row)
                with (out / "sources.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
                if kept % 20 == 0:
                    print(f"kept {kept}", flush=True)
    text = (out / "sources.jsonl").read_text(encoding="utf-8") if (out / "sources.jsonl").is_file() else ""
    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(text, encoding="utf-8")
    all_rows = [line for line in text.splitlines() if line.strip()]
    summary = {"kept_new": kept, "total": len(all_rows), "out": str(out)}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def annotate(args: argparse.Namespace) -> None:
    """CPU pre-labels. One thread so a GPU training job on the same machine keeps its cores."""
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "python" / "src"))
    from rapidstamp.pipeline import RapidStamp

    engine = RapidStamp(args.model_dir)
    out = Path(args.out)
    rows = [json.loads(line) for line in (out / "sources.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    labeled = 0
    for row in rows:
        path = out / row["file"]
        if not path.is_file():
            row["model_seals"] = 0
            continue
        try:
            seals = engine(path)
        except Exception as exc:  # noqa: BLE001 - record the failure and keep going
            row["model_error"] = str(exc)
            row["model_seals"] = 0
            continue
        row["model_seals"] = len(seals)
        row["model_text"] = [seal.text for seal in seals]
        row["model_class"] = [seal.class_name for seal in seals]
        labeled += int(bool(seals))
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (out / "sources.jsonl").write_text(text, encoding="utf-8")
    if args.manifest:
        Path(args.manifest).write_text(text, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "with_seal": labeled}, ensure_ascii=False), flush=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="Copy of sources.jsonl safe to commit (URLs only)")
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument("--limit", type=int, default=280)
    parser.add_argument("--pause", type=float, default=0.6)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these queries")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.annotate:
        if args.model_dir is None:
            raise SystemExit("--annotate needs --model-dir")
        annotate(args)
        return
    collect(args)


if __name__ == "__main__":
    main()

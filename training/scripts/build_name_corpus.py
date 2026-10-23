"""Build the seal-name corpus under the data disk. Nothing here is committed.

    python training/scripts/build_name_corpus.py \
        --dict /home/rapidstamp_data/pretrained/ppocr_keys_v1.txt \
        --kept /home/rapidstamp_data/ring_v2/kept.jsonl \
        --manifest training/manifests/user_seals_split.jsonl \
        --company /home/rapidstamp_data/synth/src/company/Company-Names-Corpus（480W）.txt \
        --org /home/rapidstamp_data/synth/src/org/Organization-Names-Corpus（110W）.txt \
        --brands /home/rapidstamp_data/synth/src/shorter.txt \
        --places /home/rapidstamp_data/synth/src/pca.json \
        --out /home/rapidstamp_data/synth/names.txt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "training"))

from rapidstamp_train.direct_aug import load_charset  # noqa: E402
from rapidstamp_train.name_corpus import (  # noqa: E402
    dedupe,
    generate_names,
    iter_accepted,
    load_blocked,
    load_places,
    reservoir,
    train_ring_texts,
    uniform_strings,
)


def _sample(path: Path, limit: int, charset, blocked, rng: random.Random) -> tuple[list[str], dict]:
    stats: Counter = Counter()
    chosen, seen = reservoir(iter_accepted(path, charset, blocked, stats), limit, rng)
    return chosen, {"accepted": seen, "kept": len(chosen), **stats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dict", type=Path, required=True)
    parser.add_argument("--kept", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--company", type=Path, required=True)
    parser.add_argument("--org", type=Path, required=True)
    parser.add_argument("--brands", type=Path, required=True)
    parser.add_argument("--places", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--companies", type=int, default=1_000_000)
    parser.add_argument("--orgs", type=int, default=250_000)
    parser.add_argument("--brand-names", type=int, default=80_000)
    parser.add_argument("--gen-companies", type=int, default=120_000)
    parser.add_argument("--gen-agencies", type=int, default=80_000)
    parser.add_argument("--per-char", type=int, default=6)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    charset = load_charset(args.dict)
    blocked = load_blocked(args.kept, args.manifest)
    places = load_places(args.places)
    print(json.dumps({"charset": len(charset), "blocked": len(blocked), "places": len(places)}), flush=True)

    company, company_stats = _sample(args.company, args.companies, charset, blocked, rng)
    print(json.dumps({"source": "company", **company_stats}, ensure_ascii=False), flush=True)
    org, org_stats = _sample(args.org, args.orgs, charset, blocked, rng)
    print(json.dumps({"source": "org", **org_stats}, ensure_ascii=False), flush=True)
    brands, brand_stats = _sample(args.brands, args.brand_names, charset, blocked, rng)
    print(json.dumps({"source": "brands", **brand_stats}, ensure_ascii=False), flush=True)
    generated, agencies = generate_names(rng, places, charset, blocked, args.gen_companies, args.gen_agencies)
    train_rows = train_ring_texts(args.kept, charset, blocked)
    uniform = uniform_strings(charset, blocked, args.per_char, rng)

    names = dedupe(company + org + brands + generated + agencies + train_rows)
    leaked = [text for text in names if text in blocked]
    leaked_uniform = [text for text in uniform if text in blocked]
    if leaked or leaked_uniform:
        raise SystemExit(f"blocked text leaked names={len(leaked)} uniform={len(leaked_uniform)}")
    if len(names) < 1_000_000:
        raise SystemExit(f"only {len(names)} unique names")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    name_path = args.out
    uniform_path = args.out.with_name("uniform.txt")
    blocked_path = args.out.with_name("blocked.txt")
    stats_path = args.out.with_name("stats.json")
    name_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    uniform_path.write_text("\n".join(uniform) + "\n", encoding="utf-8")
    blocked_path.write_text("\n".join(sorted(blocked)) + "\n", encoding="utf-8")
    stats = {
        "unique_names": len(names),
        "uniform": len(uniform),
        "blocked": len(blocked),
        "places": len(places),
        "company": company_stats,
        "org": org_stats,
        "brands": brand_stats,
        "generated_companies": len(generated),
        "generated_agencies": len(agencies),
        "train_rings": len(train_rows),
        "charset_chars": len(charset.chars),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

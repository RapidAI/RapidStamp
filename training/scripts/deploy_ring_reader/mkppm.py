import json, os, sys
from PIL import Image
which=sys.argv[1]
src={"gold":"/home/rapidstamp_data/gold_v2/labels.jsonl","csd":"/home/rapidstamp_data/external/csd/preds_csd_cand_20261031.jsonl"}[which]
d=f"/tmp/ppm_{which}"; os.makedirs(d,exist_ok=True)
with open(f"{which}.tsv","w") as f:
    for l in open(src,encoding="utf-8"):
        r=json.loads(l); p=f"{d}/{r['id']}.ppm"
        if not os.path.exists(p): Image.open(r["image"]).convert("RGB").save(p)
        f.write(f"{r['id']}\t{p}\n")

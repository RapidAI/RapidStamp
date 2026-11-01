"""End-to-end eval of a deployed ring_reader release (Python runtime). Writes preds + summary."""
import json, sys, time
from pathlib import Path
D=Path("/home/rapidstamp_data"); E=D/"deploy/ring_reader/eval"
sys.path.insert(0,str(D/"src/RapidStamp/training"))
from rapidstamp.ring_reader import RingReader
from rapidstamp_train.ring_text import one_ned
rel=Path(sys.argv[1]); tag=sys.argv[2]; which=sys.argv[3] if len(sys.argv)>3 else "gold,csd"
def groups(r):
    if r["multi"]: return []
    g=["single227"]
    if r["bucket"]=="orig63": g.append("orig63")
    elif r["split_in_pool"]!="train": g.append("unseen59")
    if r["bucket"]=="long_name": g.append("long45")
    return g
rep={}; preds=[]
def run(rr, rows, gfun, key):
    acc={}; t0=time.time(); routed=0
    for i in range(0,len(rows),16):
        batch=rows[i:i+16]
        for r,res in zip(batch, rr.read_batch([r["image"] for r in batch])):
            routed+=res.routed
            preds.append({"set":key,"id":r["id"],"gold":r["text"],"pred":res.text,"main":res.main_text,"routed":res.routed})
            for g in gfun(r):
                a=acc.setdefault(g,[0,0,0.0]); a[0]+=1; a[1]+=res.text==r["text"]; a[2]+=one_ned(res.text,r["text"])
    for g,a in acc.items(): rep[g]={"n":a[0],"exact":a[1],"1ned":round(a[2]/a[0],5)}
    rep[key+"_routed"]=routed; rep[key+"_sec_per_img"]=round((time.time()-t0)/len(rows),4)
    print(tag,key,json.dumps(rep,ensure_ascii=False),flush=True)
if "gold" in which:
    rows=[json.loads(l) for l in open(D/"gold_v2/labels.jsonl",encoding="utf-8")]
    run(RingReader(rel), rows, groups, "gold")
if "csd" in which:
    rows=[json.loads(l) for l in open(D/"external/csd/preds_csd_cand_20261031.jsonl",encoding="utf-8")]
    run(RingReader(rel, names=E/"names_lm_csd_eval.txt"), rows, lambda r:["csd5335"], "csd")
(E/f"e2e_{tag}.json").write_text(json.dumps(rep,ensure_ascii=False,indent=1),encoding="utf-8")
with open(E/f"e2e_preds_{tag}.jsonl","w",encoding="utf-8") as f:
    for p in preds: f.write(json.dumps(p,ensure_ascii=False)+"\n")

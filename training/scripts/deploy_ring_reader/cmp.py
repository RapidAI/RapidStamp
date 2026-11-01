import json, sys, statistics as st
E="/home/rapidstamp_data/deploy/ring_reader/eval/"
def cpp(p): return {a[0]:(a[1],int(a[2]),float(a[3])) for a in (l.rstrip("\n").split("\t") for l in open(p,encoding="utf-8"))}
out={}
for tag,cf,pf,sett in [x.split(",") for x in sys.argv[1:]]:
    c=cpp(cf); py={(j:=json.loads(l))["id"]:j for l in open(E+pf,encoding="utf-8") if json.loads(l)["set"]==sett}
    eq=sum(c[k][0]==py[k]["pred"] for k in py); ex=sum(c[k][0]==py[k]["gold"] for k in py)
    ms=[v[2] for v in c.values()]
    out[tag]={"n":len(py),"cpp_eq_python":eq,"cpp_exact_vs_gold":ex,"routed":sum(v[1] for v in c.values()),"ms_mean":round(st.mean(ms),1),"ms_p50":round(st.median(ms),1),"ms_p95":round(sorted(ms)[int(.95*len(ms))],1),
      "diff":[(k,c[k][0],py[k]["pred"]) for k in py if c[k][0]!=py[k]["pred"]][:5]}
    print(tag,json.dumps(out[tag],ensure_ascii=False))

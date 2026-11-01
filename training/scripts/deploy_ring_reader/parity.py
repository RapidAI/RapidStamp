"""PyTorch vs deployed ONNX parity on gold v2 (logprobs, greedy and lex strings)."""
import json, sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
D=Path("/home/rapidstamp_data"); REL=D/"deploy/ring_reader/current"
sys.path.insert(0,str(D/"src/RapidStamp/training/scripts"))
from export_ring_reader import load
from rapidstamp.ring_reader import RingReader, preprocess
from rapidstamp_train.direct_aug import load_charset
from rapidstamp_train.lexicon_decode import NameLM, constrained_text
from train_parseq import to_tensor
from PIL import Image
rows=[json.loads(l) for l in open(D/"gold_v2/labels.jsonl",encoding="utf-8")]
rr=RingReader(REL, long_threshold=0)
import onnxruntime as ort
cs=load_charset(D/"pretrained/ppocr_keys_v1.txt")
blocked={t.split("#")[0].strip() for r in rows for t in r["texts"]}
lm=NameLM([t.strip() for t in open(D/"synth_names/names_for_vl1.txt",encoding="utf-8") if t.strip() and t.strip() not in blocked])
out={}
for tag,ck,onnx in (("A","parseq_reader_large_clean_csd","reader_main.onnx"),("prod","parseq_reader_large","reader_long.onnx")):
    model,size=load(D/f"checkpoints/{ck}/finetune/best.pt",D/"pretrained/ppocr_keys_v1.txt"); model=model.cuda()
    sess=ort.InferenceSession(str(REL/onnx),providers=["CPUExecutionProvider"])
    maxd=0.0; g_eq=l_eq=pre_eq=0; lex_t=[]; lex_o=[]
    for r in rows:
        with Image.open(r["image"]) as im: t=to_tensor(im,size)
        x=preprocess(r["image"],size); pre_eq+=np.allclose(t.numpy(),x,atol=1e-6)
        with torch.inference_mode():
            mem=model.encode(t[None].cuda()); blank=torch.zeros(1,model.max_len,dtype=torch.long,device="cuda")
            lp_t=F.log_softmax(model.decode(mem,model.decode(mem,blank).argmax(-1)).float(),-1)[0].cpu()
        lp_o=sess.run(["logprobs"],{"image":x[None]})[0][0]
        maxd=max(maxd,float(np.abs(lp_t.numpy()-lp_o).max()))
        g_eq+=constrained_text(lp_t,cs,lm,0.0,4.0)==constrained_text(lp_o,cs,lm,0.0,4.0)
        a=constrained_text(lp_t,cs,lm,0.4,4.0); b=rr.decode(lp_o); l_eq+=a==b; lex_t.append(a); lex_o.append(b)
    out[tag]={"n":len(rows),"preprocess_equal":pre_eq,"max_abs_logprob_diff":maxd,"greedy_equal":g_eq,"lex_equal_torch_vs_deployed_decoder":l_eq}
    print(tag,out[tag],flush=True)
    del model; torch.cuda.empty_cache()
(D/"deploy/ring_reader/eval/parity_20261101.json").write_text(json.dumps(out,indent=1))

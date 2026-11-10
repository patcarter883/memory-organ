# Pilot of experiment G (row statistics with no model forward) on the REAL Qwen3.8-Flash-Next table, CPU only.
# Rows/embeddings fetched by HTTP range from the public bf16 checkpoint; W_v from the earlier range fetch (P/t1/fetch_real_ple.py).
import sys, os, json, struct, random, glob, ssl, time, urllib.request, collections, concurrent.futures as cf
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
sys.path.insert(0,"/home/user/memory-organ/ngram-memory/proofs/t1/src")
import torch; torch.set_num_threads(1)
from tokenizers import Tokenizer
from ngram_format import NGramTableFormat
random.seed(0); torch.manual_seed(0)
OUT=os.path.dirname(os.path.abspath(__file__))
fmt=NGramTableFormat(vocab_size=248320, eos_token_id=248044)
EOS=248044; V=248320
assert fmt.mult==[23703573157769, 20109073645365, 8052911324071]
tok=Tokenizer.from_file(f"{P}/nm/graft/tok/Qwen__Qwen3.8-Flash-Next/tokenizer.json")
# ---------- corpora (local, read-only) ----------
def load(paths, cap):
    seqs=[]; n=0
    for p in paths:
        try: s=open(p,errors="ignore").read()
        except Exception: continue
        ids=tok.encode(s).ids; seqs.append(ids); n+=len(ids)
        if n>=cap: break
    return seqs, n
code_paths=sorted(glob.glob("/home/user/patcarter883/minisglang-rdna4/python/minisgl/**/*.py",recursive=True))
random.shuffle(code_paths)
prose_paths=sorted(glob.glob("/home/user/patcarter883/minisglang-rdna4/docs/**/*.md",recursive=True)+glob.glob("/home/user/memory-organ/docs/**/*.md",recursive=True))
random.shuffle(prose_paths)
code,nc=load(code_paths,400_000); prose,npr=load(prose_paths,400_000)
print("corpus tokens code",nc,"prose",npr)
def grams(seqs,n):
    c=collections.Counter(); nxt=collections.defaultdict(collections.Counter)
    for ids in seqs:
        h=[EOS]*(n-1)+ids   # segment start = eos history (q4:1107-1123)
        for i in range(n-1,len(h)):
            g=tuple(h[i-n+1:i+1]); c[g]+=1
            if i+1<len(h): nxt[g][h[i+1]]+=1
    return c,nxt
c2,n2=grams(code,2); p2,pn2=grams(prose,2); c3,n3=grams(code,3)
def pick(c,lo,hi,k,excl_eos=True):
    pool=[g for g,v in c.items() if lo<=v<=hi and not (excl_eos and EOS in g)]
    return random.sample(pool,min(k,len(pool)))
top=lambda c,k:[g for g,_ in c.most_common(k+10) if EOS not in g][:k]
cats={"code2_top":top(c2,48),"code2_mid":pick(c2,5,20,48),"code2_once":pick(c2,1,1,48),
      "prose2_top":top(p2,32),"prose2_once":pick(p2,1,1,32),
      "code3_top":top(c3,32),"code3_once":pick(c3,1,1,32)}
# ---------- row ids (independent port; I4-checked earlier) ----------
def rows_of(g):
    n=len(g); x=list(reversed(g))           # x0 = current token, x1 = previous, ...
    mix=0
    for i in range(n): mix ^= (x[i]*fmt.mult[i]) & ((1<<64)-1)
    # torch int64 semantics: products are < 2^63 by construction, xor of non-negatives stays non-negative
    heads=range(0,8) if n==2 else range(8,16)
    return [fmt.offsets[h] + (mix % fmt.sizes[h]) for h in heads]
# ---------- HTTP range reader ----------
ctx=ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")
op=urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ctx))
URL="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
_res={}; _hdr={}
def rng(url,a,b):
    for t in range(4):
        try: return op.open(urllib.request.Request(url,headers={"Range":f"bytes={a}-{b}"}),timeout=60).read()
        except Exception as e: err=e; time.sleep(1+t)
    raise err
def header(f):
    if f not in _hdr:
        r=op.open(urllib.request.Request(URL+f,headers={"Range":"bytes=0-7"}),timeout=60); _res[f]=r.geturl()
        hl=struct.unpack("<Q",r.read())[0]; _hdr[f]=(hl,json.loads(rng(_res[f],8,8+hl-1)))
    return _hdr[f]
ple=json.load(open(f"{P}/nm/t1/hdrs/ple_tensors.json")); pre="model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
RPS=2500012
def row_loc(r):
    s,l=divmod(r,RPS); v=ple[pre+f"shard_{s}.weight"]; hl,_=header(v["file"])
    a=8+hl+v["data_offsets"][0]+l*320; return _res[v["file"]],a,a+319
def fetch_rows(ids,W=48):
    locs=[row_loc(r) for r in ids]
    t=time.time()
    with cf.ThreadPoolExecutor(W) as ex: raw=list(ex.map(lambda L:rng(*L),locs))
    dt=time.time()-t
    return torch.stack([torch.frombuffer(bytearray(b),dtype=torch.bfloat16).float() for b in raw]),dt
def fetch_block(r0,cnt):
    url,a,_=row_loc(r0); assert divmod(r0,RPS)[0]==divmod(r0+cnt-1,RPS)[0]
    return torch.frombuffer(bytearray(rng(url,a,a+cnt*320-1)),dtype=torch.bfloat16).float().view(cnt,160)
def fetch_vocab_rows(name,file,ids,W=48):
    hl,h=header(file); off=h[name]["data_offsets"][0]
    with cf.ThreadPoolExecutor(W) as ex:
        raw=list(ex.map(lambda i:rng(_res[file],8+hl+off+i*5120,8+hl+off+i*5120+5119),ids))
    return torch.stack([torch.frombuffer(bytearray(b),dtype=torch.bfloat16).float() for b in raw])
# ---------- fetch ----------
allg=[g for v in cats.values() for g in v]; ids=[r for g in allg for r in rows_of(g)]
R,dt=fetch_rows(ids); print(f"fetched {len(ids)} scattered rows in {dt:.1f}s ({len(ids)/dt:.0f} rows/s)")
G={}; k=0
for g in allg: G[g]=R[k:k+8]; k+=8
blocks=[]; t=time.time()
for h in range(16):
    r0=fmt.offsets[h]+7_000_000; r0=r0 - (r0 % RPS) + 100 if (r0 % RPS)+1024>RPS else r0
    blocks.append(fetch_block(r0,1024))
B=torch.stack(blocks)  # [16,1024,160] ~ rows of random buckets
print(f"fetched 16x1024 contiguous rows in {time.time()-t:.1f}s")
w=torch.load(f"{P}/nm/t1/real/real_ple_block.pt"); Wv=w["value_proj.weight"].float()   # [2560 out, 2560 in]
Wh=[Wv[:,h*160:(h+1)*160] for h in range(16)]
mu=B.mean(1)  # per-head mean row
print("per-head mean-row norm / mean row norm:", [round((mu[h].norm()/B[h].norm(dim=1).mean()).item(),3) for h in (0,7,8,15)])
def contrib(rows,heads,center=True): return torch.stack([Wh[h]@((rows[i]-mu[h]) if center else rows[i]) for i,h in enumerate(heads)])
def coh(C):
    Cn=C/C.norm(dim=1,keepdim=True); S=Cn@Cn.T; m=len(C); return ((S.sum()-m)/(m*(m-1))).item()
res={}
for name,gs in cats.items():
    heads=list(range(0,8)) if len(gs[0])==2 else list(range(8,16))
    norms=[G[g].norm(dim=1).mean().item() for g in gs]
    co=[coh(contrib(G[g],heads)) for g in gs]
    vnorm=[contrib(G[g],heads).sum(0).norm().item() for g in gs]
    res[name]=dict(n=len(gs),row_norm=sum(norms)/len(norms),coherence=sum(co)/len(co),
                   coh_q=[round(x,3) for x in torch.tensor(co).quantile(torch.tensor([.1,.5,.9])).tolist()],
                   centred_v_norm=sum(vnorm)/len(vnorm))
for o,heads in (("null2",list(range(0,8))),("null3",list(range(8,16)))):
    co=[];vn=[];nm=[]
    for i in range(200):
        rows=torch.stack([B[h][random.randrange(1024)] for h in heads]); C=contrib(rows,heads)
        co.append(coh(C)); vn.append(C.sum(0).norm().item()); nm.append(rows.norm(dim=1).mean().item())
    res[o]=dict(n=200,row_norm=sum(nm)/200,coherence=sum(co)/200,coh_q=[round(x,3) for x in torch.tensor(co).quantile(torch.tensor([.1,.5,.9])).tolist()],centred_v_norm=sum(vn)/200)
print("\n== per category: mean row L2, cross-head coherence (mean pairwise cos of centred W_v^h e_h over the n-gram's 8 heads), |centred v|")
for k_,v in res.items(): print(f"{k_:12s} n={v['n']:3d} row_norm={v['row_norm']:.4f} coherence={v['coherence']:+.3f} q10/50/90={v['coh_q']} |v_c|={v['centred_v_norm']:.4f}")
# ---------- lens: does the n-gram's value point at its own tokens / its continuation? ----------
lens_g=[(g,nx) for name in ("code2_top","code2_mid","prose2_top","code3_top") for g in cats[name]
        for nx in [ (n2 if len(g)==2 and name.startswith("code") else pn2 if len(g)==2 else n3)[g] ] if sum(nx.values())>=3]
lens=[(g,nx.most_common(1)[0][0],nx.most_common(1)[0][1]/sum(nx.values())) for g,nx in lens_g]
lens=[(g,y,p) for g,y,p in lens if p>=0.3]
toks=sorted({t for g,y,_ in lens for t in (g[-1],g[0],y)}); rnd=random.sample(range(V),512)
allt=sorted(set(toks)|set(rnd)); ix={t:i for i,t in enumerate(allt)}
t=time.time()
E=fetch_vocab_rows("model.language_model.embed_tokens.weight","model-00130-of-00131.safetensors",allt)
U=fetch_vocab_rows("lm_head.weight","model-00131-of-00131.safetensors",allt)
print(f"\nfetched {2*len(allt)} embed/lm_head rows in {time.time()-t:.1f}s")
rI=torch.tensor([ix[t] for t in rnd])
def z(v,M,t):
    cs=torch.nn.functional.cosine_similarity(v[None],M,dim=1); return ((cs[ix[t]]-cs[rI].mean())/cs[rI].std()).item()
rows=collections.defaultdict(list)
for g,y,p in lens:
    heads=list(range(0,8)) if len(g)==2 else list(range(8,16))
    v=contrib(G[g],heads).sum(0)   # centred value of this order
    rows["E[last]"].append(z(v,E,g[-1])); rows["E[first]"].append(z(v,E,g[0])); rows["E[next]"].append(z(v,E,y))
    rows["U[next] (direct logit)"].append(z(v,U,y)); rows["U[last]"].append(z(v,U,g[-1]))
print(f"lens over {len(lens)} n-grams with a dominant continuation (p>=0.3, count>=3); z-score of cos vs 512 random vocab rows (null: mean 0, sd 1)")
for k_,v in rows.items():
    tv=torch.tensor(v); print(f"  {k_:24s} mean z {tv.mean():+.2f}  median {tv.median():+.2f}  frac z>2 {(tv>2).float().mean():.2f}")
json.dump(dict(res=res,lens={k_:sum(v)/len(v) for k_,v in rows.items()},n_lens=len(lens)),open(f"{OUT}/row_pilot.json","w"),indent=1)
# examples
print("\nexamples (code2_top):",[tok.decode(list(g)) for g in cats["code2_top"][:12]])

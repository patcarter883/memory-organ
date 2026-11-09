import numpy as np
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex
Q='/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987'
tok=AutoTokenizer.from_pretrained(Q)
S=""
norm=normalizers.Sequence([normalizers.NFKC(),normalizers.NFD(),normalizers.StripAccents(),normalizers.Lowercase(),
  normalizers.Replace(Regex(r"[ \t\r\n]+")," "),normalizers.Replace(Regex(r"^ $"),S),normalizers.Strip(),normalizers.Replace(S," ")])
V=len(tok); k2n={}; o2n=np.empty(V,dtype=np.int64)
for t in range(V):
    tx=tok.decode([t],skip_special_tokens=False)
    key=tok.convert_ids_to_tokens(t) if '�' in tx else (norm.normalize_str(tx) or tx)
    o2n[t]=k2n.setdefault(key,len(k2n))
print('vocab',V,'->',len(k2n),f'reduction {100*(1-len(k2n)/V):.1f}%')
print('id0 =',repr(tok.decode([0])))
for w in ['Paris',' Paris',' paris']:
    ids=tok(w,add_special_tokens=False).input_ids; print(repr(w),ids,[int(o2n[i]) for i in ids])
# C7 arithmetic Qwen3-32B-like
d,ff,L,hq,hkv,hd,Vv=5120,25600,64,64,8,128,151936
per=d*hq*hd*2+d*hkv*hd*2+3*d*ff
print(f'per-layer {per/1e6:.1f}M; L0..6+emb {(7*per+Vv*d)/1e9:.2f}B -> {(7*per+Vv*d)*2/1e9:.1f} GB bf16')
tokens=8*32; fl=80*2*(2*per*tokens)
print(f'C9 fwd+act-bwd for 80 layer passes: {fl/1e12:.1f} TFLOP; at 989 TF/s: {fl/989e12*1e3:.0f} ms; 40% MFU: {fl/(0.4*989e12)*1e3:.0f} ms')

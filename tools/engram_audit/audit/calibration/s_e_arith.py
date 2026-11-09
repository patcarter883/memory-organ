# (e) C7/C9 arithmetic for a 64-layer d=5120 "27B" (and two real reference configs).
def cfg_params(L, d, nq, nkv, hd, I, V, tied):
    attn = d*nq*hd + 2*d*nkv*hd + nq*hd*d
    mlp = 3*d*I
    per = attn + mlp + 2*d
    emb = V*d*(1 if tied else 2)
    return per, emb, L*per + emb
# Solve intermediate size so a 64L/d=5120/64q/8kv/hd128/V=151936 untied model totals 27B
L, d, nq, nkv, hd, V = 64, 5120, 64, 8, 128, 151936
attn = d*nq*hd + 2*d*nkv*hd + nq*hd*d
I27 = int(((27e9 - 2*V*d)/L - attn - 2*d) / (3*d))
configs = {"hyp. 27B (64L,d5120,I=%d)" % I27: (L, d, nq, nkv, hd, I27, V, False),
           "Qwen3-32B-like (64L,d5120,I25600)": (64, 5120, 64, 8, 128, 25600, 151936, False),
           "Gemma-3-27B-like (62L,d5376,I21504)": (62, 5376, 32, 16, 128, 21504, 262144, True)}
for nm, c in configs.items():
    per, emb, tot = cfg_params(*c)
    sub = 7*per
    print(f"{nm}: total {tot/1e9:.1f}B | per layer {per/1e6:.0f}M | L0..6 = {sub/1e9:.2f}B params -> BF16 {sub*2/1e9:.2f} GB, FP8 {sub/1e9:.2f} GB; +embed_tokens {emb/(1 if c[7] else 2)/1e9:.2f}B -> BF16 {(sub+emb/(1 if c[7] else 2))*2/1e9:.2f} GB")
    print(f"    full model weights BF16 {tot*2/1e9:.1f} GB (C7 says full backprop 'spikes past 48-60 GB'; Gemini L0..6: ~3.2 GB BF16 / ~1.6 GB FP8 <=> {3.2e9/2/1e9:.1f}B params = {3.2e9/2/per:.1f} layers)")
# C9 timing: per edit, M=8 expressions x T=32 tokens; teacher prefix 64
per, emb, tot = cfg_params(*configs["hyp. 27B (64L,d5120,I=%d)" % I27])
tok_s, tok_t = 8*32, 8*(64+32)
fwd_flop = lambda ntok: 2*per*ntok
steps = 20
passes_fwd = steps*4; passes_bwd = steps*4                 # frozen weights: backward = input-grad only ~= 1x fwd matmul FLOPs
flops = fwd_flop(tok_t)*7 + fwd_flop(tok_s)*3 + (passes_fwd + passes_bwd)*fwd_flop(tok_s)
bytes_w = (7 + 3 + passes_fwd + passes_bwd) * per * 2      # weights re-read from HBM every layer pass (0.8 GB/layer >> L2)
act = 8*32*5120*2
print(f"\nC9: one [8,32,5120] hidden tensor = {act/1e6:.1f} MB (activations negligible)")
print(f"C9: Gemini counts 20x4=80 layer passes; with backward it's {passes_fwd+passes_bwd} (+10 teacher/cache) -> {flops/1e12:.1f} TFLOP, weight traffic {bytes_w/1e9:.0f} GB")
for gpu, tf, bw in [("H100 SXM", 989e12, 3.35e12), ("H200", 989e12, 4.8e12), ("A100 80GB", 312e12, 2.0e12), ("L40S", 362e12, 0.864e12), ("RTX 4090", 165e12, 1.0e12)]:
    tc, tm = flops/tf, bytes_w/bw
    print(f"  {gpu:10s}: compute floor {tc*1e3:6.1f} ms @100% MFU, HBM floor {tm*1e3:6.1f} ms -> roofline >= {max(tc,tm)*1e3:6.1f} ms; @50% MFU {max(tc/0.5,tm)*1e3:6.1f} ms (dense BF16 peak, no sparsity)")
print(f"  arithmetic intensity at 256 tokens: {2*256/2:.0f} FLOP/byte (H100 ridge ~{989/3.35:.0f}) -> memory/compute balanced; no headroom from batching edits unless M*T grows")

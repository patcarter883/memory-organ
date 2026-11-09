# (a3) The Phase-1 target sum_{l=3..6}(h_l - h_{l-1}) vs the prose's FFN-only sum: how much is attention?
import sys; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
import torch.nn.functional as F
tok, m = load_qwen(); stream, _ = repo_token_stream(tok)
eng = eg.GraftedEngramModule(d_model=896, r_mem=64, num_buckets=1000, device="cpu", dtype=torch.float32)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
ffn, attn = [], []
hooks = [m.model.layers[l].mlp.register_forward_hook(lambda mod, i, o: ffn.append(o)) for l in range(3, 7)]
hooks += [m.model.layers[l].self_attn.register_forward_hook(lambda mod, i, o: attn.append(o[0] if isinstance(o, tuple) else o)) for l in range(3, 7)]
cs_f, cs_a, nf, na = [], [], [], []
with torch.no_grad():
    for ids in batches(stream, 4, 128, 6, start=50000)[0]:
        ffn.clear(); attn.clear()
        _, _, delta = cal._run_subgraph(ids, record_ffn_accumulation=True)
        F_ = sum(ffn); A_ = sum(attn)
        d, f, a = delta[:, 1:], F_[:, 1:], A_[:, 1:]
        print_err = (d - (f + a)).norm() / d.norm()
        cs_f.append(F.cosine_similarity(d, f, dim=-1).mean()); cs_a.append(F.cosine_similarity(d, a, dim=-1).mean())
        nf.append((f.norm(dim=-1) / d.norm(dim=-1)).mean()); na.append((a.norm(dim=-1) / d.norm(dim=-1)).mean())
for h in hooks: h.remove()
mean = lambda x: torch.stack(x).mean().item()
print(f"reconstruction check ||delta - (sum attn + sum ffn)||/||delta|| = {print_err:.2e}")
print(f"positions>=1: cos(delta, FFN-only sum)={mean(cs_f):.3f}  cos(delta, attn-only sum)={mean(cs_a):.3f}  ||FFN||/||delta||={mean(nf):.3f}  ||attn||/||delta||={mean(na):.3f}")

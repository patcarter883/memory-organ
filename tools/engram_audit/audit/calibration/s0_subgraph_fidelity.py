# Does SubgraphGateCalibrator._run_subgraph reproduce the real model's layer-6 state? (sdpa vs eager, mask=None)
import sys; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
tok = AutoTokenizer.from_pretrained(QWEN); stream,_ = repo_token_stream(tok)
ids = stream[:64].view(1, 64)
for attn in ["sdpa", "eager"]:
    _, m = load_qwen(attn=attn)
    eng = eg.GraftedEngramModule(d_model=896, r_mem=64, num_buckets=1000, device="cpu", dtype=torch.float32)
    cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
    with torch.no_grad():
        hs = m(input_ids=ids, output_hidden_states=True).hidden_states
        _, h6, _ = cal._run_subgraph(ids)
        # future-leak test: change last token, see whether position 0..62 change
        ids2 = ids.clone(); ids2[0, -1] = 100
        _, h6b, _ = cal._run_subgraph(ids2)
    print(f"[{attn}] max|_run_subgraph h6 - HF hidden_states[7]| = {(h6-hs[7]).abs().max():.3e}  (rel {((h6-hs[7]).norm()/hs[7].norm()):.2e})")
    print(f"[{attn}] future-leak: max change at positions 0..62 after editing token 63 = {(h6[0,:-1]-h6b[0,:-1]).abs().max():.3e}")
    del m

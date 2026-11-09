# (c) Phase 2 sparse table SGD in bf16 (the verbatim default dtype): how many updates survive bf16 rounding?
#     and does _prefetched_e.requires_grad_(True) work / is .grad populated?
import sys, time; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
torch.manual_seed(0)
# --- requires_grad on a tensor produced by .to(...) under no_grad (both the same-dtype no-op path and a real cast)
with torch.no_grad():
    a = torch.randn(3, 4); y_same = a.to("cpu", dtype=torch.float32, non_blocking=True); y_cast = a.to("cpu", dtype=torch.bfloat16)
print("same-dtype .to returns the same object:", y_same is a, "| cast leaf:", y_cast.is_leaf)
for nm, y in [("same", y_same), ("cast", y_cast)]:
    y.requires_grad_(True); (y.float()*2).sum().backward(); print(f"  {nm}: requires_grad_ ok, grad populated: {y.grad is not None}")

DT = torch.bfloat16
tok, m = load_qwen(dtype=DT)
stream, _ = repo_token_stream(tok)

def phase2_one(eng, cal, ids, tag):
    old = {n: eng.cpu_tables[n].clone() for n in eng.orders}
    t0 = time.time()
    cal.calibrate_context_distillation([ids], prefix_len=256, lr_gates=5e-4, lr_table=5e-2)   # VERBATIM
    g = eng._prefetched_e.grad
    print(f"[{tag}] Phase-2 step {time.time()-t0:.0f}s; _prefetched_e.grad is None? {g is None}; max|grad|={g.float().abs().max():.3e} mean|grad|={g.float().abs().mean():.3e}")
    # intended fp32 update, reproduced from the same grad + hash map (exactly the verbatim scatter, but in fp32)
    gs = torch.split(g.float(), eng.r_per_order, dim=-1); idx = eng.hasher.hash_sequence(ids[:, 256:])
    for oi, n in enumerate(eng.orders):
        b = idx[n].reshape(-1)
        gf = gs[oi].unsqueeze(2).expand(-1, -1, eng.num_heads, -1).reshape(-1, eng.r_per_order) / eng.num_heads
        intended = torch.zeros(eng.cpu_tables[n].shape, dtype=torch.float32).index_add_(0, b, -5e-2 * gf)
        rows = torch.unique(b)
        it = intended[rows]; act = (eng.cpu_tables[n][rows].float() - old[n][rows].float())
        nz = it != 0
        lost = (act[nz] == 0).float().mean().item()
        rel = ((act - it).norm() / it.norm().clamp_min(1e-30)).item()
        print(f"   order {n}: touched rows {rows.numel()}, |intended upd| median {it[nz].abs().median():.2e} (table |w| median {old[n][rows].float().abs().median():.2e}, bf16 half-ulp there ~{old[n][rows].float().abs().median().item()*2**-8:.1e}); "
              f"fraction of nonzero updates rounded to ZERO = {lost:.2%}; ||applied-intended||/||intended|| = {rel:.3f}")

full = stream[30000:30000+2*512].view(2, 512)
# A) no Phase 1 (zero-init v_proj): are the table grads zero?
eng0 = eg.GraftedEngramModule(d_model=896, r_mem=512, num_buckets=300_000, device="cpu", dtype=DT)
cal0 = sc.SubgraphGateCalibrator(m, eng0, 2, 6)
phase2_one(eng0, cal0, full, "zero-init, no Phase 1")
del eng0, cal0
# Phase 1 crashes in bf16 on CPU (orthogonal_ -> QR unsupported for bf16). WORKAROUND (audit only): fp32 orthogonal_ then cast.
_orth = torch.nn.init.orthogonal_
def _orth_fp32(t, gain=1.0, generator=None):
    w = torch.empty(t.shape, dtype=torch.float32); _orth(w, gain=gain); 
    with torch.no_grad(): t.copy_(w.to(t.dtype))
    return t
torch.nn.init.orthogonal_ = _orth_fp32
# B) after Phase 1 (verbatim, 60 steps of [4,128] in bf16)
eng = eg.GraftedEngramModule(d_model=896, r_mem=512, num_buckets=300_000, device="cpu", dtype=DT)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
tr, _ = batches(stream, 4, 128, 60)
t0 = time.time(); cal.warmup_subspace_alignment(tr); print(f"Phase 1 (bf16, 60 steps) {time.time()-t0:.0f}s, ||v_proj||_F={eng.v_proj.weight.float().norm():.3f}")
for s in range(3):
    phase2_one(eng, cal, stream[40000+s*1024:40000+(s+1)*1024].view(2, 512), f"after Phase 1, Phase-2 batch {s}")

# Checks what Gemini's "dirac" short_conv init actually does under its causal left-pad, and the branch scale.
import sys, torch, torch.nn.functional as F
torch.set_num_threads(2)
torch.Tensor.pin_memory = lambda self, *a, **k: self   # CPU-only box: Gemini's pin_memory() raises without CUDA
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
import engram_graft as eg
eng = eg.GraftedEngramModule(d_model=896, device="cpu", dtype=torch.float32, num_buckets=100_000)
w = eng.short_conv.weight
print("dirac tap index with weight 1:", (w[0, 0] == 1).nonzero().flatten().tolist(), "kernel", w.shape)
torch.manual_seed(0)
ids = torch.randint(0, 150000, (1, 10))
eng.prefetch_from_host(ids)
with torch.no_grad():
    eng.v_proj.weight.normal_(0, 2.0)
    h = torch.zeros(1, 10, 896)
    br = eng.forward_fusion(h)[0]                      # branch only (h=0)
    e = eng._prefetched_e
    gv = (torch.sigmoid((eng.q_norm(eng.q_proj(h)) * eng.k_norm(eng.k_proj(e))).sum(-1, keepdim=True) * 512 ** -0.5) * eng.v_proj(e))[0]
    print("||e_t|| mean", float(e.norm(dim=-1).mean()), " ||alpha*v|| per pos", [round(float(x), 2) for x in gv.norm(dim=-1)])
    print("||branch|| per pos   ", [round(float(x), 2) for x in br.norm(dim=-1)])
    print("branch[t] == alpha*v[t-1]?  max|diff| =", float((br[1:] - gv[:-1]).abs().max()), " branch[0] norm", float(br[0].norm()))
    print("branch[t] == alpha*v[t]?    max|diff| =", float((br - gv).abs().max()))
    # incremental decode: T=1 call -> padded window [0,0,0,x_t]; dirac tap at index 2 reads a pad zero
    eng.prefetch_from_host(ids[:, 5:6])
    print("T=1 (incremental) branch norm:", float(eng.forward_fusion(torch.zeros(1, 1, 896)).norm()))
nz = (w != 0).nonzero().tolist()
print("nonzero entries in short_conv.weight after nn.init.dirac_:", len(nz), nz[:5], "of", w.numel())
with torch.no_grad():
    print("branch[t] == alpha*v[t-1] restricted to channel 0?  max|diff| =",
          float((br[1:, 0] - gv[:-1, 0]).abs().max()), "; max|branch[:,1:]| =", float(br[:, 1:].abs().max()))

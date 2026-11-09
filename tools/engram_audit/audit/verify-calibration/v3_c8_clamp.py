# F2 re-check: verbatim Phase 1 (fp32, 120 steps [4,128] of repo markdown) then the verbatim SubgraphEngramEditSolver,
# with the default clamp and with the clamp effectively removed (clamp_rho=1e4, more steps), to see whether the clamp or
# the objective is what makes C8 inert. Also measures ||v_proj||, ||e||, ||injection||/||h2||.
import sys, time, glob, torch, torch.nn.functional as F
torch.set_num_threads(2)
torch.Tensor.pin_memory = lambda self, *a, **k: self
sys.path.insert(0, "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/gemini")
import engram_graft as eg, subgraph_calibrator as sc, subgraph_edit_solver as ses
from transformers import AutoTokenizer, AutoModelForCausalLM
Q = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
tok = AutoTokenizer.from_pretrained(Q); m = AutoModelForCausalLM.from_pretrained(Q, dtype=torch.float32).eval()
for p in m.parameters(): p.requires_grad_(False)
torch.manual_seed(0)
ids = []
for f in sorted(glob.glob("/home/user/memory-organ/**/*.md", recursive=True)): ids += tok(open(f).read())["input_ids"]
stream = torch.tensor(ids)
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 120
eng = eg.GraftedEngramModule(d_model=896, r_mem=512, num_buckets=300_000, device="cpu", dtype=torch.float32)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
cal.warmup_subspace_alignment([stream[i*512:(i+1)*512].view(4, 128) for i in range(NB)])
W = eng.v_proj.weight
sv = torch.linalg.svdvals(W)
print(f"after Phase 1 ({NB} steps): ||v_proj||_F={W.norm():.3f}  sigma_max={sv[0]:.3f}  sigma_median={sv[len(sv)//2]:.4f}")
solver = ses.SubgraphEngramEditSolver(cal)
enc = lambda s: tok(s)["input_ids"]
FACTS = [("The Eiffel Tower is located in", "The Eiffel Tower", "Rome"), ("Big Ben is located in", "Big Ben", "Berlin"),
         ("The capital of France is", "The capital of France", "Berlin")]
def p_new(x, tid):
    g = eg.Layer2EngramGraft(m, m.model.layers[2], eng)
    try:
        with torch.no_grad(): lg = m(input_ids=torch.tensor([x])).logits[0, -1]
    finally: g.pre_hook.remove(); g.layer_hook.remove()
    return F.softmax(lg, -1)[tid].item()
for p, subj, new in FACTS:
    s = enc(p); tid = enc(" " + new)[0]; pre = enc(f"{p} {new}.\n"); a = [len(enc(subj)) - 1]
    t_ids, s_ids = torch.tensor([pre + s]), torch.tensor([s])
    with torch.no_grad():
        eng.prefetch_from_host(s_ids); h2 = cal._run_subgraph(s_ids)[0]
        e_norm = eng._prefetched_e[0, a[0]].norm().item()
        inj0 = (eng.forward_fusion(h2) - h2)[0, a[0]].norm().item()
    snap = {n: eng.cpu_tables[n].clone() for n in eng.orders}
    p0 = p_new(s, tid)
    for tag, kw in [("default clamp", dict(steps=20)), ("clamp off (rho=1e4), 100 steps, lr 0.5", dict(steps=100, clamp_rho=1e4, lr=0.5))]:
        L0 = solver.apply_subgraph_edit(t_ids, s_ids, len(pre), a, steps=1, clamp_rho=kw.get("clamp_rho", 0.45))
        for n in eng.orders: eng.cpu_tables[n].copy_(snap[n])
        Lf = solver.apply_subgraph_edit(t_ids, s_ids, len(pre), a, **kw)
        with torch.no_grad():
            eng.prefetch_from_host(s_ids); de = (eng._prefetched_e[0, a[0]] - torch.cat([snap[n][eng.hasher.hash_sequence(s_ids)[n][0, a[0]]].mean(0) for n in eng.orders])).norm().item()
            inj = (eng.forward_fusion(h2) - h2)[0, a[0]].norm().item()
        p1 = p_new(s, tid)
        print(f"{new:7s} [{tag}] window loss {L0:.4f}->{Lf:.4f} | ||e_anchor|| {e_norm:.3f}, written ||de|| {de:.3f} | ||inj||/||h2|| at anchor {inj0/h2[0,a[0]].norm():.2e}->{inj/h2[0,a[0]].norm():.2e} | P(new) {p0:.5f}->{p1:.5f}")
        for n in eng.orders: eng.cpu_tables[n].copy_(snap[n])

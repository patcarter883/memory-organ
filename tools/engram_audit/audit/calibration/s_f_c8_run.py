# (f) Run the VERBATIM SubgraphEngramEditSolver (C8) after a verbatim Phase 1, on real facts; measure L6 loss and the full-model effect.
import sys, time; sys.path.insert(0, __file__.rsplit('/',1)[0]); from common import *
import torch.nn.functional as F
torch.manual_seed(0)
tok, m = load_qwen()
stream, _ = repo_token_stream(tok)
eng = eg.GraftedEngramModule(d_model=896, r_mem=512, num_buckets=300_000, device="cpu", dtype=torch.float32)
cal = sc.SubgraphGateCalibrator(m, eng, 2, 6)
tr, _ = batches(stream, 4, 128, 120); cal.warmup_subspace_alignment(tr)
solver = ses.SubgraphEngramEditSolver(cal)
def ids(s): return tok(s)["input_ids"]
FACTS = [("The Eiffel Tower is located in", "The Eiffel Tower", "Rome"), ("The Colosseum is located in", "The Colosseum", "London"),
         ("Big Ben is located in", "Big Ben", "Berlin"), ("The capital of France is", "The capital of France", "Berlin"),
         ("Barack Obama was born in", "Barack Obama", "Paris")]
def p_new(x, tid, graft):
    g = eg.Layer2EngramGraft(m, m.model.layers[2], eng) if graft else None
    try:
        with torch.no_grad(): lg = m(input_ids=torch.tensor([x])).logits[0, -1]
    finally:
        if g: g.pre_hook.remove(); g.layer_hook.remove()
    return F.softmax(lg, -1)[tid].item()
for prompt, subj, new in FACTS:
    s = ids(prompt); tid = ids(" " + new)[0]; pre = ids(f"{prompt} {new}.\n")
    t_ids = torch.tensor([pre + s]); s_ids = torch.tensor([s]); a = [len(ids(subj)) - 1]
    snap = {n: eng.cpu_tables[n].clone() for n in eng.orders}
    L1 = solver.apply_subgraph_edit(t_ids, s_ids, len(pre), a, steps=1)          # loss at delta=0 (returned loss is pre-step)
    for n in eng.orders: eng.cpu_tables[n].copy_(snap[n])
    p0 = p_new(s, tid, True)
    # capture the final delta norm via the perturbation tuple each step
    norms = []; orig = eng.forward_fusion
    def ff(h):
        if eng._active_perturbation is not None: norms.append(eng._active_perturbation[0].float().norm().item())
        return orig(h)
    eng.forward_fusion = ff
    t0 = time.time(); L20 = solver.apply_subgraph_edit(t_ids, s_ids, len(pre), a, steps=20); dt = time.time() - t0
    eng.forward_fusion = orig
    p1 = p_new(s, tid, True)
    # magnitude of the L2 write this delta produces at the anchor
    with torch.no_grad():
        eng.prefetch_from_host(s_ids); h2 = cal._run_subgraph(s_ids)[0]
        inj = (eng.forward_fusion(h2) - h2)[0].norm(dim=-1)
    print(f"{prompt!r:40s} -> {new}: C8 window loss {L1:.4f} -> {L20:.4f} | max ||delta|| {max(norms):.3f} | P(new) with graft before {p0:.4f} after edit {p1:.4f} (teacher-in-context ~0.16) | ||L2 injection||/||h2|| at anchor after write {(inj[a[0]]/h2[0,a[0]].norm()).item():.2e} | {dt:.1f}s")
    for n in eng.orders: eng.cpu_tables[n].copy_(snap[n])

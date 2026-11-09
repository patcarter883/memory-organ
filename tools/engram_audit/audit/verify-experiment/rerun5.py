# Independent re-implementation (HF forward + forward hooks, NOT run.py's manual layer runner) of arms A0/A1/A2/oracle/A3
# on 5 of the 12 facts. Prompts/anchors are taken from run.py's own data code (exec'd verbatim) so the fact set is identical.
# Differences from run.py on purpose: every eval prompt is run alone (no padding at all); delta enters via a forward hook on
# model.model.layers[2]; A2 student states come from output_hidden_states[k+1] of a full HF forward.
import sys, json, time, torch, torch.nn.functional as F
EXP = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/audit/experiment"
OUT = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/audit/verify-experiment/rerun5.json"
src = open(EXP + "/run.py").read().splitlines()
cut = next(i for i, l in enumerate(src) if l.startswith("def add_deltas"))
sys.argv = ["run.py", "--n", "12", "--skip_decode"]
g = {"__name__": "x"}
exec(compile("\n".join(src[:cut]).replace("torch.set_num_threads(3)", "torch.set_num_threads(2)"), "run_head", "exec"), g)
facts, tok, model, prefix_ids = g["facts"], g["tok"], g["model"], g["prefix_ids"]
LAYERS, NORM, HEAD, D = model.model.layers, model.model.norm, model.lm_head, model.config.hidden_size
PICK = [int(x) for x in (sys.argv_pick if hasattr(sys, "argv_pick") else "0,3,4,8,10".split(","))]
LONG = 150
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


class Inject:
    """forward hook on a decoder layer: add vec[b] at position anchors[b] of the layer OUTPUT, or replace a span."""
    def __init__(self, layer):
        self.h = layer.register_forward_hook(self.f); self.add = None; self.rep = None
    def f(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.add is not None:
            vec, anchors = self.add
            off = torch.zeros_like(h)
            for b, a in enumerate(anchors):
                if a is not None:
                    off[b, a] = vec
            h = h + off
        if self.rep is not None:
            h = h.clone()
            for b, (s, e, val) in enumerate(self.rep):
                h[b, s:e] = val
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def batch(seqs):
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), tok.eos_token_id); am = torch.zeros(len(seqs), T, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s); am[i, :len(s)] = 1
    return ids, am, torch.tensor([len(s) for s in seqs])


def last_probs(seq, inj=None, vec=None, a=None):
    with torch.no_grad():
        if inj is not None:
            inj.add = (vec, [a])
        p = F.softmax(model(input_ids=torch.tensor([seq])).logits[0, -1], -1)
        if inj is not None:
            inj.add = None
    return p


def h2_norm(f):
    ids, am, lens = batch([e[0] for e in f["opt_enc"]])
    with torch.no_grad():
        hs = model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
    return torch.stack([hs[3][m, a].norm() for m, (_, a) in enumerate(f["opt_enc"])]).mean()


def opt_A1(f, inj2, rho=1.0, steps=30):
    ids, am, lens = batch([e[0] for e in f["opt_enc"]]); anc = [e[1] for e in f["opt_enc"]]
    maxn = rho * h2_norm(f)
    d = torch.zeros(D, requires_grad=True); opt = torch.optim.Adam([d], lr=0.1)
    for _ in range(steps):
        opt.zero_grad(); inj2.add = (d, anc)
        lg = model(input_ids=ids, attention_mask=am).logits[torch.arange(len(anc)), lens - 1]
        loss = -F.log_softmax(lg, -1)[:, f["tn"]].mean() + 0.01 * d.pow(2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            if d.norm() > maxn: d.mul_(maxn / d.norm())
    inj2.add = None
    return d.detach()


def teacher_k(f, k):
    pre = prefix_ids(f); P = len(pre)
    out = []
    with torch.no_grad():
        for ids_, a in f["opt_enc"]:
            hs = model(input_ids=torch.tensor([pre + ids_]), output_hidden_states=True).hidden_states
            out.append(hs[k + 1][0, P:])   # hidden_states[k+1] = output of decoder layer k (0-based)
    return out, P


def opt_A2(f, k, inj2, rho=4.0, steps=30):
    ids, am, lens = batch([e[0] for e in f["opt_enc"]]); anc = [e[1] for e in f["opt_enc"]]
    tk, P = teacher_k(f, k)
    M, T = ids.shape
    tgt = torch.ones(M, T, D); w = torch.zeros(M, T)
    for m in range(M):
        tgt[m, :lens[m]] = tk[m]
        q = anc[m]; e = min(int(lens[m]), q + 9)
        dec = torch.exp(-0.25 * torch.arange(e - q, dtype=torch.float32)); dec[0] = 2.0; w[m, q:e] = dec
    tn = F.layer_norm(tgt, (D,))
    maxn = rho * h2_norm(f)
    d = torch.zeros(D, requires_grad=True); opt = torch.optim.Adam([d], lr=0.1); hist = []
    for s in range(steps + 1):
        opt.zero_grad(); inj2.add = (d, anc)
        hs = model.model(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states[k + 1]
        lossw = (((F.layer_norm(hs, (D,)) - tn).pow(2).mean(-1) + 0.5 * (1 - F.cosine_similarity(hs, tgt, dim=-1))) * w).sum() / w.sum()
        if s in (0, steps): hist.append(float(lossw))
        if s == steps: break
        (lossw + 0.005 * d.pow(2).sum()).backward(); opt.step()
        with torch.no_grad():
            if d.norm() > maxn: d.mul_(maxn / d.norm())
    inj2.add = None
    return d.detach(), hist


def oracle(f, k, injk):
    tk, P = teacher_k(f, k); res = []
    for m, (ids_, a) in enumerate(f["opt_enc"]):
        e = min(len(ids_), a + 9)
        injk.rep = [(a, e, tk[m][a:e])]
        with torch.no_grad():
            p = F.softmax(model(input_ids=torch.tensor([ids_])).logits[0, -1], -1)
        injk.rep = None
        res.append((float(p[f["tn"]]), float(p[f["tn"]] > p[f["tt"]]), float(p.argmax() == f["tn"])))
    return [sum(x[i] for x in res) / len(res) for i in range(3)]


def evalp(f, inj2=None, vec=None, prefix=None):
    """per-prompt (group, prompt, pnew, pold, top1, kl_vs_base)"""
    rows = []
    for grp in ("opt", "held", "leak"):
        for p, (ids_, a) in zip(f[grp], f[grp + "_enc"]):
            pb = last_probs(ids_)
            if prefix is not None:
                pe = last_probs(prefix + ids_)
            elif vec is not None:
                pe = last_probs(ids_, inj2, vec, a)
            else:
                pe = pb
            kl = float((pe * (pe.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum())
            rows.append(dict(grp=grp, prompt=p, pnew=float(pe[f["tn"]]), pold=float(pe[f["tt"]]),
                             top1=float(pe.argmax() == f["tn"]), top1tok=tok.decode([int(pe.argmax())]), kl=kl))
    return rows


def held_overlap(f):
    """True if the CF paraphrase, from the subject onwards, is verbatim one of the optimisation prompts."""
    s = f["subject"]
    return [p[p.find(s):] in f["opt"] for p in f["held"]]


inj2 = Inject(LAYERS[2])
results = []
for fi in PICK:
    f = facts[fi]
    log(f"fact {fi+1} {f['subject']!r} {f['old']}->{f['new']}")
    R = dict(fact=fi + 1, subject=f["subject"], rel=f["rel"], held_overlap=held_overlap(f))
    R["A0"] = evalp(f)
    d1 = opt_A1(f, inj2, 1.0); R["A1_norm"] = float(d1.norm()); R["A1"] = evalp(f, inj2, d1)
    log("  A1 done |d|=%.2f" % d1.norm())
    for k in (6, 18):
        d2, hist = opt_A2(f, k, inj2, 4.0, 30)
        R[f"A2_k{k}"] = evalp(f, inj2, d2); R[f"A2_k{k}_hist"] = hist; R[f"A2_k{k}_norm"] = float(d2.norm())
        log(f"  A2 k{k} loss {hist} |d|={d2.norm():.2f}")
    d3, hist = opt_A2(f, 6, inj2, 1e9, LONG)   # unconstrained, 5x longer: is A2 under-optimised?
    R["A2_k6_long"] = evalp(f, inj2, d3); R["A2_k6_long_hist"] = hist; R["A2_k6_long_norm"] = float(d3.norm())
    R["cos_A2long_vs_A1"] = float(F.cosine_similarity(d3, d1, dim=0))
    log(f"  A2 k6 long({LONG}) loss {hist} |d|={d3.norm():.2f} cos(A1)={R['cos_A2long_vs_A1']:.3f}")
    for k in (6, 18):
        injk = Inject(LAYERS[k]); R[f"oracle_k{k}"] = oracle(f, k, injk); injk.h.remove()
    R["A3"] = evalp(f, prefix=prefix_ids(f))
    # logit lens on teacher, last token of each opt prompt, with final norm; check hidden_states[-1] convention
    pre = prefix_ids(f); lens_ = []
    with torch.no_grad():
        for ids_, a in f["opt_enc"]:
            o = model(input_ids=torch.tensor([pre + ids_]), output_hidden_states=True)
            hs = o.hidden_states
            pl = [float(F.softmax(HEAD(NORM(h[0, -1])), -1)[f["tn"]]) for h in hs[:-1]]
            pl.append(float(F.softmax(o.logits[0, -1], -1)[f["tn"]]))
            lens_.append(pl)
            R.setdefault("hs_last_is_normed", float((HEAD(hs[-1][0, -1]) - o.logits[0, -1]).abs().max()))
    R["teacher_lens"] = [sum(x[i] for x in lens_) / len(lens_) for i in range(len(lens_[0]))]
    results.append(R)
    json.dump(results, open(OUT, "w"), indent=1)
    log("  saved")
log("done")

"""Decisive experiment: can a free R^896 delta at L2 (last subject token) install a counterfactual fact in a
FROZEN Qwen2.5-0.5B, and can Gemini's subgraph objective (match an in-context teacher at layer k) find it?

Arms
  A0  no edit
  A1  full-NLL objective (EngramEdit Stage-1 upper bound), delta at L2 (and at L12 for depth comparison)
  A2  Gemini subgraph objective (LN-MSE + 0.5*(1-cos) vs teacher "<fact>.\n<prompt>" at layer k in {6,12,18}),
      delta at L2; position-ids convention pos0 (Gemini: student positions start at 0) and posoff (student
      positions offset by the teacher's prefix length)
  A2-oracle  patch the TEACHER's layer-k states into the student (window / anchor-only) -> ceiling of the objective
  A3  ICL ceiling (fact statement in context, no delta)
Plus: per-layer teacher-vs-student gap, teacher logit lens, n-gram hash-hit analysis, and a decode
(full forward vs incremental KV-cache) check of Gemini's GraftedEngramModule/Layer2EngramGraft.
"""
import argparse, json, math, random, sys, time, collections
import torch
import torch.nn.functional as F

torch.set_num_threads(3)
torch.manual_seed(0)
random.seed(0)

S = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"
CF = S + "/audit/experiment/data_dl/data/test-00000-of-00001-bacb83500fca49a9.parquet"
GEMINI_DIR = S + "/gemini"

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--rhos", type=str, default="0.5,1.0,4.0")
ap.add_argument("--ks", type=str, default="6,12,18")
ap.add_argument("--posoff_rhos", type=str, default="1.0")
ap.add_argument("--out", type=str, default=S + "/audit/experiment/results.json")
ap.add_argument("--skip_decode", action="store_true")
ap.add_argument("--decode_only", action="store_true")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--a2_rhos", type=str, default=None)
args = ap.parse_args()
RHOS = [float(x) for x in args.rhos.split(",")]
KS = [int(x) for x in args.ks.split(",")]
POSOFF_RHOS = [float(x) for x in args.posoff_rhos.split(",")] if args.posoff_rhos else []
A2_RHOS = [float(x) for x in args.a2_rhos.split(",")] if args.a2_rhos else RHOS
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
base = model.model
LAYERS, EMB, ROT, NORM, HEAD = base.layers, base.embed_tokens, base.rotary_emb, base.norm, model.lm_head
NL = len(LAYERS)
D = model.config.hidden_size
PAD = tok.eos_token_id
log("loaded", NL, "layers d", D)


def run(h, s, e, pos, keep=False):
    pe = ROT(h, pos)
    hs = []
    for i in range(s, e):
        h = LAYERS[i](h, position_ids=pos, position_embeddings=pe)
        h = h[0] if isinstance(h, tuple) else h
        if keep:
            hs.append(h)
    return (h, hs) if keep else h


def pad(seqs):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return ids, torch.tensor([len(s) for s in seqs])


def posids(B, T, off=None):
    p = torch.arange(T).unsqueeze(0).expand(B, T).clone()
    if off is not None:
        p = p + off.view(-1, 1)
    return p


def last_logits(h, lens):
    x = h[torch.arange(h.size(0)), lens - 1]
    return HEAD(NORM(x))


# ---------------- sanity: manual runner == HF forward ----------------
with torch.no_grad():
    sid = tok("The Eiffel Tower is located in the city of")["input_ids"]
    ids, lens = pad([sid])
    ref = model(input_ids=ids).logits[0, -1]
    mine = last_logits(run(EMB(ids), 0, NL, posids(1, ids.size(1))), lens)[0]
    SANITY = float((ref - mine).abs().max())
    log("sanity manual-vs-HF max|dlogit| =", SANITY)

# ---------------- data ----------------
TEMPL = {  # 3 optimisation templates + 1 extra held-out template, per relation
    "P103": ["The mother tongue of {} is", "{} is a native speaker of", "{} grew up speaking", "{}'s first language is"],
    "P27": ["{} is a citizen of", "{} holds citizenship of", "{} is a national of", "{} has a passport from"],
    "P1412": ["{} speaks the language", "{} writes in", "The language used by {} is", "{} is fluent in"],
    "P30": ["{} is located in the continent of", "{} is a part of the continent of", "{} belongs to the continent of", "{} can be found on the continent of"],
    "P37": ["The official language of {} is", "In {}, the language spoken is", "The people of {} speak", "In {}, most people speak"],
    "P176": ["{} is produced by", "{} is manufactured by", "{} is made by", "The maker of {} is"],
    "P17": ["{} is located in the country of", "{}, which is located in", "{} is in the country of", "{} lies in the nation of"],
    "P937": ["{} worked in the city of", "{} took up work in", "{} found employment in", "{} spent their working life in"],
    "P19": ["{} was born in", "{} is originally from", "{} originates from", "The birthplace of {} is"],
    "P20": ["{} died in the city of", "{} passed away in", "{} lost their life at", "The place of death of {} is"],
    "P159": ["The headquarters of {} is in", "{} is based in", "{}'s headquarters are in", "{} has its head office in"],
    "P364": ["The original language of {} is", "The language of {} is", "{} was originally made in the language", "{} was filmed in the language"],
    "P449": ["{} premiered on", "{} debuted on", "{} originally aired on", "{} was first broadcast on"],
    "P413": ["{} plays in the position of", "{}, who plays the position", "{} is playing as a", "On the field, {} plays as a"],
    "P178": ["{} was developed by", "{}, a product developed by", "{} is a product of", "The developer of {} is"],
    "P136": ["{} is known for performing", "The genre played by {} is", "{} performs", "The musical style of {} is"],
    "P106": ["The profession of {} is", "{} works as a", "By profession, {} is a", "{}'s occupation is"],
    "P101": ["The domain of activity of {} is", "{} works in the field of", "{}'s area of work is", "{} is known for work in"],
    "P1303": ["{} plays the instrument", "The instrument played by {} is the", "{} performs on the", "{} is a musician who plays the"],
    "P740": ["{} was formed in", "{} was founded in the city of", "{} originated in", "{} got its start in"],
    "P495": ["{} originated in the country of", "{}, that was created in", "{} comes from the country of", "The country of origin of {} is"],
}
LEAK = ["{} is a", "The most famous thing about {} is", "{} was founded in"]
NEUTRAL = "The weather today is"

import pyarrow.parquet as pq
rows = pq.read_table(CF).to_pylist()
random.Random(0).shuffle(rows)


def single(s):
    t = tok(" " + s)["input_ids"]
    return t[0] if len(t) == 1 else None


def enc_anchor(text, subj):
    """token ids and index of the token containing the subject's LAST char (first occurrence of subject)."""
    e = tok(text, return_offsets_mapping=True)
    i = text.find(subj)
    if i < 0:
        return e["input_ids"], None
    c = i + len(subj) - 1
    for j, (a, b) in enumerate(e["offset_mapping"]):
        if a <= c < b:
            return e["input_ids"], j
    return e["input_ids"], None


cands, per_rel = [], collections.Counter()
for r in rows:
    rr = r["requested_rewrite"]
    rel = rr["relation_id"]
    if rel not in TEMPL or per_rel[rel] >= 3:
        continue
    tn, tt = single(rr["target_new"]["str"]), single(rr["target_true"]["str"])
    if tn is None or tt is None or tn == tt:
        continue
    s = rr["subject"]
    rw = rr["prompt"].format(s)
    tpl = [t.format(s) for t in TEMPL[rel]]
    opt = [rw] + [t for t in tpl[:3] if t != rw]
    if len(opt) < 4:
        opt = [rw] + [t for t in tpl if t != rw][:3]
    extra = [t for t in tpl if t not in opt][:1]
    held = list(r["paraphrase_prompts"]) + extra
    leak = [t.format(s) for t in LEAK]
    allp = opt + held + leak
    enc = [enc_anchor(p, s) for p in allp]
    if any(a is None for _, a in enc):
        continue
    cands.append(dict(case_id=r["case_id"], rel=rel, subject=s, new=rr["target_new"]["str"], old=rr["target_true"]["str"],
                      tn=tn, tt=tt, fact=rw + " " + rr["target_new"]["str"], opt=opt, held=held, leak=leak,
                      opt_enc=enc[:4], held_enc=enc[4:4 + len(held)], leak_enc=enc[4 + len(held):]))
    per_rel[rel] += 1
    if len(cands) >= 3 * args.n:
        break

# validity gate: base ranks target_true above target_new on the rewrite prompt
facts = []
n_gate_fail = 0
with torch.no_grad():
    for f in cands:
        ids, lens = pad([f["opt_enc"][0][0]])
        lg = last_logits(run(EMB(ids), 0, NL, posids(1, ids.size(1))), lens)[0]
        if lg[f["tt"]] > lg[f["tn"]]:
            facts.append(f)
        else:
            n_gate_fail += 1
        if len(facts) >= args.n:
            break
log(f"facts: {len(facts)} kept (gate rejected {n_gate_fail}); relations {collections.Counter(f['rel'] for f in facts)}")

# ---------------- helpers ----------------
NL_ = "\n"
PREF_CACHE = {}


def prefix_ids(f):
    return tok(f["fact"] + "." + NL_)["input_ids"]


def add_deltas(h, anchors_per_row, delta_rows):
    """h [B,T,D]; anchors_per_row list[int or None]; delta_rows [B,D] (already gathered)."""
    bi = [b for b, a in enumerate(anchors_per_row) if a is not None]
    if not bi:
        return h
    ai = [anchors_per_row[b] for b in bi]
    bi_t, ai_t = torch.tensor(bi), torch.tensor(ai)
    off = torch.zeros_like(h).index_put((bi_t, ai_t), delta_rows[bi_t])
    return h + off


def probs_eval(seqs, anchors, configs):
    """configs: list of (layer, delta[D]) or None for no edit. Returns list (per config) of softmax probs [B, V]."""
    ids, lens = pad(seqs)
    B, T = ids.shape
    pos = posids(B, T)
    out = []
    with torch.no_grad():
        h0 = EMB(ids)
        cache = {}
        for cfg in configs:
            if cfg is None:
                if "base" not in cache:
                    cache["base"] = F.softmax(last_logits(run(h0, 0, NL, pos), lens), -1)
                out.append(cache["base"])
                continue
            L, d = cfg
            key = ("h", L)
            if key not in cache:
                cache[key] = run(h0, 0, L + 1, pos)
            h = add_deltas(cache[key], anchors, d.view(1, -1).expand(B, D))
            out.append(F.softmax(last_logits(run(h, L + 1, NL, pos), lens), -1))
            if MCB is not None:  # memory saver: reduce to metrics immediately (OOM-killed once in a shared cgroup)
                out[-1] = MCB(out[-1])
    return out
MCB = None


def metrics(P, Pb, f, idx_groups):
    res = {}
    for name, idx in idx_groups.items():
        if not idx:
            continue
        p = P[idx]; pb = Pb[idx]
        pn, po = p[:, f["tn"]], p[:, f["tt"]]
        top1 = (p.argmax(-1) == f["tn"]).float()
        kl = (p * (p.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum(-1)
        res[name] = dict(eff=float((pn > po).float().mean()), top1=float(top1.mean()), pnew=float(pn.mean()),
                         kl=float(kl.mean()))
    return res


# ---------------- optimisation ----------------
def opt_A1(f, L, rhos):
    seqs = [e[0] for e in f["opt_enc"]]; anc = [e[1] for e in f["opt_enc"]]
    ids, lens = pad(seqs); M, T = ids.shape
    pos = posids(M, T)
    with torch.no_grad():
        hc = run(EMB(ids), 0, L + 1, pos)
        anorm = torch.stack([hc[m, anc[m]].norm() for m in range(M)]).mean()
    R = len(rhos)
    maxn = torch.tensor(rhos) * anorm
    delta = torch.zeros(R, D, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.1)
    hrep = hc.repeat(R, 1, 1); lrep = lens.repeat(R); posr = pos.repeat(R, 1)
    ancr = anc * R
    rowd = torch.arange(R).repeat_interleave(M)
    hist = []
    for step in range(args.steps):
        opt.zero_grad()
        h = add_deltas(hrep, ancr, delta[rowd])
        lg = last_logits(run(h, L + 1, NL, posr), lrep)
        nll = -F.log_softmax(lg, -1)[:, f["tn"]].view(R, M).mean(1)
        loss = nll.sum() + 0.01 * delta.pow(2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            n = delta.norm(dim=1)
            sc = torch.where(n > maxn, maxn / (n + 1e-8), torch.ones_like(n))
            delta.mul_(sc.view(-1, 1))
        if step in (0, args.steps - 1):
            hist.append([float(x) for x in nll])
    return delta.detach(), float(anorm), hist


def teacher_states(f, k):
    pre = prefix_ids(f)
    P = len(pre)
    seqs = [pre + e[0] for e in f["opt_enc"]]
    ids, lens = pad(seqs); M, T = ids.shape
    with torch.no_grad():
        hk = run(EMB(ids), 0, k + 1, posids(M, T))
    return hk, P


def opt_A2(f, k, rhos, posoff):
    seqs = [e[0] for e in f["opt_enc"]]; anc = [e[1] for e in f["opt_enc"]]
    ids, lens = pad(seqs); M, T = ids.shape
    hk_t, P = teacher_states(f, k)
    # aligned teacher target [M, T, D]
    tgt = torch.ones(M, T, D)
    for m in range(M):
        Lm = int(lens[m]); tgt[m, :Lm] = hk_t[m, P:P + Lm]
    wm = torch.zeros(M, T, 1)
    for m, q in enumerate(anc):
        end = min(int(lens[m]), q + 8 + 1)
        dec = torch.exp(-0.25 * torch.arange(end - q, dtype=torch.float32)); dec[0] = 2.0
        wm[m, q:end, 0] = dec
    pos = posids(M, T, torch.full((M,), P) if posoff else None)
    with torch.no_grad():
        hc = run(EMB(ids), 0, 3, pos)
        anorm = torch.stack([hc[m, anc[m]].norm() for m in range(M)]).mean()
    t_n = F.layer_norm(tgt, (D,))
    R = len(rhos)
    maxn = torch.tensor(rhos) * anorm
    delta = torch.zeros(R, D, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=0.1)
    hrep = hc.repeat(R, 1, 1); posr = pos.repeat(R, 1); ancr = anc * R
    rowd = torch.arange(R).repeat_interleave(M)
    t_nr, tr, wr = t_n.repeat(R, 1, 1), tgt.repeat(R, 1, 1), wm.repeat(R, 1, 1)
    hist = []
    for step in range(args.steps + 1):
        opt.zero_grad()
        h = add_deltas(hrep, ancr, delta[rowd])
        s = run(h, 3, k + 1, posr)
        s_n = F.layer_norm(s, (D,))
        mse = (s_n - t_nr).pow(2).mean(-1, keepdim=True)
        cos = 1.0 - F.cosine_similarity(s, tr, dim=-1).unsqueeze(-1)
        per = ((mse + 0.5 * cos) * wr).view(R, M, T, 1).sum((1, 2, 3)) / wm.sum()
        if step == 0 or step == args.steps:
            hist.append([float(x) for x in per])
        if step == args.steps:
            break
        loss = per.sum() + 0.005 * delta.pow(2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            n = delta.norm(dim=1)
            sc = torch.where(n > maxn, maxn / (n + 1e-8), torch.ones_like(n))
            delta.mul_(sc.view(-1, 1))
    return delta.detach(), float(anorm), hist  # hist = [loss at delta=0, loss at final delta]


def oracle_patch(f, k):
    """Patch teacher layer-k states into the student at (a) window positions, (b) anchor only; P(new) etc. on opt prompts."""
    seqs = [e[0] for e in f["opt_enc"]]; anc = [e[1] for e in f["opt_enc"]]
    ids, lens = pad(seqs); M, T = ids.shape
    hk_t, P = teacher_states(f, k)
    pos = posids(M, T)
    out = {}
    with torch.no_grad():
        hs = run(EMB(ids), 0, k + 1, pos)
        for mode in ("window", "anchor"):
            h = hs.clone()
            for m, q in enumerate(anc):
                end = min(int(lens[m]), q + 9) if mode == "window" else q + 1
                h[m, q:end] = hk_t[m, P + q:P + end]
            p = F.softmax(last_logits(run(h, k + 1, NL, pos), lens), -1)
            out[mode] = dict(eff=float((p[:, f["tn"]] > p[:, f["tt"]]).float().mean()), pnew=float(p[:, f["tn"]].mean()),
                             top1=float((p.argmax(-1) == f["tn"]).float().mean()))
    return out


def layer_profile(f, configs):
    """Per-layer ||LN(h_t^l)-LN(h_s^l)|| at last prompt pos (opt prompts) and logit-lens P(new) for teacher & students.
    configs: dict name -> (L, delta) or None. Returns dict."""
    pre = prefix_ids(f); P = len(pre)
    seqs = [e[0] for e in f["opt_enc"]]; anc = [e[1] for e in f["opt_enc"]]
    ids, lens = pad(seqs); M, T = ids.shape
    tids, tl = pad([pre + s for s in seqs])
    res = {}
    with torch.no_grad():
        h0 = EMB(tids)
        _, th = run(h0, 0, NL, posids(M, tids.size(1)), keep=True)
        tH = [h0] + th
        tlast = torch.stack([x[torch.arange(M), tl - 1] for x in tH])  # [25, M, D]
        res["teacher_lens_pnew"] = F.softmax(HEAD(NORM(tlast)), -1)[..., f["tn"]].mean(1).tolist()
        for name, cfg in configs.items():
            pos = posids(M, T)
            hs0 = EMB(ids)
            if cfg is None:
                _, sh = run(hs0, 0, NL, pos, keep=True)
            else:
                L, d = cfg
                hL, sh1 = run(hs0, 0, L + 1, pos, keep=True)
                hL = add_deltas(hL, anc, d.view(1, -1).expand(M, D))
                sh1[-1] = hL
                _, sh2 = run(hL, L + 1, NL, pos, keep=True)
                sh = sh1 + sh2
            sH = [hs0] + sh
            slast = torch.stack([x[torch.arange(M), lens - 1] for x in sH])
            gap = (F.layer_norm(tlast, (D,)) - F.layer_norm(slast, (D,))).norm(dim=-1).mean(1)
            res[name + "_gap"] = gap.tolist()
            res[name + "_lens_pnew"] = F.softmax(HEAD(NORM(slast)), -1)[..., f["tn"]].mean(1).tolist()
    return res


def ngram(ids, a, n):
    return tuple(([0] * (n - 1) + list(ids))[a:a + n])


# ---------------- main loop ----------------
if args.decode_only:
    results = json.load(open(args.out)) if __import__("os").path.exists(args.out) else {}
else:
    per_fact = []
    if args.resume and __import__("os").path.exists(args.out):
        per_fact = json.load(open(args.out))["per_fact"]
        log("resumed", len(per_fact), "facts from", args.out)
    done_ids = {r["case_id"] for r in per_fact}
    for fi, f in enumerate(facts):
        if f["case_id"] in done_ids:
            continue
        tf = time.time()
        rec = dict(case_id=f["case_id"], rel=f["rel"], subject=f["subject"], new=f["new"], old=f["old"])
        configs = {"A0": None}
        info = {}
        for L in (2, 12):
            d, an, hist = opt_A1(f, L, RHOS)
            info[f"A1_L{L}"] = dict(anchor_norm=an, nll_hist=hist, delta_norms=[float(x) for x in d.norm(dim=1)])
            for i, r in enumerate(RHOS):
                configs[f"A1_L{L}_rho{r}"] = (L, d[i])
        for k in KS:
            d, an, hist = opt_A2(f, k, A2_RHOS, posoff=False)
            info[f"A2_k{k}_pos0"] = dict(anchor_norm=an, loss_hist=hist, delta_norms=[float(x) for x in d.norm(dim=1)], rhos=A2_RHOS)
            for i, r in enumerate(A2_RHOS):
                configs[f"A2_k{k}_pos0_rho{r}"] = (2, d[i])
            if POSOFF_RHOS:
                d2, an2, hist2 = opt_A2(f, k, POSOFF_RHOS, posoff=True)
                info[f"A2_k{k}_posoff"] = dict(anchor_norm=an2, loss_hist=hist2, delta_norms=[float(x) for x in d2.norm(dim=1)])
                for i, r in enumerate(POSOFF_RHOS):
                    configs[f"A2_k{k}_posoff_rho{r}"] = (2, d2[i])
                    j = A2_RHOS.index(r) if r in A2_RHOS else None
                    if j is not None:
                        info[f"A2_k{k}_posoff"]["cos_vs_pos0_rho%s" % r] = float(F.cosine_similarity(d2[i], d[j], dim=0))
                        info[f"A2_k{k}_posoff"]["reldiff_vs_pos0_rho%s" % r] = float((d2[i] - d[j]).norm() / (d[j].norm() + 1e-8))
        # ---- evaluation ----
        seqs = [e[0] for e in f["opt_enc"]] + [e[0] for e in f["held_enc"]] + [e[0] for e in f["leak_enc"]] + [tok(NEUTRAL)["input_ids"]]
        anchors = [e[1] for e in f["opt_enc"]] + [e[1] for e in f["held_enc"]] + [e[1] for e in f["leak_enc"]] + [None]
        no, nh, nl = len(f["opt_enc"]), len(f["held_enc"]), len(f["leak_enc"])
        groups = dict(rewrite=[0], opt_para=list(range(1, no)), held=list(range(no, no + nh)),
                      held_cf=list(range(no, no + len(f["held"]) - 1)), leak=list(range(no + nh, no + nh + nl)),
                      neutral=[no + nh + nl])
        names = list(configs)
        MCB = None
        Pb = probs_eval(seqs, anchors, [None])[0]
        MCB = lambda P: metrics(P, Pb, f, groups)
        Ms = probs_eval(seqs, anchors, [configs[n] for n in names[1:]])
        MCB = None
        rec["arms"] = {"A0": metrics(Pb, Pb, f, groups)}
        rec["arms"].update({n: m for n, m in zip(names[1:], Ms)})
        # A3 ICL ceiling: fact prefix + prompt (rewrite, held-out, leak)
        pre = prefix_ids(f)
        icl_seqs = [pre + s for s in seqs[:-1]]
        with torch.no_grad():
            ids, lens = pad(icl_seqs)
            Pi = F.softmax(last_logits(run(EMB(ids), 0, NL, posids(*ids.shape)), lens), -1)
        Pi = torch.cat([Pi, Pb[-1:]], 0)  # neutral unchanged (no prefix) -> KL 0
        rec["arms"]["A3_ICL"] = metrics(Pi, Pb, f, groups)
        rec["info"] = info
        rec["oracle"] = {f"k{k}": oracle_patch(f, k) for k in KS}
        # hash-hit analysis: does the 2/3-gram ending at the subject's last token match any optimisation prompt's?
        hh = {}
        for n in (2, 3):
            optg = {ngram(e[0], e[1], n) for e in f["opt_enc"]}
            hh[f"held_{n}"] = [ngram(e[0], e[1], n) in optg for e in f["held_enc"]]
            hh[f"leak_{n}"] = [ngram(e[0], e[1], n) in optg for e in f["leak_enc"]]
            hh[f"opt_distinct_{n}"] = len(optg)
        rec["hash_hit"] = hh
        prof_cfg = {"base": None, "A1_L2_rho1.0": configs.get("A1_L2_rho1.0"), "A1_L2_rho4.0": configs.get("A1_L2_rho4.0")}
        for k in KS:
            prof_cfg[f"A2_k{k}_rho4.0"] = configs.get(f"A2_k{k}_pos0_rho4.0")
        prof_cfg = {a: b for a, b in prof_cfg.items() if a == "base" or b is not None}
        rec["profile"] = layer_profile(f, prof_cfg)
        per_fact.append(rec)
        a = rec["arms"]
        log(f"fact {fi+1}/{len(facts)} [{f['rel']}] {f['subject']!r}: {f['old']}->{f['new']} ({time.time()-tf:.1f}s) | "
            f"A1L2 rho1 eff {a['A1_L2_rho1.0']['rewrite']['eff']:.0f} held {a['A1_L2_rho1.0']['held']['eff']:.2f} | "
            f"A2k6 rho4 eff {a['A2_k6_pos0_rho4.0']['rewrite']['eff']:.0f} | ICL {a['A3_ICL']['rewrite']['eff']:.0f}")
        json.dump(dict(partial=True, per_fact=per_fact), open(args.out, "w"))

    # ---------------- aggregate ----------------
    def agg(path_fn):
        v = [path_fn(r) for r in per_fact]
        v = [x for x in v if x is not None]
        return (sum(v) / len(v), len(v)) if v else (None, 0)

    arm_names = []
    for r in per_fact:
        for a_ in r["arms"]:
            if a_ not in arm_names:
                arm_names.append(a_)
    table = {}
    for an in arm_names:
        row = {}
        for g in ("rewrite", "opt_para", "held", "held_cf", "leak", "neutral"):
            for m in ("eff", "top1", "pnew", "kl"):
                mu, n = agg(lambda r: r["arms"].get(an, {}).get(g, {}).get(m))
                row[f"{g}.{m}"] = mu
        row["n"] = sum(an in r["arms"] for r in per_fact)
        table[an] = row
    oracle = {}
    for k in KS:
        for mode in ("window", "anchor"):
            oracle[f"k{k}_{mode}"] = {m: agg(lambda r: r["oracle"][f"k{k}"][mode][m])[0] for m in ("eff", "pnew", "top1")}
    prof = {}
    for key in per_fact[0]["profile"]:
        arrs = [r["profile"][key] for r in per_fact if key in r["profile"]]
        prof[key] = [sum(x[i] for x in arrs) / len(arrs) for i in range(len(arrs[0]))]
    hh = {}
    for key in ("held_2", "held_3", "leak_2", "leak_3"):
        vals = [x for r in per_fact for x in r["hash_hit"][key]]
        hh[key] = sum(vals) / len(vals)
    hh["held_cf_2"] = sum(x for r in per_fact for x in r["hash_hit"]["held_2"][:-1]) / sum(len(r["hash_hit"]["held_2"][:-1]) for r in per_fact)
    hh["held_cf_3"] = sum(x for r in per_fact for x in r["hash_hit"]["held_3"][:-1]) / sum(len(r["hash_hit"]["held_3"][:-1]) for r in per_fact)
    hh["opt_distinct_2_mean"] = agg(lambda r: r["hash_hit"]["opt_distinct_2"])[0]
    hh["opt_distinct_3_mean"] = agg(lambda r: r["hash_hit"]["opt_distinct_3"])[0]
    a2loss = {}
    for k in KS:
        for v in ("pos0", "posoff"):
            key = f"A2_k{k}_{v}"
            rs = [r for r in per_fact if key in r["info"]]
            if not rs:
                continue
            # A2 loss does not depend on rho unless the clamp binds; report the LAST rho entry (rho=4.0 / posoff 1.0)
            a2loss[key] = dict(n=len(rs), loss_at_zero=sum(r["info"][key]["loss_hist"][0][-1] for r in rs) / len(rs),
                               loss_final=sum(r["info"][key]["loss_hist"][1][-1] for r in rs) / len(rs),
                               delta_norm=sum(r["info"][key]["delta_norms"][-1] for r in rs) / len(rs))
            for kk in rs[0]["info"][key]:
                if kk.startswith("cos_vs") or kk.startswith("reldiff"):
                    a2loss[key][kk] = sum(r["info"][key][kk] for r in rs) / len(rs)
    a1norm = {}
    for key in ("A1_L2", "A1_L12"):
        a1norm[key] = [sum(r["info"][key]["delta_norms"][i] for r in per_fact) / len(per_fact) for i in range(len(RHOS))]
    anorm = {k: agg(lambda r: r["info"].get(k, {}).get("anchor_norm"))[0] for k in per_fact[0]["info"]}
    results = dict(config=vars(args), sanity_manual_vs_hf=SANITY, n_facts=len(per_fact), gate_rejected=n_gate_fail,
                   relations=dict(collections.Counter(r["rel"] for r in per_fact)), table=table, oracle=oracle,
                   profile=prof, hash_hit=hh, a2_loss=a2loss, a1_delta_norms=a1norm, anchor_norm=anorm, per_fact=per_fact)
    json.dump(results, open(args.out, "w"), indent=1)
    log("wrote", args.out)

    print("\n=== SUMMARY (mean over n=%d facts) ===" % len(per_fact))
    hdr = f"{'arm':28s} {'rw.eff':>6s} {'rw.top1':>7s} {'rw.Pnew':>7s} {'opt.eff':>7s} {'held.eff':>8s} {'heldCF.eff':>10s} {'held.Pnew':>9s} {'leak.KL':>7s} {'leak.top1':>9s} {'neut.KL':>7s}"
    print(hdr)
    for an, row in table.items():
        g = lambda k: (row.get(k) if row.get(k) is not None else float('nan'))
        print(f"{an:28s} {g('rewrite.eff'):6.3f} {g('rewrite.top1'):7.3f} {g('rewrite.pnew'):7.3f} {g('opt_para.eff'):7.3f} {g('held.eff'):8.3f} {g('held_cf.eff'):10.3f} {g('held.pnew'):9.3f} {g('leak.kl'):7.3f} {g('leak.top1'):9.3f} {g('neutral.kl'):7.3f}")
    print("oracle patch:", json.dumps(oracle))
    print("hash hits:", json.dumps(hh))
    print("A2 loss:", json.dumps(a2loss))
    print("anchor norms:", json.dumps(anorm))
    print("A1 delta norms per rho:", json.dumps(a1norm))
    for key, v in prof.items():
        print(key, " ".join(f"{x:.3f}" for x in v))

    # ---------------- decode check of Gemini's graft ----------------
if not args.skip_decode:
    sys.path.insert(0, GEMINI_DIR)
    import engram_graft as eg
    dec = {}
    pin_note = None
    try:
        eng = eg.GraftedEngramModule(d_model=D, device="cpu", dtype=torch.float32, num_buckets=100_000)
    except Exception as ex:
        pin_note = f"constructor raised {type(ex).__name__}: {str(ex)[:200]} -> monkeypatched Tensor.pin_memory=identity"
        log(pin_note)
        _pm = torch.Tensor.pin_memory
        torch.Tensor.pin_memory = lambda self, *a, **k: self
        eng = eg.GraftedEngramModule(d_model=D, device="cpu", dtype=torch.float32, num_buckets=100_000)
    dec["pin_memory_note"] = pin_note
    text = "The quick brown fox jumps over the lazy dog while the Eiffel Tower stands tall in Paris today"
    ids = torch.tensor([tok(text)["input_ids"][:20]])
    assert ids.size(1) == 20

    def decode_diff():
        with torch.no_grad():
            full = model(input_ids=ids).logits[0]
            out = model(input_ids=ids[:, :1], use_cache=True)
            pkv = out.past_key_values
            diffs = [float((out.logits[0, -1] - full[0]).abs().max())]
            inc = [out.logits[0, -1]]
            for t in range(1, 20):
                out = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                diffs.append(float((out.logits[0, -1] - full[t]).abs().max()))
                inc.append(out.logits[0, -1])
            INC[0] = torch.stack(inc)
            return diffs, float(full.abs().max())
    INC = [None]

    dec["no_graft"], _ = decode_diff()
    inc_nograft = INC[0]
    graft = eg.Layer2EngramGraft(model, LAYERS[2], eng)
    dec["graft_v0"], _ = decode_diff()
    torch.manual_seed(1)
    with torch.no_grad():
        eng.v_proj.weight.normal_(0, 0.02)
    dec["graft_v002"], mx = decode_diff()

    def branch_norm():
        with torch.no_grad():
            eng.prefetch_from_host(ids)
            hh = torch.zeros(1, 20, D)
            return float(eng.forward_fusion(hh)[0].norm(dim=-1).mean())
    dec["branch_norm_v002"] = branch_norm()
    with torch.no_grad():
        full_g = model(input_ids=ids).logits[0]
    # edit-scale branch: v_proj std 2.0 -> per-token branch norm comparable to the A1 delta* norms (~4-6)
    with torch.no_grad():
        eng.v_proj.weight.normal_(0, 2.0)
    dec["graft_v2"], _ = decode_diff()
    dec["incr_graft_v2_vs_incr_nograft_max"] = float((INC[0] - inc_nograft).abs().max())
    dec["branch_norm_v2"] = branch_norm()
    with torch.no_grad():
        full_g2 = model(input_ids=ids).logits[0]
    # post-warmup-like conv: all 4 taps active (random), v std 2.0 -> general case (wrong hashes + lost conv history)
    with torch.no_grad():
        saved_conv = eng.short_conv.weight.clone()
        eng.short_conv.weight.normal_(0, 0.5)
    dec["graft_v2_denseconv"], _ = decode_diff()
    dec["branch_norm_v2_denseconv"] = branch_norm()
    with torch.no_grad():
        full_g3 = model(input_ids=ids).logits[0]
        eng.short_conv.weight.copy_(saved_conv)
        graft.pre_hook.remove(); graft.layer_hook.remove()
        full_ng = model(input_ids=ids).logits[0]
    dec["graft_effect_full_forward_max"] = float((full_g - full_ng).abs().max())
    dec["graft_effect_full_forward_max_v2"] = float((full_g2 - full_ng).abs().max())
    dec["graft_effect_full_forward_max_v2_denseconv"] = float((full_g3 - full_ng).abs().max())
    w = eng.short_conv.weight
    dec["dirac_nonzero_entries"] = int((w != 0).sum()); dec["dirac_numel"] = int(w.numel())
    dec["dirac_nonzero_index"] = (w != 0).nonzero().tolist()
    dec["max_abs_logit"] = mx
    # incremental-mode e_t vs full-mode e_t: what does the hasher see when given 1 token?
    full_idx = eng.hasher.hash_sequence(ids)
    step_idx = eng.hasher.hash_sequence(ids[:, 5:6])
    dec["hash_mismatch_pos5_order2"] = bool((full_idx[2][0, 5] != step_idx[2][0, 0]).any())
    dec["hash_mismatch_pos5_order3"] = bool((full_idx[3][0, 5] != step_idx[3][0, 0]).any())
    results["decode"] = dec
    json.dump(results, open(args.out, "w"), indent=1)
    print("DECODE no_graft max|diff| per step:", " ".join(f"{x:.2e}" for x in dec["no_graft"]))
    print("DECODE graft v_proj=0             :", " ".join(f"{x:.2e}" for x in dec["graft_v0"]))
    print("DECODE graft v_proj~N(0,.02)      :", " ".join(f"{x:.2e}" for x in dec["graft_v002"]))
    print("DECODE graft v_proj~N(0,2.0)      :", " ".join(f"{x:.2e}" for x in dec["graft_v2"]))
    print("DECODE graft v~N(0,2), dense conv :", " ".join(f"{x:.2e}" for x in dec["graft_v2_denseconv"]))
    print("DECODE incremental graft(v2, dirac) vs incremental no-graft max|dlogit|:", dec["incr_graft_v2_vs_incr_nograft_max"])
    print("DECODE dirac_ nonzero entries:", dec["dirac_nonzero_entries"], "/", dec["dirac_numel"], dec["dirac_nonzero_index"])
    print("DECODE branch norm (std.02, std2, std2+denseconv):", dec["branch_norm_v002"], dec["branch_norm_v2"], dec["branch_norm_v2_denseconv"])
    print("DECODE graft effect full fwd (dense conv):", dec["graft_effect_full_forward_max_v2_denseconv"])
    print("DECODE graft effect on full fwd max|dlogit| (std.02, std2):", dec["graft_effect_full_forward_max"], dec["graft_effect_full_forward_max_v2"], "max|logit|", mx)
    print("DECODE hash mismatch at pos5 (order2, order3):", dec["hash_mismatch_pos5_order2"], dec["hash_mismatch_pos5_order3"])
    print("DECODE pin note:", pin_note)
log("done")

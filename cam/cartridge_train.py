"""cartridge_train — self-study generation + KL-distillation trainer for a per-edit Cartridge.

Faithful to Cartridges (arXiv:2506.06266):
  * self-study  : the FROZEN base, WITH the edit fact in-context, generates diverse text ABOUT the edit
                  from generic seeds (structuring / summary / question / use-case / creative). This is the
                  self-study curriculum x — no ground-truth labels, the base teaches itself.
  * KL-distill  : teacher = base with the edit as a text prefix in-context; student = base + attached
                  Cartridge Z (NO edit text). Loss = KL(teacher(.|edit ⊕ x[:i]) ‖ student(.|Z, x[:i]))
                  over next-token dists on x. ONLY Z optimizes (AdamW); the base stays inference_mode.

The teacher next-token distributions on each passage are constant (base frozen) -> precomputed once.
"""
import os
import sys

import torch
import torch.nn.functional as F

try:
    from .cartridge import Cartridge
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from cartridge import Cartridge  # noqa: E402


# Generic self-study seeds (Cartridges §curriculum): NO hop-specific supervision — they ask the base to
# re-express / reason about the edit from many angles. {fact} = "<prompt(subject)> <target_new>."
SEED_TEMPLATES = [
    "Restate the following fact in your own words.\nFact: {fact}\nRestatement:",
    "Summarize this fact and note what it implies.\nFact: {fact}\nSummary:",
    "Write a question about the fact below, then answer it.\nFact: {fact}\nQuestion:",
    "Describe a situation where the following fact would matter.\nFact: {fact}\nSituation:",
    "Explain the following fact to a student, step by step.\nFact: {fact}\nExplanation:",
    "List a few things that follow from this fact.\nFact: {fact}\n-",
    "Write a short paragraph that mentions the fact below.\nFact: {fact}\nParagraph:",
    "Q: {fact}\nWhat does this tell us? A:",
]


@torch.no_grad()
def _sample_gen(base, tok, prompt_ids, gen_len, temperature, device, rng):
    """Manual greedy/sampled continuation (use_cache is off on the frozen base). Returns generated ids
    (the continuation only). Stops on a blank-line / EOS."""
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = []
    eos = tok.eos_token_id
    for _ in range(gen_len):
        logits = base(input_ids=cur).logits[:, -1].float()
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            g = torch.Generator(device=probs.device)
            g.manual_seed(int(rng.integers(0, 2**31 - 1)))
            nxt = torch.multinomial(probs, 1, generator=g)[:, 0]
        else:
            nxt = logits.argmax(-1)
        t = int(nxt.item())
        if eos is not None and t == eos:
            break
        out.append(t)
        cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
        dec = tok.decode(out)
        if "\n\n" in dec:                                   # stop at a paragraph break
            break
    return out


@torch.no_grad()
def generate_selfstudy(base, tok, edit_text, device, n_seeds=6, samples_per_seed=2, gen_len=48,
                       temperature=0.8, seed=0, extra_seeds=None):
    """Generate self-study passages x about `edit_text` (the base sees the edit in-context while
    generating). Returns a list of x token-id lists (seed question ⊕ generated answer; the edit itself is
    NOT in x — it is supplied to the teacher during distillation). `extra_seeds` = already-formatted seed
    strings (NO {fact}) appended to the generic curriculum — used by the D1/D2 hop-supervision variants."""
    import numpy as np
    rng = np.random.default_rng(seed)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    fact = edit_text.strip()
    if not fact.endswith("."):
        fact = fact + "."
    passages = []
    seeds = [s.format(fact=fact) for s in SEED_TEMPLATES[:n_seeds]] + list(extra_seeds or [])
    for seed_txt in seeds:
        seed_ids = tok(seed_txt, add_special_tokens=False).input_ids
        # the base generates WITH the edit fact in context, but x records only the seed + answer
        gen_prompt = bos + tok(fact + "\n\n" + seed_txt, add_special_tokens=False).input_ids
        for s in range(samples_per_seed):
            temp = 0.0 if s == 0 else temperature            # 1 greedy + (samples-1) sampled per seed
            cont = _sample_gen(base, tok, gen_prompt, gen_len, temp, device, rng)
            if not cont:
                continue
            passages.append(seed_ids + cont)
    return passages


@torch.no_grad()
def _teacher_logprobs(base, tok, edit_text, x_ids, device, topk=0):
    """Teacher next-token log-probs on x, given the edit as an in-context text prefix. Returns
    (values, indices, surprisal): values/indices = dense log-probs [Lx, V] (indices None) or the
    sparse top-k (values,indices) if topk>0; surprisal = the teacher NLL of each REALIZED token
    x_ids[k] (= -log p_teacher(x_ids[k])), a [Lx] fp32 CPU tensor. High surprisal = the token the base
    found surprising WITH the context in-window (out-of-place facts); low = predictable filler. Used to
    weight the distillation loss so the cartridge spends capacity on the surprising content."""
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    fact = edit_text.strip()
    if not fact.endswith("."):
        fact = fact + "."
    edit_ids = tok(fact + "\n", add_special_tokens=False).input_ids
    prefix = bos + edit_ids
    ids = prefix + list(x_ids)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits = base(input_ids=inp).logits[0]                  # [T,V] (model dtype; DON'T float the whole thing)
    start = len(prefix) - 1
    Lx = len(x_ids)
    sl = logits[start:start + Lx].float()                   # predicts x_ids[0:Lx] — only the scored rows -> fp32
    lp = F.log_softmax(sl, dim=-1)
    # surprisal of the realized token at each position (before any top-k truncation)
    tgt = torch.tensor(list(x_ids), dtype=torch.long, device=lp.device)
    surp = (-lp.gather(1, tgt.view(-1, 1)).squeeze(1)).float().cpu()   # [Lx] NLL, nats
    if topk and topk > 0:
        v, idx = lp.topk(topk, dim=-1)
        return v.half().cpu(), idx.cpu(), surp
    return lp.half().cpu(), None, surp


@torch.no_grad()
def select_high_surprise_context(base, tok, ctx_text, frac=1.0, chunk_tokens=48, device=None,
                                 verbose=False):
    """SURPRISE-BASED CONTEXT SELECTION (Titans' formation-selectivity, applied before self-study):
    score fixed-width token chunks of `ctx_text` by their MEAN teacher-surprisal (causal NLL under the
    frozen base), keep only the top `frac` fraction of chunks (drop the low-surprise, predictable book
    filler), and return the reassembled reduced context text (chunks kept in original order). The bAbI
    supporting facts dropped into book prose are out-of-place -> high surprisal -> retained; the
    repetitive filler -> low surprisal -> dropped. frac>=1.0 returns the text unchanged.

    Returns (reduced_text, info) where info has n_chunks / n_kept / kept token count."""
    if frac is None or frac >= 1.0:
        return ctx_text, {"n_chunks": 0, "n_kept": 0, "selected": False}
    device = device or next(base.parameters()).device
    ids = tok(ctx_text.strip(), add_special_tokens=False).input_ids
    if len(ids) < 2 * chunk_tokens:
        return ctx_text, {"n_chunks": 0, "n_kept": 0, "selected": False}   # too short to prune
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    logits = base(input_ids=inp).logits[0].float()             # [T,V]
    # NLL of token k predicted from position k-1 (causal surprisal, context in-window)
    nll = F.cross_entropy(logits[:-1], inp[0, 1:], reduction="none").cpu()   # [T-1], aligns to ids[1:]
    n = len(ids)
    chunks = []            # (start, end, mean_nll)
    for s in range(0, n, chunk_tokens):
        e = min(s + chunk_tokens, n)
        # nll index j corresponds to ids[j+1]; the tokens of chunk [s,e) are scored by nll rows [s-1, e-1)
        seg = nll[max(s - 1, 0):e - 1]
        score = float(seg.mean()) if len(seg) else 0.0
        chunks.append((s, e, score))
    n_chunks = len(chunks)
    n_keep = max(1, int(round(frac * n_chunks)))
    keep_idx = sorted(sorted(range(n_chunks), key=lambda i: chunks[i][2], reverse=True)[:n_keep])
    kept_ids = []
    for i in keep_idx:
        s, e, _ = chunks[i]
        kept_ids.extend(ids[s:e])
    reduced = tok.decode(kept_ids)
    info = {"n_chunks": n_chunks, "n_kept": n_keep, "kept_tokens": len(kept_ids),
            "orig_tokens": n, "selected": True}
    if verbose:
        print(f"    [surprise-select] {n_chunks} chunks -> kept {n_keep} "
              f"({len(kept_ids)}/{n} tok, frac={frac})", flush=True)
    return reduced, info


def train_cartridge(base, tok, edit_text, p=128, lr=5e-3, epochs=4, max_steps=120, n_seeds=6,
                    samples_per_seed=2, gen_len=48, temperature=0.8, seed=0, topk=0, init_text=True,
                    verbose=False, passages=None, extra_seeds=None,
                    surprise_weight=0.0, surprise_temp=1.0):
    """Train a per-edit Cartridge by KL-distilling base+edit-in-context (teacher) into base+Z (student).
    Returns the trained (DETACHED) Cartridge; the caller attaches it around eval. Only Z trains.
    `passages` (list of x token-id lists) overrides self-study generation (D1 single-hop); `extra_seeds`
    appends hop-specific seeds to the generic curriculum (D2).

    SURPRISE-WEIGHTED DISTILLATION (Titans' formation-selectivity principle): with
    `surprise_weight` (alpha) > 0, each position's KL is weighted by the token's teacher-surprisal
    (NLL under the frozen base with the context in-window), normalized to per-sequence mean 1, then
    mixed as w = (1-alpha) + alpha * w_norm. alpha=0 -> uniform baseline; alpha=1 -> weight fully
    proportional to surprisal (predictable filler down-weighted, out-of-place facts dominate). Weights
    are surprisal**surprise_temp before normalizing (temp>1 sharpens, <1 softens)."""
    import numpy as np
    device = next(base.parameters()).device
    # 1) self-study curriculum (teacher-generated, edit in-context)
    if passages is None:
        passages = generate_selfstudy(base, tok, edit_text, device, n_seeds=n_seeds,
                                      samples_per_seed=samples_per_seed, gen_len=gen_len,
                                      temperature=temperature, seed=seed, extra_seeds=extra_seeds)
    if not passages:
        passages = [tok(edit_text, add_special_tokens=False).input_ids]     # fallback: distill on the edit
    # 2) precompute teacher targets (base frozen -> constant)
    teach = [_teacher_logprobs(base, tok, edit_text, x, device, topk=topk) for x in passages]
    # 3) build + init the cartridge
    cart = Cartridge(base, p=p).to(device)
    if init_text:
        cart.init_from_text(base, tok, edit_text, device=device)
    for pm in base.parameters():
        pm.requires_grad_(False)
    trainable = [q for q in cart.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    cart.attach(base)
    cart.train()
    rng = np.random.default_rng(seed + 1)
    order = list(range(len(passages)))
    step = 0
    last = float("nan")
    for ep in range(epochs):
        rng.shuffle(order)
        for j in order:
            if step >= max_steps:
                break
            x = passages[j]
            Lx = len(x)
            s_in = torch.tensor([bos + list(x)], dtype=torch.long, device=device)
            s_logits = base(input_ids=s_in).logits[0]              # student (cartridge attached) [T,V] model dtype
            # position len(bos)+k-1 predicts x[k]; with no BOS, x[0] has no context -> start at j0.
            j0 = max(0, 1 - len(bos))
            s_sl = s_logits[len(bos) + j0 - 1:len(bos) + Lx - 1].float()   # only scored rows -> fp32 (not full [T,V])
            s_lp = F.log_softmax(s_sl, dim=-1)
            tv, ti, surp = teach[j]                                 # teacher rows aligned to x[0:Lx]
            tv = tv[j0:]
            ti = ti[j0:] if ti is not None else None
            surp = surp[j0:]
            if ti is None:                                         # dense teacher
                t_lp = tv.to(device).float()
                t_p = t_lp.exp()
                kl_pos = (t_p * (t_lp - s_lp)).sum(-1)             # [num_pos] per-position KL
            else:                                                  # sparse top-k teacher
                tvd = tv.to(device).float()
                tid = ti.to(device)
                t_p = tvd.exp()
                s_sel = torch.gather(s_lp, 1, tid)
                kl_pos = (t_p * (tvd - s_sel)).sum(-1)
            if surprise_weight > 0.0 and kl_pos.numel() > 0:
                s = surp.to(device).clamp_min(0.0)
                if surprise_temp != 1.0:
                    s = s.pow(surprise_temp)
                w_norm = s / (s.mean() + 1e-8)                     # per-seq mean 1 (linear in surprisal)
                w = (1.0 - surprise_weight) + surprise_weight * w_norm   # mean 1 mix
                loss = (w * kl_pos).mean()
            else:
                loss = kl_pos.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            last = float(loss.item())
            step += 1
            if verbose and step % 20 == 0:
                print(f"    [cart] step {step}/{max_steps} KL={last:.4f}", flush=True)
        if step >= max_steps:
            break
    cart.eval()
    cart.detach()
    if verbose:
        print(f"    [cart] done: {len(passages)} passages, {step} steps, final KL={last:.4f}", flush=True)
    return cart, {"n_passages": len(passages), "steps": step, "final_kl": last}


@torch.no_grad()
def direct_cloze_delivers(base, tok, cart, cloze_text, target_str, gen_len=10):
    """Single-hop delivery check: free-gen the edit's direct cloze WITH the cartridge attached; does
    `target_str` appear? (Caller uses this both with the cartridge = counterfactual object should appear,
    and without = the base's true object.)"""
    device = next(base.parameters()).device
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + tok(cloze_text, add_special_tokens=False).input_ids
    attached = cart is not None
    if attached:
        cart.attach(base)
        cart.set_active(True)
    try:
        cur = torch.tensor([ids], dtype=torch.long, device=device)
        out = []
        for _ in range(gen_len):
            logits = base(input_ids=cur).logits[:, -1]
            nxt = logits.argmax(-1)
            out.append(int(nxt.item()))
            cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
            dec = tok.decode(out)
            if "\n" in dec:
                break
        gen = tok.decode(out)
    finally:
        if attached:
            cart.detach()
    return " ".join(target_str.split()).lower() in " ".join(gen.split()).lower(), gen

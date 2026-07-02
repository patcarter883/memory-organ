"""CAM v0 — the MAG falsifier. Does an additive ZERO-INIT Memory-as-Gate tap deliver the validated
DeepMemory binding through the FROZEN base, where boltA's Memory-as-Context prefix hit the wall
(memory ≈ no_memory)?  Full spec: titans/V0_SPEC.md.

Stage 1 (binding): train the BoltAdapter by the direct tied-unembed loss (reused from recall_boltA);
                    freeze it. This is the validated 0.86-carry binding.
Stage 2 (delivery): freeze base + memory; train ONLY the GatedMemoryTap(s) by LM-loss-through-the-
                    frozen-base on the recall task. Eval mirrors boltA: local_control / memory /
                    no_memory (NLL bits + accuracy). Default = sweep each tap depth independently.

Run:
  python -m cam.recall_mag --bind-steps 3000 --steps 3000 --tap-layers 8,16,24
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# flat package: make sibling modules importable whether run as `python -m cam.X` or `python cam/X.py`
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from m2_adapter import MODEL, DEV, load_frozen_base                       # noqa: E402
from recall_deepmem import (NAME_CANDIDATES, CARGO_CANDIDATES, MULTITOKEN_WORD_POOL,  # noqa: E402
                            single_token_ids, DocBuilder, counterfactual_single_token,
                            derange_capitals, COUNTERFACTUAL_HEADER, COUNTERFACTUAL_REL)
from recall_boltA import BoltAdapter, eval_direct                         # noqa: E402
from pk_store_adapter import PKStoreAdapter                               # noqa: E402
from gated_tap import MAGInjector                                         # noqa: E402
from realedit import (load_counterfact, as_fact_table, cf_tids_from_records)  # noqa: E402

LN2 = math.log(2.0)
# eval-batch cap: eb = EVAL_BATCH_CAP // (M*Kc), shrinking the eval forward as M/Kc grow so the
# whole-doc ceiling pass fits on a 16GB card. Env-overridable (memory-only, accuracy-neutral): lower
# it (e.g. 48 -> eb=3 at M=8 K=2) if the two-base eval OOMs on ROCm (no expandable_segments here).
EVAL_BATCH_CAP = int(os.environ.get("CAM_EVAL_BATCH_CAP", "128"))


# ---- multi-token answer helpers -----------------------------------------------------------------
# In multi-token mode direct_logits is [B,Kc,vocab] and ans is [B,Kc]; single-token stays [B,vocab]/[B].
# These collapse both shapes so the bind/eval code stays uniform.
def _dlogits(adapter, pref, ans, dec):
    """direct_logits dispatch: pass gold answer ids only to a (teacher-forced AR) decoder readout;
    linear/bolt readouts take prefix only."""
    return adapter.direct_logits(pref, ans) if dec else adapter.direct_logits(pref)


def _seq_ce(logits, ans):
    """teacher-forced CE over the answer token sequence. [B,Kc,V]/[B,Kc] -> scalar; [B,V]/[B] -> scalar."""
    if logits.dim() == 3:
        V = logits.shape[-1]
        return F.cross_entropy(logits.reshape(-1, V), ans.reshape(-1))
    return F.cross_entropy(logits, ans)


def _seq_metrics(logits, ans):
    """-> (exact_match[B] float, per_token_acc[B] float). single-token: both equal the 0/1 hit."""
    pred = logits.argmax(-1)
    if logits.dim() == 3:
        hit = (pred == ans).float()                     # [B,Kc]
        return hit.min(dim=1).values, hit.mean(dim=1)   # all-correct ; fraction-correct
    h = (pred == ans).float()
    return h, h


def _kc(builder):
    """answer length in tokens (1 single-token; Kc multi-token)."""
    return builder.cargo_tokens if getattr(builder, "multitoken", False) else 1


def _answer_logits(base, ctx_emb, Kc):
    """base forward -> the logits predicting the Kc answer tokens. single-token: [B,V] (last position).
    multi-token: [B,Kc,V] (the last Kc logit positions of a teacher-forced context).

    Only the last Kc positions are ever read, so run the LM head on JUST those (logits_to_keep=Kc):
    the full [B,T,vocab] tensor (~1 GB fp32 at V=151936) is the OOM hog on a 16GB card at the tail of
    training — keeping Kc positions collapses it to [B,Kc,vocab] (numerically identical to slicing)."""
    try:
        lg = base(inputs_embeds=ctx_emb, logits_to_keep=Kc).logits.float()  # [B,Kc,V]
    except TypeError:                                    # older HF without logits_to_keep -> slice
        lg = base(inputs_embeds=ctx_emb).logits.float()
    return lg[:, -1] if Kc == 1 else lg[:, -Kc:]


def _last_logit(base, **fwd):
    """last-position logits [B,V] as fp32, running the LM head on ONLY the last position
    (logits_to_keep=1) so the full [B,T,vocab] fp32 tensor never materializes (same OOM hog as
    _answer_logits, at the single-token probe/locality sites). Numerically identical to slicing."""
    try:
        return base(logits_to_keep=1, **fwd).logits[:, -1].float()
    except TypeError:
        return base(**fwd).logits[:, -1].float()


# ---- memory bank: K query-conditioned pooled retrieval vectors (pre out_proj), mirrors BoltAdapter.inject
def memory_bank(adapter, ids, seg_len, qa_start, answer_pos, carry=True):
    """[B,K,mem_dim] — the leak-free memory bank fed to the MAG taps (mem_dim, NOT base-embed space).

    pk-store adapters expose their OWN memory_bank() (the pooled store read, pre out_proj — the
    base-agnostic mem_dim bank); dispatch to it so the Stage-2 tap + v1 translator drive the
    product-key store unchanged. BoltAdapter has no such method -> the DeepMemory path below runs
    byte-identically (the carry/no_memory comparison vs v0 stays exact)."""
    own = getattr(adapter, "memory_bank", None)
    if own is not None:
        return own(ids, seg_len, qa_start, answer_pos, carry=carry)
    emb = adapter._e(ids)                                                 # frozen embed->in_proj->norm
    B = emb.shape[0]
    state = adapter.mem.init_state(B)
    if carry:
        for s in range(0, qa_start, seg_len):
            state = adapter.mem(emb[:, s:s + seg_len], state)             # ingest pre-QA context
    q = emb[:, qa_start:answer_pos]                                       # cargo query (leak-free)
    retrieved = adapter.mem.retrieve(q, state)                           # [B,Lq,mem_dim]
    pq = adapter.readout_q.unsqueeze(0).expand(B, -1, -1)
    attn = torch.softmax(pq @ retrieved.transpose(1, 2) / (adapter.mem_dim ** 0.5), dim=-1)
    return attn @ retrieved                                              # [B,K,mem_dim] pooled


def _leakfree_ctx(base, builder, ids, apos, end=None):
    """header (FORMAT only, no bindings) + query tokens -> base inputs_embeds. Same context boltA used.

    For multi-token TEACHER-FORCING pass end=apos+Kc-1: the context then also includes the first Kc-1
    GOLD answer tokens (ids[:,qa_start:end]) so the base's last Kc logit positions predict the full
    answer sequence ans_0..ans_{Kc-1}. end=None (default) = single-token (context ends at the query)."""
    if end is None:
        end = apos
    hlen = len(builder.bos) + len(builder.header)
    ctx_ids = torch.cat([ids[:, len(builder.bos):hlen], ids[:, builder.qa_start:end]], dim=1)
    return base.get_input_embeddings()(ctx_ids)


# ---- adapter factory: bolt (DeepMemory v0) or pk (product-key store, hub-free) -----------------
def build_adapter(args, embed_weight, H, builder=None):
    """Construct the memory front-end selected by args.store. 'bolt' = BoltAdapter (byte-identical to
    v0). 'pk' = PKStoreAdapter (product-key store) + set_builder(builder) for the bind-block positions,
    with addr-sup active during Stage-1/bind. Both satisfy the same inject/direct_logits/memory_bank
    contract Stage-2 needs."""
    store = getattr(args, "store", "bolt")
    if store == "pk":
        adapter = PKStoreAdapter(
            embed_weight, H, args.mem_dim, args.heads, args.chunk, args.expansion, args.k,
            n_sub=args.n_sub, topk=args.topk, sub_topk=args.sub_topk,
            addr_sup_weight=args.addr_sup_weight,
            read_heads=(args.pk_read_heads if args.pk_read_heads > 0 else None),
            mt_value=getattr(args, "mt_value", "mean"),
            mt_positions=max(2, getattr(args, "cargo_tokens", 1)),
            readout=getattr(args, "readout", "linear"),
            dec_layers=getattr(args, "dec_layers", 2),
            dec_heads=getattr(args, "dec_heads", 4),
            dec_dim=getattr(args, "dec_dim", 256),
            perpos_key=getattr(args, "perpos_key", "additive")).to(DEV)
        assert builder is not None, "pk adapter needs the DocBuilder (bind-block positions)"
        adapter.set_builder(builder)
        return adapter
    return BoltAdapter(embed_weight, H, args.mem_dim, args.heads, args.chunk,
                       args.expansion, args.k).to(DEV)


# ---- stage 1: bind (direct tied-unembed; no base in the loop) ----------------------------------
def bind_adapter(adapter, builder, rng, args):
    train_params = [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(train_params, lr=args.lr)
    for step in range(args.bind_steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        pref = adapter.inject(ids, args.seg_len, builder.qa_start, apos, carry=True)
        # decoder readout is autoregressive + teacher-forced -> needs the gold answer ids; linear/bolt
        # readouts ignore it (their direct_logits takes only prefix).
        _dec = getattr(adapter, "readout", "linear") == "decoder"
        lm = _seq_ce(_dlogits(adapter, pref, ans, _dec), ans)  # CE over answer token SEQUENCE
        # pk-aware addressing-supervision: PKStoreAdapter.aux_loss() returns its weighted write->read
        # InfoNCE term (None when off / not a pk adapter). BoltAdapter has no aux_loss -> untouched.
        aux_fn = getattr(adapter, "aux_loss", None)
        aux = aux_fn() if aux_fn is not None else None
        loss = lm + aux if aux is not None else lm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        if step % 200 == 0 or step == args.bind_steps - 1:
            em, pt = _seq_metrics(_dlogits(adapter, pref, ans, _dec), ans)   # exact-match ; per-token
            astr = f" addr {aux.item():.3f}" if aux is not None else ""
            print(f"[mag] bind step {step:4d} loss {lm.item():.3f}{astr} "
                  f"direct_acc(exact) {em.mean().item():.3f} per_tok {pt.mean().item():.3f}", flush=True)
    d_carry, d_abl, pt_carry, pt_abl = eval_direct(adapter, builder, rng, args)
    print(f"[mag] binding held-out: carry(exact) {d_carry:.3f} | ablated(exact) {d_abl:.3f} | "
          f"carry per-tok {pt_carry:.3f} | ablated per-tok {pt_abl:.3f} | chance {1/args.M:.3f}",
          flush=True)
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    return d_carry


# ---- stage 2: train the MAG tap(s) by LM-loss through the frozen base ---------------------------
def train_taps(base, adapter, injector, builder, rng, args, tag):
    injector.attach().train()
    Kc = _kc(builder)
    opt = torch.optim.AdamW(injector.parameters(), lr=args.lr)
    for step in range(args.steps):
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        with torch.no_grad():
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=True)
        injector.set_bank(bank)                                          # memory frozen -> bank detached
        # multi-token: teacher-force the answer prefix into the context (end=apos+Kc-1) so the last Kc
        # logit positions predict the full answer sequence.
        ctx_emb = _leakfree_ctx(base, builder, ids, apos, end=apos + Kc - 1)
        logits = _answer_logits(base, ctx_emb, Kc)                       # [B,V] or [B,Kc,V]
        loss = _seq_ce(logits, ans)
        if not torch.isfinite(loss):                                     # NaN/Inf guard: skip the step
            print(f"[mag][{tag}] step {step:4d} NON-FINITE loss -> skip", flush=True)
            opt.zero_grad(); continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(injector.parameters()), 1.0)
        if not torch.isfinite(gn):                                       # NaN grad guard
            print(f"[mag][{tag}] step {step:4d} NON-FINITE grad -> skip", flush=True)
            opt.zero_grad(); continue
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            em, pt = _seq_metrics(logits, ans)
            print(f"[mag][{tag}] step {step:4d} loss {loss.item():.3f} exact {em.mean().item():.3f} "
                  f"per_tok {pt.mean().item():.3f} gate {injector.gate_stats()}", flush=True)
    injector.set_bank(None)


def _nll_bits(logits, ans):
    """teacher-forced answer NLL in bits, mean over the answer tokens. [B,V]/[B] or [B,Kc,V]/[B,Kc]."""
    lp = F.log_softmax(logits, -1)
    if logits.dim() == 3:
        b = -lp.gather(-1, ans.unsqueeze(-1)).squeeze(-1) / LN2   # [B,Kc] per-token bits
        return b.mean(dim=1).tolist()                            # mean per-token bits per row
    return (-lp.gather(-1, ans[:, None]).squeeze(-1) / LN2).tolist()


@torch.no_grad()
def eval_generative_mag(base, adapter, injector, builder, rng, args, n=512):
    """Delivery eval through the frozen base. Multi-token: scores TEACHER-FORCED exact-match (all Kc
    answer tokens correct) AND per-token acc; single-token: both equal the 0/1 hit (byte-identical).
    Returns {cond: (nll_bits, exact_match_acc, per_token_acc)}."""
    base_embed = base.get_input_embeddings()
    Kc = _kc(builder)
    # per cond: [nll_list, exact_sum, pertok_sum]
    res = {c: [[], 0.0, 0.0] for c in ("local_control", "memory", "no_memory")}
    injector.eval()
    seen = 0
    while seen < n:
        # eval batch shrinks as M grows: M>=64 docs (qa_seg 9) OOM the full-base ceiling forward at
        # batch 16 on a 16GB card. The CEILING forward processes the WHOLE doc (~bos+header+M*bind_len+qa
        # tokens) at this batch — the memory hog — so cap conservatively for multi-token (the qa_seg-2
        # M=8 K=2 doc + accumulated frag OOM'd at eb=16). Batch size doesn't affect accuracy, only memory.
        eb = max(1, min(args.batch, EVAL_BATCH_CAP // max(1, args.M * Kc)))
        cur = min(eb, n - seen)
        ids, ans, apos = builder.build(rng, cur, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # ceiling: full in-context doc up to apos+Kc-1 (teacher-forced), last Kc logits
        injector.set_bank(None)                                          # tap OFF -> ceiling
        lc = _answer_logits(base, base_embed(ids[:, :apos + Kc - 1]), Kc)
        ctx_emb = _leakfree_ctx(base, builder, ids, apos, end=apos + Kc - 1)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=carry)
            injector.set_bank(bank)
            lg = _answer_logits(base, ctx_emb, Kc)
            res[cond][0].extend(_nll_bits(lg, ans))
            em, pt = _seq_metrics(lg, ans)
            res[cond][1] += em.sum().item(); res[cond][2] += pt.sum().item()
        injector.set_bank(None)
        res["local_control"][0].extend(_nll_bits(lc, ans))
        em, pt = _seq_metrics(lc, ans)
        res["local_control"][1] += em.sum().item(); res["local_control"][2] += pt.sum().item()
        seen += cur
        # heartbeat: the multi-token eval (small eb -> many batches) is otherwise silent for minutes
        # and trips the watchdog's log-idle STALL guard. Print progress so the run stays alive.
        print(f"[mag] eval progress {seen}/{n}", flush=True)
    return {c: (float(np.mean(res[c][0])), res[c][1] / seen, res[c][2] / seen) for c in res}


# ---- checkpoint: persist the frozen v0 memory front-end (BoltAdapter) + a passing GatedMemoryTap ----
# so v1 reuses ONE fixed memory across bases instead of re-binding it each run. The bank fed to the
# taps ([B,K,mem_dim]) is base-AGNOSTIC (DeepMemory's own mem_dim space), so the same checkpoint drives
# any base; only the per-base translator/tap geometry differs.
def save_ckpt(path, adapter, injector, tap_layer, args, d_carry, cf_meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # drop the frozen tied embed/unembed (~3GB) — rebuilt from base-1's table on load
    asd = {k: v for k, v in adapter.state_dict().items()
           if not (k.startswith("embed.") or k == "unembed")}
    torch.save({
        "adapter": asd,
        "taps": injector.taps.state_dict(),
        "tap_layer": tap_layer,
        "tap_heads": args.tap_heads,
        "mem_dim": args.mem_dim, "heads": args.heads, "chunk": args.chunk,
        "expansion": args.expansion, "k": args.k, "d_carry": d_carry,
        # store selector + pk knobs so load_ckpt rebuilds the right adapter (bolt path unchanged:
        # store defaults to 'bolt' and the pk_* keys are ignored when rebuilding a BoltAdapter).
        "store": getattr(args, "store", "bolt"),
        "n_sub": getattr(args, "n_sub", 32), "topk": getattr(args, "topk", 8),
        "sub_topk": getattr(args, "sub_topk", 4),
        "addr_sup_weight": getattr(args, "addr_sup_weight", 0.0),
        "pk_read_heads": getattr(args, "pk_read_heads", 0),
        "phrasing": getattr(args, "phrasing", "dict"),      # doc format (v1 rebuilds the same builder)
        "cargo_tokens": getattr(args, "cargo_tokens", 1),   # multi-token answer length (v1 rebuilds K)
        "mt_value": getattr(args, "mt_value", "mean"),      # multi-token value mode (mean/perpos)
        "readout": getattr(args, "readout", "linear"),      # Stage-1 value readout (linear/decoder)
        "dec_layers": getattr(args, "dec_layers", 2),
        "dec_heads": getattr(args, "dec_heads", 4),
        "dec_dim": getattr(args, "dec_dim", 256),
        "perpos_key": getattr(args, "perpos_key", "additive"),  # per-position key conditioning
        # COUNTERFACTUAL: persist the FILTERED fact table (country/capital WORD strings — tokenizer
        # agnostic) + the derangement index, so recall_v1 rebuilds the EXACT same kept-set + counterfactual
        # mapping on base-2 (intersected with base-2's single-token facts for cross-base index alignment).
        "cf_facts": ([(c, cap) for (c, cap, _ct, _kt) in cf_meta["kept"]] if cf_meta else None),
        "cf_perm": (cf_meta["perm"] if cf_meta else None),
    }, path)
    print(f"[mag] saved v0 memory checkpoint -> {path} (tap L={tap_layer}, carry {d_carry:.3f})", flush=True)


def load_ckpt(path, embed_weight, base, dev, builder=None):
    """Rebuild the frozen memory front-end (BoltAdapter or PKStoreAdapter) + GatedMemoryTap from a
    checkpoint and freeze them. Returns (adapter, injector, tap_layer, meta). For a pk-store ckpt a
    DocBuilder MUST be passed (the store needs the bind-block positions for memory_bank); bolt ckpts
    ignore it (store defaults to 'bolt' for pre-pk checkpoints)."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    H = base.config.get_text_config().hidden_size
    store = ck.get("store", "bolt")
    if store == "pk":
        adapter = PKStoreAdapter(
            embed_weight, H, ck["mem_dim"], ck["heads"], ck["chunk"], ck["expansion"], ck["k"],
            n_sub=ck.get("n_sub", 32), topk=ck.get("topk", 8), sub_topk=ck.get("sub_topk", 4),
            addr_sup_weight=ck.get("addr_sup_weight", 0.0),
            read_heads=(ck.get("pk_read_heads", 0) if ck.get("pk_read_heads", 0) > 0 else None),
            mt_value=ck.get("mt_value", "mean"),
            mt_positions=max(2, ck.get("cargo_tokens", 1)),
            readout=ck.get("readout", "linear"),
            dec_layers=ck.get("dec_layers", 2),
            dec_heads=ck.get("dec_heads", 4),
            dec_dim=ck.get("dec_dim", 256),
            perpos_key=ck.get("perpos_key", "additive")).to(dev)
        assert builder is not None, "pk-store ckpt requires a DocBuilder (set_builder) at load_ckpt"
        adapter.set_builder(builder)
    else:
        adapter = BoltAdapter(embed_weight, H, ck["mem_dim"], ck["heads"], ck["chunk"],
                              ck["expansion"], ck["k"]).to(dev)
    # embed/unembed are not in the ckpt (rebuilt from base-1's table); load the rest strictly-ish
    missing, unexpected = adapter.load_state_dict(ck["adapter"], strict=False)
    assert not unexpected, f"unexpected ckpt keys: {unexpected}"
    assert all(k.startswith("embed.") or k == "unembed" for k in missing), \
        f"unexpected MISSING adapter keys: {missing}"
    for p in adapter.parameters():
        p.requires_grad_(False)
    adapter.eval()
    L = ck["tap_layer"]
    injector = MAGInjector(base, [L], ck["mem_dim"], n_heads=ck["tap_heads"]).to(dev)
    injector.taps.load_state_dict(ck["taps"])
    for p in injector.parameters():
        p.requires_grad_(False)
    injector.eval()
    print(f"[mag] loaded v0 memory checkpoint <- {path} (tap L={L}, carry {ck.get('d_carry', float('nan')):.3f})",
          flush=True)
    return adapter, injector, L, ck


def verdict(tag, d_carry, gen, chance):
    lc = gen["local_control"][1]
    m_acc, nm_acc = gen["memory"][1], gen["no_memory"][1]            # exact-match (==hit single-token)
    m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
    print(f"\n[mag][{tag}] === generative through frozen base (acc=EXACT-MATCH; per_tok shown) ===",
          flush=True)
    print(f"{'condition':>14} {'NLL(bits)':>11} {'exact':>7} {'per_tok':>8}", flush=True)
    for c in ("local_control", "memory", "no_memory"):
        print(f"{c:>14} {gen[c][0]:>11.3f} {gen[c][1]:>7.3f} {gen[c][2]:>8.3f}", flush=True)
    print(f"[mag][{tag}] memory exact {m_acc:.3f} (per_tok {gen['memory'][2]:.3f}) / no_memory "
          f"{nm_acc:.3f} (per_tok {gen['no_memory'][2]:.3f}) / ceiling {lc:.3f}; "
          f"ΔNLL {nm_nll - m_nll:+.3f} bits", flush=True)
    if m_acc > nm_acc + 0.15 and m_acc > 0.5:
        v = "MAG WORKS — greenlight v1 (translator + 2nd base)"
    elif m_acc > nm_acc + 0.10 or (nm_nll - m_nll) > 0.5:
        v = "PARTIAL — go multi-layer / data-dependent gate / unfreeze memory gates"
    else:
        v = "WALL at this depth — escalate to multi-layer; if all depths fail, frozen-base premise is the limit"
    print(f"[mag][{tag}] => {v}\n" + "=" * 64, flush=True)
    return m_acc, nm_acc


# ---- COUNTERFACTUAL knowledge-editing: PROBE -> FILTER -> (derange) -> bind on the known set --------
@torch.no_grad()
def probe_and_filter(base, tok, facts, batch=16):
    """PROBE the FROZEN base (NO memory, tap OFF) on each candidate fact and KEEP ONLY the facts it
    answers correctly parametrically. Prompt = "The capital of <Country> is" (the exact query context the
    counterfactual eval reconstructs), gold = the TRUE capital token. This is the fix that makes the
    counterfactual probe VALID: a prior attempt was invalid because the base did NOT hold the priors it
    was tested on (no_mem prior-acc 0.107); by filtering to demonstrably-known facts, no_mem prior-acc is
    high BY CONSTRUCTION and the override test is meaningful.

    Returns (kept_facts, prior_acc_full) where kept_facts is the filtered [(country,capital,ctid,ktid)]
    list and prior_acc_full is the base's argmax accuracy over the WHOLE candidate table (diagnostic)."""
    prefix = COUNTERFACTUAL_HEADER                                     # "...The capital of"
    rel = COUNTERFACTUAL_REL                                           # " is"
    pref_ids = tok(prefix, add_special_tokens=False).input_ids
    rel_ids = tok(rel, add_special_tokens=False).input_ids
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    kept, correct = [], 0
    done = 0
    while done < len(facts):
        chunk = facts[done:done + batch]
        rows = [bos + pref_ids + [ctid] + rel_ids for (_c, _cap, ctid, _ktid) in chunk]  # ".. <Country> is"
        gold = torch.tensor([ktid for (_c, _cap, _ctid, ktid) in chunk], dtype=torch.long, device=DEV)
        ids = torch.tensor(rows, dtype=torch.long, device=DEV)        # constant length (all single-token)
        logits = _last_logit(base, input_ids=ids)                    # predict the capital token
        pred = logits.argmax(-1)
        for j, f in enumerate(chunk):
            hit = bool(pred[j].item() == gold[j].item())
            correct += int(hit)
            if hit:
                kept.append(f)
        done += len(chunk)
    return kept, correct / max(1, len(facts))


def setup_counterfactual(base, tok, args):
    """PROBE -> FILTER -> DERANGE -> DocBuilder for the counterfactual knowledge-editing run. Returns
    (builder, kept_facts, prior_acc_full). The memory (bind loop) will teach the DERANGED capitals; the
    eval scores both the counterfactual and the prior answer."""
    candidates = counterfactual_single_token(tok)
    print(f"[mag][cf] candidate facts (single-token country+capital): {len(candidates)}", flush=True)
    kept, prior_acc_full = probe_and_filter(base, tok, candidates, batch=args.cf_probe_batch)
    print(f"[mag][cf] PROBE/FILTER: base prior-acc over all {len(candidates)} candidates "
          f"= {prior_acc_full:.3f}", flush=True)
    print(f"[mag][cf] FILTERED-SET SIZE = {len(kept)} facts the base demonstrably knows "
          f"(bind the counterfactual capitals on THESE)", flush=True)
    assert len(kept) >= args.M, \
        f"filtered set ({len(kept)}) < M ({args.M}); the base knows too few facts — lower --M"
    # DERANGE the kept capitals: each country gets a counterfactual capital that is NOT its own true one
    rng_d = np.random.default_rng(args.seed)
    perm = derange_capitals(rng_d, len(kept))
    cf_tid = [kept[perm[i]][3] for i in range(len(kept))]
    builder = DocBuilder(tok, None, None, args.M, args.seg_len, args.qa_seg, phrasing="counterfactual",
                         facts=kept)
    builder.set_counterfactual(cf_tid)
    # log a few edits so the run trace shows what the memory is being asked to override
    ex = ", ".join(f"{kept[i][0]}: {kept[i][1]}->{tok.decode([cf_tid[i]]).strip()}" for i in range(min(4, len(kept))))
    print(f"[mag][cf] example edits (true->counterfactual): {ex}", flush=True)
    return builder, kept, perm, prior_acc_full


# ---- COUNTERFACTUAL on the REAL CounterFact benchmark (Track 1, issue #16) ----------------------
@torch.no_grad()
def probe_and_filter_counterfact(base, tok, records, batch=16):
    """PROBE the frozen base on each CounterFact record using the record's OWN natural prompt
    (requested_rewrite.prompt formatted with the subject) — NOT the fixed curated header. A record is
    VALID iff the base parametrically predicts its target_true token. This is the validity gate: an
    editing claim is only meaningful on a fact the base demonstrably held. Returns
    (kept_records, prior_acc_full)."""
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    kept, correct = [], 0
    done = 0
    # variable prompt length per record -> probe one-at-a-time-batched by identical length is overkill;
    # just run rows of possibly-different length as a python loop over mini-batches of EQUAL length.
    # Simplest correct approach: group by tokenized-prompt length so each forward is rectangular.
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in records:
        p_ids = tok(r.prompt_text, add_special_tokens=False).input_ids
        buckets[len(p_ids)].append((r, p_ids))
    for _plen, items in buckets.items():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            rows = [bos + p_ids for (_r, p_ids) in chunk]
            gold = torch.tensor([r.true_tid for (r, _p) in chunk], dtype=torch.long, device=DEV)
            ids = torch.tensor(rows, dtype=torch.long, device=DEV)
            pred = _last_logit(base, input_ids=ids).argmax(-1)
            for j, (r, _p) in enumerate(chunk):
                hit = bool(pred[j].item() == gold[j].item())
                correct += int(hit)
                if hit:
                    kept.append(r)
    return kept, correct / max(1, len(records))


def setup_counterfact(base, tok, args):
    """Track 1 setup: load REAL CounterFact -> single-token-subject subset -> PROBE/FILTER with each
    record's own prompt -> bind target_new (NO derangement; CounterFact supplies the counterfactual).
    Returns (builder, kept_records, prior_acc_full)."""
    path = os.path.join(args.data_dir, "counterfact.json")
    records, stats = load_counterfact(path, tok, single_token_only=True)
    print(f"[mag][cf] CounterFact <- {path}", flush=True)
    print(f"[mag][cf] survivor accounting: total {stats['total']} | objects-single {stats['objects_single']} "
          f"| subject-single {stats['subject_single']} | ALL-single (tractable) {stats['all_single']}",
          flush=True)
    print(f"[mag][cf] kept {stats['kept']} single-token-subject editable records for the probe", flush=True)
    kept, prior_acc_full = probe_and_filter_counterfact(base, tok, records, batch=args.cf_probe_batch)
    print(f"[mag][cf] PROBE/FILTER (each record's OWN prompt): base prior-acc over {len(records)} "
          f"records = {prior_acc_full:.3f}", flush=True)
    print(f"[mag][cf] FILTERED-SET SIZE = {len(kept)} facts the base demonstrably knows (across all relations)",
          flush=True)
    # Track 1 VALIDITY FIX: the eval elicits the prior via the DocBuilder's header+rel. The old code
    # hard-coded "The capital of <X> is" for EVERY fact, so non-capital facts (mother tongue, plays,
    # located-in, ...) were tested under a nonsense prompt the base can't answer -> no_mem prior-acc
    # collapsed -> gate INVALID. Fix: EDIT ONE RELATION and fold its real prompt template into the
    # header/rel, so filter and eval elicit the SAME (true) relation. Facts of one relation share the
    # template exactly -> subject stays the single-token KEY at qa_start, positions/addr-sup unchanged.
    from collections import defaultdict
    by_rel = defaultdict(list)
    for r in kept:
        by_rel[(r.relation_id, r.prompt)].append(r)
    # candidates: non-empty prefix (subject not at absolute start) + short suffix (fits the QA segment),
    # ranked by how many base-known facts the relation has.
    def _split(prompt):
        pre, _, suf = prompt.partition("{}")
        return pre.rstrip(), suf
    cand = []
    for (rid, prompt), recs in by_rel.items():
        if "{}" not in prompt:
            continue
        pre, suf = _split(prompt)
        if not pre or len(tok(suf, add_special_tokens=False).input_ids) > 6:
            continue
        cand.append((len(recs), rid, prompt, pre, suf, recs))
    dist = sorted(((len(v), k[0]) for k, v in by_rel.items()), reverse=True)[:8]
    print(f"[mag][cf] kept-set relation distribution (top, size:relation): {dist}", flush=True)
    assert cand, "no editable relation group (non-empty prefix + short suffix) in the filtered set"
    n_facts, rid, prompt, prefix, suffix, rel_kept = max(cand, key=lambda c: c[0])
    print(f"[mag][cf] EDITING relation {rid!r} — prompt {prompt!r} ({n_facts} base-known facts); "
          f"header prefix {prefix!r} | rel {suffix!r}", flush=True)
    assert len(rel_kept) >= args.M, \
        f"largest editable relation group ({len(rel_kept)}) < M ({args.M}); lower --M or widen the probe"
    kept = rel_kept
    facts = as_fact_table(kept)                       # (subject, true_str, subject_tid, true_tid)
    cf_tid = cf_tids_from_records(kept)               # target_new tids (parallel) — NO derangement
    builder = DocBuilder(tok, None, None, args.M, args.seg_len, args.qa_seg, phrasing="counterfactual",
                         facts=facts, cf_header_prefix=prefix, cf_rel=suffix)
    builder.set_counterfactual(cf_tid)
    ex = ", ".join(f"{kept[i].subject}: {kept[i].true_str}->{kept[i].new_str}" for i in range(min(4, len(kept))))
    print(f"[mag][cf] example edits (true->target_new): {ex}", flush=True)
    return builder, kept, prior_acc_full


@torch.no_grad()
def eval_locality_generalization(base, tok, injector, adapter, builder, kept, args, cap=256):
    """Track 1 metrics beyond edit-success, using the memory bound on the kept edits.

    LOCALITY: over the kept edits' neighborhood_prompts (other subjects, SAME true object; the base
      knows them, they are NOT bound in memory), score prior-acc (argmax == the record's target_true
      token) with memory ON vs OFF. Success = ON ~= OFF (editing one fact did not corrupt neighbours).
    GENERALIZATION: over the kept edits' paraphrase_prompts (rephrasings of the edited fact), score
      acc against target_new with memory ON. Success = the edit fires on paraphrases too.

    Both probes run the base directly on the natural prompt (no doc segmentation); the memory bank is
    driven from a COUNTERFACTUAL doc for the probed subject so the tap sees the edit's bank. Returns a
    dict of the four numbers + probe counts. cap bounds how many prompts we score (CPU/GPU budget)."""
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    injector.eval()

    def _score(prompts_golds, carry):
        """prompts_golds: list of (prompt_str, gold_tid, subject_tid_for_bank). Returns acc.
        The memory bank is built from a single-fact counterfactual doc keyed on subject_tid so the tap
        carries THAT edit; carry=False resets memory (floor). Batched by equal tokenized length."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for (p, g, subj) in prompts_golds:
            pid = tok(p, add_special_tokens=False).input_ids
            buckets[len(pid)].append((pid, g, subj))
        hit, seen = 0, 0
        # cap the probe batch like the other evals: each row rebuilds an M-binding cf doc for the bank,
        # so memory scales with (batch * M). Shrinks as M grows; overridable via CAM_EVAL_BATCH_CAP.
        eb = max(1, min(args.batch, EVAL_BATCH_CAP // max(1, args.M)))
        for _plen, items in buckets.items():
            for i in range(0, len(items), eb):
                chunk = items[i:i + eb]
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                ids = torch.tensor([bos + pid for (pid, _g, _s) in chunk], dtype=torch.long, device=DEV)
                gold = torch.tensor([g for (_p, g, _s) in chunk], dtype=torch.long, device=DEV)
                # drive the memory bank from a counterfactual doc so the tap is ON with the edits loaded.
                # We reuse a freshly-built cf doc (all M kept facts visible in memory) for this batch; the
                # bank is query-conditioned so it retrieves per the probe subject via the base's own attn.
                cur = ids.shape[0]
                d_ids, _cf, _pr, d_apos = builder.build_cf(np.random.default_rng(args.seed + i), cur)
                d_ids = d_ids.to(DEV)
                bank = memory_bank(adapter, d_ids, args.seg_len, builder.qa_start, d_apos, carry=carry)
                injector.set_bank(bank)
                emb = base.get_input_embeddings()(ids)
                pred = _last_logit(base, inputs_embeds=emb).argmax(-1)
                hit += (pred == gold).sum().item(); seen += cur
        injector.set_bank(None)
        return (hit / max(1, seen)), seen

    # LOCALITY probes: (neighborhood_prompt, target_true_tid). gold = the neighbour's TRUE object,
    # which shares the edited fact's true object (CounterFact construction).
    loc = []
    for r in kept:
        for p in r.neighborhood_prompts:
            loc.append((p, r.true_tid, r.subject_tid))
    loc = loc[:cap]
    # GENERALIZATION probes: (paraphrase_prompt, target_new_tid).
    gen = []
    for r in kept:
        for p in r.paraphrase_prompts:
            gen.append((p, r.new_tid, r.subject_tid))
    gen = gen[:cap]

    loc_on, n_loc = _score(loc, carry=True)
    loc_off, _ = _score(loc, carry=False)
    gen_on, n_gen = _score(gen, carry=True)
    gen_off, _ = _score(gen, carry=False)
    return {"locality_mem_on": loc_on, "locality_mem_off": loc_off, "n_locality": n_loc,
            "generalization_mem_on": gen_on, "generalization_mem_off": gen_off, "n_generalization": n_gen}


@torch.no_grad()
def eval_counterfactual(base, adapter, injector, builder, rng, args, n=512):
    """Knowledge-editing delivery eval. For each held-out doc scores, at the SAME query position, the
    logits under memory-on vs no-memory against BOTH the counterfactual capital (what the memory teaches)
    AND the true prior capital (what the base natively recalls). Returns a dict with the 4 headline
    accuracies + the ceiling and NLLs:
      memory_cf_acc     : mem-on argmax == counterfactual capital  (did the edit take?)
      no_memory_cf_acc  : no-mem argmax == counterfactual capital  (floor — should be ~0)
      no_memory_prior_acc: no-mem argmax == TRUE capital           (VALIDITY gate — must be HIGH)
      memory_prior_acc  : mem-on argmax == TRUE capital            (does delivery SUPPRESS the prior?)
    """
    base_embed = base.get_input_embeddings()
    res = {"memory_cf": 0.0, "no_memory_cf": 0.0, "memory_prior": 0.0, "no_memory_prior": 0.0,
           "ceiling_cf": 0.0, "nll_mem_cf": [], "nll_nomem_cf": [], "nll_nomem_prior": []}
    injector.eval()
    seen = 0
    while seen < n:
        eb = max(1, min(args.batch, EVAL_BATCH_CAP // max(1, args.M)))
        cur = min(eb, n - seen)
        ids, ans_cf, ans_prior, apos = builder.build_cf(rng, cur, local=False)
        ids, ans_cf, ans_prior = ids.to(DEV), ans_cf.to(DEV), ans_prior.to(DEV)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        # ceiling: the full in-context doc (bindings visible) — the base CAN read the counterfactual
        # capital from context (upper bound on delivery through pure attention, tap off).
        injector.set_bank(None)
        lc = _answer_logits(base, base_embed(ids[:, :apos]), 1)
        res["ceiling_cf"] += (lc.argmax(-1) == ans_cf).sum().item()
        # leak-free query context: header ("...The capital of") + "<Country> is" -> predict capital
        ctx_emb = _leakfree_ctx(base, builder, ids, apos)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=carry)
            injector.set_bank(bank)
            lg = _answer_logits(base, ctx_emb, 1)
            cf_hit = (lg.argmax(-1) == ans_cf).sum().item()
            prior_hit = (lg.argmax(-1) == ans_prior).sum().item()
            res[f"{cond}_cf"] += cf_hit
            res[f"{cond}_prior"] += prior_hit
            res[f"nll_{'mem' if carry else 'nomem'}_cf"].extend(_nll_bits(lg, ans_cf))
            if not carry:
                res["nll_nomem_prior"].extend(_nll_bits(lg, ans_prior))
        injector.set_bank(None)
        seen += cur
        print(f"[mag][cf] eval progress {seen}/{n}", flush=True)     # heartbeat vs the watchdog
    out = {
        "memory_cf_acc": res["memory_cf"] / seen,
        "no_memory_cf_acc": res["no_memory_cf"] / seen,
        "memory_prior_acc": res["memory_prior"] / seen,
        "no_memory_prior_acc": res["no_memory_prior"] / seen,
        "ceiling_cf_acc": res["ceiling_cf"] / seen,
        "nll_mem_cf": float(np.mean(res["nll_mem_cf"])),
        "nll_nomem_cf": float(np.mean(res["nll_nomem_cf"])),
        "nll_nomem_prior": float(np.mean(res["nll_nomem_prior"])),
    }
    return out


def verdict_counterfactual(tag, cf, chance, valid_thresh=0.6):
    """Report the 4 knowledge-editing metrics + a VALID/INVALID gate. The probe is VALID iff no_mem
    prior-acc is HIGH (the base actually holds the priors it's tested on — the fix for the earlier
    invalid 0.107 run). Given validity, the EDIT WORKS iff mem-on counterfactual-acc is high and >>
    no_mem counterfactual-acc (the memory overrides the base's own prior)."""
    m_cf = cf["memory_cf_acc"]; nm_cf = cf["no_memory_cf_acc"]
    m_pr = cf["memory_prior_acc"]; nm_pr = cf["no_memory_prior_acc"]
    print(f"\n[mag][{tag}] === COUNTERFACTUAL knowledge-editing (frozen base) ===", flush=True)
    print(f"  (a) mem-on   counterfactual-acc : {m_cf:.3f}   (did the edit take? higher=memory "
          f"overrode the prior)", flush=True)
    print(f"  (b) no_mem   counterfactual-acc : {nm_cf:.3f}   (floor; ~0 expected — base never says "
          f"the wrong capital on its own)", flush=True)
    print(f"  (c) no_mem   PRIOR-acc          : {nm_pr:.3f}   (VALIDITY gate — must be HIGH; the base "
          f"must hold the true priors)", flush=True)
    print(f"  (d) mem-on   PRIOR-acc          : {m_pr:.3f}   (does delivery SUPPRESS the true prior? "
          f"lower=stronger override)", flush=True)
    print(f"  ceiling (in-context cf, tap off): {cf['ceiling_cf_acc']:.3f}   | chance {chance:.3f}",
          flush=True)
    print(f"  NLL bits: mem cf {cf['nll_mem_cf']:.3f} | no_mem cf {cf['nll_nomem_cf']:.3f} | "
          f"no_mem prior {cf['nll_nomem_prior']:.3f}", flush=True)
    valid = nm_pr >= valid_thresh
    if not valid:
        v = (f"INVALID — no_mem prior-acc {nm_pr:.3f} < {valid_thresh:.2f}: the base does NOT reliably "
             f"hold the priors, so any override claim is meaningless. Widen the probe/filter or lower M.")
    elif m_cf > nm_cf + 0.15 and m_cf > 0.5:
        v = (f"VALID + EDIT WORKS — the base holds the priors (no_mem prior {nm_pr:.3f}) AND the memory "
             f"overrides them to the counterfactual (mem cf {m_cf:.3f} >> no_mem cf {nm_cf:.3f}).")
    elif m_cf > nm_cf + 0.10:
        v = (f"VALID + PARTIAL edit — some override (mem cf {m_cf:.3f} vs {nm_cf:.3f}) but weak; "
             f"escalate delivery (multi-layer / stronger tap).")
    else:
        v = (f"VALID but NO override — priors held (no_mem prior {nm_pr:.3f}) yet the memory did NOT "
             f"flip the base to the counterfactual (mem cf {m_cf:.3f} ~ no_mem cf {nm_cf:.3f}). The "
             f"delivery is the bottleneck, not validity.")
    print(f"[mag][{tag}] GATE: {'VALID' if valid else 'INVALID'} | => {v}\n" + "=" * 64, flush=True)
    return m_cf, nm_cf, nm_pr


def verdict_locality_generalization(tag, lg):
    """Track 1 (CounterFact) real-editing metrics report:
      LOCALITY       — neighbour prior-acc with memory ON vs OFF. Success = ON ~= OFF (editing one fact
                       did NOT corrupt unrelated facts that share the same true object).
      GENERALIZATION — paraphrase acc vs target_new with memory ON (vs the OFF floor). Success = the
                       edit fires on rephrasings, not just the exact training string."""
    loc_on, loc_off = lg["locality_mem_on"], lg["locality_mem_off"]
    gen_on, gen_off = lg["generalization_mem_on"], lg["generalization_mem_off"]
    print(f"\n[mag][{tag}] === Track 1 CounterFact LOCALITY + GENERALIZATION ===", flush=True)
    print(f"  LOCALITY (neighbour prior-acc; gold=target_true, NOT bound):", flush=True)
    print(f"    mem OFF {loc_off:.3f}  |  mem ON {loc_on:.3f}   over {lg['n_locality']} probes   "
          f"(success = ON ~= OFF; drop = collateral damage)", flush=True)
    print(f"  GENERALIZATION (paraphrase acc; gold=target_new):", flush=True)
    print(f"    mem OFF {gen_off:.3f}  |  mem ON {gen_on:.3f}   over {lg['n_generalization']} probes   "
          f"(success = ON > OFF; edit fires on rephrasings)", flush=True)
    loc_drop = loc_off - loc_on
    if loc_drop <= 0.05 and gen_on > gen_off + 0.10:
        v = "GENERALIZES + LOCAL — edit fires on paraphrases AND neighbours preserved."
    elif loc_drop <= 0.05:
        v = f"LOCAL but weak generalization (paraphrase lift {gen_on - gen_off:+.3f})."
    elif gen_on > gen_off + 0.10:
        v = f"GENERALIZES but LEAKY (neighbour acc dropped {loc_drop:.3f} — collateral damage)."
    else:
        v = f"WEAK — little generalization and/or locality damage ({loc_drop:.3f})."
    print(f"[mag][{tag}] => {v}\n" + "=" * 64, flush=True)
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bind-steps", type=int, default=3000, dest="bind_steps")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--mem-dim", type=int, default=512, dest="mem_dim")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--expansion", type=float, default=4.0)
    ap.add_argument("--store", type=str, default="bolt", choices=["bolt", "pk"],
                    help="memory mechanism: 'bolt' (DeepMemory v0, default) or 'pk' (product-key "
                         "sparse store, hub-free). pk routes the Stage-2 tap + ckpt through PKStoreAdapter.")
    ap.add_argument("--n-sub", type=int, default=32, dest="n_sub", help="pk: N=n_sub^2 slots")
    ap.add_argument("--topk", type=int, default=8, help="pk: global product-keys kept per query")
    ap.add_argument("--sub-topk", type=int, default=4, dest="sub_topk", help="pk: top-k per half")
    ap.add_argument("--addr-sup-weight", type=float, default=0.0, dest="addr_sup_weight",
                    help="pk: weight of the write->read addressing-supervision (InfoNCE) loss")
    ap.add_argument("--pk-read-heads", type=int, default=0, dest="pk_read_heads",
                    help="pk: override the store's read-head count (0 = use --heads)")
    ap.add_argument("--tap-heads", type=int, default=8, dest="tap_heads")
    ap.add_argument("--tap-layers", type=str, default="", dest="tap_layers",
                    help="comma list of decoder layers to tap; empty -> [n_layers//2]")
    ap.add_argument("--multi", action="store_true",
                    help="train ALL --tap-layers together (escalation) instead of sweeping each")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--save-ckpt", type=str, default="", dest="save_ckpt",
                    help="after a single-layer run, save the frozen BoltAdapter+tap to this path")
    ap.add_argument("--load-ckpt", type=str, default="", dest="load_ckpt",
                    help="reload a saved v0 memory checkpoint instead of re-binding; reproduces V0")
    ap.add_argument("--save-anyway", action="store_true", dest="save_anyway",
                    help="save the ckpt even if the tap didn't pass (smoke-test plumbing only)")
    ap.add_argument("--cargo-tokens", type=int, default=1, dest="cargo_tokens",
                    help="K: multi-token answer cargo phrase length (1=single-token; >1=role-swapped "
                         "'name: <K-token real-word phrase>', answer = the K-token sequence)")
    ap.add_argument("--phrasing", type=str, default="dict",
                    choices=["dict", "natural", "varied", "counterfactual"],
                    help="doc format: 'dict' (terse '<cargo>: <name>', default, byte-preserved) or "
                         "'natural' (natural-language single-relation facts '<Subject> lives in <Object>.'; "
                         "issue #1 realism probe — subject=KEY, object=VALUE, answer=object; supports "
                         "--cargo-tokens K>1 for a K-token real-word object phrase '<Subj> lives in <w0 w1>') "
                         "or 'varied' (per-fact relation drawn from a small template set — heterogeneous "
                         "facts; each binding slot m uses relations[m%R], subject=KEY, object=VALUE) "
                         "or 'counterfactual' (KNOWLEDGE EDITING: real country->capital facts the base "
                         "KNOWS, with DERANGED capitals in memory. PROBE-FILTER-EDIT: probe the frozen "
                         "base first, keep only facts it demonstrably knows, bind the counterfactual "
                         "capitals on that filtered set; metrics = mem/no_mem counterfactual-acc AND "
                         "mem/no_mem PRIOR-acc + a VALID/INVALID gate on no_mem prior-acc)")
    ap.add_argument("--cf-probe-batch", type=int, default=16, dest="cf_probe_batch",
                    help="counterfactual: batch size for the base prior-knowledge probe/filter forward")
    ap.add_argument("--dataset", type=str, default="curated", choices=["curated", "counterfact"],
                    help="counterfactual fact source: 'curated' (the hand-picked country->capital table, "
                         "default, unchanged) or 'counterfact' (REAL ROME CounterFact benchmark — Track 1: "
                         "probe with each record's own prompt, bind target_new, adds LOCALITY + "
                         "GENERALIZATION metrics). Only active with --phrasing counterfactual.")
    ap.add_argument("--data-dir", type=str, default="data", dest="data_dir",
                    help="dir holding counterfact.json (for --dataset counterfact)")
    ap.add_argument("--locality-cap", type=int, default=256, dest="locality_cap",
                    help="max locality/generalization probe prompts to score (budget cap)")
    ap.add_argument("--readout", type=str, default="linear", choices=["linear", "decoder", "perpos"],
                    help="pk multi-token VALUE readout: 'linear' (default, byte-preserved: slot t -> "
                         "answer token t in one projection) or 'decoder' (tiny AR transformer-decoder "
                         "head over the K retrieved store slots; teacher-forced causal decode of the "
                         "answer sequence — gives the value path real sequence capacity) or 'perpos' "
                         "(FACTORIZE the K-token answer into K single-token (name,position) bindings: "
                         "per-position store value slots + per-position read queries + PER-POSITION "
                         "addressing supervision; each slot decoded by the single-token linear readout. "
                         "Shorthand for '--pk-mt-value perpos' with per-position addr-sup.)")
    ap.add_argument("--dec-layers", type=int, default=2, dest="dec_layers", help="decoder readout layers")
    ap.add_argument("--dec-heads", type=int, default=4, dest="dec_heads", help="decoder readout heads")
    ap.add_argument("--dec-dim", type=int, default=256, dest="dec_dim", help="decoder readout model dim")
    ap.add_argument("--pk-mt-value", type=str, default="mean", dest="mt_value", choices=["mean", "perpos"],
                    help="pk multi-token value mode: 'mean' (one store value = mean of the K cargo "
                         "embeds; K-slot head disentangles) or 'perpos' (K position-tagged store "
                         "associations; each answer slot reads its own token)")
    ap.add_argument("--perpos-key", type=str, default="additive", dest="perpos_key",
                    choices=["additive", "gated", "codebook", "disjoint"],
                    help="perpos per-position KEY conditioning (Thrust 1 #3): how answer position t is "
                         "folded into the name key/query. 'additive' (default, byte-preserved): "
                         "key=name+pos_tag[t] (the weak code exp#2 ruled out). 'gated': "
                         "key=name*pos_gate[t]+pos_tag[t] (elementwise per-position scale). 'codebook': "
                         "key=pos_proj[t](name)+pos_tag[t] (per-position LINEAR map ~identity-init, the "
                         "strongest separation — each position addresses a different codebook subspace "
                         "BEFORE the product-key half-split). 'disjoint' (Thrust 1 #4): each position "
                         "owns an ENTIRELY SEPARATE ProductKeyStore (own codebook/heads/bank); position t "
                         "reads/writes/addr-sups against store t ONLY, so the per-position address cannot "
                         "be contaminated by the other position's slots (key = L2(name)+pos_tag[t]). "
                         "Only active under perpos.")
    args = ap.parse_args()

    # '--readout perpos' is shorthand: it selects the per-position VALUE STORE (mt_value=perpos) +
    # per-position addressing supervision, decoded by the single-token LINEAR readout (slot t @ unembed).
    # Normalize to the underlying (mt_value, readout=linear) the adapter/ckpt understand.
    if args.readout == "perpos":
        args.mt_value = "perpos"
        args.readout = "linear"

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base, tok = load_frozen_base()
    H = base.config.get_text_config().hidden_size
    n_layers = base.config.get_text_config().num_hidden_layers
    embed_weight = base.get_input_embeddings().weight.detach().float().clone()

    names = single_token_ids(tok, NAME_CANDIDATES)
    # natural/varied phrasing places the object MID-SENTENCE (space-prefixed single token); dict places
    # cargo line-initial (NO-space). Pick the object/cargo pool encoding to match the phrasing.
    cargo_prefix = " " if args.phrasing in ("natural", "varied") else ""
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix=cargo_prefix)
    # multi-token object words are drawn from MULTITOKEN_WORD_POOL, space-prefixed (the natural OBJECT is
    # mid-sentence, same as the DICT multi-token cargo phrase — both use the space-prefixed pool).
    cargo_words = single_token_ids(tok, MULTITOKEN_WORD_POOL) if args.cargo_tokens > 1 else None
    assert not (args.phrasing == "varied" and args.cargo_tokens > 1), \
        "varied phrasing is single-token only (no multi-token cargo)"
    counterfactual = args.phrasing == "counterfactual"
    cf_meta = None
    cf_records = None                    # Track 1 (CounterFact) records for locality/generalization eval
    if counterfactual:
        assert args.cargo_tokens == 1, "counterfactual phrasing is single-token only"
        # PROBE the frozen base -> FILTER to demonstrably-known facts -> DERANGE the memory capitals.
        # (this is the GPU probe forward the orchestrator runs; the filtered set size + example edits
        # are logged above the bind loop.)
        if args.dataset == "counterfact":
            # Track 1: REAL CounterFact benchmark (probe with each record's own prompt, bind target_new).
            builder, cf_records, prior_acc_full = setup_counterfact(base, tok, args)
            cf_meta = {"kept": [(r.subject, r.true_str) for r in cf_records], "perm": None}
        else:
            builder, kept_facts, cf_perm, prior_acc_full = setup_counterfactual(base, tok, args)
            cf_meta = {"kept": kept_facts, "perm": cf_perm}  # persisted in the ckpt for v1 transfer
            cf_records = None
    else:
        builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing=args.phrasing,
                             cargo_tokens=args.cargo_tokens, cargo_words=cargo_words)
    if args.cargo_tokens > 1:
        print(f"[mag] MULTI-TOKEN cargo: K={args.cargo_tokens} word_pool={len(cargo_words)} "
              f"(answer = K-token real-word phrase; acc = exact-match)", flush=True)

    # ---- RELOAD path: reuse a fixed v0 memory checkpoint (no re-bind) and reproduce the V0 eval ----
    if args.load_ckpt:
        adapter, injector, L, ck = load_ckpt(args.load_ckpt, embed_weight, base, DEV, builder=builder)
        print(f"[mag] {MODEL} | H={H} n_layers={n_layers} | RELOAD tap L={L} | "
              f"K={ck['k']} mem_dim={ck['mem_dim']} | chance acc={1/args.M:.3f}", flush=True)
        injector.attach()
        gen = eval_generative_mag(base, adapter, injector, builder, rng, args)
        m_acc, nm_acc = verdict(str(L), ck.get("d_carry", float("nan")), gen, 1 / args.M)
        if counterfactual:
            cf = eval_counterfactual(base, adapter, injector, builder, rng, args)
            verdict_counterfactual(str(L), cf, 1 / args.M)
            if cf_records is not None:
                lg = eval_locality_generalization(base, tok, injector, adapter, builder, cf_records,
                                                  args, cap=args.locality_cap)
                verdict_locality_generalization(str(L), lg)
        injector.detach()
        print("\n[mag] RELOAD SANITY (tap -> memory / no_memory / ceiling):", flush=True)
        print(f"  L={L:>8}  {m_acc:.3f} / {nm_acc:.3f} / {gen['local_control'][1]:.3f}", flush=True)
        print("[mag] PASS if memory ≫ no_memory (reproduces V0 on base-1 from the saved memory).", flush=True)
        return

    layers = ([int(x) for x in args.tap_layers.split(",") if x != ""]
              if args.tap_layers else [n_layers // 2])
    print(f"[mag] {MODEL} | H={H} n_layers={n_layers} | tap_layers={layers} multi={args.multi} | "
          f"K={args.k} mem_dim={args.mem_dim} | chance acc={1/args.M:.3f}", flush=True)

    # ---- stage 1: bind once ----
    adapter = build_adapter(args, embed_weight, H, builder=builder)
    d_carry = bind_adapter(adapter, builder, rng, args)

    # ---- stage 2: MAG delivery ----
    configs = [layers] if args.multi else [[L] for L in layers]
    summary = []
    for cfg in configs:
        tag = "+".join(map(str, cfg))
        injector = MAGInjector(base, cfg, args.mem_dim, n_heads=args.tap_heads).to(DEV)
        train_taps(base, adapter, injector, builder, rng, args, tag)
        gen = eval_generative_mag(base, adapter, injector, builder, rng, args)
        m_acc, nm_acc = verdict(tag, d_carry, gen, 1 / args.M)
        if counterfactual:
            # the 4 knowledge-editing metrics + VALID/INVALID gate (scores mem/no_mem against BOTH the
            # counterfactual capital and the true prior at the SAME query position).
            cf = eval_counterfactual(base, adapter, injector, builder, rng, args)
            verdict_counterfactual(tag, cf, 1 / args.M)
            if cf_records is not None:
                # Track 1 real-editing metrics: LOCALITY (neighbours preserved) + GENERALIZATION (edit
                # fires on paraphrases). Only for --dataset counterfact (records carry the probe prompts).
                lg = eval_locality_generalization(base, tok, injector, adapter, builder, cf_records,
                                                  args, cap=args.locality_cap)
                verdict_locality_generalization(tag, lg)
        summary.append((tag, m_acc, nm_acc, gen["local_control"][1]))
        # save the FIRST passing single-layer tap as the reusable v0 memory checkpoint
        if args.save_ckpt and len(cfg) == 1 and (args.save_anyway or (m_acc > nm_acc + 0.15 and m_acc > 0.5)):
            save_ckpt(args.save_ckpt, adapter, injector, cfg[0], args, d_carry, cf_meta=cf_meta)
            args.save_ckpt = ""  # save only once (the first passing depth)
        injector.detach()

    print("\n[mag] SUMMARY (tap -> memory / no_memory / ceiling):", flush=True)
    for tag, m, nm, lc in summary:
        print(f"  L={tag:>8}  {m:.3f} / {nm:.3f} / {lc:.3f}", flush=True)
    print(f"[mag] boltA reference (MAC): memory ≈ no_memory ≈ 0.000 (the wall this run tests against).",
          flush=True)


if __name__ == "__main__":
    main()

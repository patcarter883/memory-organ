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

# flat package: sibling imports resolve relatively when imported as cam.X (`python -m cam.X`,
# `import cam.X`) and fall back to a path-hacked absolute import when run as a file (`python cam/X.py`).
try:
    from .m2_adapter import MODEL, DEV, StageCost, load_frozen_base
    from .recall_deepmem import (NAME_CANDIDATES, CARGO_CANDIDATES, MULTITOKEN_WORD_POOL,
                                 single_token_ids, DocBuilder, counterfactual_single_token,
                                 derange_capitals, COUNTERFACTUAL_HEADER, COUNTERFACTUAL_REL)
    from .recall_boltA import BoltAdapter, eval_direct
    from .pk_store_adapter import PKStoreAdapter
    from .gated_tap import MAGInjector
    from .realedit import load_counterfact, as_fact_table, cf_tids_from_records
except ImportError:
    if __package__:  # real ImportError inside a sibling, not "run as a file" — don't mask it
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import MODEL, DEV, StageCost, load_frozen_base       # noqa: E402
    from recall_deepmem import (NAME_CANDIDATES, CARGO_CANDIDATES, MULTITOKEN_WORD_POOL,  # noqa: E402
                                single_token_ids, DocBuilder, counterfactual_single_token,
                                derange_capitals, COUNTERFACTUAL_HEADER, COUNTERFACTUAL_REL)
    from recall_boltA import BoltAdapter, eval_direct                     # noqa: E402
    from pk_store_adapter import PKStoreAdapter                           # noqa: E402
    from gated_tap import MAGInjector                                     # noqa: E402
    from realedit import load_counterfact, as_fact_table, cf_tids_from_records  # noqa: E402

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
    lg = lg.to(DEV)                                      # model-parallel: logits (last shard) -> cuda:0 (targets)
    return lg[:, -1] if Kc == 1 else lg[:, -Kc:]


def _last_logit(base, **fwd):
    """last-position logits [B,V] as fp32, running the LM head on ONLY the last position
    (logits_to_keep=1) so the full [B,T,vocab] fp32 tensor never materializes (same OOM hog as
    _answer_logits, at the single-token probe/locality sites). Numerically identical to slicing."""
    try:
        return base(logits_to_keep=1, **fwd).logits[:, -1].float().to(DEV)
    except TypeError:
        return base(**fwd).logits[:, -1].float().to(DEV)


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


def _set_bank(injector, adapter, bank):
    """Set the tap bank AND forward the adapter's per-example store-confidence scalar (pk-store only;
    None for bolt) + the queried relation index (for the per-relation conf-gate EMA; None outside
    multi-relation). The tap uses conf/relidx only when its confidence gate is enabled."""
    injector.set_bank(bank, conf=getattr(adapter, "_last_conf", None),
                      relidx=getattr(adapter, "_last_relidx", None))


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
def _loc_neg_batch(loc_buckets, rng, batch):
    """Sample a same-length batch of negative (out-of-store) probe ids from the length-bucketed pool.
    Returns [b, L] long on DEV (or None if empty)."""
    lens = [L for L, v in loc_buckets.items() if v]
    if not lens:
        return None
    L = lens[int(rng.integers(0, len(lens)))]
    items = loc_buckets[L]
    idx = rng.choice(len(items), size=min(batch, len(items)), replace=False)
    return torch.tensor([items[i] for i in idx], dtype=torch.long, device=DEV)


def train_taps(base, adapter, injector, builder, rng, args, tag, loc_buckets=None):
    injector.attach().train()
    Kc = _kc(builder)
    lw = getattr(args, "locality_weight", 0.0)
    opt = torch.optim.AdamW(injector.parameters(), lr=args.lr)
    _timing = os.environ.get("CAM_STEP_TIMING") == "1"                  # per-phase step profiler (diag)
    import time as _time

    def _t(sync=True):
        if _timing and sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        return _time.perf_counter()
    for step in range(args.steps):
        _p = {}
        _s = _t(sync=False)
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        _p["build"] = _t() - _s; _s = _t(sync=False)
        with torch.no_grad():
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=True)
        _set_bank(injector, adapter, bank)                              # memory frozen -> bank detached
        _p["bank"] = _t() - _s; _s = _t(sync=False)
        # multi-token: teacher-force the answer prefix into the context (end=apos+Kc-1) so the last Kc
        # logit positions predict the full answer sequence.
        ctx_emb = _leakfree_ctx(base, builder, ids, apos, end=apos + Kc - 1)
        logits = _answer_logits(base, ctx_emb, Kc)                       # [B,V] or [B,Kc,V]
        edit_loss = _seq_ce(logits, ans)
        _p["fwd"] = _t() - _s; _s = _t(sync=False)
        if not torch.isfinite(edit_loss):                                # NaN/Inf guard: skip the step
            print(f"[mag][{tag}] step {step:4d} NON-FINITE edit loss -> skip", flush=True)
            opt.zero_grad(); continue
        edit_loss.backward()                                            # backprop + FREE the edit graph
        _p["bwd"] = _t() - _s; _s = _t(sync=False)
        # RETRIEVAL-STRENGTH LOCALITY: teach the tap to gate on the STORE READ, not the prompt. Build a
        # negative with the SAME prompt type as the positive (an edited-subject query) but with the edit
        # NOT bound in the doc -> the episodic store read is WEAK. Match tap-on to the frozen base tap-off
        # (KL) so the tap injects nothing when the store lacks the edit. Since positive (strong bank) and
        # negative (weak bank) share the prompt distribution and differ ONLY in retrieval strength, the tap
        # learns strength-gating -> at eval it DELIVERS paraphrases (their edit IS retrievable) yet stays
        # inert on neighbours (not retrievable). Backprop separately so one graph is alive at a time.
        loc_val = 0.0
        neg_null = float("nan")
        if lw > 0 and getattr(builder, "facts", None):
            nfac = len(builder.facts)
            if getattr(builder, "phrasing", None) == "counterfactual_multi":
                # build_cf_query needs one relation per call -> draw this step's negatives from ONE relation
                rid = builder.rel_order[int(rng.integers(0, builder.R))]
                pool = builder.rel_groups[rid]
                tgt = [int(pool[int(rng.integers(0, len(pool)))]) for _ in range(args.batch)]
            else:
                tgt = [int(t) for t in rng.integers(0, nfac, size=args.batch)]
            neg_ids, neg_apos = builder.build_cf_query(rng, tgt, args.batch, bind_target=False)  # weak bank
            neg_ids = neg_ids.to(DEV)
            neg_ctx = _leakfree_ctx(base, builder, neg_ids, neg_apos)     # header + "<subject> is"
            with torch.no_grad():
                neg_bank = memory_bank(adapter, neg_ids, args.seg_len, builder.qa_start, neg_apos, carry=True)
                injector.set_bank(None)
                off = _answer_logits(base, neg_ctx, 1)                    # tap OFF -> the base's true prior
            _set_bank(injector, adapter, neg_bank)                        # weak bank + its (low) confidence
            on = _answer_logits(base, neg_ctx, 1)                         # tap ON (weak bank), differentiable
            loc_loss = F.kl_div(F.log_softmax(on, -1), F.softmax(off, -1), reduction="batchmean")
            if torch.isfinite(loc_loss):
                (lw * loc_loss).backward()                               # accumulate grad; FREE the loc graph
                loc_val = float(loc_loss.detach())                       # scalar read post-backward (no grad warning)
            neg_null = float(np.mean(list(injector.null_attn_stats().values())))
        _p["loc"] = _t() - _s; _s = _t(sync=False)
        gn = torch.nn.utils.clip_grad_norm_(list(injector.parameters()), 1.0)
        if not torch.isfinite(gn):                                       # NaN grad guard
            print(f"[mag][{tag}] step {step:4d} NON-FINITE grad -> skip", flush=True)
            opt.zero_grad(); continue
        opt.step()
        if _timing:
            _p["opt"] = _t() - _s
            tot = sum(_p.values())
            print(f"[mag][{tag}][timing] step {step:3d} total {tot:.2f}s | "
                  + " ".join(f"{k}={v:.2f}" for k, v in _p.items()), flush=True)
        if step % 200 == 0 or step == args.steps - 1:
            em, pt = _seq_metrics(logits, ans)
            extra = (f" loc_kl {loc_val:.3f} neg_null {neg_null:.3f}" if lw > 0 else "")
            # confidence gate: cgate_pos = c on the STRONG (positive) bank just delivered — want ->1;
            # neg_cgate = c on the WEAK negative bank (last set) — want ->0. The spread is the gate working.
            if getattr(args, "conf_gate", False):
                neg_cg = float(np.mean(list(injector.cgate_stats().values())))
                extra += f" neg_cgate {neg_cg:.3f}"
            print(f"[mag][{tag}] step {step:4d} loss {edit_loss.item():.3f} exact {em.mean().item():.3f} "
                  f"per_tok {pt.mean().item():.3f} gate {injector.gate_stats()}{extra}", flush=True)
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
        # ceiling: full in-context doc up to apos+Kc-1 (teacher-forced), last Kc logits. The whole-doc
        # forward is the eval's memory hog; CAM_SKIP_CEILING=1 skips it (a diagnostic) so the eval fits at
        # high M on a 16GB card.
        injector.set_bank(None)                                          # tap OFF -> ceiling
        skip_ceiling = os.environ.get("CAM_SKIP_CEILING") == "1"
        lc = None if skip_ceiling else _answer_logits(base, base_embed(ids[:, :apos + Kc - 1]), Kc)
        ctx_emb = _leakfree_ctx(base, builder, ids, apos, end=apos + Kc - 1)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=carry)
            _set_bank(injector, adapter, bank)
            lg = _answer_logits(base, ctx_emb, Kc)
            res[cond][0].extend(_nll_bits(lg, ans))
            em, pt = _seq_metrics(lg, ans)
            res[cond][1] += em.sum().item(); res[cond][2] += pt.sum().item()
        injector.set_bank(None)
        if not skip_ceiling:
            res["local_control"][0].extend(_nll_bits(lc, ans))
            em, pt = _seq_metrics(lc, ans)
            res["local_control"][1] += em.sum().item(); res["local_control"][2] += pt.sum().item()
        seen += cur
        # heartbeat: the multi-token eval (small eb -> many batches) is otherwise silent for minutes
        # and trips the watchdog's log-idle STALL guard. Print progress so the run stays alive.
        print(f"[mag] eval progress {seen}/{n}", flush=True)
    return {c: (float(np.mean(res[c][0])) if res[c][0] else float("nan"),
                res[c][1] / seen, res[c][2] / seen) for c in res}


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
        "conf_gate": getattr(args, "conf_gate", False),     # store-confidence gate (retrieval-strength delivery)
        "n_rel": getattr(injector.taps[str(tap_layer)], "n_rel", 1),  # per-relation conf-gate EMA count
        "mem_dim": args.mem_dim, "heads": args.heads, "chunk": args.chunk,
        "expansion": args.expansion, "k": args.k, "d_carry": d_carry,
        # donor id + embed-table shape: loaders rebuild the SAME base-1 the memory was bound on and
        # can VERIFY it — a same-hidden-size donor swap is otherwise invisible to load_state_dict
        # (embed/unembed are excluded from the ckpt). Absent in pre-flag checkpoints -> loaders
        # fall back to the historical default (MODEL) and skip the shape check.
        "base1": args.base1,
        "embed_shape": tuple(adapter.embed.weight.shape),
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
    # donor-mismatch guard: embed/unembed are rebuilt from the SUPPLIED table (never checked by the
    # strict-ish load below), so a wrong-donor embed_weight with matching hidden size would silently
    # rebuild a garbage adapter. New ckpts record the bound table's shape; verify when present.
    if ck.get("embed_shape") is not None:
        got, want = tuple(embed_weight.shape), tuple(ck["embed_shape"])
        assert got == want, \
            (f"embed table {got} != ckpt-recorded donor table {want} — this memory was bound on "
             f"{ck.get('base1', 'the historical default donor')}; pass the matching --base1.")
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
    injector = MAGInjector(base, [L], ck["mem_dim"], n_heads=ck["tap_heads"],
                           conf_gate=ck.get("conf_gate", False), n_rel=ck.get("n_rel", 1)).to(dev)
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
@torch.no_grad()
def apitask_probe(base, tok, path):
    """API-OVERRIDE PREMISE GATE (issue #19 use case). Before building the full edit+eval on library-API
    facts, ask the cheap decisive question: is the frozen base actually CONFIDENTLY WRONG about library APIs
    (so there is something to fix)? If the base already gets APIs right, the override use case is moot.

    Loads a curated API-facts JSON in CounterFact schema (target_true.str = the CORRECT current API answer),
    forwards the base on each record's natural prompt, and logs the base's top-1 prediction vs correct
    (first-token match). Low base accuracy = many editable candidates = the use case is real. No editing."""
    records, _stats = load_counterfact(path, tok, single_token_only=False)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    print(f"\n[apitask] === API-OVERRIDE PREMISE PROBE ({len(records)} curated facts, base={MODEL}) ===",
          flush=True)
    hit = 0
    for r in records:
        ids = torch.tensor([bos + tok(r.prompt_text, add_special_tokens=False).input_ids],
                           dtype=torch.long, device=DEV)
        pred = int(_last_logit(base, input_ids=ids).argmax(-1).item())
        gold = r.true_ids[0] if r.true_ids else r.true_tid
        ok = int(pred == gold)
        hit += ok
        print(f"  [{'OK   ' if ok else 'WRONG'}] {r.prompt_text!r}  correct={r.true_str!r}  "
              f"base->{tok.decode([pred])!r}", flush=True)
    n = max(1, len(records))
    print(f"  --> base ALREADY CORRECT on {hit}/{len(records)} ({hit/n:.2f}); "
          f"WRONG = editable candidates {len(records) - hit}/{len(records)} ({1 - hit/n:.2f})", flush=True)
    print("  (high WRONG rate => the base is stale on APIs => the override use case is real)", flush=True)
    print("=" * 64, flush=True)


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
            # base-known gate: base predicts the true object's FIRST token (Phase M proxy for multi-token
            # objects; identical to true_tid for single-token). Full-sequence base-known is Phase M1.
            gold = torch.tensor([(r.true_ids[0] if r.true_ids else r.true_tid) for (r, _p) in chunk],
                                dtype=torch.long, device=DEV)
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


def setup_counterfact_multi(base, tok, args):
    """Track 1 MULTI-RELATION setup (#16): probe/filter the base, then keep the top-N base-known relations
    and edit them TOGETHER in one memory. MULTI-TOKEN SUBJECTS: the single-token-subject filter drops ~96%
    of CounterFact and skews to a few relations (P37/P36), so here we keep single-token OBJECTS but allow
    MULTI-TOKEN subjects (the store keys on the subject's LAST token). To keep batches rectangular, each
    relation fixes ONE subject length (the length with the most base-known facts). One relation per
    relation_id (semantic diversity). Returns (builder, kept, prior_acc)."""
    path = os.path.join(args.data_dir, "counterfact.json")
    records, stats = load_counterfact(path, tok, single_token_only=False)
    # Phase M (M0): allow objects up to CAM_MAX_OBJ_TOK tokens (default 1 = legacy single-token behaviour).
    # Lifting this is the productionization gate (§3.18/§6f): it unlocks the ~96% of CounterFact facts the
    # single-token filter drops. The base-known probe uses the first-token proxy (probe_and_filter_counterfact).
    K = max(1, int(os.environ.get("CAM_MAX_OBJ_TOK", "1")))
    obj = [r for r in records if 1 <= len(r.true_ids) <= K and 1 <= len(r.new_ids) <= K]
    n_multi = sum(1 for r in records if len(r.true_ids) > 1 or len(r.new_ids) > 1)
    print(f"[mag][cf-multi] CounterFact <- {path} | MAX_OBJ_TOK={K} -> {len(obj)} candidate facts of "
          f"{stats['total']} (objects-single {stats['objects_single']}; multi-token-object supply {n_multi})",
          flush=True)
    # cap the probe pool (probing ~20k is expensive) — deterministic sample surfaces many relations.
    rng = np.random.default_rng(args.seed)
    cap = min(len(obj), getattr(args, "cf_probe_cap", 8000))
    pool = [obj[i] for i in rng.permutation(len(obj))[:cap]]
    # PROBE CACHE: forwarding `cap` (~21k) records through the frozen base to find base-known facts is the
    # single biggest recurring GPU cost (minutes at low occupancy) and is DETERMINISTIC in
    # (base, seed, cap, dataset) -> memoize `kept`. Disable with CAM_PROBE_CACHE=0; dir via
    # CAM_PROBE_CACHE_DIR (the data mount is often read-only).
    kept = prior_acc_full = None
    model_id = getattr(base.config, "_name_or_path", "") or getattr(args, "base1", "") or MODEL
    cache_on = os.environ.get("CAM_PROBE_CACHE", "1") != "0"
    cache_dir = os.environ.get("CAM_PROBE_CACHE_DIR") or os.path.join(args.data_dir, "probe_cache")
    import hashlib as _hl, pickle as _pk
    key = _hl.sha1(f"{model_id}|{args.seed}|{cap}|{len(obj)}|K{K}".encode()).hexdigest()[:16]
    cpath = os.path.join(cache_dir, f"cfmulti_{key}.pkl")
    if cache_on and os.path.exists(cpath):
        try:
            with open(cpath, "rb") as _f:
                _blob = _pk.load(_f)
            kept, prior_acc_full = _blob["kept"], _blob["prior_acc"]
            print(f"[mag][cf-multi] PROBE CACHE HIT {cpath} | {len(kept)} known facts "
                  f"(skipped the {cap}-record probe)", flush=True)
        except Exception as _e:  # noqa
            print(f"[mag][cf-multi] probe cache read failed ({_e}); re-probing", flush=True)
            kept = None
    if kept is None:
        kept, prior_acc_full = probe_and_filter_counterfact(base, tok, pool, batch=args.cf_probe_batch)
        print(f"[mag][cf-multi] PROBE/FILTER base prior-acc = {prior_acc_full:.3f} | {len(kept)} known facts "
              f"(of {cap} probed)", flush=True)
        if cache_on:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cpath, "wb") as _f:
                    _pk.dump({"kept": kept, "prior_acc": prior_acc_full, "model": model_id,
                              "seed": args.seed, "cap": cap}, _f)
                print(f"[mag][cf-multi] probe cached -> {cpath}", flush=True)
            except Exception as _e:  # noqa
                print(f"[mag][cf-multi] probe cache write failed ({_e})", flush=True)
    from collections import defaultdict

    def _split(prompt):
        pre, _, suf = prompt.partition("{}")
        return pre.rstrip(), suf
    # rid -> (prompt, subject_len) -> [recs]; pick, per relation_id, the (prompt,len) with the most facts.
    # Phase N (N0b): the relation prompt/suffix filter is what caps the fact pool at ~10 relations (§6g N0
    # result). CAM_MAX_SUFFIX_TOK raises the suffix-length cap (legacy 6); CAM_ALLOW_EMPTY_PREFIX admits
    # sentence-initial-subject prompts (pre==""). Both default to legacy. The store keys on the subject
    # (last-token/pooled), so a longer suffix / empty prefix only affects the query-prompt text, not
    # addressing — safe to relax for measurement; N1 validates for bind.
    max_suf = int(os.environ.get("CAM_MAX_SUFFIX_TOK", "6"))
    allow_empty_pre = os.environ.get("CAM_ALLOW_EMPTY_PREFIX") == "1"
    max_subj = int(os.environ.get("CAM_MAX_SUBJ_LEN", "999"))   # N1: cap subject len so the bind block fits
    n_skip_suf = n_skip_pre = n_skip_subj = 0
    by_rid = defaultdict(lambda: defaultdict(list))
    for r in kept:
        if "{}" not in r.prompt:
            continue
        if len(r.subject_tids) > max_subj:
            n_skip_subj += 1; continue
        pre, suf = _split(r.prompt)
        if not pre and not allow_empty_pre:
            n_skip_pre += 1; continue
        if len(tok(suf, add_special_tokens=False).input_ids) > max_suf:
            n_skip_suf += 1; continue
        by_rid[r.relation_id][(r.prompt, len(r.subject_tids))].append(r)
    print(f"[mag][cf-multi] relation-filter (MAX_SUFFIX_TOK={max_suf}, allow_empty_prefix={allow_empty_pre}, "
          f"MAX_SUBJ_LEN={max_subj}): {len(by_rid)} relations pass; skipped {n_skip_suf} long-suffix + "
          f"{n_skip_pre} empty-prefix + {n_skip_subj} long-subject facts", flush=True)
    R = max(2, args.multi_relations)
    per_rel_min = max(2, (args.M + R - 1) // R + 1)      # distinct subjects for this relation's doc-slot share
    # Phase N (N0, CAM_ALL_SUBJ_LENGTHS=1): the legacy grouping keeps only each relation's ONE largest
    # (prompt, subject-length) bucket — a rectangular-bind-batching shortcut that collapses ~2936 base-known
    # facts to ~147 (§3.19). Relaxed: per relation keep ALL subject-lengths of its DOMINANT prompt (the
    # store keys length-agnostically on the subject last-token/pooled span). slen=-1 flags mixed lengths;
    # bind must then length-bucket (N1). Default off = byte-identical legacy behaviour.
    all_lengths = os.environ.get("CAM_ALL_SUBJ_LENGTHS") == "1"
    length_split = os.environ.get("CAM_LENGTH_SPLIT") == "1"   # N1: bind-ready per-(rid,len) sub-relations
    # ceiling diagnostic: base-known facts kept per relation across ALL lengths (dominant prompt).
    # Each `best` entry: (n, relkey, rid, prompt, slen, recs). relkey is the DocBuilder relation key —
    # legacy/N0 use rid; N1 (length_split) uses "rid#L<slen>" so every fact in a relkey shares ONE subject
    # length (satisfies the DocBuilder rel_subj_len assertion) while ALL lengths are bound.
    tot_all = 0
    best = []
    for rid, variants in by_rid.items():
        if length_split or all_lengths:
            by_prompt = defaultdict(lambda: defaultdict(list))    # prompt -> slen -> recs
            for (prompt, slen), recs in variants.items():
                by_prompt[prompt][slen].extend(recs)
            dom = max(by_prompt.items(), key=lambda kv: sum(len(v) for v in kv[1].values()))[0]
            slen_buckets = by_prompt[dom]
            if length_split:                             # N1: one fixed-length relkey PER length bucket
                # N1b (CAM_REL_INTERLEAVE=1): rank each rid's length buckets by size (round 0 = largest) and
                # embed the round in the relkey (R00#rid#L..). rel_order=sorted(relkeys) then INTERLEAVES rids
                # — the first M slots (slot_relid=rel_order[:M]) span M DISTINCT rids' round-0 buckets instead
                # of one rid's length buckets. Fixes the N1 efficacy drop (bind saw only ~2 rids at N=441).
                interleave = os.environ.get("CAM_REL_INTERLEAVE") == "1"
                ranked = sorted(slen_buckets.items(), key=lambda kv: -len(kv[1]))
                for rnd, (slen, recs) in enumerate(ranked):
                    tot_all += len(recs)
                    if len(recs) >= per_rel_min:
                        rk = f"R{rnd:02d}#{rid}#L{slen}" if interleave else f"{rid}#L{slen}"
                        best.append((len(recs), rk, rid, dom, slen, recs))
            else:                                        # N0 (measure only): merge lengths (slen=-1)
                recs = [r for rs in slen_buckets.values() for r in rs]
                tot_all += len(recs)
                if len(recs) >= per_rel_min:
                    best.append((len(recs), rid, rid, dom, -1, recs))
        else:                                            # legacy: single largest (prompt,length) bucket
            (prompt, slen), recs = max(variants.items(), key=lambda kv: len(kv[1]))
            tot_all += len(recs)
            if len(recs) >= per_rel_min:
                best.append((len(recs), rid, rid, prompt, slen, recs))
    best.sort(reverse=True, key=lambda c: c[0])
    chosen = best[:R]
    mode = "LENGTH-SPLIT(N1)" if length_split else ("ALL-LENGTHS" if all_lengths else "one-length")
    print(f"[mag][cf-multi] grouping={mode} | {len(best)} relation-buckets pass per_rel_min>={per_rel_min} "
          f"(of {len(by_rid)} rids); top-{R} kept; fact pool across kept = {sum(c[0] for c in chosen)}",
          flush=True)
    assert len(chosen) >= 2, (f"need >= 2 relation-buckets with >= {per_rel_min} base-known facts "
                              f"(got {[(c[1], c[0]) for c in best[:6]]}); lower --multi-relations/--M")
    print(f"[mag][cf-multi] EDITING {len(chosen)} relation-buckets (relkey, subj_len, #facts): "
          f"{[(rk, slen, n) for (n, rk, _rid, _p, slen, _r) in chosen]}", flush=True)
    # HELD-OUT SUBJECT split (Track 4 #19): default OFF (byte-identical). When CAM_HELDOUT_FRAC>0, split
    # each relation-bucket's records DETERMINISTICALLY (seeded by args.seed) into a BIND portion (1-frac)
    # and a HELD-OUT portion (frac). The bind (DocBuilder facts + projection/tap training) sees ONLY bind
    # subjects; the persistent evals write+query ONLY held-out subjects — subjects the bind NEVER saw. We
    # split WITHIN each bucket (each relkey has ONE fixed subject length, N1 assertion) so both portions
    # cover the same relations/lengths/templates: this isolates NOVEL SUBJECT, not novel relation, and
    # keeps every held-out relkey's template present in rel_templates (built from the bind portion).
    heldout_frac = float(os.environ.get("CAM_HELDOUT_FRAC", "0") or "0")
    heldout_on = heldout_frac > 0.0
    if heldout_on:
        assert 0.0 < heldout_frac < 1.0, f"CAM_HELDOUT_FRAC must be in (0,1); got {heldout_frac}"
    # HELD-OUT RELATION split (novel-RELATION test, the private/codebase-facts axis): CAM_HELDOUT_RELATIONS=k
    # holds out the k SMALLEST relation-buckets ENTIRELY from the bind — they contribute NO DocBuilder facts
    # (and no template), and ALL their records go to kept_multi. The persistent evals query them via each
    # record's OWN prompt_text (subject-keyed store read; no bind template needed), so this measures whether
    # the trained readout delivers facts of a relation SHAPE the bind never saw. (bind keeps the biggest
    # buckets so it stays robust.) Composes with / overrides CAM_HELDOUT_FRAC per-relkey.
    heldout_rels = int(os.environ.get("CAM_HELDOUT_RELATIONS", "0") or "0")
    held_rel_keys = {c[1] for c in chosen[len(chosen) - heldout_rels:]} if heldout_rels > 0 else set()
    # Per-bucket bind/held record lists (default: bind==held==recs -> byte-identical legacy construction).
    split_buckets = []                                   # (relkey, rid, prompt, slen, bind_recs, held_recs)
    n_held_drop = 0
    for (_n, relkey, rid, prompt, slen, recs) in chosen:
        if relkey in held_rel_keys:                      # novel-RELATION held-out: whole bucket -> held, no bind
            split_buckets.append((relkey, rid, prompt, slen, [], recs))
            continue
        if not heldout_on:
            split_buckets.append((relkey, rid, prompt, slen, recs, recs))
            continue
        # Deterministic per-bucket shuffle seeded by (args.seed, relkey) so the split is stable across runs
        # and independent of bucket order. hashlib (not builtin hash(): PYTHONHASHSEED-salted -> non-repro).
        # n_held rounds so at least 1 held-out when the bucket is splittable.
        _bseed = int(_hl.sha1(f"{int(args.seed)}|{relkey}".encode()).hexdigest()[:8], 16)
        brng = np.random.default_rng(_bseed)
        order = brng.permutation(len(recs))
        shuffled = [recs[i] for i in order]
        n_held = int(round(len(recs) * heldout_frac))
        n_bind = len(recs) - n_held
        # Guard: a bucket must leave >=1 record in EACH portion to be usable (held needs a bind template; bind
        # needs subjects to train). Otherwise keep the whole bucket in BIND and drop it from held-out.
        if n_held < 1 or n_bind < 1:
            split_buckets.append((relkey, rid, prompt, slen, recs, []))
            n_held_drop += 1
            continue
        bind_recs = shuffled[:n_bind]
        held_recs = shuffled[n_bind:]
        split_buckets.append((relkey, rid, prompt, slen, bind_recs, held_recs))
    facts, fact_relid, cf_tid, fact_subj_tids, kept_multi = [], [], [], [], []
    rel_templates, rel_subj_len = {}, {}
    for (relkey, rid, prompt, slen, bind_recs, held_recs) in split_buckets:
        if bind_recs:                                    # BIND bucket -> template + DocBuilder facts.
            pre, suf = _split(prompt)                    # (held-out-RELATION buckets have bind_recs==[] -> no
            rel_templates[relkey] = (pre, suf)           #  template, no facts: their relation is NEVER bound.)
            # N1: relkey is per-length so slen>0 (assertion holds). N0-merge (slen=-1) uses max (probe-only).
            _all = bind_recs if bind_recs is held_recs else (bind_recs + held_recs)
            rel_subj_len[relkey] = slen if slen > 0 else max(len(r.subject_tids) for r in _all)
            for r in bind_recs:
                # Phase M: object's FIRST token as the single-token stand-in (KEY is subject_last_tid).
                _tt = r.true_ids[0] if r.true_ids else r.true_tid
                _nt = r.new_ids[0] if r.new_ids else r.new_tid
                facts.append((r.subject, r.true_str, r.subject_last_tid, _tt))  # KEY = subject LAST token
                fact_relid.append(relkey)
                cf_tid.append(_nt)
                fact_subj_tids.append(list(r.subject_tids))
        # HELD-OUT portion -> kept_multi (persistent-eval write/query subjects). Default: held is recs.
        for r in held_recs:
            r._relkey = relkey                           # tag the record so eval can bucket by relation
            kept_multi.append(r)
    if heldout_on or heldout_rels > 0:
        n_held_rel = len({rk for (rk, _r, _p, _s, _b, h) in split_buckets if h})
        assert len(facts) >= args.M, (f"HELD-OUT split left only {len(facts)} bind facts (< M={args.M}); "
                                      f"lower CAM_HELDOUT_FRAC/RELATIONS or --M, or raise --cf-probe-cap")
        assert kept_multi, "HELD-OUT split produced ZERO held-out records; lower the held-out amount"
        # FRAC held-outs (novel SUBJECT) MUST have a bind template; RELATION held-outs (novel relation) must
        # NOT (that is the point) — exempt held_rel_keys from the template-presence check.
        _missing = [r._relkey for r in kept_multi
                    if r._relkey not in rel_templates and r._relkey not in held_rel_keys]
        assert not _missing, f"held-out records reference relkeys with no bind template: {set(_missing)}"
        if heldout_rels > 0:
            print(f"[mag][cf-multi][HELD-OUT-RELATION] {heldout_rels} relation-bucket(s) held out ENTIRELY "
                  f"({sorted(held_rel_keys)}): BIND={len(facts)} subj / {len(rel_templates)} rels vs "
                  f"HELD-OUT={len(kept_multi)} subj / {n_held_rel} NOVEL rels (bind never saw their relation)",
                  flush=True)
        print(f"[mag][cf-multi][HELD-OUT] frac={heldout_frac} seed={args.seed} | BIND={len(facts)} subjects "
              f"across {len(rel_templates)} relations vs HELD-OUT={len(kept_multi)} subjects across "
              f"{n_held_rel} relations | {n_held_drop} bucket(s) too small to split (kept whole in bind)",
              flush=True)
    builder = DocBuilder(tok, None, None, args.M, args.seg_len, args.qa_seg,
                         phrasing="counterfactual_multi", facts=facts, fact_relid=fact_relid,
                         rel_templates=rel_templates, fact_subj_tids=fact_subj_tids, rel_subj_len=rel_subj_len)
    builder.set_counterfactual(cf_tid)
    ex = "; ".join(f"{r.subject} [{r._relkey}]: {r.true_str}->{r.new_str}" for r in kept_multi[:5])
    print(f"[mag][cf-multi] {len(kept_multi)} edits across {len(chosen)} relations | e.g. {ex}", flush=True)
    return builder, kept_multi, prior_acc_full


@torch.no_grad()
def build_locality_split(records, tok, frac_train=0.5):
    """Split each record's neighborhood_prompts into TRAIN (for the locality-preservation loss) and
    EVAL (for the metric) halves, disjoint per record so there is no train/eval leak. Returns
    (train_buckets, eval_probes): train_buckets = {token_len: [bos+prompt_ids,...]} out-of-store
    negatives for train_taps; eval_probes = [(prompt_str, true_tid, subject_tid)] the held-out metric."""
    from collections import defaultdict
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    train_buckets = defaultdict(list)
    eval_probes = []
    for r in records:
        nb = list(r.neighborhood_prompts)
        cut = int(len(nb) * frac_train)
        for p in nb[:cut]:
            pid = bos + tok(p, add_special_tokens=False).input_ids
            train_buckets[len(pid)].append(pid)
        for p in nb[cut:]:
            eval_probes.append((p, r.true_tid, r.subject, getattr(r, "_relkey", None)))   # subj STRING key
    return dict(train_buckets), eval_probes


def eval_locality_generalization(base, tok, injector, adapter, builder, kept, args, cap=256,
                                 loc_override=None):
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
    multi = getattr(builder, "phrasing", None) == "counterfactual_multi"
    # subject-tid -> fact index, so a probe's bank can be conditioned on the probe's OWN subject. For
    # MULTI-RELATION a subject can appear in >1 relation, so key by (subject_tid, relation) instead.
    if not getattr(builder, "facts", None):
        subj2fact = {}
    elif multi:
        # MULTI-RELATION: key by (subject STRING, relation) — subjects may be multi-token (subject_tid=-1)
        # and can appear in >1 relation. facts[i][0] is the subject string.
        subj2fact = {(builder.facts[i][0], builder.fact_relid[i]): i for i in range(len(builder.facts))}
    else:
        subj2fact = {builder.facts[i][0]: i for i in range(len(builder.facts))}   # subject string

    def _key(subj, relkey):                          # subj = subject STRING (see probe tuples below)
        return (subj, relkey) if multi else subj

    def _score(prompts_golds, carry, cond=False, weak=False):
        """prompts_golds: list of (prompt_str, gold_tid, subject_tid, relkey). Returns (acc, n). Batched by
        equal tokenized length (AND relation, for multi — build_cf_query needs one relation per call).
        cond=True (RETRIEVAL-CONDITIONED banking): build each probe's bank from a cf doc that QUERIES the
        probe's own subject (build_cf_query). weak=False -> subject's edit BOUND (STRONG, GENERALIZATION);
        weak=True -> queried but NOT bound (WEAK, LOCALITY). cond=False: shared random cf-doc bank.
        carry=False resets memory (floor)."""
        from collections import defaultdict
        buckets = defaultdict(list)
        for (p, g, subj, relkey) in prompts_golds:
            pid = tok(p, add_special_tokens=False).input_ids
            buckets[(len(pid), relkey if multi else None)].append((pid, g, subj, relkey))
        hit, seen = 0, 0
        eb = max(1, min(args.batch, EVAL_BATCH_CAP // max(1, args.M)))
        for _bkey, items in buckets.items():
            for i in range(0, len(items), eb):
                chunk = items[i:i + eb]
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                ids = torch.tensor([bos + pid for (pid, _g, _s, _r) in chunk], dtype=torch.long, device=DEV)
                gold = torch.tensor([g for (_p, g, _s, _r) in chunk], dtype=torch.long, device=DEV)
                cur = ids.shape[0]
                if cond and all(_key(s, r) in subj2fact for (_p, _g, s, r) in chunk):
                    fidx = [subj2fact[_key(s, r)] for (_p, _g, s, r) in chunk]
                    d_ids, d_apos = builder.build_cf_query(np.random.default_rng(args.seed + i), fidx, cur,
                                                           bind_target=not weak)
                else:
                    d_ids, _cf, _pr, d_apos = builder.build_cf(np.random.default_rng(args.seed + i), cur)
                d_ids = d_ids.to(DEV)
                bank = memory_bank(adapter, d_ids, args.seg_len, builder.qa_start, d_apos, carry=carry)
                _set_bank(injector, adapter, bank)
                emb = base.get_input_embeddings()(ids)
                pred = _last_logit(base, inputs_embeds=emb).argmax(-1)
                hit += (pred == gold).sum().item(); seen += cur
        injector.set_bank(None)
        return (hit / max(1, seen)), seen

    # LOCALITY probes: (neighborhood_prompt, target_true_tid). gold = the neighbour's TRUE object,
    # which shares the edited fact's true object (CounterFact construction). loc_override = the held-out
    # EVAL half when the locality-preservation loss trains on the other half (no leak); else all neighbours.
    if loc_override is not None:
        loc = [t if len(t) == 4 else (t[0], t[1], t[2], None) for t in loc_override]
    else:
        loc = []
        for r in kept:
            for p in r.neighborhood_prompts:
                loc.append((p, r.true_tid, r.subject, getattr(r, "_relkey", None)))
    loc = loc[:cap]
    # GENERALIZATION probes: (paraphrase_prompt, target_new_tid, subject STRING, relkey).
    gen = []
    for r in kept:
        for p in r.paraphrase_prompts:
            gen.append((p, r.new_tid, r.subject, getattr(r, "_relkey", None)))
    gen = gen[:cap]

    # LOCALITY with WEAK (out-of-store) banking: the neighbour's subject is not in the store, so the read
    # is weak and the tap must stay inert (retrieval-strength gating). GENERALIZATION with STRONG banking:
    # the paraphrase retrieves its OWN edit. Both mirror deployment (query the memory with the subject).
    loc_on, n_loc = _score(loc, carry=True, cond=True, weak=True)
    loc_off, _ = _score(loc, carry=False, cond=True, weak=True)
    gen_on, n_gen = _score(gen, carry=True, cond=True)
    gen_off, _ = _score(gen, carry=False, cond=True)
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
        # capital from context (upper bound on delivery through pure attention, tap off). It runs the WHOLE
        # doc (M*bind_len tokens) so it's the eval's memory hog; CAM_SKIP_CEILING=1 skips it (a diagnostic,
        # not a headline metric) so the gate/edit metrics fit at high M on a 16GB card.
        injector.set_bank(None)
        if os.environ.get("CAM_SKIP_CEILING") != "1":
            lc = _answer_logits(base, base_embed(ids[:, :apos]), 1)
            res["ceiling_cf"] += (lc.argmax(-1) == ans_cf).sum().item()
        # leak-free query context: header ("...The capital of") + "<Country> is" -> predict capital
        ctx_emb = _leakfree_ctx(base, builder, ids, apos)
        for cond, carry in (("memory", True), ("no_memory", False)):
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=carry)
            _set_bank(injector, adapter, bank)
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
        "ceiling_cf_acc": (float("nan") if os.environ.get("CAM_SKIP_CEILING") == "1"
                           else res["ceiling_cf"] / seen),
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


def _n_disjoint_banks():
    """Track 4 #19 Phase C: how many DISJOINT persistent value banks. The N=137 ceiling is shared-store
    CROWDING (Phase A/B falsified every key-encoding fix). Routing each subject to one of B disjoint value
    banks by a STABLE hash of its token-ids turns one crowded N=137 store into ~B parallel N≈137/B stores,
    each in the low-crowding regime the store already handles well (~0.5 @ N≈9). Shared trained store
    projections; only the VALUE bank is disjoint -> a PERSISTENT-PATH change, NO retraining. Default 1."""
    return max(1, int(os.environ.get("CAM_DISJOINT_BANKS", "1")))


def _subject_bank(subject_tids, B):
    """Stable subject-identity hash -> bank index in [0,B). On the discrete token-ids (not the learned
    embedding) so write and read route identically regardless of encoder state."""
    if B <= 1:
        return 0
    import hashlib
    h = hashlib.md5(",".join(map(str, subject_tids)).encode()).hexdigest()
    return int(h, 16) % B


def _init_banks(adapter, B):
    """B disjoint persistent value banks (list); B=1 -> a single bank (byte-identical to the pre-Phase-C path)."""
    return [adapter.store.init_state(1, DEV, dtype=torch.float32) for _ in range(B)]


@torch.no_grad()
def _persistent_write_val(adapter, V, r, val_tid, pooled):
    """Incremental error-correcting write of subject(`r`) -> `val_tid` into the standing bank(s) V (a LIST
    of B disjoint banks; subject routed by stable hash — Phase C). Key = pooled subject span (mean, or the
    learned attention pool when CAM_LEARNED_KEY_POOL=1), else last token. `val_tid` lets the overwrite test
    write a SECOND value for the same key."""
    tids = torch.tensor([r.subject_tids], dtype=torch.long, device=DEV)
    if getattr(adapter, "_gte_keys", None) is not None:                                    # DECOUPLED: GTE key
        key = adapter._gte_key(tids).unsqueeze(1)                                          # [1,1,mem_dim]
    else:
        subj_emb = adapter._e(tids)                                                        # [1,S,mem_dim]
        key = adapter._pool_subject(subj_emb, keepdim=True) if pooled else subj_emb[:, -1:]  # [1,H,mem] or [1,1,mem]
    val = adapter._e(torch.tensor([[val_tid]], dtype=torch.long, device=DEV))              # [1,1,mem_dim]
    lam = float(os.environ.get("CAM_VALUE_SUPPRESS", "0"))    # R1-prior-v2: promote new, SUPPRESS original
    if lam > 0 and getattr(r, "true_tid", -1) >= 0:           # value = new - lam*original (damps the base's
        val = val - lam * adapter._e(torch.tensor([[r.true_tid]], dtype=torch.long, device=DEV))  # confident prior)
    if os.environ.get("CAM_VALUE_UNIT_NORM") == "1":         # store a UNIT value so retrieval conf reflects
        val = torch.nn.functional.normalize(val, dim=-1)    # ADDRESSING quality, not the object token's
                                                            # embedding norm (weak-edit diagnostic §3.15)
    if key.shape[1] > 1:                        # multi-vector keys: repeat the value across the H key slots
        val = val.expand(-1, key.shape[1], -1)
    b = _subject_bank(r.subject_tids, len(V))
    V[b] = adapter.persistent_write(V[b], key, val)
    return V


def _persistent_write_one(adapter, V, r, pooled):
    """One incremental write of edit `r` (value = its own new_tid) into the standing bank(s) V."""
    return _persistent_write_val(adapter, V, r, r.new_tid, pooled)


def _persistent_preds(base, adapter, injector, tok, V, cohort, bank_ids=None):
    """Query each edit in `cohort` against standing bank V (subject-keyed read + tap) and return the list
    of predicted next-token ids at each prompt's last position. Non-mutating.

    The 4B base forward is the cost; a batch-1 forward per edit is occupancy-starved (100% util / ~34W).
    So batch it: the per-edit STORE reads stay a cheap B=1 loop (variable subject length, small mem_dim
    ops), then the EXPENSIVE base forwards run in groups — the tap is already per-row batched (bank
    [B,K,mem], conf [B]). RIGHT-pad is causal-safe: the logit at each row's true last token depends only
    on tokens <= that position, so trailing pad (and pad-position tap injections) can't perturb it.
    CAM_PERSISTENT_EVAL_BATCH=1 reproduces the old exact batch-1 path (parity)."""
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    base_embed = base.get_input_embeddings()
    bs = max(1, int(os.environ.get("CAM_PERSISTENT_EVAL_BATCH", "1")))   # default 1: safe on a full 16GB card
                                                                          # (batching raises peak mem — opt in only with headroom, e.g. 2-card)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    # cheap per-edit store reads (B=1; variable subject length) -> collect per-row bank/conf + prompt ids
    banks, confs, id_lists = [], [], []
    learned_pool = os.environ.get("CAM_LEARNED_KEY_POOL") == "1"
    gte = getattr(adapter, "_gte_keys", None) is not None
    for i, r in enumerate(cohort):
        tids = torch.tensor([r.subject_tids], dtype=torch.long, device=DEV)
        if gte:                                                          # DECOUPLED: GTE query (matches write)
            q = adapter._gte_key(tids).unsqueeze(1)                      # [1,1,mem]
        else:
            q = adapter._e(tids)
            if learned_pool:                                            # symmetric with the pooled write key
                q = adapter._pool_subject(q, keepdim=True)              # [1,1,mem]
        b = bank_ids[i] if bank_ids is not None else _subject_bank(r.subject_tids, len(V))  # R0 solo: identity
        banks.append(adapter.persistent_bank(V[b], q))                    # [1,K,mem]
        confs.append(getattr(adapter, "_last_conf", None))               # [1] or None
        id_lists.append(bos + tok(r.prompt_text, add_special_tokens=False).input_ids)
    preds = [0] * len(cohort)
    for i in range(0, len(cohort), bs):
        rows = list(range(i, min(i + bs, len(cohort))))
        B = len(rows)
        lens = [len(id_lists[j]) for j in rows]
        Tmax = max(lens)
        ids = torch.full((B, Tmax), pad_id, dtype=torch.long, device=DEV)   # RIGHT-pad (causal-safe)
        for b, j in enumerate(rows):
            ids[b, :lens[b]] = torch.tensor(id_lists[j], dtype=torch.long, device=DEV)
        bank = torch.cat([banks[j] for j in rows], dim=0)                 # [B,K,mem]
        conf = None if any(confs[j] is None for j in rows) else torch.cat([confs[j] for j in rows], dim=0)
        injector.set_bank(bank, conf=conf, relidx=0)
        logits = base(inputs_embeds=base_embed(ids)).logits               # [B,Tmax,vocab]
        ld = logits.device                                                # MODEL-PARALLEL: lm_head may be on card 1
        last = torch.tensor([lens[b] - 1 for b in range(B)], device=ld)
        last_logits = logits[torch.arange(B, device=ld), last]            # [B,vocab]
        alpha = float(os.environ.get("CAM_LOGIT_INJECT", "0"))           # PARADIGM test: add the retrieved
        if alpha > 0:                                                     # value's contribution straight to the
            bh = adapter.out_proj(bank).mean(1)                          # OUTPUT logits (bypass the residual
            lm = base.get_output_embeddings().weight                     # site). Does logit-space break ~0.7?
            inj = alpha * (bh.to(lm.device, lm.dtype) @ lm.t()).to(ld)   # [B,vocab]
            c0 = os.environ.get("CAM_LOGIT_GATE_C0")                     # CONF-GATE (#67 fix): scale the
            if c0 is not None and conf is not None:                      # injection by retrieval strength so it
                cc = conf.to(ld)                                         # fires on the edited subject (high conf)
                if os.environ.get("CAM_LOGIT_GATE_HARD") == "1":         # but stays inert on out-of-store
                    g = (cc > float(c0)).to(inj.dtype)                   # neighbours (low conf) -> keeps locality.
                else:                                                    # HARD step exploits the ~122-vs-0
                    k = float(os.environ.get("CAM_LOGIT_GATE_K", "1"))   # in/out-of-store conf separation
                    g = torch.sigmoid(k * (cc - float(c0)))
                inj = inj * g.view(-1, 1)
            last_logits = last_logits + inj
        pred = last_logits.argmax(-1)                                     # gather each row's true last token
        for b, j in enumerate(rows):
            preds[j] = int(pred[b].item())
    injector.set_bank(None)
    return preds


def _persistent_score(base, adapter, injector, tok, V, cohort):
    """cf_delivery, prior_recall over `cohort` — fraction predicting the edit's new_tid / true_tid."""
    preds = _persistent_preds(base, adapter, injector, tok, V, cohort)
    n = max(1, len(cohort))
    cf_hit = sum(int(preds[i] == cohort[i].new_tid) for i in range(len(cohort)))
    pr_hit = sum(int(preds[i] == cohort[i].true_tid) for i in range(len(cohort)))
    return cf_hit / n, pr_hit / n


def eval_persistent(base, adapter, injector, tok, kept, args):
    """Track 4 (#19) — PERSISTENT / online memory. Write ALL N kept edits into ONE standing store V
    (incremental error-correcting writes; NO episodic doc, NO reset), then query each edit's NATURAL
    prompt and score counterfactual delivery + prior recall. Tests the step from a per-doc scratchpad to
    a standing memory: does a persistent store of N edits deliver each one at once? Reuses the trained
    store projections + conf-gate tap. Key = pooled subject span (CAM_POOLED_SUBJ_KEY) or last token.

    With --persistent-sweep, ALSO run a RETENTION/INTERFERENCE curve: at each checkpoint (fraction of N)
    during the incremental write phase, re-query a FIXED early cohort (first C edits) — same edits, a
    growing store — so a decaying early-cohort curve isolates interference (does edit #1 survive writing
    edit #N?), decoupled from the cumulative all-so-far delivery."""
    injector.eval()
    pooled = os.environ.get("CAM_POOLED_SUBJ_KEY") == "1"
    N = len(kept)
    sweep = getattr(args, "persistent_sweep", False)
    B = _n_disjoint_banks()                                           # Phase C: B disjoint value banks
    V = _init_banks(adapter, B)                                       # subject-hash routed (B=1 -> one bank)
    if B > 1:
        print(f"[mag][persistent] Phase C: {B} disjoint value banks (subject-hash routed, ~{N/B:.1f} edits/bank)",
              flush=True)

    curve = []
    if sweep:
        C = min(int(getattr(args, "persistent_cohort", 10) or 10), N)   # fixed early cohort tracked over N
        early = kept[:C]
        # checkpoints in # of edits written: unique, monotone, always ending at N
        fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
        ckpts = sorted({max(C, int(round(f * N))) for f in fracs} | {N})
        ci = 0
        for i, r in enumerate(kept, start=1):
            V = _persistent_write_one(adapter, V, r, pooled)
            if ci < len(ckpts) and i == ckpts[ci]:
                cf_e, pr_e = _persistent_score(base, adapter, injector, tok, V, early)
                cf_a, pr_a = _persistent_score(base, adapter, injector, tok, V, kept[:i])
                curve.append({"written": i, "early_cf": cf_e, "early_pr": pr_e,
                              "all_cf": cf_a, "all_pr": pr_a})
                print(f"[mag][persistent][sweep] written={i:4d}  early(1..{C}) cf={cf_e:.3f} pr={pr_e:.3f}"
                      f"   all(1..{i}) cf={cf_a:.3f} pr={pr_a:.3f}", flush=True)
                ci += 1
        cf_hit_rate, pr_hit_rate = curve[-1]["all_cf"], curve[-1]["all_pr"]
    else:
        for r in kept:
            V = _persistent_write_one(adapter, V, r, pooled)
        cf_hit_rate, pr_hit_rate = _persistent_score(base, adapter, injector, tok, V, kept)

    print(f"\n[mag][persistent] === Track 4: {N} edits in ONE standing store (online) ===", flush=True)
    print(f"  cf-delivery {cf_hit_rate:.3f} | prior-recall {pr_hit_rate:.3f}   "
          f"(cf HIGH + prior LOW = the persistent store overrides across all {N} edits at once)", flush=True)
    if sweep and curve:
        e0, e1 = curve[0]["early_cf"], curve[-1]["early_cf"]
        print(f"  retention: early cohort(1..{C}) cf {e0:.3f} @ {curve[0]['written']} written "
              f"-> {e1:.3f} @ {N} written  (interference drop = {e0 - e1:+.3f})", flush=True)
    print("=" * 64, flush=True)
    return {"n_edits": N, "cf_delivery": cf_hit_rate, "prior_recall": pr_hit_rate, "curve": curve}


def eval_persistent_overwrite(base, adapter, injector, tok, kept, args):
    """Track 4 (#19) incr#3 — ONLINE UPDATE / overwrite. The pk-store write is an error-correcting DELTA;
    does writing a SECOND value for a key cleanly REPLACE the first, or blend/stale? Two passes into ONE
    standing store: (A) subject_i -> new_tid_i, then (B) subject_i -> a DIFFERENT value B_i (the NEXT
    edit's new_tid, cyclic; skip pairs where B==A). Query each subject's prompt after each pass and score
    which value wins: UPDATED (pred==B, the goal), STALE (pred==A, old value survived), or prior/other."""
    injector.eval()
    pooled = os.environ.get("CAM_POOLED_SUBJ_KEY") == "1"
    N = len(kept)
    # deterministic second value: the next edit's new_tid (cyclic). Keep only pairs with B != A so
    # "updated" and "stale" are distinguishable; also B must differ from the base prior true_tid to read.
    idx = [i for i in range(N) if kept[(i + 1) % N].new_tid != kept[i].new_tid]
    cohort = [kept[i] for i in idx]
    valA = [r.new_tid for r in cohort]
    valB = [kept[(idx[j] + 1) % N].new_tid for j in range(len(cohort))]
    n = max(1, len(cohort))
    V = _init_banks(adapter, _n_disjoint_banks())          # Phase C: B disjoint banks (subject-hash routed)
    for r in cohort:                                        # pass A: write the first value
        V = _persistent_write_val(adapter, V, r, r.new_tid, pooled)
    predsA = _persistent_preds(base, adapter, injector, tok, V, cohort)
    a_deliver = sum(int(predsA[j] == valA[j]) for j in range(len(cohort))) / n
    for j, r in enumerate(cohort):                          # pass B: overwrite with a different value
        V = _persistent_write_val(adapter, V, r, valB[j], pooled)
    predsB = _persistent_preds(base, adapter, injector, tok, V, cohort)
    updated = sum(int(predsB[j] == valB[j]) for j in range(len(cohort))) / n   # new value won (goal)
    stale = sum(int(predsB[j] == valA[j]) for j in range(len(cohort))) / n     # old value survived
    prior = sum(int(predsB[j] == cohort[j].true_tid) for j in range(len(cohort))) / n
    print(f"\n[mag][persistent] === Track 4 incr#3: ONLINE UPDATE over {len(cohort)} subjects ===", flush=True)
    print(f"  after write-A: A-delivery {a_deliver:.3f}", flush=True)
    print(f"  after write-B (same keys, new value): UPDATED(B) {updated:.3f} | STALE(A) {stale:.3f} | "
          f"prior {prior:.3f}   (clean update = UPDATED high, STALE low)", flush=True)
    print("=" * 64, flush=True)
    return {"n": len(cohort), "a_deliver": a_deliver, "updated": updated, "stale": stale, "prior": prior}


@torch.no_grad()
def eval_persistent_solo(base, adapter, injector, tok, kept, args):
    """Phase R / R0 — SINGLE-FACT FIDELITY. Write each edit ALONE into its own fresh store and query it,
    so NO collision is possible. Isolates the per-fact retrieval-fidelity ceiling (store/value/tap readout
    + the frozen-base single-injection mechanism) from all addressing. If solo-delivery is ~1.0 the B=32
    ceiling was residual collision; if ~0.7 it's the store/tap fidelity floor and Phase R attacks it."""
    injector.eval()
    pooled = os.environ.get("CAM_POOLED_SUBJ_KEY") == "1"
    per_V = []
    for r in kept:
        V = [adapter.store.init_state(1, DEV, dtype=torch.float32)]       # its own 1-bank store
        per_V.append(_persistent_write_one(adapter, V, r, pooled)[0])     # write this edit alone
    preds = _persistent_preds(base, adapter, injector, tok, per_V, kept, bank_ids=list(range(len(kept))))
    if os.environ.get("CAM_PRIORCONF_LOG") == "1":           # R-univ within-model: does edit success fall
        injector.set_bank(None)                               # as the base's PRE-EDIT confidence in the
        base_embed = base.get_input_embeddings()              # ORIGINAL answer rises? (architecture-controlled)
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        for i, r in enumerate(kept):
            ids = torch.tensor([bos + tok(r.prompt_text, add_special_tokens=False).input_ids],
                               dtype=torch.long, device=DEV)
            lg = _last_logit(base, inputs_embeds=base_embed(ids))[0]      # [V] no-tap base logits
            pconf = float(torch.softmax(lg, -1)[r.true_tid])             # base P(original) pre-edit
            print(f"[priorconf] pconf={pconf:.4f} hit={int(preds[i]==r.new_tid)} rid={r.relation_id}", flush=True)
        injector.set_bank(None)
    n = max(1, len(kept))
    solo = sum(int(preds[i] == kept[i].new_tid) for i in range(len(kept))) / n
    prior = sum(int(preds[i] == kept[i].true_tid) for i in range(len(kept))) / n
    if os.environ.get("CAM_SOLO_LOG") == "1":                # per-fact outcomes: is the ~30% failure a
        for i, r in enumerate(kept):                          # CONSISTENT subset or noise? (subject-tids key)
            key = "_".join(map(str, r.subject_tids))
            print(f"[solofact] rid={r.relation_id} subj={key} hit={int(preds[i]==r.new_tid)} "
                  f"prior={int(preds[i]==r.true_tid)} slen={len(r.subject_tids)}", flush=True)
    print(f"\n[mag][persistent] === Phase R/R0: SINGLE-FACT FIDELITY over {len(kept)} edits (each ALONE) ===",
          flush=True)
    print(f"  solo-delivery {solo:.3f} | prior {prior:.3f}   (per-fact store/tap ceiling — no collision "
          f"possible; gap to 1.0 = the retrieval-fidelity lever)", flush=True)
    print("=" * 64, flush=True)
    return {"solo": solo, "prior": prior}


@torch.no_grad()
def eval_persistent_generate(base, adapter, injector, tok, kept, args):
    """GENERATION-COHERENCE check (the reality test): does the edited memory produce FLUENT text carrying
    the edit, or only flip the next-token argmax? All prior metrics are single-token. Here we write all
    edits into the standing store, then for a sample of edits GENERATE CAM_GEN_LEN tokens (greedy) from the
    edit's natural prompt (a) with memory OFF (base) and (b) with memory ON (trained tap + hard-conf-gated
    logit injection at the operating alpha). Prints both continuations + the edit so a human can eyeball
    fluency and whether the NEW object is produced instead of the base's TRUE object.

    NOTE: the logit injection is applied at EVERY generation step (constant, keyed on the fixed subject) —
    the naive deployment. If that over-injects (repetition past the object), that itself is a coherence
    finding about the blunt mechanism."""
    injector.eval()
    pooled = os.environ.get("CAM_POOLED_SUBJ_KEY") == "1"
    V = _init_banks(adapter, _n_disjoint_banks())
    for r in kept:
        V = _persistent_write_one(adapter, V, r, pooled)
    base_embed = base.get_input_embeddings()
    lm = base.get_output_embeddings().weight
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    learned_pool = os.environ.get("CAM_LEARNED_KEY_POOL") == "1"
    gte = getattr(adapter, "_gte_keys", None) is not None
    alpha = float(os.environ.get("CAM_LOGIT_INJECT", "0"))
    c0env, hard = os.environ.get("CAM_LOGIT_GATE_C0"), os.environ.get("CAM_LOGIT_GATE_HARD") == "1"
    gen_len = int(os.environ.get("CAM_GEN_LEN", "16"))
    sample = kept[:int(os.environ.get("CAM_GEN_SAMPLE", "16"))]

    def _bank_conf(r):
        tids = torch.tensor([r.subject_tids], dtype=torch.long, device=DEV)
        if gte:
            q = adapter._gte_key(tids).unsqueeze(1)
        else:
            q = adapter._e(tids)
            if learned_pool:
                q = adapter._pool_subject(q, keepdim=True)
        bank = adapter.persistent_bank(V[_subject_bank(r.subject_tids, len(V))], q)
        return bank, getattr(adapter, "_last_conf", None)

    # CAM_GEN_INJECT_STEPS: how many leading generation steps get the logit injection. 0 = ALL steps
    # (naive constant — degenerates into repetition of the object). >0 = inject only the first K steps (the
    # ANSWER span) then let the base continue fluently — the fix for the repetition failure.
    def _gen(prompt_ids, bank, conf, inj_steps):
        injector.set_bank(bank, conf=conf, relidx=0) if bank is not None else injector.set_bank(None)
        inj = None
        if bank is not None and alpha > 0:
            inj = alpha * (adapter.out_proj(bank).mean(1).to(lm.device, lm.dtype) @ lm.t())   # [1,vocab]
            if c0env is not None and conf is not None:
                cc = conf.to(inj.device)
                if hard:
                    g = (cc > float(c0env)).to(inj.dtype)
                else:
                    g = torch.sigmoid(float(os.environ.get("CAM_LOGIT_GATE_K", "1")) * (cc - float(c0env)))
                inj = inj * g.view(-1, 1)
        cur = torch.tensor([prompt_ids], dtype=torch.long, device=DEV)
        out = []
        for t in range(gen_len):
            if inj_steps > 0 and t == inj_steps and bank is not None:
                injector.set_bank(None)                                       # disable the TAP too after the
                inj = None                                                    # answer span -> base continues fluent
            logits = base(inputs_embeds=base_embed(cur)).logits[:, -1]        # [1,vocab] (no KV cache; small)
            if inj is not None:
                logits = logits + inj.to(logits.device)
            nxt = logits.argmax(-1)
            out.append(int(nxt.item()))
            cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
        injector.set_bank(None)
        return tok.decode(out).replace("\n", " ").strip()

    k_ans = int(os.environ.get("CAM_GEN_INJECT_STEPS", "2"))                  # answer-span injection width
    print(f"\n[mag][persistent] === GENERATION COHERENCE ({len(sample)} edits, {gen_len} tok, α={alpha}, "
          f"answer-inject-steps={k_ans}) ===", flush=True)
    new_const = new_ans = true_off = 0
    for r in sample:
        pid = bos + tok(r.prompt_text, add_special_tokens=False).input_ids
        bank, conf = _bank_conf(r)
        g_off = _gen(pid, None, None, 0)
        g_const = _gen(pid, bank, conf, 0)                                   # inject EVERY step (naive)
        g_ans = _gen(pid, bank, conf, k_ans)                                 # inject only the answer span
        nl = r.new_str.strip().lower()
        new_const += int(nl in g_const.lower()); new_ans += int(nl in g_ans.lower())
        true_off += int(r.true_str.strip().lower() in g_off.lower())
        print(f"  [{r.relation_id}] {r.prompt_text!r}  edit {r.true_str!r}->{r.new_str!r}", flush=True)
        print(f"     OFF        : {g_off!r}", flush=True)
        print(f"     ON constant: {g_const!r}", flush=True)
        print(f"     ON answer-{k_ans}: {g_ans!r}   [new: {nl in g_ans.lower()}]", flush=True)
    n = max(1, len(sample))
    print(f"  --> NEW object present — constant-inject: {new_const}/{len(sample)} ({new_const/n:.2f}) | "
          f"answer-span-inject: {new_ans}/{len(sample)} ({new_ans/n:.2f}); base TRUE-recall {true_off}/{len(sample)}",
          flush=True)
    print(f"  (constant-inject = repetition/degenerate; answer-span-inject should read FLUENT with the edit)",
          flush=True)
    print("=" * 64, flush=True)
    return {"n": len(sample), "new_constant": new_const / n, "new_answer": new_ans / n, "true_in_base": true_off / n}


class _NbrRec:
    """Lightweight pseudo-record so a neighbour prompt can ride the exact `_persistent_preds` path
    (store read keyed by subject_tids -> tap + logit-injection).

    Two keying modes for the locality question:
      * ADVERSARIAL (subject_tids = the EDITED subject): forces the store to return the edit's STRONG
        value and inject it onto a same-true-object neighbour — the UPPER BOUND on damage ('if the
        edit's value reaches a neighbour, how bad?'). Bypasses addressing.
      * DEPLOYMENT (subject_tids = the NEIGHBOUR's OWN subject, parsed from the shared relation
        template): the neighbour is out-of-store, so its own subject drives retrieval exactly as in
        production — measures whether the store's ADDRESSING keeps injection off unrelated prompts.
    Both score keep = pred==true_tid (neighbour held) and leak = pred==new_tid (flipped to the EDIT's
    counterfactual object)."""
    __slots__ = ("subject_tids", "prompt_text", "new_tid", "true_tid", "relation_id")

    def __init__(self, subject_tids, prompt, new_tid, true_tid, relation_id):
        self.subject_tids = subject_tids
        self.prompt_text = prompt
        self.new_tid = new_tid
        self.true_tid = true_tid
        self.relation_id = relation_id


def _nbr_subject_tids(tok, r, prompt):
    """Parse the NEIGHBOUR's own subject tids out of a neighbourhood prompt by stripping the edit's
    relation template (r.prompt = 'The mother tongue of {} is'; neighbour = same template, other
    subject). Returns the space-prefixed subject token ids, or None if the template doesn't match."""
    tmpl = getattr(r, "prompt", "")
    if "{}" not in tmpl:
        return None
    pre, suf = tmpl.split("{}", 1)
    p = prompt
    if pre and not p.startswith(pre):
        return None
    p = p[len(pre):]
    if suf:
        j = p.rfind(suf)
        if j < 0:
            return None
        p = p[:j]
    subj = p.strip()
    if not subj:
        return None
    return tok(" " + subj, add_special_tokens=False).input_ids


def _cohort_confs(adapter, V, cohort, pooled, bank_ids=None):
    """Per-item store retrieval-confidence (factual-head magnitude) for a cohort, using the SAME subject
    query as _persistent_preds. Used to (a) show whether in-store (edited) subjects retrieve more
    strongly than out-of-store neighbours and (b) self-calibrate the conf-gate threshold."""
    learned_pool = os.environ.get("CAM_LEARNED_KEY_POOL") == "1"
    gte = getattr(adapter, "_gte_keys", None) is not None
    out = []
    for i, r in enumerate(cohort):
        tids = torch.tensor([r.subject_tids], dtype=torch.long, device=DEV)
        if gte:
            q = adapter._gte_key(tids).unsqueeze(1)
        else:
            q = adapter._e(tids)
            if learned_pool:
                q = adapter._pool_subject(q, keepdim=True)
        b = bank_ids[i] if bank_ids is not None else _subject_bank(r.subject_tids, len(V))
        adapter.persistent_bank(V[b], q)
        c = getattr(adapter, "_last_conf", None)
        out.append(float(c.item()) if c is not None else float("nan"))
    return out


def eval_persistent_locality(base, adapter, injector, tok, kept, args):
    """Phase R / logit-injection LOCALITY — the decisive follow-up to CAM_LOGIT_INJECT (#67).

    Logit injection breaks the residual wall (solo 0.65 -> 0.87) by ADDING alpha*out_proj(bank)@lm_head
    straight to the output logits. That is blunt: the real question is whether forcing the object logit
    also damages NEIGHBOURING facts (a paradigm shift only if delivery rises WITHOUT locality
    collapsing). So sweep alpha and measure delivery and locality TOGETHER:

      * DELIVERY  — over the edited prompts: fraction predicting new_tid (the edit fires).
      * NBR-KEEP  — over the edits' neighbourhood_prompts (other subjects, SAME true object; gold =
                    true_tid): fraction still predicting the neighbour's TRUE answer (locality held).
      * NBR-LEAK  — same neighbours: fraction wrongly flipped to the EDIT's new_tid (the leak we fear).

    The neighbour is queried against the standing store keyed by the EDITED subject (see _NbrRec) —
    the ADVERSARIAL upper bound (the edit's strong value IS retrieved and injected). If NBR-LEAK stays
    low even here, logit injection is specific; if it rises with alpha, the fidelity wall has merely
    been traded for a locality wall and there may (or may not) be a usable operating point."""
    injector.eval()
    pooled = os.environ.get("CAM_POOLED_SUBJ_KEY") == "1"
    N = len(kept)
    V = _init_banks(adapter, _n_disjoint_banks())
    for r in kept:
        V = _persistent_write_one(adapter, V, r, pooled)

    cap = int(os.environ.get("CAM_LOCALITY_NBR_CAP", "3"))           # neighbours per edit (budget)
    # DEPLOYMENT cohort: neighbour keyed on its OWN parsed subject (out-of-store; addressing in play).
    # ADVERSARIAL cohort: same neighbour prompt keyed on the EDIT's subject (edit's value forced in).
    # GEN cohort: the edit's PARAPHRASE prompts (rephrasings of the SAME fact) — the third leg of the
    # editing triad. A paraphrase is about the edit's OWN subject, so it routes to the edit's bank and
    # retrieves strongly (unlike a neighbour); we WANT it to fire (gold = new_tid). Keyed on the edit's
    # subject_tids (deployment-faithful: the engine parses the paraphrase's subject = the edit's subject).
    # Bound cohort work + hang exposure at scale: skip pathologically long prompts (long CounterFact
    # paraphrase preambles are the likely RDNA4 kernel-flake trigger AND are not needed) and cap each
    # cohort to a fixed subset regardless of N. Delivery/confdiag still run over ALL N.
    maxtok = int(os.environ.get("CAM_PROMPT_MAXTOK", "64"))
    cohort_cap = int(os.environ.get("CAM_COHORT_CAP", "0")) or None       # 0 = uncapped
    def _short(p):
        return bool(p) and len(tok(p, add_special_tokens=False).input_ids) <= maxtok
    dep, adv, gen, n_unparsed = [], [], [], 0
    for r in kept:
        for p in r.neighborhood_prompts[:cap]:
            if not _short(p):
                continue
            adv.append(_NbrRec(r.subject_tids, p, r.new_tid, r.true_tid, r.relation_id))
            st = _nbr_subject_tids(tok, r, p)
            if st:
                dep.append(_NbrRec(st, p, r.new_tid, r.true_tid, r.relation_id))
            else:
                n_unparsed += 1
        for p in getattr(r, "paraphrase_prompts", [])[:cap]:
            if _short(p):
                gen.append(_NbrRec(r.subject_tids, p, r.new_tid, r.true_tid, r.relation_id))
    if cohort_cap:                                                        # bound eval at scale
        dep, adv, gen = dep[:cohort_cap], adv[:cohort_cap], gen[:cohort_cap]
    if not adv:
        print("[mag][persistent] locality: no neighbourhood_prompts on records; skipping.", flush=True)
        return {}
    alphas = [float(x) for x in os.environ.get("CAM_LOGIT_INJECT_SWEEP", "0,2,8,20").split(",")]
    saved_a = os.environ.get("CAM_LOGIT_INJECT")
    saved_c0 = os.environ.get("CAM_LOGIT_GATE_C0")

    # --- conf separation + self-calibrated gate threshold -----------------------------------------
    # Calibrate C0 to include weakly-retrieved-but-REAL edits (the delivery gap: §3.15 diagnostic shows
    # ~17/137 edits retrieve at conf 1-30 yet deliver unconditionally) while excluding the neighbour BULK
    # (median ~0). The neighbour conf p95 is NOT usable here — it ≈ the edited median (a ~5% tail of
    # neighbours FULLY collide with an edit in the same bank; irreducible, no threshold excludes them). So
    # optimise the edited side: C0 = em/12 (~sweet spot 10 from the threshold sweep — C0=10 recovers +0.04
    # delivery over the midpoint 59 at ~0 locality cost), floored above the neighbour bulk. CAM_LOGIT_GATE_C0
    # overrides for the production knob.
    import statistics as _st
    def _pct(xs, q):
        if not xs:
            return float("nan")
        s = sorted(xs); i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[i]
    ec = [c for c in _cohort_confs(adapter, V, kept, pooled) if c == c]            # edited (in-store) subj
    dc = [c for c in _cohort_confs(adapter, V, dep, pooled) if c == c] if dep else []  # neighbour (out-of-store)
    em = _st.median(ec) if ec else float("nan")
    dm = _st.median(dc) if dc else float("nan")
    dp95 = _pct(dc, 0.95) if dc else float("nan")
    c0 = max(em / 12.0, 10.0 * (dm if dm == dm else 0.0), 0.5) if ec else None     # ~em/12, above nbr bulk
    spread = abs(em - dm) if (ec and dc) else 0.0
    gate_k = 4.0 / spread if spread > 1e-6 else 1.0                                # ~saturate across the gap
    print(f"\n[mag][persistent] conf(factual-head magnitude): edited median={em:.3f}  "
          f"neighbour median={dm:.3f} p95={dp95:.3f}  separation={em - dm:+.3f}  -> gate "
          f"C0={c0 if c0 is None else round(c0, 3)} (~em/12, above nbr bulk) K={round(gate_k, 3)}", flush=True)

    dbg = os.environ.get("CAM_TRIAD_DEBUG") == "1"

    def _score(cohort, a, tag=""):
        if not cohort:
            return None, None
        if dbg:
            print(f"[triad-dbg] score {tag} a={a} n={len(cohort)} ...", flush=True)
        os.environ["CAM_LOGIT_INJECT"] = str(a)
        pr = _persistent_preds(base, adapter, injector, tok, V, cohort)
        if dbg:
            print(f"[triad-dbg] score {tag} a={a} DONE", flush=True)
        n = max(1, len(cohort))
        keep = sum(int(pr[i] == cohort[i].true_tid) for i in range(len(cohort))) / n
        leak = sum(int(pr[i] == cohort[i].new_tid) for i in range(len(cohort))) / n
        return keep, leak

    saved_hard = os.environ.get("CAM_LOGIT_GATE_HARD")

    def _sweep(gated, hard=False, thr=None):
        t = c0 if thr is None else thr
        if gated and t is not None:
            os.environ["CAM_LOGIT_GATE_C0"] = str(t); os.environ["CAM_LOGIT_GATE_K"] = str(gate_k)
            os.environ["CAM_LOGIT_GATE_HARD"] = "1" if hard else "0"
        else:
            os.environ.pop("CAM_LOGIT_GATE_C0", None); os.environ.pop("CAM_LOGIT_GATE_HARD", None)
        rows = []
        for a in alphas:
            if dbg:
                print(f"[triad-dbg] sweep gated={gated} hard={hard} a={a} -> delivery(kept n={N}) ...", flush=True)
            os.environ["CAM_LOGIT_INJECT"] = str(a)
            ep = _persistent_preds(base, adapter, injector, tok, V, kept)   # edit prompts (own bank) = delivery
            delivery = sum(int(ep[i] == kept[i].new_tid) for i in range(N)) / max(1, N)
            dk, dl = _score(dep, a, "dep"); ak, al = _score(adv, a, "adv")
            g_prior, g_hit = _score(gen, a, "gen")                          # GEN: hit = pred==new (fires on
            rows.append({"alpha": a, "delivery": delivery,                  # paraphrase); prior = pred==true
                         "dep_keep": dk, "dep_leak": dl, "adv_keep": ak, "adv_leak": al,
                         "gen_hit": g_hit, "gen_prior": g_prior})
        return rows

    def _print(title, rows):
        print(f"\n  --- {title} ---", flush=True)
        print(f"  {'alpha':>6}  {'delivery':>9} | {'DEP-keep':>9} {'DEP-leak':>9} | {'GEN-hit':>8} {'GEN-prior':>9} | "
              f"{'ADV-keep':>9} {'ADV-leak':>9}", flush=True)
        for row in rows:
            dk = f"{row['dep_keep']:.3f}" if row['dep_keep'] is not None else "  -  "
            dl = f"{row['dep_leak']:.3f}" if row['dep_leak'] is not None else "  -  "
            gh = f"{row['gen_hit']:.3f}" if row.get('gen_hit') is not None else "  -  "
            gp = f"{row['gen_prior']:.3f}" if row.get('gen_prior') is not None else "  -  "
            print(f"  {row['alpha']:>6.1f}  {row['delivery']:>9.3f} | {dk:>9} {dl:>9} | {gh:>8} {gp:>9} | "
                  f"{row['adv_keep']:>9.3f} {row['adv_leak']:>9.3f}", flush=True)

    print(f"\n[mag][persistent] === Logit-injection TRIAD: {N} edits | dep={len(dep)} adv={len(adv)} "
          f"neighbours ({n_unparsed} unparsed) | gen={len(gen)} paraphrases ===", flush=True)
    print(f"  (delivery=efficacy | DEP=locality (keep the neighbour) | GEN-hit=generality (edit fires on "
          f"paraphrase, gold=new) | ADV=adversarial upper bound)", flush=True)
    ungated = _sweep(gated=False)
    _print("UNCONDITIONAL injection", ungated)
    gated = _sweep(gated=True) if c0 is not None else None
    if gated is not None:
        _print(f"CONF-GATED (soft, C0={round(c0,3)}, K={round(gate_k,3)})", gated)
    hardg = _sweep(gated=True, hard=True) if c0 is not None else None
    if hardg is not None:
        _print(f"CONF-GATED (HARD step at C0={round(c0,3)})", hardg)

    # THRESHOLD sweep: the midpoint C0 is over-conservative given the ~122-vs-0 separation. Lowering the
    # hard threshold toward the neighbour level admits more sub-conf EDITED subjects (recovers the delivery
    # gap) while conf~0.04 neighbours stay excluded. Find the C0 that maxes delivery at baseline locality.
    thr_rows = None
    if c0 is not None and dc:
        amax = alphas[-1]                                                  # hard-gate delivery saturates in a
        os.environ["CAM_LOGIT_INJECT"] = str(amax)                         # -> sweep C0 at the top alpha
        cands = sorted({round(c0, 3), 20.0, 10.0, 5.0, 2.0,
                        round(max(0.5, 3 * (dm if dm == dm else 0.04)), 3)}, reverse=True)
        thr_rows = []
        for t in cands:
            os.environ["CAM_LOGIT_GATE_C0"] = str(t); os.environ["CAM_LOGIT_GATE_HARD"] = "1"
            os.environ["CAM_LOGIT_GATE_K"] = str(gate_k)
            ep = _persistent_preds(base, adapter, injector, tok, V, kept)
            delivery = sum(int(ep[i] == kept[i].new_tid) for i in range(N)) / max(1, N)
            dk, dl = _score(dep, amax)
            thr_rows.append({"c0": t, "delivery": delivery, "dep_keep": dk, "dep_leak": dl})
        print(f"\n  --- HARD-gate C0 THRESHOLD sweep (alpha={amax}; edited conf~{round(em,1)}, "
              f"neighbour conf~{round(dm,3)}) ---", flush=True)
        print(f"  {'C0':>8}  {'delivery':>9} | {'DEP-keep':>9} {'DEP-leak':>9}", flush=True)
        for row in thr_rows:
            print(f"  {row['c0']:>8.3f}  {row['delivery']:>9.3f} | {row['dep_keep']:>9.3f} "
                  f"{row['dep_leak']:>9.3f}", flush=True)
        print(f"  (lower C0 -> more edits clear the gate; neighbours (conf~{round(dm,3)}) stay excluded "
              f"until C0 approaches their level)", flush=True)

    # PER-EDIT conf DIAGNOSTIC (#19 next lever): the hard-gate delivery gap = EDITED subjects that retrieve
    # with conf < C0 (false negatives). What makes an edit low-conf? Log per-edit conf vs subject length /
    # relation / whether the value is readable at all (unconditional max-alpha delivery). If low-conf edits
    # DO deliver unconditionally, the fix is raising in-store retrieval strength (writes/addressing), not
    # the readout. Correlate offline from [confdiag] lines.
    if os.environ.get("CAM_CONF_DIAG") == "1" and ec:
        os.environ.pop("CAM_LOGIT_GATE_C0", None); os.environ.pop("CAM_LOGIT_GATE_HARD", None)
        os.environ["CAM_LOGIT_INJECT"] = str(alphas[-1])                   # unconditional, max alpha
        epu = _persistent_preds(base, adapter, injector, tok, V, kept)     # is the value readable at all?
        os.environ["CAM_LOGIT_INJECT"] = "0"
        ep0 = _persistent_preds(base, adapter, injector, tok, V, kept)     # residual-only baseline
        from collections import defaultdict
        by_rel = defaultdict(lambda: [0, 0.0])                             # rid -> [n, sum_conf]
        lo_deliverable = 0                                                 # conf<C0 but readable unconditionally
        lo_n = 0
        for i, r in enumerate(kept):
            conf = ec[i]
            hu = int(epu[i] == r.new_tid); h0 = int(ep0[i] == r.new_tid)
            by_rel[r.relation_id][0] += 1; by_rel[r.relation_id][1] += conf
            below = c0 is not None and conf < c0
            if below:
                lo_n += 1; lo_deliverable += hu
            print(f"[confdiag] rid={r.relation_id} slen={len(r.subject_tids)} conf={conf:.2f} "
                  f"below_c0={int(below)} hit_uncond={hu} hit_base={h0}", flush=True)
        print(f"\n  --- per-edit conf diagnostic (C0={c0 if c0 is None else round(c0,3)}) ---", flush=True)
        print(f"  edits below C0: {lo_n}/{N}; of those, {lo_deliverable} DELIVER under unconditional "
              f"max-alpha (=> readable, just weakly RETRIEVED -> addressing/write lever, not readout)",
              flush=True)
        for rid, (n, s) in sorted(by_rel.items()):
            print(f"    {rid}: n={n:3d}  mean-conf={s / max(1, n):8.2f}", flush=True)

    # BANK SWEEP (§3.15 capstone): the delivery gap = weakly-retrieved edits (bank crowding). Scaling
    # disjoint banks de-crowds them -> raises in-store conf -> more edits clear the gate AND the bimodal
    # neighbour tail shrinks (fewer same-bank collisions). For each B: re-route into B banks, recompute the
    # per-B C0=em/12 hard gate, report gated delivery + DEP locality at the top alpha. (Disjoint banks are
    # persistent-path only — bind/tap unchanged — so this is a pure re-routing sweep, no re-bind.)
    bank_list = os.environ.get("CAM_BANK_SWEEP", "")
    if bank_list and dep:
        Bs = [int(x) for x in bank_list.split(",")]
        amax = alphas[-1]
        print(f"\n  --- BANK sweep (alpha={amax}, hard gate C0=em/12 per B) ---", flush=True)
        print(f"  {'B':>5}  {'edit-conf':>9} {'C0':>7} {'below':>6} | {'delivery':>9} {'DEP-keep':>9} {'DEP-leak':>9}",
              flush=True)
        for B in Bs:
            Vb = _init_banks(adapter, B)
            for r in kept:
                Vb = _persistent_write_one(adapter, Vb, r, pooled)
            ecb = [c for c in _cohort_confs(adapter, Vb, kept, pooled) if c == c]
            emb = _st.median(ecb) if ecb else float("nan")
            c0b = max(emb / 12.0, 0.5) if ecb else 0.5
            below = sum(1 for c in ecb if c < c0b)
            os.environ["CAM_LOGIT_INJECT"] = str(amax); os.environ["CAM_LOGIT_GATE_HARD"] = "1"
            os.environ["CAM_LOGIT_GATE_C0"] = str(c0b); os.environ["CAM_LOGIT_GATE_K"] = str(gate_k)
            ep = _persistent_preds(base, adapter, injector, tok, Vb, kept)
            delivery = sum(int(ep[i] == kept[i].new_tid) for i in range(N)) / max(1, N)
            prb = _persistent_preds(base, adapter, injector, tok, Vb, dep)
            nn = max(1, len(dep))
            keep = sum(int(prb[i] == dep[i].true_tid) for i in range(len(dep))) / nn
            leak = sum(int(prb[i] == dep[i].new_tid) for i in range(len(dep))) / nn
            print(f"  {B:>5}  {emb:>9.2f} {c0b:>7.2f} {below:>6} | {delivery:>9.3f} {keep:>9.3f} {leak:>9.3f}",
                  flush=True)
        print(f"  (more banks -> de-crowd -> higher edit-conf, fewer below-gate, delivery up at flat locality)",
              flush=True)

    for var, val in (("CAM_LOGIT_INJECT", saved_a), ("CAM_LOGIT_GATE_C0", saved_c0),
                     ("CAM_LOGIT_GATE_HARD", saved_hard)):
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val
    print(f"\n  DEP = neighbour's OWN subject drives retrieval (deployment — decides usability). "
          f"ADV = forced onto the EDIT's bank (upper bound).", flush=True)
    print(f"  want: delivery UP, DEP-keep flat, DEP-leak ~0. Gate helps iff conf separates edited vs "
          f"neighbour (see separation above).", flush=True)
    print("=" * 64, flush=True)
    return {"n_edits": N, "n_dep": len(dep), "n_adv": len(adv),
            "conf_edited": em, "conf_nbr": dm, "gate_c0": c0,
            "ungated": ungated, "gated": gated, "hard": hardg, "thr_sweep": thr_rows}


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
    ap.add_argument("--base1", type=str, default="",
                    help=f"donor (frozen base-1) HF model id; default = the donor recorded in "
                         f"--load-ckpt (the base the memory was bound on), else {MODEL} — the donor "
                         f"every published number used. Swapping it is an untested-donor "
                         f"experiment, not a reproduction.")
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
                    choices=["dict", "natural", "varied", "counterfactual", "counterfactual_multi"],
                    help="doc format: 'dict' (terse '<cargo>: <name>', default, byte-preserved) or "
                         "'natural' (natural-language single-relation facts '<Subject> lives in <Object>.'; "
                         "issue #1 realism probe — subject=KEY, object=VALUE, answer=object; supports "
                         "--cargo-tokens K>1 for a K-token real-word object phrase '<Subj> lives in <w0 w1>') "
                         "or 'varied' (per-fact relation drawn from a small template set — heterogeneous "
                         "facts; each binding slot m uses relations[m%%R], subject=KEY, object=VALUE) "
                         "or 'counterfactual' (KNOWLEDGE EDITING: real country->capital facts the base "
                         "KNOWS, with DERANGED capitals in memory. PROBE-FILTER-EDIT: probe the frozen "
                         "base first, keep only facts it demonstrably knows, bind the counterfactual "
                         "capitals on that filtered set; metrics = mem/no_mem counterfactual-acc AND "
                         "mem/no_mem PRIOR-acc + a VALID/INVALID gate on no_mem prior-acc)")
    ap.add_argument("--cf-probe-batch", type=int, default=16, dest="cf_probe_batch",
                    help="counterfactual: batch size for the base prior-knowledge probe/filter forward")
    ap.add_argument("--cf-probe-cap", type=int, default=8000, dest="cf_probe_cap",
                    help="counterfactual_multi: how many objects-single records to probe for base-known "
                         "facts (bigger -> more facts per relation -> higher M reachable; slower probe)")
    ap.add_argument("--multi-relations", type=int, default=4, dest="multi_relations",
                    help="counterfactual_multi (#16): how many DISTINCT CounterFact relations to edit "
                         "together in one memory (top-N base-known relations by fact count). Each doc "
                         "slot m cycles relations[m%%N]; each fact keeps its own real prompt.")
    ap.add_argument("--dataset", type=str, default="curated", choices=["curated", "counterfact"],
                    help="counterfactual fact source: 'curated' (the hand-picked country->capital table, "
                         "default, unchanged) or 'counterfact' (REAL ROME CounterFact benchmark — Track 1: "
                         "probe with each record's own prompt, bind target_new, adds LOCALITY + "
                         "GENERALIZATION metrics). Only active with --phrasing counterfactual.")
    ap.add_argument("--data-dir", type=str, default="data", dest="data_dir",
                    help="dir holding counterfact.json (for --dataset counterfact)")
    ap.add_argument("--locality-cap", type=int, default=256, dest="locality_cap",
                    help="max locality/generalization probe prompts to score (budget cap)")
    ap.add_argument("--locality-weight", type=float, default=0.0, dest="locality_weight",
                    help="Track 1 SURGICAL editing: weight on the locality-preservation KL loss during "
                         "tap training (tap-on ≈ frozen-base tap-off on HELD-OUT neighbour prompts). >0 "
                         "teaches the tap's null slot to leave out-of-store facts alone; 0 = edit-only "
                         "(current behavior). Neighbours are split 50/50 train/eval (no leak).")
    ap.add_argument("--conf-gate", action="store_true", dest="conf_gate",
                    help="Track 1: gate the tap injection by an explicit STORE-CONFIDENCE scalar "
                         "(pk_store factual-head pre-norm retrieval magnitude) instead of relying on the "
                         "learned null slot. c=sigmoid(scale*(conf/EMA-bias)) scales the whole injection: "
                         "a paraphrase retrieves its own edit (strong->deliver) while a neighbour retrieves "
                         "nothing (weak->inert), decoupling delivery from PROMPT NOVELTY (the null slot's "
                         "proxy) -> closes the locality<->generalization gap. pk-store only.")
    ap.add_argument("--persistent-eval", action="store_true", dest="persistent_eval",
                    help="Track 4 (#19): after training, write ALL kept edits into ONE standing store "
                         "(incremental, no episodic doc) and query each edit's natural prompt — the "
                         "online/persistent memory test. counterfactual_multi + --dataset counterfact only.")
    ap.add_argument("--persistent-sweep", action="store_true", dest="persistent_sweep",
                    help="Track 4 (#19): RETENTION/INTERFERENCE curve — checkpoint during the incremental "
                         "write phase and re-query a FIXED early cohort (--persistent-cohort edits) as the "
                         "store grows. A decaying early-cohort curve = interference (does edit #1 survive "
                         "writing edit #N?), separated from cumulative all-so-far delivery. Implies "
                         "--persistent-eval.")
    ap.add_argument("--persistent-cohort", type=int, default=10, dest="persistent_cohort",
                    help="Track 4 sweep: size of the fixed early cohort (first C edits) tracked across "
                         "checkpoints for the retention/interference curve.")
    ap.add_argument("--probe-only", action="store_true", dest="probe_only",
                    help="counterfactual_multi: run + CACHE the base-known probe/filter, then exit before "
                         "stage-1/2. Cheap way to pre-warm the probe cache on ONE card (the probe is the "
                         "big recurring GPU cost) so a later 2-card sweep is a cache hit.")
    ap.add_argument("--persistent-overwrite", action="store_true", dest="persistent_overwrite",
                    help="Track 4 (#19) incr#3: after the retention sweep, run the ONLINE UPDATE test — "
                         "write subject->A then subject->B (different value) into the SAME standing store "
                         "and score whether the delta-write cleanly UPDATES to B (vs stale A). Implies "
                         "--persistent-sweep.")
    ap.add_argument("--persistent-solo", action="store_true", dest="persistent_solo",
                    help="Phase R/R0 (#19): SINGLE-FACT FIDELITY — write each edit ALONE in its own store "
                         "and query it (no collision). Isolates the per-fact store/tap ceiling. Implies "
                         "--persistent-sweep.")
    ap.add_argument("--persistent-locality", action="store_true", dest="persistent_locality",
                    help="Phase R (#19/#67): LOGIT-INJECTION LOCALITY — after writing all edits, sweep "
                         "CAM_LOGIT_INJECT alpha and score edit DELIVERY together with neighbour KEEP/LEAK "
                         "(does forcing the object logit damage same-true-object neighbours?). The decisive "
                         "delivery<->locality trade for the logit-injection paradigm. Implies "
                         "--persistent-sweep.")
    ap.add_argument("--apitask", type=str, default="",
                    help="API-override PREMISE PROBE (issue #19 use case): path to a curated library-API "
                         "facts JSON (CounterFact schema; target_true.str = the CORRECT current answer). "
                         "Probe the frozen base + report how often it is already correct vs WRONG (editable), "
                         "then exit. The cheap gate before building the full API-override edit+eval.")
    ap.add_argument("--persistent-generate", action="store_true", dest="persistent_generate",
                    help="GENERATION-COHERENCE (reality check): after writing all edits, GENERATE "
                         "CAM_GEN_LEN tokens (greedy) from a sample of edit prompts with memory OFF vs ON "
                         "(tap + hard-conf-gated logit injection) so a human can eyeball fluency + whether "
                         "the edit takes in free generation, not just next-token argmax. Implies "
                         "--persistent-sweep.")
    ap.add_argument("--query-rel", type=str, default="", dest="query_rel",
                    help="ADVERSARIAL paraphrase probe (issue #10, natural phrasing only): bind and "
                         "train exactly as normal (bindings + training queries use ' lives in'), then "
                         "ALSO evaluate with the query phrased as this relation (e.g. ' resides in') — "
                         "does the memory address by the subject or by the literal relation tokens? "
                         "Reported next to the standard eval; leading space matters for tokenization.")
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

    # resolve the donor BEFORE loading it: explicit --base1 wins; a RELOAD falls back to the donor
    # recorded in the checkpoint (the base the memory was actually bound on); else the historical
    # default. An explicit override that disagrees with the record is almost certainly a mistake
    # (the adapter rebuilds on the wrong embedding table and load_state_dict cannot catch a
    # same-hidden-size donor swap), so say so loudly.
    if args.load_ckpt:
        _peek = torch.load(args.load_ckpt, map_location="cpu", weights_only=False)
        _recorded = _peek.get("base1")
        del _peek
        if args.base1 and _recorded and args.base1 != _recorded:
            print(f"[mag] WARNING: --base1 {args.base1} != ckpt-recorded donor {_recorded} — "
                  f"the memory was bound on {_recorded}; results on a different donor are garbage "
                  f"unless you know exactly why you are doing this.", flush=True)
        args.base1 = args.base1 or _recorded or MODEL
    else:
        args.base1 = args.base1 or MODEL

    base, tok = load_frozen_base(args.base1)
    if getattr(args, "apitask", ""):                      # API-override premise probe -> report + exit
        apitask_probe(base, tok, args.apitask)
        return
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
    counterfactual = args.phrasing in ("counterfactual", "counterfactual_multi")
    multi_relation = args.phrasing == "counterfactual_multi"
    cf_meta = None
    cf_records = None                    # Track 1 (CounterFact) records for locality/generalization eval
    loc_buckets, loc_eval = None, None   # surgical editing: train/eval split of neighbour prompts
    if counterfactual:
        assert args.cargo_tokens == 1, "counterfactual phrasing is single-token only"
        # PROBE the frozen base -> FILTER to demonstrably-known facts -> DERANGE the memory capitals.
        # (this is the GPU probe forward the orchestrator runs; the filtered set size + example edits
        # are logged above the bind loop.)
        if args.dataset == "counterfact" and multi_relation:
            # Track 1 MULTI-RELATION (#16): edit N distinct relations in ONE memory (faithful prefix).
            builder, cf_records, prior_acc_full = setup_counterfact_multi(base, tok, args)
            if getattr(args, "probe_only", False):
                # cache-build run: the probe (the costly part) ran + cached inside setup; exit before the
                # single-card-fragile stage-2 so the cache is ready for a subsequent (2-card) sweep.
                print(f"[mag][cf-multi] --probe-only: {len(cf_records)} edits ready + cached; exiting.",
                      flush=True)
                return
        elif args.dataset == "counterfact":
            # Track 1: REAL CounterFact benchmark (probe with each record's own prompt, bind target_new).
            builder, cf_records, prior_acc_full = setup_counterfact(base, tok, args)
            cf_meta = {"kept": [(r.subject, r.true_str) for r in cf_records], "perm": None}
            # Always split neighbours 50/50 so LOCALITY is scored on the HELD-OUT half — identical eval
            # set whether or not the locality-preservation loss is on (a clean lw=0 vs lw>0 control). The
            # train half is only consumed by train_taps when --locality-weight > 0.
            loc_buckets, loc_eval = build_locality_split(cf_records, tok)
            n_tr = sum(len(v) for v in loc_buckets.values())
            print(f"[mag][cf] locality split: {n_tr} train-neighbour negatives, {len(loc_eval)} "
                  f"held-out eval neighbours | locality-weight {args.locality_weight}"
                  f"{' (SURGICAL)' if args.locality_weight > 0 else ' (edit-only control)'}", flush=True)
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
    # PARAPHRASE probe (#10): an EVAL-ONLY builder whose docs are identical except the query relation.
    # Bind/train never see it — the store must address by the subject, not the literal relation tokens.
    builder_pq = None
    if args.query_rel:
        assert args.phrasing == "natural", "--query-rel is a natural-phrasing probe"
        builder_pq = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="natural",
                                cargo_tokens=args.cargo_tokens, cargo_words=cargo_words,
                                query_rel=args.query_rel)
        print(f"[mag] PARAPHRASE probe: bindings ' lives in' | eval query '{args.query_rel}'", flush=True)

    # ---- RELOAD path: reuse a fixed v0 memory checkpoint (no re-bind) and reproduce the V0 eval ----
    if args.load_ckpt:
        adapter, injector, L, ck = load_ckpt(args.load_ckpt, embed_weight, base, DEV, builder=builder)
        print(f"[mag] {args.base1} | H={H} n_layers={n_layers} | RELOAD tap L={L} | "
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
        if builder_pq is not None:
            pq_carry, pq_abl, _pc, _pa = eval_direct(adapter, builder_pq, rng, args)
            print(f"[mag][{L}] PARAPHRASE carry: {pq_carry:.3f} (ablated {pq_abl:.3f})", flush=True)
            gen_pq = eval_generative_mag(base, adapter, injector, builder_pq, rng, args)
            verdict(f"{L}|paraphrase", pq_carry, gen_pq, 1 / args.M)
        injector.detach()
        print("\n[mag] RELOAD SANITY (tap -> memory / no_memory / ceiling):", flush=True)
        print(f"  L={L:>8}  {m_acc:.3f} / {nm_acc:.3f} / {gen['local_control'][1]:.3f}", flush=True)
        print("[mag] PASS if memory ≫ no_memory (reproduces V0 on base-1 from the saved memory).", flush=True)
        return

    layers = ([int(x) for x in args.tap_layers.split(",") if x != ""]
              if args.tap_layers else [n_layers // 2])
    print(f"[mag] {args.base1} | H={H} n_layers={n_layers} | tap_layers={layers} multi={args.multi} | "
          f"K={args.k} mem_dim={args.mem_dim} | chance acc={1/args.M:.3f}", flush=True)

    # ---- stage 1: bind once ----
    adapter = build_adapter(args, embed_weight, H, builder=builder)
    with StageCost(f"stage-1 bind (M={args.M}, {args.bind_steps} steps, batch {args.batch})"):
        d_carry = bind_adapter(adapter, builder, rng, args)
    # free stage-1's AdamW state + autograd fragments before stage-2 — on a single 16GB card the leftover
    # fragmentation (bind runs many steps) is what pushed stage-2's first backward over the wall + wedged.
    import gc as _gc0
    _gc0.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- stage 2: MAG delivery ----
    if (getattr(args, "persistent_overwrite", False) or getattr(args, "persistent_solo", False)
            or getattr(args, "persistent_locality", False) or getattr(args, "persistent_generate", False)):
        args.persistent_sweep = True         # the online-update / solo / locality tests run after the sweep
    if getattr(args, "persistent_sweep", False):
        args.persistent_eval = True          # the sweep is a superset of the persistent eval
    configs = [layers] if args.multi else [[L] for L in layers]
    # per-relation conf-gate EMA: one slot per edited relation (multi), else a single global EMA.
    n_rel = getattr(builder, "R", 1) if multi_relation else 1
    summary = []
    for cfg in configs:
        tag = "+".join(map(str, cfg))
        injector = MAGInjector(base, cfg, args.mem_dim, n_heads=args.tap_heads,
                               conf_gate=getattr(args, "conf_gate", False), n_rel=n_rel).to(DEV)
        with StageCost(f"stage-2 tap fit L={tag} ({args.steps} steps, batch {args.batch})"):
            train_taps(base, adapter, injector, builder, rng, args, tag, loc_buckets=loc_buckets)
        # free the training-phase CUDA memory (AdamW optimizer states + autograd graph fragments) before
        # eval — on a 16GB card the leftover fragmentation is what OOMs the high-M eval forwards.
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if getattr(args, "persistent_sweep", False):
            # Track 4 (#19) retention/interference sweep is the ONLY eval we want here. SKIP the delivery /
            # counterfactual / locality-generalization evals: each forwards the whole M*bind_len doc (and
            # loc/gen the neighbour set) at many edits, which fragments a 16GB card — the OOM/wedge source.
            # The retention question (does edit #1 survive writing edit #N?) uses none of them.
            if cf_records is None:
                print(f"[mag][{tag}] --persistent-sweep needs --dataset counterfact records; skipping.",
                      flush=True)
            else:
                with StageCost(f"persistent (online) retention sweep L={tag}"):
                    eval_persistent(base, adapter, injector, tok, cf_records, args)
                if getattr(args, "persistent_overwrite", False):
                    with StageCost(f"persistent online-update L={tag}"):
                        eval_persistent_overwrite(base, adapter, injector, tok, cf_records, args)
                if getattr(args, "persistent_solo", False):
                    with StageCost(f"persistent solo-fidelity L={tag}"):
                        eval_persistent_solo(base, adapter, injector, tok, cf_records, args)
                if getattr(args, "persistent_locality", False):
                    with StageCost(f"persistent logit-injection locality L={tag}"):
                        eval_persistent_locality(base, adapter, injector, tok, cf_records, args)
                if getattr(args, "persistent_generate", False):
                    with StageCost(f"persistent generation-coherence L={tag}"):
                        eval_persistent_generate(base, adapter, injector, tok, cf_records, args)
            injector.detach()
            continue
        with StageCost(f"delivery eval L={tag}"):
            gen = eval_generative_mag(base, adapter, injector, builder, rng, args)
        m_acc, nm_acc = verdict(tag, d_carry, gen, 1 / args.M)
        if builder_pq is not None:
            # the store/tap were trained with ' lives in' queries ONLY; this asks the same questions
            # phrased differently. carry = the store's own paraphrase addressing; the verdict block =
            # paraphrase delivery through the frozen base.
            pq_carry, pq_abl, _pc, _pa = eval_direct(adapter, builder_pq, rng, args)
            print(f"[mag][{tag}] PARAPHRASE carry: {pq_carry:.3f} (ablated {pq_abl:.3f}) "
                  f"vs standard {d_carry:.3f}", flush=True)
            with StageCost(f"paraphrase delivery eval L={tag}"):
                gen_pq = eval_generative_mag(base, adapter, injector, builder_pq, rng, args)
            verdict(f"{tag}|paraphrase", pq_carry, gen_pq, 1 / args.M)
        if counterfactual:
            # the 4 knowledge-editing metrics + VALID/INVALID gate (scores mem/no_mem against BOTH the
            # counterfactual capital and the true prior at the SAME query position).
            cf = eval_counterfactual(base, adapter, injector, builder, rng, args)
            verdict_counterfactual(tag, cf, 1 / args.M)
            if cf_records is not None:
                if getattr(args, "persistent_eval", False):
                    # Track 4 (#19): write ALL edits into ONE standing store, query each — online memory.
                    # Run BEFORE loc/gen (which fragments a 16GB card at many edits) so it always reports.
                    import gc as _gc
                    _gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    with StageCost(f"persistent (online) eval L={tag}"):
                        eval_persistent(base, adapter, injector, tok, cf_records, args)
                # Track 1 real-editing metrics: LOCALITY (neighbours preserved) + GENERALIZATION (edit
                # fires on paraphrases). Only for --dataset counterfact (records carry the probe prompts).
                lg = eval_locality_generalization(base, tok, injector, adapter, builder, cf_records,
                                                  args, cap=args.locality_cap, loc_override=loc_eval)
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

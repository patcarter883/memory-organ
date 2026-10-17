"""E1 (#104) — trigger-local DISTILLED delivery for the frozen-base in-distribution ripple protocol.

The delivery GEOMETRY under test (spec: docs/CAM_TRIGGER_DELIVERY_SPEC.md; analysis note:
docs/research/2026-10-17-engram-tech-transfer.md): a CONTENT-TRIGGERED read — the trained tap fires
only at positions where the staged edit's subject (its hashed multi-order n-gram keys) appears in the
CURRENT sequence — trained by KL DISTILLATION (the settled lever) instead of the next-token delivery
loss:

    teacher   = frozen base + edit fact as a text prefix (in-context)
    student   = frozen base + TriggerInjector masked delivery, NO edit text in the prompt
    loss      = KL(teacher || student) over next-token distributions on per-edit self-study passages
    trainable = ONLY the injector params (base + store frozen, exactly as every CAM stage-2)

Self-study passages are recorded FACT-FREE (the seed stems never contain the edit; the fact is
prepended only to the GENERATION prompt so the base writes about it) — the student's only route to
the fact is the trigger. Eval is the SAME in-distribution language-pivot 2-hop the tap was scored on
(mquake_ripple helpers reused verbatim; _gen / _single_hop_control call injector.set_trigger_ids per
forward), with no_edit + RAG scored in the SAME pass, the single-hop delivery GATE
(CAM_MQUAKE_CTRL_MIN, default 0.40), the confound filter (gold_cf != gold_true), and per-case fire
diagnostics (a dead mask silently reproduces no_edit — fire stats are mandatory before any ripple
number is read).

Arms (--trigger-arms):
  trigger : masked content-triggered read       (A2 — the E1 geometry)
  nomask  : identical module + objective, no mask (A1 — isolates the geometry from the objective)

Both scorers are reported for every condition: the strict _match (the July in-dist tap runs) and the
head-word _hit (cartridge_ripple), so numbers are readable against BOTH prior result sets.

Entry point is recall_mag.py (the single setup path); see tools/trigger_ripple.sh for the launch.
"""
import gc
import os
import sys
import time

import numpy as np
import torch

try:
    from . import mquake_ripple as MR
    from .m2_adapter import DEV
    from .gated_tap import TriggerInjector
    from .cartridge_train import _teacher_logprobs
    from .cartridge_ripple import _hit
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    import mquake_ripple as MR                                    # noqa: E402
    from m2_adapter import DEV                                    # noqa: E402
    from gated_tap import TriggerInjector                          # noqa: E402
    from cartridge_train import _teacher_logprobs                  # noqa: E402
    from cartridge_ripple import _hit                              # noqa: E402


# Fact-FREE seed stems for the self-study recording (x). The edit fact is prepended ONLY to the
# generation prompt; these stems are all the STUDENT ever sees besides the generated answer.
_CLEAN_STEMS = [
    "Restate the fact above in your own words.\nRestatement:",
    "Summarize the fact above and note what it implies.\nSummary:",
    "Write a question about the fact above, then answer it.\nQuestion:",
    "Describe a situation where the fact above would matter.\nSituation:",
    "Explain the fact above to a student, step by step.\nExplanation:",
    "List a few things that follow from the fact above.\n-",
    "Write a short paragraph that mentions the fact above.\nParagraph:",
    "Q: What does the fact above tell us? A:",
]


def _orders_from(args):
    raw = getattr(args, "trigger_orders", "2,3") or "2,3"
    return tuple(int(x) for x in str(raw).replace(";", ",").split(",") if x.strip())


def _sample_continuation(base, tok, prompt_ids, gen_len, temperature, device, rng):
    """Mirrors cartridge_train._sample_gen (greedy when temperature == 0). CACHED incremental decode by
    default — the full-sequence re-prefill loop re-enters the chunked-GDN path every step, which wedges
    intermittently on this box (observed 2026-10-17; see tools/gen_cache_probe.py for the parity check:
    token-exact vs the no-cache loop). CAM_TRIG_NO_CACHE=1 restores the historical loop."""
    out = []
    eos = tok.eos_token_id
    dbg = os.environ.get("CAM_TRIG_DEBUG") == "1"
    if os.environ.get("CAM_TRIG_NO_CACHE", "0") == "1":
        cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        for _i in range(gen_len):
            with torch.no_grad():
                logits = base(input_ids=cur).logits[:, -1].float()
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                g = torch.Generator(device=probs.device)
                g.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
                nxt = torch.multinomial(probs, 1, generator=g)[:, 0]
            else:
                nxt = logits.argmax(-1)
            t = int(nxt.item())
            if dbg:
                print(f"[ssgen] T{cur.shape[1]} step{_i} tok{t}", flush=True)
            if eos is not None and t == eos:
                break
            out.append(t)
            cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
            if "\n\n" in tok.decode(out):
                break
        return out
    step = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    past = None
    with torch.no_grad():
        for _i in range(gen_len):
            outp = base(input_ids=step, past_key_values=past, use_cache=True)
            past = outp.past_key_values
            logits = outp.logits[:, -1].float()
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                g = torch.Generator(device=probs.device)
                g.manual_seed(int(rng.integers(0, 2 ** 31 - 1)))
                nxt = torch.multinomial(probs, 1, generator=g)[:, 0]
            else:
                nxt = logits.argmax(-1)
            t = int(nxt.item())
            if dbg:
                print(f"[ssgen] T{int(past.get_seq_length())} step{_i} tok{t}", flush=True)
            if eos is not None and t == eos:
                break
            out.append(t)
            step = nxt.view(1, 1)
            if "\n\n" in tok.decode(out):
                break
    return out


def build_selfstudy(base, tok, fact, args, device, rng):
    """Per-edit self-study passages x: [fact-free stem ⊕ generated answer]. The generation prompt gets
    the fact prepended (teacher-side knowledge), the RECORDED x does not."""
    n = int(getattr(args, "trigger_nseeds", 6))
    spp = int(getattr(args, "trigger_spp", 2))
    gen_len = int(getattr(args, "trigger_genlen", 48))
    fact = fact.strip()
    if not fact.endswith("."):
        fact = fact + "."
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    dbg = os.environ.get("CAM_TRIG_DEBUG") == "1"
    passages = []
    for si, stem in enumerate(_CLEAN_STEMS[:n]):
        stem_ids = tok(stem, add_special_tokens=False).input_ids
        gen_prompt = bos + tok(fact + "\n\n" + stem, add_special_tokens=False).input_ids
        for s in range(spp):
            temp = 0.0 if s == 0 else 0.8
            if dbg:
                print(f"[ssgen] PASSAGE stem{si} s{s} promptlen {len(gen_prompt)} temp {temp}", flush=True)
            cont = _sample_continuation(base, tok, gen_prompt, gen_len, temp, device, rng)
            if not cont:
                continue
            passages.append(stem_ids + cont)
    return passages


def build_arms(base, args, n_rel):
    """One TriggerInjector per requested arm ({'trigger': masked A2, 'nomask': unmasked A1})."""
    want = [a.strip() for a in str(getattr(args, "trigger_arms", "trigger")).split(",") if a.strip()]
    arms = {}
    for a in want:
        if a not in ("trigger", "nomask"):
            print(f"[trig] unknown arm {a!r} (known: trigger, nomask) — skipped.", flush=True)
            continue
        arms[a] = TriggerInjector(
            base, int(getattr(args, "trigger_layer", 12)), args.mem_dim,
            n_heads=args.tap_heads, conf_gate=getattr(args, "conf_gate", False), n_rel=n_rel,
            orders=_orders_from(args), apply_mask=(a == "trigger")).to(DEV)
        print(f"[trig] arm {a!r}: TriggerInjector @L{getattr(args, 'trigger_layer', 12)} "
              f"apply_mask={a == 'trigger'} orders={_orders_from(args)}", flush=True)
    return arms


def build_work(base, tok, adapter, edits, args, verbose=True):
    """Per-edit work set shared by ALL arms: episodic bank + fact-free self-study passages + dense
    teacher targets. Deterministic (seed+7); computed ONCE per run — the bank, passages and teacher
    log-probs are identical across arms (only the injector differs)."""
    rng = np.random.default_rng(int(getattr(args, "seed", 0)) + 7)
    seg_len, qa_seg = int(args.seg_len), int(args.qa_seg)
    work = []
    t_ss = time.time()
    for i, r in enumerate(edits):
        eb = MR._episodic_bank(adapter, tok, r, seg_len, qa_seg, rng)
        if eb is None:
            continue
        bank, conf, relidx = eb
        edit_text = r.prompt_text + " " + r.new_str
        passages = build_selfstudy(base, tok, edit_text, args, DEV, rng)
        teach = [_teacher_logprobs(base, tok, edit_text, x, DEV) for x in passages]
        if os.environ.get("CAM_TRIG_DEBUG") == "1":
            _c = None if conf is None else conf.detach().float()
            print(f"[trig][dbg] self-study {i} ({r.subject!r}): bank_absmax "
                  f"{float(bank.detach().float().abs().max().item()):.3e} conf "
                  + ("None" if _c is None else f"nan {int(torch.isnan(_c).sum())} "
                     f"absmax {float(_c.abs().max().item()):.3e} vals {_c.flatten()[:4].tolist()}"),
                  flush=True)
        work.append((r, bank, conf, relidx, passages, teach))
        if verbose:
            print(f"[trig]   self-study {i + 1}/{len(edits)}: {r.subject!r} "
                  f"{r.true_str!r}->{r.new_str!r} ({len(passages)} passages)", flush=True)
    if verbose:
        print(f"[trig] work set ready: {len(work)}/{len(edits)} edits (shared across arms) in "
              f"{time.time() - t_ss:.0f}s", flush=True)
    return work


def train_trigger(base, tok, adapter, injector, edits, args, verbose=True, work=None):
    """KL-distill the injector over the shared work set (module docstring). Returns stats or None."""
    if work is None:
        work = build_work(base, tok, adapter, edits, args, verbose=verbose)
    if not work:
        return None
    rng = np.random.default_rng(int(getattr(args, "seed", 0)) + 7)
    max_steps = int(getattr(args, "trigger_steps", 300))
    lr = float(getattr(args, "trigger_lr", 1e-3))
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    seg_len, qa_seg = int(args.seg_len), int(args.qa_seg)

    for p in base.parameters():
        p.requires_grad_(False)
    trainable = [q for q in injector.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    injector.attach()
    injector.train()
    step, last, skipped, scrubbed = 0, float("nan"), 0, 0
    debug = os.environ.get("CAM_TRIG_DEBUG", "0") == "1"
    t0 = time.time()
    while step < max_steps:
        order = list(range(len(work)))
        rng.shuffle(order)
        for wi in order:
            if step >= max_steps:
                break
            r, bank, conf, relidx, passages, teach = work[wi]
            j = int(rng.integers(0, len(passages)))
            x = passages[j]
            tv, _ti, _surp = teach[j]                      # dense teacher log-probs [Lx, V] (fp16 cpu)
            Lx = len(x)
            s_in = torch.tensor([bos + list(x)], dtype=torch.long, device=DEV)
            injector.set_bank(bank, conf=conf, relidx=relidx)
            injector.set_trigger_keys(r.subject_tids)
            injector.set_trigger_ids(s_in[0])
            s_logits = base(input_ids=s_in).logits[0]      # student: base + trigger delivery, no fact
            j0 = max(0, 1 - len(bos))
            s_sl = s_logits[len(bos) + j0 - 1:len(bos) + Lx - 1].float()
            s_lp = torch.nn.functional.log_softmax(s_sl, dim=-1)
            t_lp = tv[j0:].to(DEV).float()
            t_p = t_lp.exp()
            loss = (t_p * (t_lp - s_lp)).sum(-1).mean()    # per-position KL (mirrors train_cartridge)
            finite = bool(torch.isfinite(loss).item())
            if debug and step < 4:                         # CAM_TRIG_DEBUG=1: one line per early step
                print(f"[trig][dbg] step {step}: t_lp[min {t_lp.min().item():.2f} ninf "
                      f"{int(torch.isinf(t_lp).sum())} nnan {int(torch.isnan(t_lp).sum())}] "
                      f"s_lp[min {s_lp.min().item():.2f} nnan {int(torch.isnan(s_lp).sum())}] "
                      f"s_sl[absmax {float(s_sl.abs().max().item()):.2e} nnan "
                      f"{int(torch.isnan(s_sl).sum())}] t_p_nan {int(torch.isnan(t_p).sum())} "
                      f"loss {float(loss.item()):.4f} finite={finite}", flush=True)
            opt.zero_grad(set_to_none=True)
            if not finite:                                 # NaN-guard: never let a bad step poison the taps
                skipped += 1
                if verbose and skipped <= 3:
                    print(f"[trig]   step {step}: non-finite loss -> skipped ({skipped} so far)", flush=True)
                step += 1
                continue
            loss.backward()
            # E1 (#104): the first real-gradient backprop through the frozen base + tap read path can
            # emit ±inf in a param grad (observed: to_o.weight, grad-norm ~5e3). clip_grad_norm_ then
            # scales with 1/inf -> inf*0 = NaN -> the param goes NaN and every later forward is NaN.
            # Scrub non-finite grads to 0 BEFORE the clip (a standard defensive pass); count the steps.
            if scrubbed == 0 and debug:
                nnf = sum(int((~torch.isfinite(p.grad)).any().item()) for p in trainable if p.grad is not None)
                print(f"[trig][dbg] step {step}: grads non-finite {nnf} (pre-scrub)", flush=True)
            step_scrubbed = 0
            for p in trainable:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    step_scrubbed += 1
            if step_scrubbed:
                scrubbed += 1
                if verbose and scrubbed <= 3:
                    print(f"[trig]   step {step}: scrubbed non-finite grads on {step_scrubbed} param(s) "
                          f"({scrubbed} step(s) total)", flush=True)
            gn = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if debug and step < 4:
                ngn = sum(int(torch.isnan(p.grad).any().item()) for p in trainable if p.grad is not None)
                print(f"[trig][dbg] step {step}: grad_norm {float(gn):.3e} grads_nan {ngn}", flush=True)
            opt.step()
            if debug and step < 4:
                npn = sum(int((~torch.isfinite(p)).any().item()) for p in trainable)
                amax = max((float(p.detach().abs().max().item()) for p in trainable), default=0.0)
                print(f"[trig][dbg] step {step}: params nonfinite {npn} | weights absmax {amax:.3e}", flush=True)
                if npn:
                    for nm, pp in injector._mag.taps["12"].named_parameters():
                        if not torch.isfinite(pp).all():
                            print(f"[trig][dbg]   non-finite param: {nm} "
                                  f"absmax {float(pp.detach().abs().max().item()):.3e}", flush=True)
            if step % 25 == 0:                                 # allocator pool hygiene (no compaction on
                torch.cuda.empty_cache()                       # this platform -> return blocks early)
            last = float(loss.item())
            step += 1
            if verbose and step % 50 == 0:
                print(f"[trig]   step {step}/{max_steps} KL={last:.4f} ({time.time() - t0:.0f}s)",
                      flush=True)
    injector.set_bank(None)
    injector.eval()
    return {"steps": step, "final_kl": last, "edits": len(work), "secs": time.time() - t0,
            "skipped": skipped, "scrubbed": scrubbed}


def train_trigger_delivery(base, tok, adapter, injector, work, args, verbose=True):
    """Masked DELIVERY trainer — the July tap-fit objective (CE on the target object at the edit's own
    cloze), made mask-compatible: with the edit's bank staged and the fire mask installed on
    "The mother tongue of <subject> is", cross-entropy on the new-object token. batch=1; short
    sequences; the gradient reaches the tap through the subject-span injections — the exact geometry
    the eval's single-hop control gate probes. Use via --trigger-objective delivery."""
    if not work:
        return None
    rng = np.random.default_rng(int(getattr(args, "seed", 0)) + 11)
    steps = int(getattr(args, "trigger_steps", 300))
    lr = float(getattr(args, "trigger_lr", 1e-3))
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    for p in base.parameters():
        p.requires_grad_(False)
    trainable = [q for q in injector.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    injector.attach()
    injector.train()
    step, last, skipped, scrubbed = 0, float("nan"), 0, 0
    t0 = time.time()
    while step < steps:
        wi = int(rng.integers(0, len(work)))
        r, bank, conf, relidx, _passages, _teach = work[wi]
        s_in = torch.tensor([bos + tok(r.prompt_text, add_special_tokens=False).input_ids],
                            dtype=torch.long, device=DEV)
        injector.set_bank(bank, conf=conf, relidx=relidx)
        injector.set_trigger_keys(r.subject_tids)
        injector.set_trigger_ids(s_in[0])
        logits = base(input_ids=s_in).logits[0, -1].float()
        loss = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([int(r.new_tid)], dtype=torch.long, device=DEV))
        finite = bool(torch.isfinite(loss).item())
        opt.zero_grad(set_to_none=True)
        if not finite:
            skipped += 1
            step += 1
            continue
        loss.backward()
        step_scrubbed = 0
        for p in trainable:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                step_scrubbed += 1
        if step_scrubbed:
            scrubbed += 1
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if step % 25 == 0:
            torch.cuda.empty_cache()
        last = float(loss.item())
        step += 1
        if verbose and step % 100 == 0:
            print(f"[trig]   [delivery] step {step}/{steps} CE={last:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)
    injector.set_bank(None)
    injector.eval()
    return {"objective": "delivery", "steps": step, "final_ce": last, "edits": len(work),
            "secs": time.time() - t0, "skipped": skipped, "scrubbed": scrubbed}


def eval_trigger_ripple(base, adapter, injector, tok, args, cf_records):
    """In-distribution language-pivot 2-hop for the trigger condition (+ no_edit/RAG same-pass, both
    scorers + fire diagnostics). Mirrors mquake_ripple.eval_indist_ripple's layout."""
    assert cf_records is not None, "trigger eval needs the counterfactual_multi records (cf_records)"
    injector.eval()
    lang = MR._lang_edits(cf_records, int(getattr(args, "mquake_limit", 0) or 0))
    if not lang:
        print("[trig] no single-token language-relation edits; nothing to eval.", flush=True)
        return {"n": 0}
    gen_len = int(os.environ.get("CAM_MQUAKE_GEN_LEN", "40"))
    show = int(os.environ.get("CAM_MQUAKE_SHOW", "12"))
    ctrl_min = float(os.environ.get("CAM_MQUAKE_CTRL_MIN", "0.40"))
    seg_len, qa_seg = int(args.seg_len), int(args.qa_seg)
    rng = np.random.default_rng(int(getattr(args, "seed", 0)))
    base_embed = base.get_input_embeddings()
    lm = base.get_output_embeddings().weight
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    hop, qtmpl, omap = MR._select_twohop(base, adapter, tok, injector, args, lang, tag="trig",
                                         force=os.environ.get("CAM_RIPPLE_EVAL_HOP"))

    rows = []
    ctrl_hits = n_ctrl = n_collide = n_skip = 0
    for i, r in enumerate(lang):
        gold_true, gold_cf = omap.get(r.true_str), omap.get(r.new_str)
        if not gold_true or not gold_cf or MR._norm(gold_true) == MR._norm(gold_cf):
            n_collide += 1
            continue
        eb = MR._episodic_bank(adapter, tok, r, seg_len, qa_seg, rng)
        if eb is None:
            n_skip += 1
            continue
        bank, conf, relidx = eb
        injector.set_trigger_keys(r.subject_tids)
        # ---- single-hop delivery control (GATE), with its own fire snapshot ----
        injector.reset_stats()
        gh, _ah, _gctrl = MR._single_hop_control(base, adapter, tok, injector, base_embed, lm,
                                                 r, bank, conf, relidx, bos, 0.0)
        fs_ctrl = injector.fire_stats()
        ctrl_hits += int(gh)
        n_ctrl += 1
        # ---- 2-hop: no_edit + RAG (bank off), then the trigger condition ----
        NP = MR._first_hop_np(r)
        Q = qtmpl.format(NP=NP)
        p_no = MR.FEWSHOT + f"Question: {Q}\n"
        p_rag = MR.FEWSHOT + f"Suppose {r.prompt_text} {r.new_str}.\n" + f"Question: {Q}\n"
        ids_no = bos + tok(p_no, add_special_tokens=False).input_ids
        ids_rag = bos + tok(p_rag, add_special_tokens=False).input_ids
        g_no = MR._gen(base, adapter, tok, injector, base_embed, lm, ids_no, None, None, None, 0.0, gen_len)
        g_rag = MR._gen(base, adapter, tok, injector, base_embed, lm, ids_rag, None, None, None, 0.0, gen_len)
        injector.reset_stats()
        g_t = MR._gen(base, adapter, tok, injector, base_embed, lm, ids_no, bank, conf, relidx, 0.0, gen_len)
        fs2 = injector.fire_stats()
        rows.append(dict(
            subj=r.subject,
            # strict _match (the July in-dist tap scorer)
            base_true=MR._match(g_no, [gold_true]), no_cf=MR._match(g_no, [gold_cf]),
            rag_cf=MR._match(g_rag, [gold_cf]), rag_true=MR._match(g_rag, [gold_true]),
            t_cf=MR._match(g_t, [gold_cf]), t_true=MR._match(g_t, [gold_true]),
            # head-word _hit (the cartridge scorer; verbose-gold friendly)
            h_base_true=_hit(gold_true, g_no, gold_cf),
            h_rag_cf=_hit(gold_cf, g_rag, gold_true),
            h_t_cf=_hit(gold_cf, g_t, gold_true), h_t_true=_hit(gold_true, g_t, gold_cf),
            ctrl=bool(gh), ctrl_fires=fs_ctrl["fire_positions"] and fs_ctrl["forwards"] or 0,
            fire_positions=fs2["fire_positions"], zero_fwd=fs2["zero_forwards"], n_fwd=fs2["forwards"],
            ans=MR._answer_span(g_t).strip()))
        if i < show:
            print(f"[trig] {r.subject!r} {r.true_str!r}->{r.new_str!r} | gold {gold_true!r}->{gold_cf!r} "
                  f"| ctrl[{'OK' if gh else '..'}] fires_ctrl={fs_ctrl['fire_positions']} "
                  f"fires_2hop={fs2['fire_positions']} | no_edit[{'T' if rows[-1]['base_true'] else '.'}] "
                  f"rag_cf[{'Y' if rows[-1]['rag_cf'] else '.'}] trig_cf[{'Y' if rows[-1]['t_cf'] else '.'}]: "
                  f"{rows[-1]['ans'][:28]!r}", flush=True)
        if i % 5 == 4:                                         # allocator pool hygiene between cases
            torch.cuda.empty_cache()

    def _m(xs, fn):
        return (sum(fn(x) for x in xs) / len(xs)) if xs else float("nan")

    ctrl_gen = ctrl_hits / max(1, n_ctrl)
    valid = ctrl_gen >= ctrl_min
    filt = [x for x in rows if x["base_true"] and x["ctrl"]]
    n_zero2 = sum(1 for x in rows if x["fire_positions"] == 0)
    n_zeroc = sum(1 for x in rows if x["ctrl_fires"] == 0)
    fire_frac = _m(rows, lambda x: x["fire_positions"] / max(1, x["n_fwd"] * 1))
    summary = dict(n_lang=len(lang), scored=len(rows), hop=hop, ctrl_gen=ctrl_gen, valid=valid,
                   n_filtered=len(filt), n_collide=n_collide, n_skip=n_skip,
                   zero_fire_2hop=n_zero2, zero_fire_ctrl=n_zeroc)
    print("\n[trig] " + "=" * 74, flush=True)
    print(f"[trig] SUMMARY 2-hop={hop}  language-edits={len(lang)}  scored={len(rows)}", flush=True)
    print(f"[trig]   dropped: confound-collision {n_collide} | episodic-doc-unfit {n_skip}", flush=True)
    print(f"[trig]   *** SINGLE-HOP DELIVERY CONTROL (n={n_ctrl}) — GATE ***  free-gen {ctrl_gen:.3f} "
          f"(native in-dist ~0.6-0.8; gate >= {ctrl_min:.2f})", flush=True)
    print(f"[trig]   fire diagnostics: zero-fire ctrl cases {n_zeroc}/{len(rows)} | zero-fire 2-hop cases "
          f"{n_zero2}/{len(rows)}  (a dead mask silently reproduces no_edit — check BEFORE reading ripple)",
          flush=True)
    print(f"[trig]   base-composes-true (no_edit==gold_true, strict): {_m(rows, lambda x: x['base_true']):.3f}",
          flush=True)
    print(f"[trig]   confound-controlled FILTERED set (base-composes ∩ delivers): n={len(filt)}", flush=True)
    if valid and filt:
        t_cf_s, t_cf_h = _m(filt, lambda x: x["t_cf"]), _m(filt, lambda x: x["h_t_cf"])
        t_tr_s, t_tr_h = _m(filt, lambda x: x["t_true"]), _m(filt, lambda x: x["h_t_true"])
        rag_s, rag_h = _m(filt, lambda x: x["rag_cf"]), _m(filt, lambda x: x["h_rag_cf"])
        leak = _m(filt, lambda x: x["no_cf"])
        print("[trig]   --- RIPPLE vs gold_cf (did the edit propagate through the 2-hop?) ---", flush=True)
        print(f"[trig]     TRIGGER ripple  strict {t_cf_s:.3f} | head-word {t_cf_h:.3f}", flush=True)
        print(f"[trig]     shortcut(true)  strict {t_tr_s:.3f} | head-word {t_tr_h:.3f}", flush=True)
        print(f"[trig]     other (neither) strict {1 - t_cf_s - t_tr_s:.3f}", flush=True)
        print(f"[trig]     RAG ripple     strict {rag_s:.3f} | head-word {rag_h:.3f}  (same pass)", flush=True)
        print(f"[trig]     no_edit→gold_cf leak (strict) {leak:.3f}", flush=True)
        print(f"[trig]   unfiltered (all scored): trigger strict {_m(rows, lambda x: x['t_cf']):.3f} "
              f"head-word {_m(rows, lambda x: x['h_t_cf']):.3f} | delivery {ctrl_gen:.3f}", flush=True)
        summary.update(dict(t_cf_strict=t_cf_s, t_cf_hit=t_cf_h, t_true_strict=t_tr_s, t_true_hit=t_tr_h,
                            rag_strict=rag_s, rag_hit=rag_h, leak=leak,
                            t_cf_all=_m(rows, lambda x: x["t_cf"]), h_t_cf_all=_m(rows, lambda x: x["h_t_cf"])))
    else:
        why = (f"single-hop delivery {ctrl_gen:.3f} < {ctrl_min:.2f}" if not valid else "empty filtered set")
        print(f"[trig]   *** VERDICT WITHHELD — {why}: cannot read a ripple verdict.", flush=True)
    print("[trig] " + "=" * 74, flush=True)
    return summary


def run_trigger_eval(base, tok, adapter, builder, cf_records, args, rng):
    """E1 flow: build arm(s) -> KL distillation -> in-dist 2-hop eval. Called from recall_mag."""
    if cf_records is None:
        print("[trig] --trigger-eval requires --dataset counterfact --phrasing counterfactual_multi "
              "(so cf_records exist). Aborting.", flush=True)
        return None
    assert getattr(args, "store", "bolt") == "pk", "trigger eval needs --store pk"
    n_rel = getattr(builder, "R", 1) if getattr(args, "phrasing", "") == "counterfactual_multi" else 1
    lang = MR._lang_edits(cf_records, int(getattr(args, "mquake_limit", 0) or 0))
    n_train = int(getattr(args, "trigger_train_edits", 0) or 0)
    train_edits = lang[:n_train] if n_train > 0 else lang
    print(f"[trig] E1 trigger-local delivery: {len(train_edits)} train edits / {len(lang)} eval edits "
          f"| arms={getattr(args, 'trigger_arms', 'trigger')} | layer L{getattr(args, 'trigger_layer', 12)} "
          f"| orders {_orders_from(args)} | steps {getattr(args, 'trigger_steps', 300)}", flush=True)

    arms = build_arms(base, args, n_rel)
    work = build_work(base, tok, adapter, train_edits, args)
    torch.cuda.empty_cache()
    if not work:
        print("[trig] no deliverable training edits — aborting.", flush=True)
        return None
    results = {}
    for name, injector in arms.items():
        print(f"\n[trig] ================== ARM {name!r} (apply_mask={injector.apply_mask}) "
              f"==================", flush=True)
        t0 = time.time()
        if str(getattr(args, "trigger_objective", "distill")) == "delivery":
            tinfo = train_trigger_delivery(base, tok, adapter, injector, work, args)
        else:
            tinfo = train_trigger(base, tok, adapter, injector, train_edits, args, work=work)
        train_s = time.time() - t0
        print(f"[trig] arm {name!r}: distill {tinfo} in {train_s:.0f}s", flush=True)
        if tinfo is None:
            results[name] = {"error": "no deliverable training edits"}
            continue
        res = eval_trigger_ripple(base, adapter, injector, tok, args, cf_records)
        res["train_s"] = train_s
        res["train"] = tinfo
        results[name] = res
        injector.detach()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[trig] ============ E1 SUMMARY (per arm) ============", flush=True)
    for name, res in results.items():
        if "error" in res:
            print(f"[trig]   {name}: {res['error']}", flush=True)
            continue
        print(f"[trig]   {name}: ctrl {res.get('ctrl_gen', float('nan')):.3f} | "
              f"ripple strict {res.get('t_cf_strict', float('nan')):.3f} / head-word "
              f"{res.get('t_cf_hit', float('nan')):.3f} | RAG strict {res.get('rag_strict', float('nan')):.3f}"
              f" | n={res.get('n_filtered')}", flush=True)
    print(f"[trig] RESULT_JSON {results}", flush=True)
    return results

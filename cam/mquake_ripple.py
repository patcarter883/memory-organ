"""MQuAKE multi-hop RIPPLE eval — does a trained CAM residual tap integrate an edit into MULTI-HOP
reasoning better than generation-time RAG (edit-in-prompt)?

=====================================================================================================
DESIGN NOTE (read before interpreting the numbers)
=====================================================================================================

WHAT THIS MEASURES
------------------
A single-hop editor flips "Ellie Kemper is a citizen of" from "United States" to "Croatia". The RIPPLE
question is whether that edit propagates through a chain the model composes itself: "Who is the head of
state of the country where Ellie Kemper holds a citizenship?" — post-edit answer = Croatia's head of
state, NOT the USA's. MQuAKE-CF-3k supplies exactly this per case: edit(s) (`requested_rewrite`), 3
paraphrased multi-hop `questions`, the PRE-edit `answer`, the POST-edit `new_answer` (+ aliases).

Three ways of making the edit available, all answering the SAME question by free greedy generation
under an IDENTICAL few-shot chain-of-thought preamble:
  * no_edit : question only. Base composes the chain from parametric knowledge.
  * rag     : edit PREPENDED as text ("Suppose <prompt(subject)> <target_new>.") then the question —
              generation-time retrieval-augmentation (edit is literally in the prompt).
  * tap     : edit delivered through the trained CAM residual tap; NO edit text in the prompt. The
              tap prompt is byte-identical to no_edit's. Under test: does the injected association get
              USED by the base's own multi-hop reasoning?

DELIVERY — THE EPISODIC PATH (this is the load-bearing correctness fix)
----------------------------------------------------------------------
The tap is TRAINED on the EPISODIC store path: recall_mag.eval_generative_mag builds a counterfactual
doc, `adapter.memory_bank(doc_ids, seg_len, qa_start, answer_pos, carry=True)` writes a FRESH episodic
store from the doc (`_write_episode`) and reads it by the subject query. That path is what delivers the
native belief-override (~0.64–0.79). An earlier version of this eval delivered the tap via the
PERSISTENT store (`_persistent_write_one` + `persistent_bank`), which writes with the read-query key
convention (`head_query`) — a DIFFERENT key space than the tap was trained on — and single-hop delivery
collapsed to ~0.09. So the tap wasn't firing; multi-hop numbers were understated, NOT evidence of
integration failure.

THE FIX: for each edit we synthesize a one-fact, one-relation, M=1 `counterfactual_multi` DocBuilder
(the SAME builder class + doc format the tap trained on) whose single binding states the edit
("<relation-prefix><subject><relation-suffix> <target_new>.\n"), temporarily point the adapter at it,
and take the bank from `adapter.memory_bank(...)` — the RELIABLE, trained encoding. Building the doc
from an arbitrary (subject, relation-prompt, object) triple is exact because `_write_episode` /
`memory_bank` for counterfactual_multi locate the subject KEY and object VALUE via the builder's own
`binding_positions` / `binding_key_spans` / `q_subj_off` (they do NOT assume a fixed relation), so a
per-edit builder with `rel_templates={rid:(prompt.split('{}'))}`, `fact_subj_tids=[subject_ids]`,
`rel_subj_len={rid:len(subject_ids)}` reproduces the trained layout. Prefix rstripped + subject
space-prefixed via realedit._sp_tokens, mirroring setup_counterfact_multi's tokenization exactly. The
same bank is then injected during free-gen (CAM_SUBJ_ONLY_QUERY / CAM_LEARNED_KEY_POOL honored inside
memory_bank), with conf + q_relidx forwarded to the tap's conf-gate exactly as native eval does.

NON-NEGOTIABLE GATE — the single-hop control
--------------------------------------------
Delivered the SAME (episodic) way, we FIRST check single-hop delivery: inject the edit's bank while
free-generating the edit's OWN cloze ("Ellie Kemper is a citizen of") and see if `target_new` appears
(plus a 1-step argmax==new_tid cross-check). This must land in the ballpark of the native ~0.6–0.8 for
the ripple number to mean anything. It is printed prominently; if it is below CAM_MQUAKE_CTRL_MIN
(default 0.40) the multi-hop tap-vs-rag verdict is declared INVALID (delivery failure, not integration
failure) and NO verdict is drawn.

DELIVERY MODE IS SELECTABLE (side-by-side): CAM_MQUAKE_EPISODIC=1 (default) = the trained episodic
path; =0 = the old persistent path (expected ~0.09 single-hop — the bug, kept for comparison).
CAM_MQUAKE_BOTH=1 additionally prints the OTHER mode's single-hop control (control-only) in the same
run so episodic vs persistent delivery sit next to each other.

SCOPE / FILTERS
---------------
* TAP restricted to SINGLE-EDIT, single-token `target_new` cases — one fact per doc, the cleanest
  episodic doc (no multi-edit concatenation heuristic). N reported. (rag/no_edit run the FULL set, so
  RAG is visible on the superset AND like-for-like on the tap subset.)
* base-compose validity (mirrors probe_and_filter_counterfact's spirit): ripple scored only where
  `no_edit` reproduces the PRE-edit `answer` (base can compose the TRUE chain unaided). Both filtered
  and unfiltered numbers + the base-compose pass rate are reported.

FEW-SHOT / MATCHING
-------------------
FEWSHOT (module constant) = 3 hand-written multi-hop CoT exemplars, IDENTICAL bytes across all three
conditions (RAG only additionally inserts "Suppose ...." lines — intrinsic to the baseline; the tap
prompt equals no_edit's). Greedy free-gen ≤ CAM_MQUAKE_GEN_LEN (default 40) tokens, early-stopped at
the model's Answer line or a fabricated next Question. Correct iff any gold alias (≥2 chars,
whitespace-normalized, lowercased) is a substring of the post-"Answer:" span (else the whole cut gen).
rag/tap vs new_answer(+alias); no_edit vs answer(+alias).

CONFOUNDS I could NOT fully control
-----------------------------------
* The tap bank is set globally per forward and keyed by the edit SUBJECT — not positionally addressed
  to where the subject appears in the question. A tap "miss" can be an addressing/routing failure, not
  an integration failure. (The single-hop control isolates raw delivery; a healthy control + a low
  ripple is the interesting "does-not-propagate" signal.)
* q_relidx is forced to 0 for the per-edit builder (one relation), so the conf-gate uses training
  relation-0's EMA scale, not the MQuAKE relation's — a soft mis-scaling that can dampen delivery.
  Set CAM_MQUAKE_NO_CONFGATE=1 to inject ungated (conf=None) if the control comes back low.
* Constant every-step injection (the naive deployment) can over-steer into repetition; the default is
  the pure residual tap (CAM_LOGIT_INJECT=0). Optional logit-space injection with the same conf gate.
* Substring alias matching can false-positive on very short golds (mitigated: ≥2 chars, prefer the
  Answer span). rag and tap are matched identically, so any bias is symmetric.
* Base-compose pass rate on a 4B base is modest → the filtered subset can be small; raise
  --mquake-limit for tighter CIs. Free-gen ripple is noisier than single-token argmax — treat gaps
  below a few points as ties.
=====================================================================================================
"""
import json
import os

import numpy as np
import torch

# flat-package import shim (mirrors recall_mag.py): work as `cam.mquake_ripple` and as a bare file.
try:
    from .m2_adapter import DEV
    from .realedit import EditRecord, _sp_tokens
    from .recall_deepmem import DocBuilder
    from .recall_mag import _persistent_write_one
except ImportError:
    if __package__:
        raise
    import os as _os
    import sys as _sys
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    if _HERE not in _sys.path:
        _sys.path.insert(0, _HERE)
    from m2_adapter import DEV                                    # noqa: E402
    from realedit import EditRecord, _sp_tokens                   # noqa: E402
    from recall_deepmem import DocBuilder                          # noqa: E402
    from recall_mag import _persistent_write_one                  # noqa: E402


# Three hand-written multi-hop CoT exemplars. IDENTICAL bytes across no_edit / rag / tap conditions.
FEWSHOT = (
    "Answer each question by reasoning one hop at a time, then give the final answer on a line "
    "starting with \"Answer:\".\n\n"
    "Question: What is the capital of the country where Mount Fuji is located?\n"
    "Mount Fuji is located in Japan. The capital of Japan is Tokyo.\n"
    "Answer: Tokyo\n\n"
    "Question: Who is the head of state of the country where the Eiffel Tower is located?\n"
    "The Eiffel Tower is located in France. The head of state of France is Emmanuel Macron.\n"
    "Answer: Emmanuel Macron\n\n"
    "Question: Which continent is the country where Table Mountain is located part of?\n"
    "Table Mountain is located in South Africa. South Africa is part of Africa.\n"
    "Answer: Africa\n\n"
)


def _norm(s):
    """lowercase + collapse whitespace + strip — the normalization used on both text and gold alias."""
    return " ".join(str(s).split()).lower().strip()


def _build_records(case, tok):
    """One EditRecord per requested_rewrite (MQuAKE == CounterFact schema); realedit space-prefixed
    tokenization so keys/values match the trained CounterFact path exactly."""
    recs = []
    for rr in case.get("requested_rewrite", []):
        subject, prompt = rr["subject"], rr["prompt"]
        true_str, new_str = rr["target_true"]["str"], rr["target_new"]["str"]
        subj_ids = _sp_tokens(tok, subject)
        true_ids = _sp_tokens(tok, true_str)
        new_ids = _sp_tokens(tok, new_str)
        recs.append(EditRecord(
            case_id=case.get("case_id", -1),
            subject=subject, prompt=prompt, relation_id=rr.get("relation_id", ""),
            true_str=true_str, new_str=new_str,
            subject_tid=(subj_ids[0] if len(subj_ids) == 1 else -1),
            subject_last_tid=subj_ids[-1], subject_ntok=len(subj_ids),
            subject_tids=list(subj_ids),
            true_tid=(true_ids[0] if len(true_ids) == 1 else -1),
            new_tid=(new_ids[0] if len(new_ids) == 1 else -1),
            true_ids=list(true_ids), new_ids=list(new_ids)))
    return recs


def _golds(*fields):
    """Flatten answer + alias fields into a de-duped, >=2-char, normalized-nonempty gold list."""
    out = []
    for f in fields:
        if not f:
            continue
        for a in ([f] if isinstance(f, str) else list(f)):
            if a and len(str(a).strip()) >= 2:
                out.append(str(a))
    return out


def _answer_span(gen):
    """Scored span: cut at a fabricated next question, then the text after the FIRST 'Answer:'
    (else the whole cut generation)."""
    g = gen
    for marker in ("\nQuestion:", "\nQuestion", "\nQ:"):
        i = g.find(marker)
        if i != -1:
            g = g[:i]
            break
    low = g.lower()
    j = low.find("answer:")
    if j != -1:
        return g[j + len("answer:"):].split("\n")[0]
    return g


def _match(gen, golds):
    """True iff any normalized gold alias is a substring of the normalized answer span."""
    span = _norm(_answer_span(gen))
    return bool(span) and any(_norm(a) in span for a in golds)


# ---- EPISODIC delivery (the trained path) -------------------------------------------------------
def _edit_builder(tok, r, seg_len, qa_seg):
    """A one-fact, one-relation, M=1 counterfactual_multi DocBuilder stating edit `r` in the SAME doc
    format the tap trained on. rel_templates from the edit's own prompt (prefix rstripped + suffix as-is
    — mirrors setup_counterfact_multi._split); subject space-prefixed. Returns None if the doc does not
    fit the seg budget (assertion in __init__) or the prompt lacks '{}'."""
    if "{}" not in r.prompt:
        return None
    rid = r.relation_id or "mq"
    pre, _, suf = r.prompt.partition("{}")
    true_key = r.true_tid if r.true_tid >= 0 else (r.true_ids[0] if r.true_ids else 0)  # facts[i][3]=ans_prior (unused)
    facts = [(r.subject, r.true_str, r.subject_tids[-1], true_key)]
    try:
        b = DocBuilder(tok, None, None, 1, seg_len, qa_seg, phrasing="counterfactual_multi",
                       facts=facts, fact_relid=[rid], rel_templates={rid: (pre.rstrip(), suf)},
                       fact_subj_tids=[list(r.subject_tids)], rel_subj_len={rid: len(r.subject_tids)})
    except AssertionError:
        return None                                    # doc doesn't fit qa_start / seg_len
    b.set_counterfactual([r.new_tid])
    return b


def _episodic_bank(adapter, tok, r, seg_len, qa_seg, rng):
    """Deliver edit `r` through the trained EPISODIC memory_bank path. Builds the per-edit doc, points
    the adapter at it, writes+reads a fresh episodic store, returns (bank[1,K,mem], conf, relidx) — the
    reliable trained encoding. Restores the adapter's original builder. None if the doc can't be built."""
    cb = _edit_builder(tok, r, seg_len, qa_seg)
    if cb is None:
        return None
    orig = getattr(adapter, "builder", None)
    adapter.set_builder(cb)
    try:
        ids, _ans, apos = cb.build(rng, 1, local=False)
        bank = adapter.memory_bank(ids.to(DEV), seg_len, cb.qa_start, apos, carry=True)   # [1,K,mem]
        conf = getattr(adapter, "_last_conf", None)
        relidx = getattr(adapter, "_last_relidx", None)
    except AssertionError:
        adapter.set_builder(orig)
        return None
    adapter.set_builder(orig)
    if os.environ.get("CAM_MQUAKE_NO_CONFGATE") == "1":       # escape hatch: inject ungated (full)
        conf = None
    return bank, conf, relidx


# ---- PERSISTENT delivery (the OLD path — kept selectable for side-by-side comparison) -------------
def _persistent_bank(adapter, tok, records, learned_pool, gte):
    """The old persistent-store delivery: write all edits into a fresh per-case bank, concatenate each
    edit's subject-keyed read. Kept ONLY for the CAM_MQUAKE_EPISODIC=0 / CAM_MQUAKE_BOTH comparison
    (delivers ~0.09 single-hop — the wrong key space)."""
    pooled = os.environ.get("CAM_POOLED_SUBJ_KEY") == "1"
    V = [adapter.store.init_state(1, DEV, dtype=torch.float32)]
    for r in records:
        V = _persistent_write_one(adapter, V, r, pooled)
    banks, confs = [], []
    for r in records:
        tids = torch.tensor([r.subject_tids], dtype=torch.long, device=DEV)
        if gte:
            q = adapter._gte_key(tids).unsqueeze(1)
        else:
            q = adapter._e(tids)
            if learned_pool:
                q = adapter._pool_subject(q, keepdim=True)
        banks.append(adapter.persistent_bank(V[0], q))
        confs.append(getattr(adapter, "_last_conf", None))
    bank = torch.cat(banks, dim=1)
    conf = None if any(c is None for c in confs) else torch.cat(confs).max().view(1)
    return bank, conf, 0


@torch.no_grad()
def _gen(base, adapter, tok, injector, base_embed, lm, prompt_ids, bank, conf, relidx, alpha, gen_len):
    """Greedy free-gen. bank=None -> tap OFF. bank set -> tap ON: constant residual injection every step
    + optional conf-gated LOGIT injection when alpha>0. Early-stops at the model's Answer line or a
    fabricated next Question."""
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=DEV)
    c0env, hard = os.environ.get("CAM_LOGIT_GATE_C0"), os.environ.get("CAM_LOGIT_GATE_HARD") == "1"
    out = []
    for _ in range(gen_len):
        inj = None
        if bank is None:
            injector.set_bank(None)
        else:
            injector.set_bank(bank, conf=conf, relidx=relidx)
            if alpha > 0:
                v = alpha * (adapter.out_proj(bank).mean(1).to(lm.device, lm.dtype) @ lm.t())
                if c0env is not None and conf is not None:
                    cc = conf.to(v.device)
                    if hard:
                        g = (cc > float(c0env)).to(v.dtype)
                    else:
                        g = torch.sigmoid(float(os.environ.get("CAM_LOGIT_GATE_K", "1")) * (cc - float(c0env)))
                    v = v * g.view(-1, 1)
                inj = v
        logits = base(inputs_embeds=base_embed(cur)).logits[:, -1]
        if inj is not None:
            logits = logits + inj.to(logits.device)
        nxt = logits.argmax(-1)
        out.append(int(nxt.item()))
        cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
        dec = tok.decode(out)
        low = dec.lower()
        if "\nquestion" in low:
            break
        j = low.find("answer:")
        if j != -1 and "\n" in dec[j:]:
            break
    injector.set_bank(None)
    return tok.decode(out)


@torch.no_grad()
def _single_hop_control(base, adapter, tok, injector, base_embed, lm, r, bank, conf, relidx, bos, alpha):
    """Does the delivered edit fire single-hop? Free-gen the edit's OWN cloze with the bank injected and
    check target_new appears (+ a 1-step argmax==new_tid cross-check). The delivery-validity GATE."""
    pid = bos + tok(r.prompt_text, add_special_tokens=False).input_ids
    g = _gen(base, adapter, tok, injector, base_embed, lm, pid, bank, conf, relidx, alpha,
             int(os.environ.get("CAM_MQUAKE_CTRL_LEN", "10")))
    gen_hit = _norm(r.new_str) in _norm(g)
    injector.set_bank(bank, conf=conf, relidx=relidx)                       # 1-step argmax cross-check
    logits = base(inputs_embeds=base_embed(torch.tensor([pid], dtype=torch.long, device=DEV))).logits[:, -1]
    argmax_hit = int(logits.argmax(-1).item()) == r.new_tid
    injector.set_bank(None)
    return gen_hit, argmax_hit, g


# =================================================================================================
# EXPERIMENT A — IN-DISTRIBUTION language-pivot ripple (--indist-ripple / CAM_INDIST_RIPPLE=1)
# -------------------------------------------------------------------------------------------------
# The tap is COUPLED TO ITS TRAINING DISTRIBUTION: on an OOD MQuAKE edit (US->Croatia) it emits a
# LANGUAGE (the CounterFact object-type it trained on), so OOD ripple can't be tested with it (delivery
# ~0.024). This mode tests ripple IN-DISTRIBUTION, where the tap actually delivers (~0.6): edit a
# LANGUAGE relation (P103 mother tongue / P37 official language / P364 original language) it trained on,
# then test whether that single-hop delivery ripples through a LANGUAGE-PIVOT 2-hop whose second hop is
# a property of a language the BASE reliably knows (oracle). CONFOUND CONTROL: require the 2-hop answer
# for the CF language to DIFFER from the true language's, so ripple (gold_cf) is distinguishable from
# the subject-shortcut (gold_true = routing via the subject's real language). Reuses _episodic_bank,
# _gen, _match, FEWSHOT, and the single-hop delivery GATE unchanged.
# =================================================================================================
LANG_RELS = ("P103", "P37", "P364")   # mother tongue / official language / original language (object=language)

# candidate 2-hops: (name, language-property CLOZE for the base oracle, composed-question template).
# {L}=language string, {NP}=first-hop noun phrase for the language (e.g. "the mother tongue of X").
_TWOHOPS = {
    "family":  ("The {L} language belongs to the",        "What language family does {NP} belong to?"),
    "script":  ("The {L} language is written in the",      "What script is {NP} written in?"),
    "country": ("The {L} language is primarily spoken in", "In which country is {NP} primarily spoken?"),
}


def _oracle_span(gen):
    """Short entity from a base oracle free-gen: cut at the first sentence/line/clause boundary."""
    g = gen.strip()
    cut = len(g)
    for sep in ("\n", ".", ",", ";", "("):
        k = g.find(sep)
        if k != -1:
            cut = min(cut, k)
    return g[:cut].strip()


def _first_hop_np(r):
    """The language's noun phrase from the edit prompt: 'The mother tongue of X is' -> 'the mother
    tongue of X' (trailing copula stripped, leading char lowercased for mid-sentence embedding)."""
    pt = r.prompt_text
    for suf in (" is", " was", " are", " were"):
        if pt.endswith(suf):
            pt = pt[:-len(suf)]
            break
    return (pt[0].lower() + pt[1:]) if pt else pt


@torch.no_grad()
def _oracle(base, adapter, tok, injector, base_embed, lm, cloze, bos, olen):
    """Base-only (tap OFF) short generation from a cloze -> the oracle entity span."""
    pid = bos + tok(cloze, add_special_tokens=False).input_ids
    g = _gen(base, adapter, tok, injector, base_embed, lm, pid, None, None, None, 0.0, olen)
    return _oracle_span(g)


@torch.no_grad()
def eval_indist_ripple(base, adapter, injector, tok, args, cf_records):
    """IN-DISTRIBUTION language-pivot ripple. Edits are language-relation CounterFact records the tap
    trained on (base-known, single-token true+new). Delivered via the SAME episodic path; the base is
    the oracle for a language's 2-hop property; ripple (gold_cf) is separated from the subject-shortcut
    (gold_true) by requiring gold_cf != gold_true."""
    assert getattr(args, "store", "bolt") == "pk", "in-dist ripple needs the pk store; run with --store pk"
    assert cf_records is not None, "in-dist ripple needs the trained CounterFact records (cf_records)"
    injector.eval()
    lang = [r for r in cf_records if getattr(r, "relation_id", "") in LANG_RELS
            and getattr(r, "new_tid", -1) >= 0 and getattr(r, "true_tid", -1) >= 0
            and r.true_str.strip().lower() != r.new_str.strip().lower()]
    limit = int(getattr(args, "mquake_limit", 0) or 0)
    if limit > 0:
        lang = lang[:limit]
    from collections import Counter
    rel_ct = Counter(r.relation_id for r in lang)
    gen_len = int(os.environ.get("CAM_MQUAKE_GEN_LEN", "40"))
    olen = int(os.environ.get("CAM_INDIST_ORACLE_LEN", "8"))
    show = int(os.environ.get("CAM_MQUAKE_SHOW", "12"))
    alpha = float(os.environ.get("CAM_LOGIT_INJECT", "0"))
    ctrl_min = float(os.environ.get("CAM_MQUAKE_CTRL_MIN", "0.40"))
    seg_len, qa_seg = int(args.seg_len), int(args.qa_seg)
    rng = np.random.default_rng(int(getattr(args, "seed", 0)))
    base_embed = base.get_input_embeddings()
    lm = base.get_output_embeddings().weight
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    print(f"\n[indist] === IN-DISTRIBUTION language-pivot ripple ({len(lang)} language edits) ===", flush=True)
    print(f"[indist] relations: {dict(rel_ct)} | tap delivers in-dist (unlike OOD MQuAKE) — testing "
          f"whether that delivery RIPPLES through a language 2-hop (base as oracle)", flush=True)
    if not lang:
        print("[indist] no single-token language-relation edits in the trained set; nothing to do.", flush=True)
        return {"n": 0}

    # ---- choose the 2-hop with the most confound-eligible cases (gold_cf != gold_true, both non-empty) ----
    uniq = sorted({r.true_str for r in lang} | {r.new_str for r in lang})
    best_hop, best_maps, best_elig = None, None, -1
    for name, (cloze_t, _q) in _TWOHOPS.items():
        omap = {L: _oracle(base, adapter, tok, injector, base_embed, lm, cloze_t.format(L=L), bos, olen)
                for L in uniq}
        elig = sum(1 for r in lang if omap[r.true_str] and omap[r.new_str]
                   and _norm(omap[r.true_str]) != _norm(omap[r.new_str]))
        print(f"[indist] 2-hop candidate {name!r}: confound-eligible (gold_cf!=gold_true) = {elig}/{len(lang)}",
              flush=True)
        if elig > best_elig:
            best_hop, best_maps, best_elig = name, omap, elig
    hop = best_hop
    omap = best_maps
    _cloze_t, qtmpl = _TWOHOPS[hop]
    print(f"[indist] CHOSEN 2-hop = {hop!r} (confound-eligible {best_elig}); composed Q = {qtmpl!r}", flush=True)

    rows = []          # dict per case
    ctrl_hits = n_ctrl = 0
    n_collide = n_deliver_skip = 0
    for i, r in enumerate(lang):
        gold_true, gold_cf = omap[r.true_str], omap[r.new_str]
        if not gold_true or not gold_cf or _norm(gold_true) == _norm(gold_cf):
            n_collide += 1
            continue                                              # confound control: undistinguishable
        eb = _episodic_bank(adapter, tok, r, seg_len, qa_seg, rng)
        if eb is None:
            n_deliver_skip += 1
            continue
        bank, conf, relidx = eb
        gh, ah, gctrl = _single_hop_control(base, adapter, tok, injector, base_embed, lm, r,
                                            bank, conf, relidx, bos, alpha)
        ctrl_hits += int(gh); n_ctrl += 1

        NP = _first_hop_np(r)
        Q = qtmpl.format(NP=NP)
        p_no = FEWSHOT + f"Question: {Q}\n"
        p_rag = FEWSHOT + f"Suppose {r.prompt_text} {r.new_str}.\n" + f"Question: {Q}\n"
        ids_no = bos + tok(p_no, add_special_tokens=False).input_ids
        ids_rag = bos + tok(p_rag, add_special_tokens=False).input_ids

        g_no = _gen(base, adapter, tok, injector, base_embed, lm, ids_no, None, None, None, 0.0, gen_len)
        base_true = _match(g_no, [gold_true])
        g_rag = _gen(base, adapter, tok, injector, base_embed, lm, ids_rag, None, None, None, 0.0, gen_len)
        rag_cf, rag_true = _match(g_rag, [gold_cf]), _match(g_rag, [gold_true])
        g_tap = _gen(base, adapter, tok, injector, base_embed, lm, ids_no, bank, conf, relidx, alpha, gen_len)
        tap_cf, tap_true = _match(g_tap, [gold_cf]), _match(g_tap, [gold_true])
        rows.append(dict(ctrl=gh, base_true=base_true, rag_cf=rag_cf, rag_true=rag_true,
                         tap_cf=tap_cf, tap_true=tap_true))

        if i < show:
            print(f"[indist] {r.relation_id} {r.subject!r}: lang {r.true_str!r}->{r.new_str!r} | "
                  f"2hop gold {gold_true!r}->{gold_cf!r} | ctrl[{'OK' if gh else '..'}]", flush=True)
            print(f"[indist]   Q: {Q!r}", flush=True)
            print(f"[indist]   no_edit[{'T' if base_true else '.'}]: {_answer_span(g_no).strip()!r}", flush=True)
            print(f"[indist]   rag[cf={'Y' if rag_cf else 'n'} true={'Y' if rag_true else 'n'}]: "
                  f"{_answer_span(g_rag).strip()!r}", flush=True)
            print(f"[indist]   tap[cf={'Y' if tap_cf else 'n'} true={'Y' if tap_true else 'n'}]: "
                  f"{_answer_span(g_tap).strip()!r}", flush=True)

    ctrl_gen = ctrl_hits / max(1, n_ctrl)
    valid = ctrl_gen >= ctrl_min
    # filtered set: base composes the TRUE chain AND the tap delivered single-hop for this case
    filt = [x for x in rows if x["base_true"] and x["ctrl"]]

    def _frac(xs, key):
        return (sum(x[key] for x in xs) / len(xs)) if xs else float("nan")

    print("\n[indist] " + "=" * 72, flush=True)
    print(f"[indist] SUMMARY  2-hop={hop}  language-edits={len(lang)}  scored={len(rows)}", flush=True)
    print(f"[indist]   dropped: confound-collision (gold_cf==gold_true) {n_collide} | "
          f"episodic-doc-unfit {n_deliver_skip}", flush=True)
    print(f"[indist]   *** SINGLE-HOP DELIVERY CONTROL (n={n_ctrl}) — GATE ***  free-gen {ctrl_gen:.3f} "
          f"(native in-dist ~0.6-0.8)", flush=True)
    print(f"[indist]   base-composes-true (no_edit==gold_true): {_frac(rows, 'base_true'):.3f} "
          f"(n={len(rows)})", flush=True)
    print(f"[indist]   confound-controlled FILTERED set (base-composes ∩ tap-delivers): n={len(filt)}", flush=True)
    if valid and filt:
        print(f"[indist]   --- RIPPLE vs gold_cf (did the edit propagate through the 2-hop?) ---", flush=True)
        print(f"[indist]     TAP ripple (gold_cf) {_frac(filt, 'tap_cf'):.3f}  |  shortcut (gold_true) "
              f"{_frac(filt, 'tap_true'):.3f}  |  other {1 - _frac(filt, 'tap_cf') - _frac(filt, 'tap_true'):.3f}",
              flush=True)
        print(f"[indist]     RAG ripple (gold_cf) {_frac(filt, 'rag_cf'):.3f}  |  shortcut (gold_true) "
              f"{_frac(filt, 'rag_true'):.3f}", flush=True)
        print(f"[indist]   HEADLINE: TAP {_frac(filt, 'tap_cf'):.3f} vs RAG {_frac(filt, 'rag_cf'):.3f} "
              f"(Δ = {_frac(filt, 'tap_cf') - _frac(filt, 'rag_cf'):+.3f})", flush=True)
        print(f"[indist]   interpret: tap_cf≈rag_cf => in-dist edit ripples as well as in-context text; "
              f"tap_cf≪rag_cf with tap delivering single-hop => delivers but does NOT propagate.", flush=True)
    else:
        why = (f"single-hop delivery {ctrl_gen:.3f} < {ctrl_min:.2f}" if not valid
               else "empty filtered set")
        print(f"[indist]   *** VERDICT WITHHELD — {why}: cannot read a ripple verdict.", flush=True)
    print("[indist] " + "=" * 72, flush=True)
    return {"twohop": hop, "n_lang": len(lang), "scored": len(rows), "ctrl_gen": ctrl_gen, "valid": valid,
            "n_filtered": len(filt), "tap_cf": _frac(filt, "tap_cf"), "rag_cf": _frac(filt, "rag_cf"),
            "tap_true": _frac(filt, "tap_true")}


@torch.no_grad()
def eval_mquake_ripple(base, adapter, injector, tok, args):
    """MQuAKE multi-hop ripple eval. Reuses the trained tap (hooks attached by train_taps) + trained
    pk-store. Delivers each edit through the EPISODIC path (default), gates on single-hop delivery, then
    compares tap-vs-rag ripple accuracy vs new_answer on the base-can-compose subset."""
    assert getattr(args, "store", "bolt") == "pk", \
        "MQuAKE tap condition needs the pk store (episodic memory_bank + tap); run with --store pk"
    injector.eval()
    with open(args.mquake_eval) as f:
        data = json.load(f)
    limit = int(getattr(args, "mquake_limit", 0) or 0)
    if limit > 0:
        data = data[:limit]
    nq = max(1, int(os.environ.get("CAM_MQUAKE_NQ", "1")))
    gen_len = int(os.environ.get("CAM_MQUAKE_GEN_LEN", "40"))
    show = int(os.environ.get("CAM_MQUAKE_SHOW", "12"))
    alpha = float(os.environ.get("CAM_LOGIT_INJECT", "0"))
    ctrl_min = float(os.environ.get("CAM_MQUAKE_CTRL_MIN", "0.40"))
    episodic = os.environ.get("CAM_MQUAKE_EPISODIC", "1") != "0"
    both = os.environ.get("CAM_MQUAKE_BOTH") == "1"
    learned_pool = os.environ.get("CAM_LEARNED_KEY_POOL") == "1"
    gte = getattr(adapter, "_gte_keys", None) is not None
    seg_len, qa_seg = int(args.seg_len), int(args.qa_seg)
    rng = np.random.default_rng(int(getattr(args, "seed", 0)))
    base_embed = base.get_input_embeddings()
    lm = base.get_output_embeddings().weight
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []

    prim = "episodic" if episodic else "persistent"
    print(f"\n[ripple] === MQuAKE multi-hop ripple eval ({len(data)} cases, {nq} q/case, {gen_len} tok, "
          f"α={alpha}) <- {args.mquake_eval} ===", flush=True)
    print(f"[ripple] delivery={prim} (tap trained on episodic memory_bank) | tap=store-bound edit, NO "
          f"edit text | rag=edit prepended as text | no_edit=question only | same few-shot all three",
          flush=True)

    def _deliver(mode, records):
        """(bank, conf, relidx) or None. episodic: single-edit only (records[0]); persistent: all edits."""
        if mode == "episodic":
            return _episodic_bank(adapter, tok, records[0], seg_len, qa_seg, rng)
        return _persistent_bank(adapter, tok, records, learned_pool, gte)

    rows = []                       # (base_ok, rag_ok, tap_ok|None, tap_eligible, noedit_new_ok)
    ctrl_gen_hits, ctrl_arg_hits, n_ctrl = 0, 0, 0
    ctrl_other_gen, n_ctrl_other = 0, 0        # side-by-side other-mode control (CAM_MQUAKE_BOTH)
    n_single_token_cases = n_deliverable = 0
    for ci, case in enumerate(data):
        records = _build_records(case, tok)
        if not records:
            continue
        # TAP eligibility: single-edit + single-token target_new (the cleanest per-edit episodic doc).
        cand = len(records) == 1 and len(records[0].new_ids) == 1
        n_single_token_cases += int(cand)
        ans_golds = _golds(case.get("answer"), case.get("answer_alias"))
        new_golds = _golds(case.get("new_answer"), case.get("new_answer_alias"))
        rag_block = "".join(f"Suppose {r.prompt_text} {r.new_str}.\n" for r in records)

        bank = conf = relidx = None
        tap_eligible = False
        if cand:
            d = _deliver(prim, records)
            if d is not None:
                bank, conf, relidx = d
                tap_eligible = True
                n_deliverable += 1
                # single-hop delivery GATE (once per deliverable case)
                gh, ah, gctrl = _single_hop_control(base, adapter, tok, injector, base_embed, lm,
                                                    records[0], bank, conf, relidx, bos, alpha)
                ctrl_gen_hits += int(gh); ctrl_arg_hits += int(ah); n_ctrl += 1
                if both:
                    do = _deliver("persistent" if episodic else "episodic", records)
                    if do is not None:
                        gh2, _ah2, _g2 = _single_hop_control(base, adapter, tok, injector, base_embed,
                                                            lm, records[0], do[0], do[1], do[2], bos, alpha)
                        ctrl_other_gen += int(gh2); n_ctrl_other += 1
                if ci < show:
                    print(f"[ripple] ctrl case {case.get('case_id', ci)} edit "
                          f"{records[0].true_str!r}->{records[0].new_str!r}  single-hop gen[{'OK' if gh else '..'}] "
                          f"argmax[{'OK' if ah else '..'}]: {gctrl.strip()!r}", flush=True)

        for qi, question in enumerate(case.get("questions", [])[:nq]):
            p_no = FEWSHOT + f"Question: {question}\n"
            p_rag = FEWSHOT + rag_block + f"Question: {question}\n"
            ids_no = bos + tok(p_no, add_special_tokens=False).input_ids
            ids_rag = bos + tok(p_rag, add_special_tokens=False).input_ids

            g_no = _gen(base, adapter, tok, injector, base_embed, lm, ids_no, None, None, None, 0.0, gen_len)
            base_ok = _match(g_no, ans_golds)
            noedit_new_ok = _match(g_no, new_golds)
            g_rag = _gen(base, adapter, tok, injector, base_embed, lm, ids_rag, None, None, None, 0.0, gen_len)
            rag_ok = _match(g_rag, new_golds)
            tap_ok = None
            g_tap = ""
            if tap_eligible:
                # tap prompt == no_edit prompt; the edit lives ONLY in the injected episodic bank.
                g_tap = _gen(base, adapter, tok, injector, base_embed, lm, ids_no, bank, conf, relidx,
                             alpha, gen_len)
                tap_ok = _match(g_tap, new_golds)
            rows.append((base_ok, rag_ok, tap_ok, tap_eligible, noedit_new_ok))

            if ci < show and qi == 0:
                print(f"[ripple] case {case.get('case_id', ci)} ({len(records)} edit, tap_elig={tap_eligible}) "
                      f"{question!r} | gold new={case.get('new_answer')!r} true={case.get('answer')!r}", flush=True)
                print(f"[ripple]   no_edit[{'OK' if base_ok else '..'}]: {_answer_span(g_no).strip()!r}", flush=True)
                print(f"[ripple]   rag    [{'OK' if rag_ok else '..'}]: {_answer_span(g_rag).strip()!r}", flush=True)
                if tap_eligible:
                    print(f"[ripple]   tap    [{'OK' if tap_ok else '..'}]: {_answer_span(g_tap).strip()!r}", flush=True)

    def _acc(sel, key):
        xs = [key(r) for r in rows if sel(r)]
        return (sum(xs) / len(xs) if xs else float("nan")), len(xs)

    T = len(rows)
    base_rate, _ = _acc(lambda r: True, lambda r: r[0])
    rag_all, n_rag_all = _acc(lambda r: True, lambda r: r[1])
    tap_all, n_tap_all = _acc(lambda r: r[3], lambda r: r[2])
    noedit_leak, _ = _acc(lambda r: True, lambda r: r[4])
    rag_bc, n_rag_bc = _acc(lambda r: r[0], lambda r: r[1])
    tap_bc, n_tap_bc = _acc(lambda r: r[0] and r[3], lambda r: r[2])
    rag_ll, n_ll = _acc(lambda r: r[0] and r[3], lambda r: r[1])
    ctrl_gen = ctrl_gen_hits / max(1, n_ctrl)
    ctrl_arg = ctrl_arg_hits / max(1, n_ctrl)

    print("\n[ripple] " + "=" * 72, flush=True)
    print(f"[ripple] SUMMARY  delivery={prim}  cases={len(data)}  trials(T)={T}  q/case={nq}", flush=True)
    print(f"[ripple]   single-edit+single-token cases: {n_single_token_cases}/{len(data)}; "
          f"deliverable (doc fits): {n_deliverable}; tap-eligible trials={n_tap_all}", flush=True)
    print(f"[ripple]   *** SINGLE-HOP DELIVERY CONTROL (n={n_ctrl}) — GATE ***", flush=True)
    print(f"[ripple]       free-gen delivery: {ctrl_gen:.3f}   1-step argmax delivery: {ctrl_arg:.3f}   "
          f"(native belief-override ~0.64-0.79)", flush=True)
    if both and n_ctrl_other:
        other = "persistent" if episodic else "episodic"
        print(f"[ripple]       side-by-side {other} free-gen delivery: {ctrl_other_gen / n_ctrl_other:.3f} "
              f"(n={n_ctrl_other})", flush=True)
    valid = ctrl_gen >= ctrl_min
    print(f"[ripple]   base-compose pass rate (no_edit == PRE-edit answer): {base_rate:.3f}  "
          f"(no_edit already says new_answer: {noedit_leak:.3f})", flush=True)
    print(f"[ripple]   --- UNFILTERED (all trials) ---", flush=True)
    print(f"[ripple]     rag ripple-acc vs new_answer : {rag_all:.3f}  (n={n_rag_all})", flush=True)
    print(f"[ripple]     tap ripple-acc vs new_answer : {tap_all:.3f}  (n={n_tap_all}, tap-eligible only)", flush=True)
    print(f"[ripple]   --- FILTERED (base can compose the true chain) ---", flush=True)
    print(f"[ripple]     rag ripple-acc vs new_answer : {rag_bc:.3f}  (n={n_rag_bc})", flush=True)
    print(f"[ripple]     tap ripple-acc vs new_answer : {tap_bc:.3f}  (n={n_tap_bc}, tap-eligible only)", flush=True)
    if valid:
        print(f"[ripple]   --- HEADLINE: like-for-like on base-compose ∩ tap-eligible (n={n_ll}) ---", flush=True)
        print(f"[ripple]     TAP {tap_bc:.3f}  vs  RAG {rag_ll:.3f}   (Δ tap-rag = {tap_bc - rag_ll:+.3f})", flush=True)
        print(f"[ripple]   interpret: tap≈rag => the residual edit integrates into reasoning as well as "
              f"in-context text; tap≪rag (with healthy control) => it does not propagate through the chain.",
              flush=True)
    else:
        print(f"[ripple]   *** VERDICT WITHHELD — single-hop delivery {ctrl_gen:.3f} < {ctrl_min:.2f}: the "
              f"tap is NOT firing in this harness. The multi-hop tap numbers are UNDERSTATED by a delivery "
              f"failure, NOT integration failure. Fix delivery before reading a ripple verdict.", flush=True)
    print("[ripple] " + "=" * 72, flush=True)
    return {"delivery": prim, "trials": T, "ctrl_gen": ctrl_gen, "ctrl_arg": ctrl_arg, "valid": valid,
            "base_rate": base_rate, "rag_all": rag_all, "tap_all": tap_all, "rag_bc": rag_bc,
            "tap_bc": tap_bc, "rag_ll": rag_ll, "n_ll": n_ll, "n_deliverable": n_deliverable}

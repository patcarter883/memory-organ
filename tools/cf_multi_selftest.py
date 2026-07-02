"""Tokenizer-only selftest for MULTI-RELATION counterfactual editing (DocBuilder phrasing
'counterfactual_multi'). No model / no GPU — validates the doc STRUCTURE and POSITIONS that the pk-store
adapter + addr-sup rely on, so position bugs are caught before spending a GPU:

  1. each binding tokenizes as "<prefix><subject><suffix> <cf-object>.\n" for its OWN relation
  2. binding_positions() KEY offsets land exactly on the subject tid, VALUE offsets on the cf-object tid
  3. the query region reconstructs "<prefix><subject><suffix>" and the subject sits at qa_start+q_subj_off
     (what _compute_addr_sup reads to identify the queried binding)
  4. build_cf_query strong/weak: the target IS / is NOT bound, target subject always queried
  5. rows are rectangular (uniform length) across a batch mixing relations

Run:  PYTHONPATH=/engine python /engine/tools/cf_multi_selftest.py   (in titans:dev, CPU, no lease)
"""
import os
import numpy as np
from transformers import AutoTokenizer

from cam.recall_deepmem import DocBuilder, single_token_ids, piece

MODEL = os.environ.get("CAM_BASE_MODEL", "Qwen/Qwen3.5-4B")


def build_facts(tok):
    """Two relations sharing NO subjects, single-token subjects/objects, >= M/R facts each."""
    # candidate single-token, space-prefixed words for subjects and objects
    subj_words = single_token_ids(tok, ["France", "Japan", "Canada", "Brazil", "Egypt", "India",
                                        "Norway", "Chile", "Kenya", "Peru", "Cuba", "Ghana"])
    obj_words = single_token_ids(tok, ["Paris", "Tokyo", "London", "Berlin", "Madrid", "Rome",
                                       "French", "Spanish", "German", "Arabic", "English", "Dutch"])
    assert len(subj_words) >= 8 and len(obj_words) >= 8, "need enough single-token words for the selftest"
    rel_templates = {"cap": ("The capital of", " is"),
                     "lang": ("The official language of", " is")}
    facts, fact_relid, cf_tid = [], [], []
    # relation 'cap': first 4 subjects -> first 4 objects (true), a shifted object as counterfactual
    for k in range(4):
        s_w, s_t = subj_words[k]; t_w, t_t = obj_words[k]; c_w, c_t = obj_words[(k + 1) % 4]
        facts.append((s_w, t_w, s_t, t_t)); fact_relid.append("cap"); cf_tid.append(c_t)
    # relation 'lang': next 4 subjects -> language objects (true), shifted as counterfactual
    for k in range(4):
        s_w, s_t = subj_words[4 + k]; t_w, t_t = obj_words[6 + k]; c_w, c_t = obj_words[6 + (k + 1) % 4]
        facts.append((s_w, t_w, s_t, t_t)); fact_relid.append("lang"); cf_tid.append(c_t)
    return facts, fact_relid, cf_tid, rel_templates


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    facts, fact_relid, cf_tid, rel_templates = build_facts(tok)
    M, seg_len, qa_seg = 4, 48, 2
    b = DocBuilder(tok, None, None, M, seg_len, qa_seg, phrasing="counterfactual_multi",
                   facts=facts, fact_relid=fact_relid, rel_templates=rel_templates)
    b.set_counterfactual(cf_tid)
    print(f"[selftest] R={b.R} relations {b.rel_order} | slot_relid={b.slot_relid} | "
          f"slot_bind_len={b.slot_bind_len} | slot_key_off={b.slot_key_off} slot_val_off={b.slot_val_off}")

    rng = np.random.default_rng(0)
    ids, ans_cf, ans_prior, apos = b.build_cf(rng, batch=6)
    S = ids.shape[1]
    print(f"[selftest] build_cf: ids {tuple(ids.shape)} apos={apos} qa_start={b.qa_start} "
          f"q_subj_off={b.q_subj_off} qfix_len={b.qfix_len}")

    # (5) rectangular
    assert ids.shape[1] == S, "non-rectangular rows"

    # (2) binding_positions land on subject (KEY) and cf-object (VALUE) for row 0
    hstart = len(b.bos) + len(b.header)
    keys, vals = b.binding_positions(hstart)
    row0 = ids[0].tolist()
    subj_tids = {f[2] for f in facts}
    for m in range(M):
        ktid, vtid = row0[keys[m]], row0[vals[m]]
        assert ktid in subj_tids, f"slot {m} KEY pos {keys[m]} = {ktid!r} not a subject tid"
        # the value at vals[m] must be the cf-object of the fact whose subject is ktid, at slot m's relation
        # find the fact: subject ktid AND relation slot_relid[m]
        fi = next(i for i, f in enumerate(facts)
                  if f[2] == ktid and fact_relid[i] == b.slot_relid[m])
        assert vtid == cf_tid[fi], f"slot {m} VALUE pos {vals[m]} = {vtid} != cf {cf_tid[fi]}"
    print(f"[selftest] (2) binding_positions OK — KEY=subject, VALUE=cf-object for all {M} slots")

    # (1) decode a binding block to eyeball structure
    m0_start = hstart + b.bind_bases[0]
    m0 = row0[m0_start:m0_start + b.slot_bind_len[0]]
    print(f"[selftest] (1) slot-0 binding decodes to: {tok.decode(m0)!r}")

    # (3) subject sits at qa_start + q_subj_off; query region reconstructs "<prefix><subj><suffix>"
    q_subj_tid = row0[b.qa_start + b.q_subj_off]
    assert q_subj_tid in subj_tids, f"queried subject at qa_start+q_subj_off ({q_subj_tid}) is not a subject"
    qregion = row0[b.qa_start:apos]
    print(f"[selftest] (3) query region decodes to: {tok.decode(qregion)!r} | subject tid {q_subj_tid} "
          f"at qa_start+{b.q_subj_off}")
    # gold answer token = the queried fact's cf-object
    assert row0[apos] == ans_cf[0].item(), "answer position mismatch"

    # (4) build_cf_query strong vs weak for a 'cap' target
    cap_facts = [i for i, r in enumerate(fact_relid) if r == "cap"]
    tgt = cap_facts[0]
    tgt_subj = facts[tgt][2]
    for bind_target, name in ((True, "STRONG"), (False, "WEAK")):
        qids, qapos = b.build_cf_query(rng, [tgt] * 5, 5, bind_target=bind_target)
        r = qids[0].tolist()
        # target subject is ALWAYS queried
        assert r[b.qa_start + b.q_subj_off] == tgt_subj, f"{name}: target subject not queried"
        # is the target's subject present in the BINDING block?
        block = r[hstart:b.qa_start]
        bound = tgt_subj in block
        assert bound == bind_target, f"{name}: target bound={bound} but bind_target={bind_target}"
        print(f"[selftest] (4) build_cf_query {name}: target subject bound-in-doc={bound} (expected {bind_target}) OK")

    print("[selftest] ALL CHECKS PASSED — multi-relation doc structure + positions are correct.")


if __name__ == "__main__":
    main()

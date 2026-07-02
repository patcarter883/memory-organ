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

    print("[selftest] ALL CHECKS PASSED (single-token subjects).")
    _multitoken_subject(tok)


def _multitoken_subject(tok):
    """MULTI-TOKEN SUBJECT: a relation whose subjects are 2 tokens each. Verify the KEY lands on the
    subject's LAST token, positions stay rectangular, and the query carries the full subject."""
    # build 2-token subjects: pick words whose ' <w>' is 2 tokens (or concatenate two single-token words).
    st = single_token_ids(tok, ["York", "Delhi", "Jersey", "Orleans", "Zealand", "Guinea", "Mexico", "Hampshire"])
    pre1 = single_token_ids(tok, ["New", "Old", "West", "East", "South", "North"])
    assert len(st) >= 4 and len(pre1) >= 2
    obj = single_token_ids(tok, ["French", "German", "Spanish", "Arabic", "English", "Dutch", "Italian", "Korean"])
    # subject = "<pre> <second>" -> tokenized as 2 tokens (space-prefixed). Build facts for relation 'lang2'.
    facts, fact_relid, cf_tid, fact_subj_tids = [], [], [], []
    rel_templates = {"lang2": ("The official language of", " is")}
    rel_subj_len = {"lang2": 2}
    for k in range(6):
        pw, pt = pre1[k % len(pre1)]; sw, s2 = st[k % len(st)]
        subj_str = f"{pw} {sw}"
        subj_ids = tok(" " + subj_str, add_special_tokens=False).input_ids
        if len(subj_ids) != 2:            # skip if tokenizer merges differently
            continue
        t_w, t_t = obj[k % len(obj)]; c_w, c_t = obj[(k + 1) % len(obj)]
        facts.append((subj_str, t_w, subj_ids[-1], t_t))   # KEY = last subject token
        fact_relid.append("lang2"); cf_tid.append(c_t); fact_subj_tids.append(subj_ids)
    assert len(facts) >= 4, "need >=4 two-token-subject facts for the selftest"
    # need a 2nd relation for rectangular multi (R>=2); reuse single-token 'cap'
    sing = single_token_ids(tok, ["France", "Japan", "Canada", "Brazil"])
    capobj = single_token_ids(tok, ["Paris", "Tokyo", "London", "Berlin"])
    rel_templates["cap"] = ("The capital of", " is"); rel_subj_len["cap"] = 1
    for k in range(4):
        sw, s_t = sing[k]; t_w, t_t = capobj[k]; c_w, c_t = capobj[(k + 1) % 4]
        facts.append((sw, t_w, s_t, t_t)); fact_relid.append("cap"); cf_tid.append(c_t)
        fact_subj_tids.append([s_t])
    import numpy as np
    from cam.recall_deepmem import DocBuilder
    M = 4
    b = DocBuilder(tok, None, None, M, 48, 3, phrasing="counterfactual_multi", facts=facts,
                   fact_relid=fact_relid, rel_templates=rel_templates,
                   fact_subj_tids=fact_subj_tids, rel_subj_len=rel_subj_len)
    b.set_counterfactual(cf_tid)
    print(f"[selftest-mt] slot_relid={b.slot_relid} rel_subj_len={rel_subj_len} slot_key_off={b.slot_key_off} "
          f"slot_val_off={b.slot_val_off} slot_bind_len={b.slot_bind_len}")
    rng = np.random.default_rng(1)
    ids, ans_cf, ans_prior, apos = b.build_cf(rng, batch=6)
    row0 = ids[0].tolist()
    hstart = len(b.bos) + len(b.header)
    keys, vals = b.binding_positions(hstart)
    subj_last = {f[2] for f in facts}
    for m in range(M):
        assert row0[keys[m]] in subj_last, f"[mt] slot {m} KEY pos not a subject LAST token"
    m0 = row0[hstart + b.bind_bases[0]: hstart + b.bind_bases[0] + b.slot_bind_len[0]]
    print(f"[selftest-mt] slot-0 binding: {tok.decode(m0)!r}")
    # query subject last token at qa_start + q_key_off
    assert row0[b.qa_start + b.q_key_off] in subj_last, "[mt] query subject last-token offset wrong"
    print(f"[selftest-mt] query region: {tok.decode(row0[b.qa_start:apos])!r} | q_bind_idx={b.q_bind_idx} "
          f"q_subj_off={b.q_subj_off} q_key_off={b.q_key_off}")
    # rectangular
    assert ids.shape[1] == len(row0)
    print("[selftest-mt] ALL CHECKS PASSED — multi-TOKEN subject positions correct (KEY=last token, rectangular).")


if __name__ == "__main__":
    main()

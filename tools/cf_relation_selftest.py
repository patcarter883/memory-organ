"""Tokenizer-only check of the Track 1 per-relation counterfactual DocBuilder (no base, no GPU).

Picks the largest single-token-subject relation group in CounterFact, folds its real prompt template
into the DocBuilder header/rel (as setup_counterfact does), builds a doc, and verifies: round-trip
tokenization stability, KEY(subject)/VALUE(object) positions, the eval eliciting text matches the fact's
TRUE relation, and the query-key at qa_start is the subject (so addr-sup is unchanged).
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cam"))
from transformers import AutoTokenizer  # noqa: E402

from realedit import load_counterfact, as_fact_table, cf_tids_from_records  # noqa: E402
from recall_deepmem import DocBuilder  # noqa: E402

MODEL = os.environ.get("CAM_BASE_MODEL", "Qwen/Qwen3.5-4B")
M, SEG, QASEG = 8, 48, 2


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    records, stats = load_counterfact(os.path.join(os.environ.get("CF_DATA", "/data"), "counterfact.json"),
                                      tok, single_token_only=True)
    print(f"loaded {len(records)} single-token-subject-and-object records")
    by_rel = defaultdict(list)
    for r in records:
        by_rel[(r.relation_id, r.prompt)].append(r)

    def _split(p):
        pre, _, suf = p.partition("{}")
        return pre.rstrip(), suf
    cand = []
    for (rid, prompt), recs in by_rel.items():
        if "{}" not in prompt:
            continue
        pre, suf = _split(prompt)
        if not pre or len(tok(suf, add_special_tokens=False).input_ids) > 6:
            continue
        cand.append((len(recs), rid, prompt, pre, suf, recs))
    n, rid, prompt, prefix, suffix, rel_recs = max(cand, key=lambda c: c[0])
    print(f"picked relation {rid!r} prompt={prompt!r} ({n} recs) prefix={prefix!r} suffix={suffix!r}")
    assert len(rel_recs) >= M, f"largest group {len(rel_recs)} < M {M}"

    facts = as_fact_table(rel_recs)
    cf_tid = cf_tids_from_records(rel_recs)
    b = DocBuilder(tok, None, None, M, SEG, QASEG, phrasing="counterfactual", facts=facts,
                   cf_header_prefix=prefix, cf_rel=suffix)
    b.set_counterfactual(cf_tid)
    print(f"header={tok.decode(b.header)!r} rel={tok.decode(b.rel)!r} bind_len={b.bind_len} "
          f"qfix_len={b.qfix_len} qa_start={b.qa_start} key_off={b.key_off} val_off={b.val_off}")

    ids, ans_cf, ans_prior, apos = b.build_cf(np.random.default_rng(0), 3, local=False)
    fact_ctids = {f[2] for f in facts}
    for row in range(ids.shape[0]):
        # answer at apos
        assert ids[row, apos].item() == ans_cf[row].item(), "cf answer not at apos"
        # query key at qa_start is the subject (KEY) — addr-sup unchanged
        assert ids[row, b.qa_start].item() in fact_ctids, "qa_start token is not a subject KEY"
        # per-binding KEY/VALUE
        hstart = len(b.bos) + len(b.header)
        for m in range(b.M):
            base = hstart + m * b.bind_len
            assert ids[row, base + b.key_off].item() in fact_ctids, f"slot {m} KEY not a subject"
        # eval eliciting text (header + query region) must read as the TRUE relation prompt
        hlen = len(b.bos) + len(b.header)
        ctx = ids[row, len(b.bos):hlen].tolist() + ids[row, b.qa_start:apos].tolist()
        text = tok.decode(ctx)
        subj = tok.decode([ids[row, b.qa_start].item()]).strip()
        expect = prompt.replace("{}", subj)
        assert expect in text, f"eliciting text {text!r} does not contain the true prompt {expect!r}"
        if row == 0:
            print(f"  eval eliciting text: {text!r}")
            print(f"  (true relation prompt for subject {subj!r}: {expect!r})")
    # round-trip stability
    row0 = ids[0].tolist()
    body = row0[1:] if (b.bos and row0[0] == b.bos[0]) else row0
    retok = tok(tok.decode(body), add_special_tokens=False).input_ids
    assert retok == body, f"NOT round-trip stable:\n built={body}\n retok={retok}"
    print("OK — per-relation counterfactual doc: round-trip stable, subject@qa_start (addr-sup intact), "
          "eval elicits the TRUE relation.")


if __name__ == "__main__":
    main()

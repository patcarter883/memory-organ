"""Track 1 (issue #16): CounterFact real-editing loader + survivor accounting.

The curated country->capital knowledge-editing path (recall_mag.py phrasing=counterfactual) proves
the memory can OVERRIDE a prior the frozen base parametrically holds — but on a hand-picked, single
fixed-prompt table. Track 1 replaces that curated table with the REAL ROME CounterFact benchmark
(21919 records) and adds the two metrics a real editing benchmark demands:

  * LOCALITY      — editing one fact must NOT corrupt unrelated facts. CounterFact ships
                    `neighborhood_prompts`: other subjects that share the SAME true object. With the
                    memory attached, the base must STILL answer these correctly.
  * GENERALIZATION — the edit should fire on PARAPHRASES of the edited fact, not just the exact
                     training string. CounterFact ships `paraphrase_prompts` (gold = target_new).

## The single-token-KEY constraint (read recall_deepmem.DocBuilder carefully)

The store addresses a fact by a SINGLE-TOKEN KEY at the query position (`qa_start`), and binds a
SINGLE-TOKEN VALUE at `val_off` (= 1+len(rel)). In the counterfactual DocBuilder:
    "<Country> is <Capital>.\n"   KEY=country@0 (single tok), VALUE=capital (single tok)
    query "<Country> is" -> predict the capital token.
Mapping CounterFact onto this contract:  subject -> KEY, object -> VALUE. So a record is tractable
for the CURRENT store iff:
    * the SUBJECT reduces to a single space-prefixed token  (it is the addressable KEY), AND
    * target_true.str  is a single space-prefixed token     (the prior VALUE we probe/score), AND
    * target_new.str   is a single space-prefixed token     (the counterfactual VALUE we bind).

CounterFact subjects are mostly multi-token names ("Danielle Darrieux"), so the single-token-SUBJECT
subset is expected to be small. We therefore ALSO report an "objects-only" survivor count (both
objects single-token; the subject may be multi-token — its LAST token would serve as the key, or a
future multi-token-key store handles it). Surfacing BOTH lets the orchestrator choose the GPU run
shape (single-token-subset vs multi-token-key) instead of a silent truncation.

Run the CPU selftest (tokenizer only, no model forward):
    python -m cam.realedit --selftest --data-dir /home/pat/code/memory-organ/data
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


@dataclass
class EditRecord:
    """One CounterFact edit reduced to the fields the probe/bind/eval machinery needs."""
    case_id: int
    subject: str
    prompt: str                       # requested_rewrite.prompt, e.g. "The mother tongue of {} is"
    relation_id: str
    true_str: str                     # target_true.str  (the prior the base is probed for)
    new_str: str                      # target_new.str   (the counterfactual we bind)
    subject_tid: int                  # single-token subject KEY tid (space-prefixed), or -1 if multi-tok
    subject_last_tid: int             # last token of the (possibly multi-token) subject
    subject_ntok: int                 # subject token count (space-prefixed)
    true_tid: int                     # single-token target_true VALUE tid (space-prefixed), or -1
    new_tid: int                      # single-token target_new  VALUE tid (space-prefixed), or -1
    true_ids: list = field(default_factory=list)       # FULL space-prefixed target_true object token ids (Phase M)
    new_ids: list = field(default_factory=list)        # FULL space-prefixed target_new  object token ids (Phase M)
    subject_tids: list = field(default_factory=list)   # FULL space-prefixed subject token ids (multi-token OK)
    neighborhood_prompts: list = field(default_factory=list)   # LOCALITY probes (gold = true_str)
    paraphrase_prompts: list = field(default_factory=list)     # GENERALIZATION probes (gold = new_str)

    @property
    def prompt_text(self) -> str:
        """The record's OWN editing prompt with the subject filled in ("The mother tongue of Danielle Darrieux is")."""
        return self.prompt.replace("{}", self.subject)


def _sp_tokens(tok, s):
    """token ids for a space-prefixed string ' <s>' with NO special tokens (the mid-sentence encoding
    the DocBuilder uses for names/objects). Returns the id list."""
    return tok(" " + s, add_special_tokens=False).input_ids


def load_counterfact(path, tok, single_token_only=True, limit=None):
    """Load ROME CounterFact and reduce each record to an EditRecord.

    `single_token_only=True` (the tractable subset for the CURRENT single-token-KEY store) keeps only
    records where the SUBJECT is a single space-prefixed token AND both target_true / target_new are
    each a single space-prefixed token. This is the set the DocBuilder (subject=KEY, object=VALUE) can
    bind and score exactly.

    `single_token_only=False` returns ALL records (each still carrying the tokenizer stats), so callers
    can compute alternative survivor regimes (objects-only) without re-tokenizing.

    Always returns (records, stats) where stats reports BOTH filtering regimes over the full file:
        stats["total"]                    — records in the file
        stats["objects_single"]           — both objects single-token (subject may be multi-token)
        stats["subject_single"]           — subject single-token (objects unconstrained)
        stats["all_single"]               — subject AND both objects single-token (the tractable subset)
    """
    with open(path) as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]

    all_recs = []
    n_obj_single = 0
    n_subj_single = 0
    n_all_single = 0
    for rec in data:
        rr = rec["requested_rewrite"]
        subject = rr["subject"]
        true_str = rr["target_true"]["str"]
        new_str = rr["target_new"]["str"]

        subj_ids = _sp_tokens(tok, subject)
        true_ids = _sp_tokens(tok, true_str)
        new_ids = _sp_tokens(tok, new_str)

        subj_single = len(subj_ids) == 1
        obj_single = len(true_ids) == 1 and len(new_ids) == 1
        all_single = subj_single and obj_single

        n_obj_single += int(obj_single)
        n_subj_single += int(subj_single)
        n_all_single += int(all_single)

        er = EditRecord(
            case_id=rec.get("case_id", -1),
            subject=subject,
            prompt=rr["prompt"],
            relation_id=rr.get("relation_id", ""),
            true_str=true_str,
            new_str=new_str,
            subject_tid=(subj_ids[0] if subj_single else -1),
            subject_last_tid=subj_ids[-1],
            subject_ntok=len(subj_ids),
            subject_tids=list(subj_ids),
            true_tid=(true_ids[0] if len(true_ids) == 1 else -1),
            new_tid=(new_ids[0] if len(new_ids) == 1 else -1),
            true_ids=list(true_ids),                     # FULL object token lists (multi-token OK; Phase M)
            new_ids=list(new_ids),
            neighborhood_prompts=list(rec.get("neighborhood_prompts", [])),
            paraphrase_prompts=list(rec.get("paraphrase_prompts", [])),
        )
        if single_token_only:
            if all_single:
                all_recs.append(er)
        else:
            all_recs.append(er)

    stats = {
        "total": len(data),
        "objects_single": n_obj_single,
        "subject_single": n_subj_single,
        "all_single": n_all_single,
        "kept": len(all_recs),
    }
    return all_recs, stats


def as_fact_table(records):
    """Project single-token-subject EditRecords onto the (country, capital, country_tid, capital_tid)
    4-tuple shape the counterfactual DocBuilder / probe_and_filter consume, treating
    subject->'country'(KEY) and target_true->'capital'(true VALUE). The counterfactual VALUE
    (target_new) is carried separately (see cf_tids_from_records) so the bind step installs it.

    Only single-token-subject records are valid here (subject_tid != -1); asserts otherwise."""
    facts = []
    for r in records:
        assert r.subject_tid != -1 and r.true_tid != -1, \
            f"as_fact_table needs single-token subject+true object (case {r.case_id})"
        facts.append((r.subject, r.true_str, r.subject_tid, r.true_tid))
    return facts


def cf_tids_from_records(records):
    """Parallel list of target_new (counterfactual) VALUE tids for as_fact_table's records.

    Unlike the curated country->capital path (which DERANGES the kept true capitals to synthesize a
    counterfactual), CounterFact SUPPLIES target_new directly — no derangement. Each record's
    counterfactual value is its own new_tid."""
    for r in records:
        assert r.new_tid != -1, f"record {r.case_id} target_new is multi-token — filter first"
    return [r.new_tid for r in records]


# --------------------------------------------------------------------------------------------------
# CPU selftest — tokenizer only, no model forward. Prints both survivor regimes + example records.
# --------------------------------------------------------------------------------------------------
def _selftest(data_dir):
    from transformers import AutoTokenizer
    from m2_adapter import MODEL

    path = os.path.join(data_dir, "counterfact.json")
    print(f"[realedit] loading tokenizer {MODEL} (offline) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[realedit] loading CounterFact <- {path}", flush=True)

    # full pass (single_token_only=False) so we get every regime count from one tokenization sweep
    all_recs, stats = load_counterfact(path, tok, single_token_only=False)
    print("\n[realedit] ===== SURVIVOR ACCOUNTING (single-token filtering regimes) =====")
    print(f"  total records in file .................. {stats['total']}")
    print(f"  objects-only single-token .............. {stats['objects_single']}   "
          f"(target_true AND target_new each 1 tok; subject may be multi-token)")
    print(f"  subject-only single-token .............. {stats['subject_single']}   "
          f"(subject 1 tok; objects unconstrained)")
    print(f"  ALL single-token (subj + both objs) .... {stats['all_single']}   "
          f"<== the tractable subset for the CURRENT single-token-KEY store")
    print(f"  => kept {stats['all_single']} / {stats['total']} single-token-subject editable records", flush=True)

    # the tractable subset (what recall_mag --dataset counterfact will bind)
    kept = [r for r in all_recs if r.subject_tid != -1 and r.true_tid != -1 and r.new_tid != -1]
    print(f"\n[realedit] {len(kept)} records are fully bindable (single-token subject KEY + both objects). "
          f"Showing 3 examples with locality/generalization probes:\n")
    for r in kept[:3]:
        print(f"  --- case {r.case_id} [{r.relation_id}] ---")
        print(f"    prompt        : {r.prompt_text!r}")
        print(f"    edit          : {r.true_str!r} (tid {r.true_tid})  ->  {r.new_str!r} (tid {r.new_tid})")
        print(f"    subject KEY   : {r.subject!r} (tid {r.subject_tid}, {r.subject_ntok} tok)")
        print(f"    LOCALITY (neighborhood, gold=true={r.true_str!r}):")
        for p in r.neighborhood_prompts[:3]:
            print(f"        - {p!r}")
        print(f"    GENERALIZATION (paraphrase, gold=new={r.new_str!r}):")
        for p in r.paraphrase_prompts[:3]:
            print(f"        - {p!r}")
        print()

    # objects-only diagnostic: how many editable if we relax the subject to multi-token (last-token key)
    obj_only = [r for r in all_recs if r.true_tid != -1 and r.new_tid != -1]
    multi_subj = [r for r in obj_only if r.subject_tid == -1]
    print(f"[realedit] MULTI-TOKEN-KEY headroom: {len(obj_only)} records have both objects single-token; "
          f"of those {len(multi_subj)} have a MULTI-token subject (need a multi-token-key store or "
          f"last-token key).", flush=True)
    if kept:
        import statistics
        subj_lens = [r.subject_ntok for r in obj_only]
        print(f"[realedit] subject token-length over objects-single set: "
              f"min={min(subj_lens)} median={statistics.median(subj_lens):.0f} max={max(subj_lens)}",
          flush=True)
    print("\n[realedit] selftest OK (tokenizer-only, no GPU).", flush=True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="CPU tokenizer-only survivor + example dump")
    ap.add_argument("--data-dir", default="data", dest="data_dir",
                    help="dir holding counterfact.json (default 'data'; pass the MAIN checkout's data dir)")
    args = ap.parse_args()
    if args.selftest:
        _selftest(args.data_dir)
    else:
        ap.error("nothing to do; pass --selftest")


if __name__ == "__main__":
    main()

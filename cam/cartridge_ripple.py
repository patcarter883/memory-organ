"""cartridge_ripple — Path D runner: per-edit Cartridge edit-ripple, reusing the mquake_ripple
disambiguation machinery (oracle map, 2-hop selection, confound filter, scoring) but delivering the edit
through a TRAINED KV-prefix Cartridge instead of the residual tap.

For each language-relation CounterFact edit it: (1) trains a per-edit Cartridge by KL-distilling
base+edit-in-context into base+Z (cam/cartridge_train), (2) scores the language-pivot 2-hop ripple in the
SAME four conditions the tap eval uses — no_edit / rag(=teacher ceiling) / cartridge — reusing
_select_twohop / _oracle / _match / _first_hop_np / FEWSHOT unchanged.

Variants (env CAM_CART_VARIANT): D0 generic self-study (no hop supervision); D1 distill only the direct
cloze; D2 distill a self-study 2-hop on CAM_CART_SELFSTUDY_HOP (default country), EVAL on a held-out hop
(CAM_RIPPLE_EVAL_HOP, default family). Ceiling = RAG; floor reference = the old tap 0.31.

Run (local gfx1201, via the shared arbiter):
  CAM_BASE_MODEL=Qwen/Qwen3-4B-Base HF_HUB_OFFLINE=1 \
  gpu-lease -n 1 -- python -m cam.cartridge_ripple --data-dir /home/pat/code/memory-organ/data \
      --limit 8 --p 128 --max-steps 120
"""
import argparse
import os
import sys

import numpy as np
import torch

try:
    from .m2_adapter import DEV, load_frozen_base
    from .realedit import load_counterfact
    from . import mquake_ripple as MR
    from .cartridge_train import train_cartridge, direct_cloze_delivers
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import DEV, load_frozen_base            # noqa: E402
    from realedit import load_counterfact                   # noqa: E402
    import mquake_ripple as MR                              # noqa: E402
    from cartridge_train import train_cartridge, direct_cloze_delivers  # noqa: E402


class _NoOpInjector:
    """Satisfies _gen / _select_twohop's injector contract (set_bank/eval) for the base-only conditions;
    the cartridge is delivered orthogonally via attach()/detach(), not through a bank."""
    def set_bank(self, *a, **k):
        pass

    def eval(self, *a, **k):
        return self

    def train(self, *a, **k):
        return self


def _frac(xs, key):
    return (sum(x[key] for x in xs) / len(xs)) if xs else float("nan")


# The base-oracle golds for the family/script/country hops are VERBOSE phrases ("Romance group of
# languages", "East Slavic group of the Indo-European"), while the model's canonical answer is short
# ("Romance language family"). MR._match's full-substring test therefore misses correct answers (and
# empties the base_true filter). _hit is a head-word match: any DISTINCTIVE (non-stopword) gold token
# present in the answer span. Confound inequality (gold_cf != gold_true) is still enforced on the full
# strings, so a shared generic head can't collapse the two.
import re as _re
_HOP_STOP = set("language languages family group branch of the is a an spoken written script primarily "
                "in to belong belongs which what does dialect and or".split())


def _words(s):
    return {w for w in _re.sub(r"[^a-z0-9\- ]", " ", s.lower()).split() if w and w not in _HOP_STOP}


# D3 "entailment self-study": diverse 2-hop-THROUGH-THE-BRIDGE questions about the subject's language
# (the edit installs a new bridge language; these route subject -> CF-language -> downstream property,
# across MANY relations). {NP} = the subject's language noun phrase (e.g. "the official language of
# Gibraltar"). The eval taxonomic hops (family, script) are HELD OUT so a positive result is genuine
# generalization, not coverage. Names matching held-out hops are excluded at build time.
D3_TEMPLATES = [
    ("country",  "In which country is {NP} primarily spoken?"),
    ("continent", "On which continent is {NP} primarily spoken?"),
    ("region",   "In which part of the world is {NP} mainly spoken?"),
    ("city",     "Name a major city where {NP} is spoken."),
    ("speakers", "Roughly how many people speak {NP}?"),
    ("related",  "Which language is most closely related to {NP}?"),
    ("neighbour", "Name another language spoken near where {NP} is spoken."),
    ("direction", "Is {NP} normally written left-to-right or right-to-left?"),
    ("currency", "What is the currency of the country where {NP} is spoken?"),
    ("capital",  "What is the capital of the country where {NP} is spoken?"),
]


def _hit(gold, gen, other=None):
    """head-word match of `gold` against the answer span, using only the tokens that DISTINGUISH gold
    from `other` (so a generic family both share, e.g. 'indo-european', can't count as a ripple hit)."""
    distinctive = _words(gold) - (_words(other) if other else set())
    return bool(distinctive & _words(MR._answer_span(gen)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/home/pat/code/memory-organ/data")
    ap.add_argument("--limit", type=int, default=8, help="max language edits to score (small = fast signal)")
    ap.add_argument("--p", type=int, default=128, help="cartridge prefix length p in {128,512}")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--samples-per-seed", type=int, default=2)
    ap.add_argument("--gen-len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default=os.environ.get("CAM_CART_VARIANTS", "D2"),
                    help="comma list from {D0,D1,D2}; trained per-edit in ONE pass so no_edit/RAG/oracle "
                         "are shared (apples-to-apples).")
    ap.add_argument("--selfstudy-hop", default=os.environ.get("CAM_CART_SELFSTUDY_HOP", "country"))
    ap.add_argument("--eval-hop", default=os.environ.get("CAM_RIPPLE_EVAL_HOP", "family"))
    ap.add_argument("--no-init-text", action="store_true", help="disable init-from-edit (random init)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    args.mquake_limit = args.limit          # for MR._lang_edits / MR._select_twohop
    torch.manual_seed(args.seed)

    print(f"[cart-ripple] loading base {os.environ.get('CAM_BASE_MODEL', '<default>')} ...", flush=True)
    base, tok = load_frozen_base()
    base_embed = base.get_input_embeddings()
    lm = base.get_output_embeddings().weight
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    inj = _NoOpInjector()

    path = os.path.join(args.data_dir, "counterfact.json")
    all_recs, stats = load_counterfact(path, tok, single_token_only=True)
    print(f"[cart-ripple] CounterFact {path}: kept {stats['kept']}/{stats['total']} single-token", flush=True)
    lang = MR._lang_edits(all_recs, args.limit)
    if not lang:
        print("[cart-ripple] no single-token language-relation edits; nothing to do.", flush=True)
        return
    print(f"[cart-ripple] {len(lang)} language edits | variants={variants} "
          f"selfstudy-hop={args.selfstudy_hop} eval-hop={args.eval_hop}", flush=True)

    # --- oracle map + eval 2-hop (reuse mquake_ripple machinery; base-only, cartridge OFF) ---
    eval_hop, qtmpl, omap = MR._select_twohop(base, None, tok, inj, args, lang, tag="cart",
                                              force=args.eval_hop)
    ss_qtmpl = MR._TWOHOPS.get(args.selfstudy_hop, (None, None))[1]

    gen_len = int(os.environ.get("CAM_MQUAKE_GEN_LEN", "40"))
    show = int(os.environ.get("CAM_MQUAKE_SHOW", "12"))

    def _passages_extra(variant, edit_text, NP):
        if variant == "D1":                                     # distill ONLY the direct edit cloze
            cloze = tok(edit_text + ".", add_special_tokens=False).input_ids
            return [cloze] * 12, None                            # repeated so step-budget ~matches D0/D2
        if variant == "D2" and ss_qtmpl:                        # distill a self-study 2-hop on selfstudy-hop
            return None, [f"Question: {ss_qtmpl.format(NP=NP)}\nAnswer:"]
        if variant == "D3":                                     # entailment self-study: MANY through-bridge
            held = {eval_hop, "family", "script"}               # hold out the taxonomic/eval hops
            extra = [f"Question: {qt.format(NP=NP)}\nAnswer:"
                     for name, qt in D3_TEMPLATES if name not in held]
            return None, extra
        return None, None                                       # D0 generic (no hop supervision)

    import time as _time
    edits = []                          # per-edit: gold + no_edit/RAG scores + per-variant cart scores
    n_collide = 0
    train_secs = {v: 0.0 for v in variants}
    n_trained = {v: 0 for v in variants}
    for i, r in enumerate(lang):
        gold_true, gold_cf = omap.get(r.true_str), omap.get(r.new_str)
        if not gold_true or not gold_cf or MR._norm(gold_true) == MR._norm(gold_cf):
            n_collide += 1
            continue                                            # confound control (undistinguishable)
        edit_text = r.prompt_text + " " + r.new_str
        NP = MR._first_hop_np(r)
        Q = qtmpl.format(NP=NP)                                 # eval (held-out) 2-hop question

        # ---- no_edit + RAG (SHARED across variants; base-only) ----
        ids_no = bos + tok(MR.FEWSHOT + f"Question: {Q}\n", add_special_tokens=False).input_ids
        ids_rag = bos + tok(MR.FEWSHOT + f"Suppose {r.prompt_text} {r.new_str}.\n" + f"Question: {Q}\n",
                            add_special_tokens=False).input_ids
        g_no = MR._gen(base, None, tok, inj, base_embed, lm, ids_no, None, None, None, 0.0, gen_len)
        g_rag = MR._gen(base, None, tok, inj, base_embed, lm, ids_rag, None, None, None, 0.0, gen_len)
        e = dict(subj=r.subject, base_true=_hit(gold_true, g_no, gold_cf),
                 no_cf=_hit(gold_cf, g_no, gold_true),
                 rag_cf=_hit(gold_cf, g_rag, gold_true), rag_true=_hit(gold_true, g_rag, gold_cf),
                 var={})

        # ---- per-variant: train a fresh cartridge, gate on single-hop delivery, score the 2-hop ----
        for v in variants:
            passages, extra = _passages_extra(v, edit_text, NP)
            t0 = _time.time()
            cart, tinfo = train_cartridge(
                base, tok, edit_text, p=args.p, lr=args.lr, epochs=args.epochs, max_steps=args.max_steps,
                n_seeds=args.n_seeds, samples_per_seed=args.samples_per_seed, gen_len=args.gen_len,
                seed=args.seed + i, init_text=not args.no_init_text, verbose=args.verbose,
                passages=passages, extra_seeds=extra)
            train_secs[v] += _time.time() - t0
            n_trained[v] += 1
            ctrl_hit, _gc = direct_cloze_delivers(base, tok, cart, r.prompt_text, r.new_str)
            cart.attach(base); cart.set_active(True)
            try:
                g_cart = MR._gen(base, None, tok, inj, base_embed, lm, ids_no, None, None, None, 0.0, gen_len)
            finally:
                cart.detach()
            e["var"][v] = dict(ctrl=ctrl_hit, cart_cf=_hit(gold_cf, g_cart, gold_true),
                               cart_true=_hit(gold_true, g_cart, gold_cf),
                               ans=MR._answer_span(g_cart).strip())
        edits.append(e)
        if i < show:
            vs = " ".join(f"{v}[{'Y' if e['var'][v]['cart_cf'] else '.'}:{e['var'][v]['ans'][:16]!r}]"
                          for v in variants)
            print(f"[cart] {r.subject!r} {r.true_str!r}->{r.new_str!r} | gold {gold_true!r}->{gold_cf!r} "
                  f"| base_true={'Y' if e['base_true'] else '.'} rag_cf={'Y' if e['rag_cf'] else '.'} | {vs}",
                  flush=True)

    # ================================ REPORT ================================
    scored = len(edits)
    print("\n[cart-ripple] " + "=" * 72, flush=True)
    print(f"[cart-ripple] eval-hop={eval_hop} p={args.p} scored={scored} (collide-dropped {n_collide}) "
          f"| head-word matcher (NOT the strict substring scorer; old-tap 0.31 was a different scorer)",
          flush=True)
    print(f"[cart-ripple] base-composes-true (no_edit gold_true)={_frac(edits, 'base_true'):.3f}  "
          f"no_edit ripple(gold_cf)={_frac(edits, 'no_cf'):.3f}  "
          f"RAG ripple(gold_cf)={_frac(edits, 'rag_cf'):.3f} (full n={scored})", flush=True)
    hdr = f"{'variant':8s} | {'n_filt':>6s} | {'CART cf':>7s} {'true':>5s} {'oth':>5s} | " \
          f"{'RAG cf':>6s} {'true':>5s} | {'noedit cf':>9s} | {'CART full':>9s}  {'deliver':>7s}  {'train/edit':>10s}"
    print("[cart-ripple] " + hdr, flush=True)
    print("[cart-ripple] " + "-" * len(hdr), flush=True)
    def _m(rows, fn):
        return (sum(fn(r) for r in rows) / len(rows)) if rows else float("nan")
    for v in variants:
        filt = [e for e in edits if e["base_true"] and e["var"][v]["ctrl"]]  # base composes true ∩ delivers
        n_f = len(filt)
        c_cf = _m(filt, lambda r: r["var"][v]["cart_cf"]); c_tr = _m(filt, lambda r: r["var"][v]["cart_true"])
        c_ot = (1 - c_cf - c_tr) if n_f else float("nan")
        r_cf = _m(filt, lambda r: r["rag_cf"]); r_tr = _m(filt, lambda r: r["rag_true"])
        n_cf = _m(filt, lambda r: r["no_cf"])
        full_cf = _m(edits, lambda r: r["var"][v]["cart_cf"])
        deliv = _m(edits, lambda r: r["var"][v]["ctrl"])
        tpe = train_secs[v] / max(1, n_trained[v])
        print(f"[cart-ripple] {v:8s} | {n_f:6d} | {c_cf:7.3f} {c_tr:5.3f} {c_ot:5.3f} | "
              f"{r_cf:6.3f} {r_tr:5.3f} | {n_cf:9.3f} | {full_cf:9.3f}  {deliv:7.3f}  {tpe:8.1f}s",
              flush=True)
    print("[cart-ripple] " + "=" * 72, flush=True)
    print("[cart-ripple] read: CART cf/true/oth on the base-composes∩delivers FILTERED set (same set scores "
          "RAG & no_edit); CART full = ripple over ALL scored; deliver = single-hop delivery rate.", flush=True)


if __name__ == "__main__":
    main()

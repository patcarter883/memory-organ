# CAM — E1: trigger-local distilled delivery (Engram-style) — spec

**What this is.** The one delivery geometry the frozen-base program never ran: a *content-triggered*
read — the edit fires into the residual stream **only at positions where the fact's subject (its trigger
n-gram) appears**, trained by the lever that is settled as the one that matters (distillation). It is a
controlled test of reorient **lever 3** ("fire at the right reasoning step" — position-/content-addressed
injection) under the **distillation-vs-injection** finding, aimed at the two axes where the current
artifacts are weakest: free-gen **degeneration** (the always-on tap's failure) and **distractor
selectivity** (the cartridge's BABILong shortfall).

Lineage: `memory-organ/docs/research/2026-10-17-engram-tech-transfer.md` (item 1 of the engram transfer
note); Engram paper `~/kb/pdf/memory-organ/engram-paper.pdf`; reorient `CAM_TITANS_REORIENT.md`
(levers 1–4); Path-D harness `CAM_CARTRIDGE_EDITRIPPLE_SPEC.md`; distance program
`CAM_TITANS_DISTANCE_SCORECARD.md`; the bar: RESULTS §10 (memory-organ), Phase 18.

## Positioning — what this is NOT

The boundary is **settled**: a frozen base reaches in-context quality and cannot exceed it
(perturb-vs-recompute; §10 / Phase 18; all activation-write forms converge at ≈RAG on bridge-routed hops,
below RAG on competing-path hops). **E1 does not reopen that.** Expected ceiling = RAG. The bet is
*within-ceiling quality at lower cost, without degeneration*, on:

- **Degeneration / over-steer** — the always-on tap's failure mode (in-dist B0 reference: tap ripple
  0.352 vs RAG 0.426, "other" [degeneration] **0.611**; every-step injection collapses free-gen into
  repetition — the seed-once wall).
- **Selectivity** — the cartridge memorizes filler (BABILong 0.25/0.10 @1k/4k vs RAG 0.60/0.45; the
  surprise experiment failed: surprisal ≠ relevance).
- One **acknowledged confound it removes** — `mquake_ripple.py`'s own design note: "the tap bank is set
  globally per forward and keyed by the edit SUBJECT — not positionally addressed to where the subject
  appears in the question. A tap 'miss' can be an addressing/routing failure, not an integration
  failure."

## Why it is not a re-tread

Every landed delivery differs on both axes this one changes:

| artifact | read placement | read math | trained by |
|---|---|---|---|
| MAC (`recall_boltA`) | input prefix, untrained | latent prefix | — (falsified, ignored) |
| residual tap (MAG) | **every position**, always-on | softmax over bank | next-token delivery loss |
| KV inject (1c) | append attendable K/V | base's own softmax | LM loss |
| cartridge (D) | global KV prefix | base's own attention | **KL distillation** |
| **trigger (E1)** | **content-matched positions only** | tap math, gated by an O(1) content match | **KL distillation** |

New combination: content-addressed (not global), sparse/transient (not sustained), distill-trained (not
delivery-trained). The one experiment E1 is closest to is A1 below, which exists precisely to attribute
the effect (geometry vs objective).

## Mechanism

1. **Trigger keys** come from the staged edit, not from a guess: take the subject span's token ids from
   the per-edit builder (the same `binding_positions` / `binding_key_spans` / `q_subj_off` machinery the
   `_episodic_bank` path already uses to build the trained-encoding bank). Build a match set of
   **multi-order n-gram hashes — orders {2,3}** (Engram's N), each to 64-bit; single-token subjects fall
   back to the unigram. (This is the engram addressing half: deterministic hashing; collisions are
   *measured* — multi-head prime-bucket hashing is the hardening knob if false fires appear.)
2. **Trigger mask at forward time**: every forward, recompute matches over the CURRENT token ids
   (prompt + any text the model has generated) → `trigger_mask [B,T]`. Subject re-mentions inside the
   model's own chain-of-thought **re-fire the read** (re-ignition: influence as a function of mention,
   not a fixed window or an always-on gate).
3. **Injection**: extend `GatedMemoryTap` with the mask. At masked-out positions the read is **exactly
   no-op** (null-slot routing — queries on the zero-value slot; or a gate multiply — builder's choice,
   but the no-op must be exact and testable: empty trigger set ⇒ byte-identical to no-bank). At
   masked-in positions the tap math is unchanged (q=Wq·h; k/v from the staged bank; tanh(γ) gate;
   conf-gate honored).
4. **Everything else frozen**: base, bank path (episodic = the trained encoding), read math. New
   trainable params: only the tap module's (its existing set).

## Training (the settled lever)

Only the injector trains, by **KL distillation**: teacher = frozen base + edit-in-context; student =
frozen base + trigger injection (NO edit text in prompt); loss = KL over next-token dists on the
self-study distribution — reuse `cam/cartridge_train.py`'s generator/loop (generic seeds as in D0;
D3 entailment self-study as the stronger arm; the curriculum is provably not the lever, so keep the set
fixed for comparability). No prefix tokens, no KV build; at run time the trigger match is a per-position hash comparison and the
read is the existing small-k attention over the staged bank.

Optional (flagged, off by default): a short delivery-loss warmup to open gates before KL (guards gate
starvation from the MAC/tap history; if used, the reported run must say so).

**Attribution arms:** **A1** = distillation-trained tap WITHOUT the mask (isolates the objective
change; expect ≈RAG on bridge hops per the converged set). **A2** = trigger mask, distillation-trained
(the full geometry — the primary). **A2−** (optional) = trigger mask, delivery-trained (mask-only
ablation under the old objective).

## Eval (same-pass protocol, unchanged machinery)

- Reuse `mquake_ripple.py` verbatim: `eval_indist_ripple` (in-dist language-pivot 2-hop; ripple vs
  subject-shortcut separated by `gold_cf ≠ gold_true`; confound filter) for the in-dist arm,
  `eval_mquake_ripple` for the OOD arm. Add the trigger as one more condition where `g_tap = _gen(...)`
  sits, feeding it the **same bank/conf/relidx the tap uses** — isolating the geometry change.
- **Non-negotiable gates** (house style): single-hop control ≥ `CAM_MQUAKE_CTRL_MIN` (0.40) else run is
  INVALID; `no_edit`/`no_memory` floor pinned; **same-run baselines** — floor/oracle/cartridge/tap
  re-measured in the SAME pass (small-n is noise; sub-few-point gaps are ties).
- Bar (powered same-pass, §10): family n=113: floor 0.124 / oracle 0.531 / cartridge 0.575;
  country n=130: floor 0.015 / oracle 0.346 / cartridge 0.231. In-dist B0 degeneration reference:
  tap 0.352 / RAG 0.426 / other 0.611.
- **E1-specific instrumentation** (mandatory before reading any ripple number): trigger fire-rate;
  **trigger-position recall** (fraction of subject mentions masked-in — a dead mask silently reproduces
  `no_edit`); hash-collision fire precision; gate stats (`gate_stats()` / `cgate_stats()`); free-gen
  repetition probe (the F3 channel).
- If the primary passes: BABILong (`cam/babilong_eval.py` — distractor selectivity vs the cartridge's
  below-RAG 0.25/0.10) + NIAH (`cam/niah_eval.py` — parity check + read cost).

## Success / falsifiers

- **SUCCESS:** A2 ≈ cartridge/oracle on bridge-routed hops with "other" ≪ tap's 0.611 (target ≤0.25)
  and a healthy control — then selectivity ≥ cartridge on BABILong. That is "RAG-class ripple, no
  degeneration, cheapest delivery" and it enters the best-of stack.
- **FALSIFY:** control passes but ripple ≤ tap (and other stays high) → content-triggered input-space
  delivery joins the converged set as a fourth measured point; the engram-delivery question closes with
  a clean negative; move on. (Either outcome feeds the scorecard.)
- **RED FLAG:** trigger > oracle on any hop → measure artifact until proven otherwise (nothing frozen
  exceeds in-context; §10).

## Build

1. `cam/gated_tap.py` — mask support in `GatedMemoryTap` (+ exact no-op path); `TriggerInjector`
   (MAGInjector-compatible surface: `set_bank/attach/detach/eval/train/parameters`; carries mask state;
   optional `set_bank(..., trigger_ids=...)`).
2. `cam/trigger_match.py` (or fold into gated_tap) — n-gram hashing + subsequence-id match + fire stats.
3. `cam/recall_mag.py` — `--inject-mode trigger` (+ `--trigger-layer`, default = the same layer the tap
   arm uses in the same-pass run [12 in the in-dist protocol; sweep {12,24} as secondary], mirroring the
   kv/gdnstate wiring at ~2492–2505).
4. `cam/trigger_ripple.py` — runner mirroring `cartridge_ripple.py`: per edit, build bank → distill
   trigger → score `no_edit / rag / tap / cartridge / trigger` in ONE pass (keep `_hit` head-word
   matching for the verbose family/script golds).
5. `tools/trigger_selftest.py` — mask correctness (subject span located; fire stats; empty-set no-op
   byte-identical), mirroring `kv_inject_selftest.py` / `ripple_obj_selftest.py`.
6. `tools/trigger_ripple.sh` + `tools/trigger_wait_run.sh` queue runner (gpu-free poll → `gpu-lease -n 1`),
   mirroring `li_wait_run.sh` / `p2_wait_run.sh`.

**Checkout:** the harness lives on `wip/ripple-scorecard-harness` (worktree
`~/code/memory-organ-tapmetric-preserve` — holds `mquake_ripple.py`, `cartridge_*.py`, the injected
`gated_tap.py`). Branch `exp/e1-trigger-delivery` off it (or rebase that branch onto `main` first — its
18-file diff IS the harness and is unmerged). **Cost:** Qwen3-4B-Base (standard attention, no fla),
1×16GB gfx1201, local; per-edit distill + eval ≈ cartridge-run size — `--limit 8` first for signal,
then the full set; no cloud needed.

## Risks / confounds

- Trigger recall on paraphrased mentions (start id-subsequence + hashed orders; measure recall first).
- Hash collisions → false fires (measure precision; multi-head prime buckets if needed).
- Gate starvation (optional warmup lever; monitor gate stats).
- Dead mask machinery (selftest + fire stats mandatory before numbers — a dead mask is silent).
- Free-gen noise on a 4B base (existing caveat; keep `_hit` matching + same-pass baselines).
- Scope guard: single layer only (multi-site residual is falsified); don't touch the base; don't change
  the bank path; do not attempt above-ceiling.

## Deliverables / definition of done

1. Selftest green (mask correctness + no-op guarantees).
2. Same-pass table [floor / oracle / cartridge / tap / A1 / A2] on family + country, with per-case
   rows, CIs, control-gate printout, degeneration buckets.
3. Trigger diagnostics (recall / precision / fire-rate) + gate stats + repetition probe.
4. (If passed) BABILong + NIAH arms.
5. Verdict folded back: `docs/research/2026-10-17-engram-tech-transfer.md` updated; RESULTS §10 /
   distance scorecard gains a fifth condition row. Wrong turns stay in the record.

## Follow-on (not this experiment)

**E1b** — trigger selects *(position × fact)* among MANY staged facts (the full engram table read);
natural home is the BABILong multi-fact arm. **Serving** — the mask machinery is directly transplantable
to the deployed CAM serving plane (positional addressing of delivery inside the running engine),
gated on this arm landing.

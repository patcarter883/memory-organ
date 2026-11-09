# Relaunching memory-organ around Engram-style memory: a proposal and its verdict

*A research note, 2026-11-09. It has no new measurements. It follows
[engram-graft-assessment.md](engram-graft-assessment.md) (the "assessment"), which judged Engram as an
add-on arm. This note asks the harder question the owner posed: should Engram (hashed n-gram tables plus
EngramEdit writes on a frozen base) become memory-organ's **new core thesis**? Five candidate designs
were each red-teamed through three lenses (repo evidence, literature, feasibility) and then ranked.*

*Provenance. Repo facts cite `file:section` or `file:line`. arXiv numbers come from the assessment §1 or
from the red-team literature reads; I did not re-fetch them, and figures that only appeared in a
secondary summary are marked **unverified**. The "systems inventory" and "literature inventory" cited
below are working dossiers for this review. **They are not files in the repo**, and every cost or latency
figure taken from them is an estimate. The 15 red-team verdicts, the judge scores in §1 and §6, the critic report
and the fixes applied are saved verbatim in [engram-relaunch-dossier.json](engram-relaunch-dossier.json).*

## 1. TL;DR

**Verdict: NO-GO on relaunching now. GO on two bounded tracks, and a relaunch only if specific numbers
come back.**

- **What the relaunch would have to beat.** The incumbent is the deployed **arm-B pointer**: a
  whitened-GTE subject key whose value is the exact object token ids, delivered at the output and gated
  by presence (RESULTS §7 A/B/C bake-off; CAM_100_DISJOINT_RESULT "THE UNLOCK"; RESULTS §8).
  - All five candidate designs were benchmarked against **A1**, the trained `out_proj` logit-inject arm
    (0.81 → 0.565 from N=137 to N=441; key-encoding §0(D), §6g).
  - That is the wrong opponent. The pointer has no shared readout, and it is exact by construction.
- **All 15 red-team verdicts refuted their concept at major severity** (verbatim in
  [engram-relaunch-dossier.json](engram-relaunch-dossier.json)).
- **The incumbent itself has never been measured on the CounterFact triad.** No arm-B triad numbers
  exist at N=441/1105. `tools/bakeoff_abc.py` and `cam/export_serving.py`, which RESULTS §7 and §8 cite,
  are absent from this checkout (checked with `ls`).

**Conditions, in order:**

- **G0 (required first).** Measure arm B in-harness: the triad plus SSOR, alias, CounterFact+ and a
  powered format-transfer cohort, at N=441/1105 with 3 reps. This needs new code (§7 R1); arm B does
  not run on the existing harness today.
  - *"Weak" (placeholder thresholds set by this note):* arm-B generality < 0.82 or alias recall < 0.70
    at N=441. Either result would reopen the case for a δ\*-valued design (§10).
- **G1 (Engram-site relaunch).** Relaunch around hashed residual rows only if a training-free per-fact
  δ\* row at some layer ℓ ≤ 8 reaches all of the following. The first two bars are **absolute floors
  derived from A1**, not offsets from arm B; they are restated against R1's arm-B values once R1 exists:
  - generality ≥ 0.82 (A1's triad generality at N=137, key-encoding §0(D));
  - efficacy ≥ 0.88 at N=441 (A1's *solo* logit-inject fidelity 0.876, key-encoding §3.14; A1's own
    N=441 triad efficacy is ~0.55–0.565, §0(D)/§6g);
  - efficacy(1105) − efficacy(137) ≥ −0.05 (this note's isolation bar);
  - SSOR (same-subject other-relation) agreement ≥ 0.80 (this note's placeholder);
  - CounterFact+ keep within 0.02 of base (this note's placeholder);
  - **a win over arm B by ≥ 0.05** on format transfer or non-base-known multi-token span-exact
    (relative to R1);
  - survival under the served 35B numerics, i.e. replayed efficacy no more than 0.10 below in-HF
    efficacy (R2), with solve time ≤ 10 s. If full-depth solves from L2 really take the unverified
    25–90 s estimated in §6, this bar fails automatically at ℓ ≤ 8.

  If those bars are met only at ℓ ≥ 16, ship δ\* rows as a value option inside the pointer system,
  without the Engram brand. If generality is < 0.70 at every ℓ, the Engram site is dead on a frozen base.
- **G2 (key front end, an evolution and not a relaunch).** Adopt a hash-first (subject, relation) key
  with a GTE fallback only if all of these hold:
  - false-fire < 0.05;
  - alias recall within 0.02 of whitened GTE;
  - generality ≥ arm B;
  - per-step trigger overhead ≤ 3 % of p50 TPOT, with 0 per-step host syncs (placeholder bar).
- **G3 (unfreeze / co-train).** Only if G1 stalls (solo < 0.88 at every depth) **and** E1 shows raw-d
  rows hold flat in N where `out_proj` drops. Even then, the pass bar is ripple above the same-pass RAG
  oracle on held-out relations, per hop: family > 0.531 and country > 0.346 (RESULTS §10, measured on
  Qwen3-4B-Base).

**If G1 fails, which is the predicted outcome, the one-line pitch stays as it is.** The cheapest
valuable thing in this note is G0. It is a baseline the project needs whatever happens to Engram.

## 2. Why this would be a relaunch

The current thesis appears in README "Where this is going", ROADMAP "What is proven today" and RESULTS
§7–8. It
reads: *editable, routed, non-interfering memory for frozen LLMs*. The memory sits **outside** the
circuit: an exact table, delivered at the output (forced tokens or a conf-gated logit bias), gated by
store presence, base-agnostic and training-free, with millisecond writes. The founding claim that a
frozen circuit **reads** a memory channel, from the "Titans for everyone" MAG tap, was in effect
retired when delivery moved off the residual (key-encoding §0(C), §3.14: 0.65 → 0.876 solo).

An Engram relaunch would reverse that retirement and change four commitments:

| Commitment | Today | Engram core thesis |
|---|---|---|
| Who produces the object | The store, by forcing exact ids | The frozen circuit, reading an early additive row |
| Value | Exact token ids (lossless) | Continuous d- or r-dim row, specific to one base |
| Site | Logits / sampler | Residual at embed–L2 (Engram 2601.07372 §6.2; EngramEdit "before block 1") |
| Write | GTE encode + id insert, ms, no autograd | δ\* solve through the frozen base plus ridge allocation, seconds, autograd |

**The candidate new pitch:** *"Every fact is a hashed n-gram row that a frozen LLM reads at an early
layer and phrases in its own words."* That pitch is only worth adopting if the model's own phrasing buys
something the pointer cannot: format transfer, inflection, embedding the object mid-sentence, and
non-base-known multi-token objects. Otherwise Engram contributes a **key** (hash, presence bit,
prefetch) and the thesis does not change.

## 3. What carries over and what is retired

Sources: the asset inventory and the five designs' carry/discard lists, checked against `ls` of `cam/`
and `tools/`.

| Asset | Status under the recommended tracks | Notes |
|---|---|---|
| CounterFact loader `cam/realedit.py:46-183` | **Carries, with additions** | Add subject n-gram tuples and a tag for "paraphrase preserves the subject verbatim" |
| Validity gate `recall_mag.py:739` / `verdict_counterfactual` :1247 | **Carries as-is** | Mandatory. It guarantees argmax = o_old's **first token** on each record's own rewrite prompt only (:739-770); paraphrase and CounterFact+ prompts are not filtered |
| N-supply configs and probe-cache tools (`build_probe_cache.sh`, `n0*_probe.sh`, `m0_probe.sh`) | **Carries** | Gives N=137/441/1105 on Qwen3.5-4B (key-encoding §0(D)) |
| Triad / cohort code `eval_persistent_locality` :2674, `_NbrRec` :2606 | **Carries with a shim** | Needs an arm-agnostic delivery call. Every eval today takes `(adapter, injector)` |
| `eval_persistent_generate` :2199, `_gen_ptr` :2419 | **Carries, needs a port** | Today it reads the **PK-store id-bank** through `PKStoreAdapter` (`_ptr_ids` :2254, `store.read_ids` :2264-2266; per-position :2495), not the whitened-GTE pointer with the 0.90 gate. It becomes an arm-B baseline only after the R1 shim. Its span_exact is tautological for a pointer |
| `eval_persistent_rescue` :2025 | **Carries** | Presence is redefined as a verified hash hit (+0.464 target, RESULTS §8) |
| `CAM_LOGIT_INJECT` + `_GATE_HARD` :1545-1562 | **Carries as comparator A1** | Also the template for a sparse id bias without `out_proj` |
| `gate_router.py` | **Carries (logit site only)** | Its no-backward trick needs a logit-space term |
| `_subject_bank` md5 hash :1322 | **Generalised** | Becomes a salted, fingerprinted, exact n-gram key |
| `gated_tap.py` `decoder_layers` :22, γ=0 / dtype / NaN discipline | **Carries as an instrument** | Used as a pre-hook for E0 rows. Cross-attention readout retired |
| `lens.py` | **Carries** | Use a tuned lens at early layers (the file's own header warning) |
| `m2_adapter.py` `load_frozen_base`, `CAM_NATIVE_GDN` | **Carries** | Required on RDNA4 (fla segfaults; DIARY Ph.16) |
| `tools/pk_proxy.py` | **Rewritten** | Today it is a ~40-line PK slot-overlap script that loads a hard-coded `/engine/.probe_cache/cfmulti_03d36da327fc3ce4.pkl` (`tools/pk_proxy.py:6`). It becomes a CPU n-gram collision screen |
| `tools/whiten_precompute.py`, `CAM_GTE_KEYS` | **Carries as arm B / fallback key** | — |
| ProductKeyStore addressing, `PKStoreAdapter`, K1/K2, key heads/repulsion/MaxSim | **Retired after R1** | Exact hashing gives write ≡ read by construction. Kept until the R1 shim replaces it, because the carried `(adapter, injector)` evals, `_gen_ptr` and A1 all read the PK store (and `CAM_GTE_KEYS` lives inside `PKStoreAdapter`, `cam/pk_store_adapter.py:159-167`) |
| Titans lineage (`deep_memory*`, `store_recurrence`, `recall_boltA`, `bind_msweep`) | **Retired** | Already superseded (RESULTS §2) |
| `cam/translator.py` | **Retired** | Reuse answered NO (ROADMAP "Translator reuse"). Portability now means re-tokenising strings |
| Learned-addressing sweep drivers, multigate, cf_*_selftest | **Retired** | — |
| Bind/tap training as an eval prerequisite | **Decoupled** | Still needed for A1 |
| `bakeoff_abc.py`, `export_serving.py`, `CAM_REMEMBER_GATE=rank` | **Missing; must be rebuilt for G0** | Cited in RESULTS §7–8, absent here |
| minisgl serving (`/cam/*`, rank-0 broadcast) | **External, unverified** | `minisgl-rdna4` PR #2 (CAM_100_CONTINUATION). In-repo serving docs describe the 4B tap MVP, not the 35B pointer |
| **New:** whitened-GTE pointer harness arm, arm-agnostic delivery shim, exact hasher, δ\* solver, ridge allocator, corpus n-gram frequency table, SSOR/SNG/alias (RippleEdits/UniEdit)/CounterFact+/coreference-multi-turn/format-transfer cohorts | **Build** | No code exists for any of these |

## 4. Evidence for and against

**Site: strongly against an early frozen site; untested for per-fact δ\*.**
- *Against:*
  - The trained shared tap tops out at solo 0.69. Depth gives L18 0.565 / L24 0.69 / L28 0.701 /
    L30 0.623, and multi-layer injection 0.698 (key-encoding §3.7, §3.9).
  - Seven levers failed (§3.8–3.10). The wall is worse on a base that knows more: Llama-3.2-3B 0.574
    (§3.11).
  - Early L3 garbles (scorecard "The mechanism first"; RESULTS §10).
  - STAPLE's backbone co-adapts: removing every table drops it to 32.17, below dense 44.39
    (2610.12599 §4.1; **unverified**, full text not re-read).
  - Frozen-backbone Engram-style grafts **do exist for task gains**, but none is shown for **fact
    editing**:
    - the Engram Adapter is a "post-hoc adapter for frozen LLMs" (Qwen3-4B/8B) with n-gram matching,
      residual injection and a learned scalar gate (2608.29327 abstract). Its gate and projection
      attenuate the injected residual to **about 0.08 % of the hidden-state norm** (abstract). That
      directly supports the "frozen circuit barely reads an early additive channel" wall;
    - NGM is a training-free n-gram module plugged into Qwen3 and Llama-3.2 backbones (2605.16893
      abstract; that the backbone is left frozen is implied, not stated);
    - Cross-Model Memory Transfer freezes an Engram memory trained on a source model, attaches it to a
      different target model and trains only a lightweight reader (2608.17050 abstract).
  - An independent frozen-base Engram replication reports zero downstream gain
    (github.com/t-hertel/Engram, **unverified**).
- *For:*
  - Every repo wall number comes from a **trained, shared** `GatedMemoryTap`. No per-fact optimised δ\*
    was ever run here (assessment §3: E0 is unrun; grep finds no v\*/ROME run).
  - Frozen bases demonstrably read per-fact optimised subject-position perturbations:
    - ROME, ~99.9 ES on GPT-J (as cited in Larimar 2403.11901 Table 2);
    - NeuralDB, a "non-linear gated retrieval module" over a neural KV store, run at 10k edits and
      reported effective at 100k edits on GPT2-XL, GPT-J and Llama-3-8B (2507.18028 abstract). The
      per-N figures (efficacy 99.9 → 99.2 from 2k to 10k on Llama-3-8B, at layer 7) are **unverified**
      (full text not re-read; not in the abstract);
    - SWEA adds token-id-keyed subject embeddings at the **input** (ℓ=0) (2401.17809). The figures
      99.6/98.3/79.0 on GPT-J at 10k edits are **unverified** (full text not re-read; the abstract
      names no model or numbers);
    - REMEDI uses a trained layer-1 editor (2304.00740).
  - So "the frozen circuit cannot read an early channel" is **false for per-fact solves**. The repo's
    §3.10 claim that the wall is intrinsic needs to be narrowed to trained shared readouts.
- *Still unknown:* single-hop delivery at ℓ ≤ 16 on this repo's bases. Also whether the
  prior-confidence and vocabulary factors of the wall (r = −0.60 and r = −0.89; key-encoding §3.10–3.12)
  describe the downstream circuit, in which case they also hit δ\* (an extrapolation), or the shared tap.

**Value: against continuous values, for row isolation.**
- *Against:*
  - Continuous per-position values wall at t ≥ 1: isolated 0.50, delivery 0/4, cosine-NN 0.33
    (CAM_100_DISJOINT "The result", "Lever 1/3/4").
  - A pooled latent decodes "Normandy" as "Moscow" (key-encoding §3.28).
  - In realistic evaluation, editing methods drop sharply: 38.5 % vs 96.8 % (Mirage 2502.11177
    abstract). The ROME-class QAEdit figures of 69.1 reliability / 57.0 generalisation (Table 3) are
    **unverified** (full text not re-read).
- *For:* one fact per row is purer. Delivery is 0.635 → 0.723 going from B=32 to B=137
  (key-encoding §3.15). NeuralDB's isolated rows stay flat in N (per-N figures **unverified**, above).
- *Caveat:* the pointer is already lossless and isolated, so isolation is not a gain over arm B. The
  in-repo pointer evidence is CAM_100_DISJOINT_RESULT "THE UNLOCK": a PK-store id-bank delivering 4/4
  invented multi-token objects exactly. The 0.917 counterfactual delivery with 48 multi-token objects
  coexisting (RESULTS §4 "Coexistence at scale") belongs to the per-position disjoint **product-key
  store** (BANKS=1024 + bf16), not to arm B.
- *Multi-token supply is thin.* About 46 % of objects are multi-token in the A/B/C bake-off set
  (RESULTS.md:538; whether that set was validity-gated is **unverified**). But at `CAM_MAX_OBJ_TOK` ≤ 4
  only 598 of 21,919 candidate CounterFact facts (2.7 %) have multi-token objects (key-encoding §3.19),
  and the base-known filter leaves about 1 per 137 (§3.28). Multi-token cohorts must be built outside
  the filter or they are underpowered.

**Key: exact hashing is sound but solves a solved problem, at an alias cost.**
- *For:*
  - Hashed token-id banks were the winning addressing lever: 0.24 → 0.66 (key-encoding §0(A)).
  - K1 shows the value of write ≡ read: below-gate 0/441 (§3.16, §6g).
  - A hit bit is the presence/provenance signal (+0.464; RESULTS §8).
  - It needs no encoder forward and no subject span (integration_design.md §3a/§7 risk #1).
- *Against:*
  - Addressing is already closed, so hashing fixes nothing that is open (assessment §2 "Key").
  - Realistic routing may be bottlenecked on aliases: retrieval 0.70, misses on "same-structure
    aliases" (scorecard finding 3). That is a **different router**, the cartridge router on
    Qwen3-4B-Base with mean-centred subject keys, not whitened-GTE arm B, and reading "same-structure
    aliases" as surface aliases rather than structurally confusable facts is an interpretation
    (**unverified**).
  - A claimed result that subject-string gating on CounterFact covers 100 % of paraphrases but only
    32 % of aliases, against 99.3 % for a learned scope, was attributed to 2609.11063 SN13. The
    2609.11063 abstract does not mention gating or aliases, so the source is **unverified** and the
    figure is not used as evidence here.
  - Token-id keys are bypassed by alternative tokenisations, 38.6 % (2609.29045, **unverified**,
    secondary summary).
  - NFKC + casefold does not strip diacritics; NFD with mark removal is needed.

**Gate: presence wins; trained context gates lose.**
- *For:* the hard presence gate holds keep exactly at the 0.466 baseline (key-encoding §3.15).
- *Against:*
  - A learned **prompt-novelty** gate (the null slot) costs generality: 0.611/0.667 against 0.907/0.926
    for the store-confidence gate (RESULTS §7 "Closing the gap"). Engram's trained α\_t is a context
    gate of this broad kind. Applying the same cost to a relation or prompt-surface AND-gate is an
    **extrapolation**; no such gate was measured.
  - Always-on firing degenerates ("French French…", §3.20).
  - The o\_old displacement trigger is tautological on the records' own rewrite prompts, because the
    validity gate guarantees argmax = o\_old's first token there (recall_mag.py:739-770). It is not
    tautological on paraphrase or CounterFact+ prompts, which the gate does not filter. For novel
    subjects o\_old is a generic default such as "English" (key-encoding §3.29).

**Integration: against every injection.**
- KV-append ripple 0.000, GDN seed 0.067, residual tap 0.21–0.31, all below same-pass RAG (country 0.346, family 0.531) (RESULTS
  §10; scorecard F2).
- Trained readouts are relation-specific: novel relations 0.000 against 0.77 (key-encoding §3.23);
  ripple 0.85 → 0.31 on a held-out relation (scorecard F3).
- EngramEdit's native MQuAKE-CoT is 36.9/20.1/15.1 (assessment §1).
- No Engram variant promises ripple above in-context.

**Portability: against, unless keys and values are text.**
- δ\* rows are specific to one base and token-id keys to one tokenizer (assessment §2).
- Translator reuse already failed (ROADMAP).
- Arm B is already base-agnostic and training-free (RESULTS §7 A/B/C bake-off). The serving docs
  (integration_design.md; online_api.md §1.3/§4.3) describe the PK-store tap MVP, whose warm start
  needs "the matching adapter checkpoint" (online_api §4.3), so they are not evidence for arm B's
  rebuild-from-strings path. How the deployed 35B pointer stores and rebuilds edits is **unverified**.
- A text-keyed Engram adds nothing here. A δ\*-valued Engram is a regression.

## 5. Recommended core design: keep arm B, run two tracks

The core stays the **arm-B pointer**: exact object ids, output delivery, presence gate. Two tracks run
against it.

### Track 1: evolution. An Engram-style key front end on arm B

1. **Normaliser.** NFD → strip combining marks → NFKC → casefold, then Unicode `\w+` word segmentation.
   Map positions to words through the fast-tokenizer offset mapping.
2. **Keys.**
   - `key64 = blake2b-64(ns_salt ‖ relation_id? ‖ last ≤4 canonical subject words)`, exact, with the
     full canonical subject stored for backward-span verification. Python `hash()` is never used
     (online_api.md §0).
   - The relation tag is a **disambiguator, never an AND gate**. Consult it only when a subject has
     more than one live relation, which closes online_api.md §1.2 (one live value per subject). An AND
     gate would likely pay a generality cost like the one measured for the learned novelty gate
     (0.611 against 0.907, RESULTS §7 "Closing the gap"); that transfer is an extrapolation.
3. **Lookup order.** Hash first; on a miss, fall back to whitened GTE at the 0.90 gate. Report every
   fire with its source (hash, GTE, or none).
4. **Hit bit = provenance.** It feeds the presence-gated delivery and the rescue protocol.
5. **Multi-token continuation.** Stateless per-position prefix keys `(edit_id, emitted o_new[<t])`
   replace the `_gen_ptr` cursor. Seed-once release after the span (RESULTS §8).
6. **Ambiguity rule.** Subjects whose key has corpus frequency above τ (common 1-word nouns) need the
   relation tag or a GTE confirmation. This takes EngramEdit's frequency penalty (2610.10533 A.3.3)
   as a key-selection rule.
7. **Edit-plane format.** An auditable cartridge file (manifest with normaliser, Unicode and salt
   versions; `facts.parquet` with provenance and tombstones; `rows.parquet` with `ngram_text`).
   Delete is exact. This answers BadEngram's point (2609.13478) that the write surface is an attack
   surface.

What it keeps: base-agnosticism, ms writes, no KV / prefix / GDN cache contact, losslessness. What it
risks: alias recall and per-step serving cost (§8).

### Track 2: falsification. A training-free per-fact hashed residual row (assessment E0/E1/E2)

1. **Row.** One raw d\_model δ\* per fact, keyed by an exact verified hash of the last min(n\_subj, 3)
   subject token ids, with the full span verified. Hooked as a pre-hook on `decoder_layers(base)[ℓ]`
   at the last verified subject token. A miss means no hook output, which is a byte-exact no-op.
2. **Solve** (base frozen, ROME/EngramEdit-style):
   - objective `−Σ_j log P(o_new span | prefix_j ⊕ prompt; h_ℓ[t] += δ) + λ_KL·KL(SSOR-train ∪ "{s} is a")`;
   - clamp `‖δ‖ ≤ ρ‖h_ℓ[t]‖` with ρ ∈ {1, 4};
   - K = 10 base-generated prefixes, about 20–25 Adam steps (ROME defaults, **unverified**);
   - SSOR-train templates are disjoint from SSOR-eval templates;
   - facts sharing a subject share one row, solved jointly.
3. **Sites:** ℓ ∈ {embed, 2, 8, 16, 24}.
4. **Arms in the same session:**
   - **E0** private raw-d rows;
   - **E1** a fixed random orthonormal r=512 projection, and separately the trained `out_proj`. This
     directly tests the key-encoding §0(D) inference;
   - **E2** EngramEdit ridge `U* = (AᵀA+Λ)⁻¹AᵀB`, with λ\_j = 0.05·w\_j + 0.01 over unverified
     2/3-gram rows (2610.10533 Eqs. 6–10);
   - **NeuralDB control:** the same δ\*, gated by activation cosine > 0.65. The abstract describes
     NeuralDB's gate only as a "non-linear gated retrieval module" (2507.18028), so the cosine form and
     the 0.65 threshold are **unverified**. This leaves key type as the only variable;
   - **A1** and **arm B** as references.
5. **Framing.** Present this as *"NeuralDB/SWEA with an exact verified token-span key"*, not as
   Engram. Engram's context gate, short conv, multi-head summed hashing and offload are all absent,
   deliberately: summed colliding slices become locality hits on a frozen base.

### Ideas carried in from the runners-up

- **From CF-Engram:** the SSOR cohort becomes the primary locality metric for any subject-keyed write,
  because DEP neighbours name other subjects and are near-vacuous for exact keys (realedit.py:9). Also
  the SNG cohort (next-token KL over the 8 tokens after incidental n-gram hits in generic text), and a
  Phase-0 writability screen.
- **From EKP:** the o\_old-rank displacement gate, kept **only as an ablation**. It is measured per
  position in free generation and multi-turn, against an ATBias-style rank-constrained bias
  (2406.12468).
- **From the Cartridge:** negative-row text-space ridge allocation as an *offline false-fire
  calibrator*, not a memory thesis. With binary B it is a bag-of-n-grams ridge classifier.
- **From Mode B:** the A3 "LoRA-only" control logic, an empty-table capability check, and held-out
  relation efficacy as the primary metric. All three are reserved for G3.
- **From every red-team:** CounterFact+ prefixed neighbours (2305.17553); RippleEdits/UniEdit alias
  cohorts (2307.12976, 2505.12345); coreference and multi-turn prompts; a powered format-transfer
  cohort (≥ 150 prompts). The 12-prompt gen-coherence set has SE ≈ 0.13 around the 9/12 baseline.
- **Test hygiene:** an "exact no-op on miss" logit byte-identity test and a CPU/GPU hasher-parity test
  in `tests/test_smoke.py`.

## 6. Alternatives considered and their red-team outcomes

Judge ranks are on a 10-point scale. Every concept was refuted at major severity on all three lenses.

**Delta-Engram (overall 6; adopted as Track 2).** The premise holds: the wall was never measured for
per-fact δ\*. Against it:
- The core is published. NeuralDB has flat-in-N gated per-fact residuals on the CounterFact triad
  (2507.18028), and SWEA has token-id-keyed subject embeddings at the input (2401.17809). Neither is
  cited anywhere in the repo's docs (grep).
- Its relaunch bar was set against A1 rather than arm B.
- Its locality and false-fire kills pass by construction.
- The 35B write path needs a differentiable copy of about 70 GB against 2×16 GB local, and HF-to-serve
  residual parity has never been run (parity_spike.md status: plan only).

Outcome: a valuable falsification arm and a weak core thesis.

**Engram-Keyed Pointer (overall 6; adopted as Track 1).** It keeps every measured escape and is honest
that it is an evolution. Its Stage-1 test is tautological on the records' own rewrite prompts: the
validity gate guarantees argmax = o\_old's first token there (recall_mag.py:739-770), so the
displacement trigger always fires on them, and different-subject neighbours never arm. Paraphrase and
CounterFact+ prompts are not filtered, so the test is not tautological there.
- Its one discriminating axis is alias coverage. A published negative answer (32 % against 99.3 %) was
  attributed to 2609.11063, but that paper's abstract does not cover gating or aliases, so the source
  is **unverified** and the point is not used as a kill argument. R0/R1 measure alias recall directly.
- The displacement gate resembles ATBias (2406.12468), and "memory outside the circuit" is SERAC
  (2206.06520).
- A per-decode-step trigger reverses the prefill-only store read that keeps decode graph-capturable
  (integration_design.md §5b/§5d). Those sections describe the 4B tap MVP; that the deployed 35B
  pointer uses the same prefill-only capture design is **unverified**.

Outcome: a key front end, not a thesis.

**Engram Cartridge (overall 5; parts adopted).** "One memory, any frozen model" is already a property
of arm B (RESULTS §7). NFKC word-n-gram hashing was attributed to 2608.17050, whose abstract does not
mention NFKC or word n-grams (**unverified**); byte-level tokenizer-agnostic Engram hashing is published
(2607.29065 abstract). The relation AND-gate is a prompt-surface gate; the repo measured a learned novelty gate as costly to
generality (RESULTS §7 "Closing the gap"), and extending that cost to an AND-gate is an extrapolation. A W=64 window scan fires on every CounterFact+ prefixed neighbour. Its kills are
near-tautological: pointer span\_exact is true by construction (recall_mag.py:2419-2438), and
cross-base parity follows because the key never touches the base.

Outcome: the file format, the (subject, relation) disambiguator and the negative-row calibrator are
kept.

**CF-Engram, a calibrated fusion reader at L2 (overall 3; parked).** It reverses both measured escapes,
back to a continuous value and back to the earliest site. Its justification, that a calibrated reader
generalises to held-out relations, is contradicted twice by repo data (key-encoding §3.23; scorecard
F3). Its training-free core is SWEA, and its calibrated editor resembles REMEDI, whose essence score of
24.8 shows a trained early editor did not fix same-subject bleed (2304.00740). Its SITE KILL cannot fire,
because the concept expects per-fact solves to work trivially. Feasibility re-estimate (**unverified**):
the red-team assumed about 1 s/step for a backward above L24 only, giving roughly 25–90 s per full-depth
solve from L2 and a few hundred GPU-h in total, against the 25–35 claimed. No source for the 1 s/step
anchor was found: DIARY Phase 16 gives only stage-2 wall time (527 s → 147 s), and
`tools/diag_bwd_timing.sh` exists but its output is not in the repo. R2 measures this. There is no hardware home for a 35B
write and calibration path.

Outcome: only SSOR/SNG survive.

**Read-Port Retrofit, Mode B: world-swap co-training of a 1.7–4B checkpoint, then freeze (overall 3;
conditional on G3).** It is the only route with a credible path to ripple above injection, and its
question, whether the wall comes from frozen-ness or from the site, has never been tested with a
trained backbone. Against it:
- It drops every defining invariant: frozen base, attach-anywhere, removable organ, the deployed 35B.
- Repo data already points to the site: the same frozen base gets 0.65 → 0.876 solo by moving to
  logits (key-encoding §3.14). Frozen ROME is about 99.9 ES, so A0 probably saturates and the A0-vs-A2
  contrast tells nothing.
- Prior art covers the thesis:
  - Larimar: train once, closed-form writes, exact forget (2403.11901);
  - KBLaM: synthetic-KB reader training (2410.10450);
  - LTE: its own A3 control (2402.11905).
- A negative at 200M tokens cannot be told apart from under-training, since STAPLE uses about 200B.
- The site and solo thresholds (key-encoding §3.7–3.15) were measured on Qwen3.5-4B, not on the
  proposed Qwen3-4B-Base. Only the ripple/RAG bars (RESULTS §10) were measured on Qwen3-4B-Base.

Outcome: run it only after E0/E1 make a specific case for it.

## 7. Experiment ladder

No rung before R8 trains base weights or any new memory module; the A1 and trained-`out_proj`
references (R1, R4/E1) need the existing stage-2 tap bind, and R8 is the only training rung. Base is
Qwen3.5-4B with `CAM_NATIVE_GDN=1` under `watchdog_run.sh` unless stated. Bars marked "vs arm B" are
relative to R1's arm-B numbers; the G1 floors are absolute and A1-derived (§1). Costs are estimates.
Only R0 runs on existing code, and even R0 depends on the hard-coded probe-cache pickle in
`tools/pk_proxy.py:6`.

| Rung | What | Base, N | Metrics | Cost | Kill / decision (numeric) |
|---|---|---|---|---|---|
| **R0** CPU key screen | Rewrite `pk_proxy.py`. Compare exact canonical-text key, token-id last-n (n = 1/2/3) and whitened GTE over: edit, paraphrase, surface variants (case, diacritics, surname, retokenisation), DEP neighbours, CounterFact+ prefixed, SSOR, out-of-store, generic text | CPU, N = 137/441/1105 + full CounterFact | hit rate, false-hit rate, alias recall, writability after frequency cut | ~1–2 eng-days, minutes of CPU | Hash key dead if paraphrase hit < 0.95 even with aliasing, **or** out-of-store false-hit > 0.05 (the 0.90 GTE gate, RESULTS §7), **or** > 10 % of subjects have no writable 2/3-gram |
| **R1 = G0** incumbent baseline | Build items (none exist today): (a) a whitened-GTE pointer harness arm with the 0.90 gate (the in-harness `_gen_ptr` reads the PK-store id-bank via `PKStoreAdapter`, `recall_mag.py:2254-2268`, :2419; `bakeoff_abc` and the rank-R gate are missing); (b) five cohort loaders: SSOR, alias (RippleEdits/UniEdit), CounterFact+, coreference/multi-turn, and format-transfer (≥ 150 prompts, built **outside** the base-known filter so multi-token is powered); (c) the arm-agnostic delivery shim replacing `(adapter, injector)`. Then run arm B, A1 and base | N = 441/1105, 3 reps | efficacy, generality (split by whether the subject string is preserved), DEP keep/leak, SSOR agreement, CF+ keep, alias recall, span-exact, judged format-transfer | ~8–12 eng-days build (estimate) + < 1 GPU-day | No kill on the rung itself. Produces every "vs arm B" reference below. Arm B counts as **weak** (§1 G0, §10) if generality < 0.82 or alias recall < 0.70 at N=441 |
| **R2** write feasibility | Time δ\* solves on 20 records on RDNA4. One cloud spot check: solve on Qwen3.6-35B-A3B in HF bf16 and replay through the served (likely quantised; **unverified**) minisgl TP=2 path, by adapting the never-run parity_spike.md probe | 4B local; 35B cloud, 20 facts | s/solve; efficacy retention after replay | 1–2 days, < $50 cloud | Systems kill for online writes if > 10 s/solve on the 4B. **35B relaunch dead whatever the 4B result** if replayed efficacy falls > 0.10 below in-HF efficacy |
| **R3** E0 site sweep | Private raw-d rows, ℓ ∈ {embed, 2, 8, 16, 24}, ρ ∈ {1, 4}, NeuralDB cosine control, arm B in the same run. Tuned-lens check of propagation at embed/L2 | N = 137, 1 rep | efficacy, generality, SSOR, CF+ keep, prior-confidence stratification (the 12 % always-fail subset, key-encoding §3.10) | ~2–3 eng-days. 137 × 5 sites × 2 ρ ≈ 1,370 solves. Solve time is unmeasured until R2: at 1–3 s/solve about 0.5–1 GPU-h; at the unverified 25–90 s/solve full-depth estimate (§6) about 10–35 GPU-h, so subsample to 137 × 3 sites if R2 is slow. At > 10 s/solve the G1 time bar already fails for that site | **Site dead** if generality < 0.70 at every ℓ. **Early-site dead** (Engram identity gone) if embed/L2 trail the best ℓ by > 0.10 generality or have SSOR < 0.80 |
| **R4** E0/E1 scale | Best 2 sites. Raw-d vs fixed r=512 vs trained `out_proj`. NeuralDB control. Arm B | N = 137/441/1105, 3 reps | as R3, plus efficacy slope over N | 2–10 GPU-h | **G1 bars (§1).** Isolation claim dead if efficacy(1105) − efficacy(137) < −0.05. §0(D) diagnosis wrong if raw-d drops like `out_proj` does |
| **R5** where δ\* could beat the pointer | Powered format-transfer / free-generation cohort and non-base-known multi-token cohort (private-demo invented objects + `CAM_MAX_OBJ_TOK` ≤ 4), δ\* vs arm B (with seed-once) | ≥ 150 prompts, 3 reps | judged span-exact, fluency, CounterFact+ drift in long-form (cf. 2402.09394) | ~1 GPU-day | **Relaunch dead** unless δ\* beats arm B by ≥ 0.05 on one of these. Multi-token stays on the pointer if span-exact < 0.75 |
| **R6** E2 ridge | Ridge allocation over unverified 2/3-gram rows with the frequency penalty (new corpus frequency table) vs private verified rows | N = 441, 3 reps | surface-variant generality vs shared-n-gram leak, SNG KL, ZsRE-style specificity vs pre-edit | ~2 eng-days, ~2 GPU-h | Kill if generality gains < 0.03 over E0, **or** SNG KL > 0.3 nats/token (**placeholder**: key-encoding §3.25's 0.3 is the confident-bucket KL on *edited* prompts at α = 0.25, not per-token KL on generic text, so it only sets the order of magnitude), **or** shared-n-gram leak > 2× DEP leak. Rationale for the leak bars: in EngramEdit's own native setting > 92 % of new unrelated errors involve updated rows, and ZsRE specificity falls 41.6 → 38.3 (assessment §1–2) |
| **R7 = G2** serving spike | Per-step trigger (hash-first, GTE fallback, prefix-key continuation) as static GPU buffers in minisgl's captured sampler under TP=2 | 35B, live traffic replay | TPOT overhead, host syncs, false-force per turn | 2–3 eng-days (external code, **unverified**) | Kill if p50 TPOT overhead > 3 % (placeholder), **or** any per-step host sync is needed. Otherwise keep the prefill-only read |
| **R8 = G3** (conditional) co-train | Pre-registered late-port-only A1/A2/A3 contrast; ≥ 2 training seeds; thresholds re-measured on the retrofit base; Larimar/KBLaM reference arms | Qwen3-4B-Base, cloud | held-out-relation efficacy; ripple vs same-pass RAG; empty-table capability | ~35–40 H100-h claimed; eval alone 30–110 H100-h without a batched solver (**unverified**) | Enter only if R3/R4 show solo < 0.88 at every ℓ and raw-d flat in N. **Pass only** if held-out ripple beats same-pass RAG per hop (family > 0.531, country > 0.346; RESULTS §10, Qwen3-4B-Base) and capability drop ≤ 1 pt |

Order: R0 and R1 run in parallel, then R2, then R3 through R6. R7 depends only on R0 and R1.

## 8. Serving implications

Source: the systems inventory (an estimate dossier). The 35B config values (d = 2048, ~40 layers) are
**unverified**.

- **Track 1 (key front end).**
  - Host RAM only, a few hundred bytes per edit. No VRAM, no KV/GDN contact, rank-0 decision and
    broadcast as deployed (README:173-176).
  - The real cost is moving from a **prefill-only** read to a **per-decode-step** trigger. Capture
    safety came from hoisting the store read to prefill (integration_design.md §5b/§5d, written for the
    4B tap MVP; the 35B pointer's capture design is **unverified**).
  - Either the trigger lives as static GPU buffers in the captured sampler, or a per-step host sync
    appears. R7 measures which.
- **Track 2, if it ever serves.**
  - *Read path:* cheap, an integer hash plus gather. The table fits in VRAM at edit scale: 10^6 d-wide
    rows ≈ 4 GB at d = 2048 bf16. CPU offload pays only for multi-tenant tables of about 10^7 rows or
    more.
  - *TP:* add δ on the owner rank (hash % TP) before the existing row-parallel all-reduce. That is
    rank-authoritative by construction with no extra collective. It is not byte-identical to eager HF,
    and it needs care if minisgl fuses the all-reduce with RMSNorm (**unverified**).
  - *Hazards:*
    - **Stale caches.** An early δ is baked into every downstream KV/GDN state, so prefix-cache keys
      must carry `(namespace, table_epoch)`, and every write invalidates that tenant's cached
      prefixes.
    - **Spec decode.** A draft head without memory collapses acceptance exactly on memory hits
      (deployment config **unverified**).
    - **Writes.** Autograd through the 35B in a sidecar, about 1–5 s/edit on GPU (estimate). Bulk
      import of 10^4 edits takes about 3–14 h, against seconds for the pointer. Writes become
      asynchronous (`edit_pending`).
    - **Graph capture.** Fixed table capacity; growth means re-capture.
  - *Portability:* every base bump re-solves every δ\*. The `(subject, relation, object)` strings stay
    the source of truth (online_api.md §1.3).

## 9. Prior art and differentiation

**What would NOT be novel**, so we should not claim it:

| Claim | Prior art |
|---|---|
| Per-fact δ at the last subject token, base frozen, ~100 % efficacy | ROME 2202.05262 (from memory); SUIT 2509.24502 (snippet) |
| Isolated gated per-fact residual rows, flat in N on the CounterFact triad | NeuralDB 2507.18028 |
| Token-id-keyed per-subject embeddings added at the input of a frozen base | SWEA 2401.17809 |
| A trained early editor that is then frozen | REMEDI 2304.00740 |
| Discrete keyed codebook of edits on a frozen model | GRACE 2211.11031; MEMOIR 2506.07899 (TopHash masks); WISE 2405.14768 |
| Memory outside the circuit with a scope gate and output override | SERAC 2206.06520, including its training-free lookup-cache baseline |
| Subject-string gating on CounterFact | attributed to 2609.11063 SN13; **unverified** (the abstract does not cover it) |
| o\_old top-k displacement biasing | ATBias 2406.12468 |
| Tokenizer-agnostic Engram keys | byte-level hashing, Tokenizer-Agnostic Engram 2607.29065; NFKC word-boundary keys attributed to 2608.17050, **unverified** (not in its abstract) |
| Suffix-match forcing of continuations | REST 2311.08252; terminology-constrained decoding 1704.07138 / 1906.01105 |
| Train once to read, then closed-form writes with exact delete | Larimar 2403.11901; KBLaM 2410.10450; LTE 2402.11905 |
| Ridge allocation into n-gram rows with a frequency penalty | EngramEdit 2610.10533 (native base only) |
| Post-hoc row overlays per user | User-as-Engram 2606.19172 (Engram-native base) |
| Frozen-base Engram adapters for task gains | Engram Adapter 2608.29327; NGM 2605.16893; Cross-Model Memory Transfer 2608.17050 (frozen memory, trained reader); NeMo RFC #4292 |

**What could still be defensible** (narrow, and only with measurements):
1. Whether an **exact verified presence bit** can replace NeuralDB's cosine threshold and remove its
   fluency/specificity trade-off at 10^4 edits on a GDN-hybrid or MoE base. R3/R4 run the cosine gate
   as the in-run control.
2. A controlled head-to-head of text-hash and whitened-GTE keys with value and site held fixed, on
   alias, CounterFact+ and SSOR cohorts.
3. The presence/provenance rescue (+0.464, RESULTS §8). This is the repo's least-trodden claim
   (novelty-positioning.md Claim 4), and it predates Engram.
4. Rank-authoritative, multi-tenant, implicitly triggered serving of an exact edit table in a TP=2 MoE
   engine.

The repo's earlier prior-art map, in the assessment and in novelty-positioning.md, **missed NeuralDB,
SWEA, REMEDI, Larimar, KBLaM and LTE** (grep finds none). That gap should be fixed in
novelty-positioning.md regardless of what happens to this proposal.

### Reconciliation with the assessment

- **Agrees:** do not replace the current system; run E0 → E1 → E2 before any fusion training; keep key
  properties if the site dies.
- **Changes:**
  1. The relaunch bar is now a win over arm B, not over "conf-gated logit inject". The assessment's E0
     kill ("below the conf-gated logit inject") is replaced by the G1 bars, whose generality and
     efficacy floors are still absolute A1-derived numbers until R1 supplies arm-B values. G0 is added
     as a hard precondition.
  2. The assessment's §1 verdict "no frozen-backbone graft exists" is narrowed. Frozen-backbone
     Engram-style grafts exist for **task gains** (Engram Adapter 2608.29327, NGM 2605.16893,
     Cross-Model Memory Transfer 2608.17050; §4); none is shown for **fact editing**. The Engram
     Adapter's ~0.08 % residual-to-hidden-norm ratio is consistent with the predicted wall. The
     *frozen-base, token-keyed, per-fact editing* cell is occupied by non-Engram work (NeuralDB, SWEA),
     so E0 is a replication-plus-key-swap, not a discovery. A positive E0 does **not** by itself
     justify the Engram name.
  3. E3 (calibrated fusion) moves from "run if E0 leaks locality" to **parked**, because of the
     relation-shortcut evidence (§3.23, F3) and REMEDI's essence result.
  4. E5 portability via the translator is dropped in favour of text keys and re-tokenised strings.
  5. Mode B moves from "out of scope" to "conditional on G3 with a ripple-above-RAG bar".
  6. SSOR, CounterFact+ and alias cohorts become mandatory. The assessment's DEP-only locality is
     near-vacuous for exact keys.
  7. E4 (#100 per-position keys `hash(o_{t-2}, o_{t-1})`) is absorbed into Track 1 step 5 as
     stateless prefix keys on the pointer, with exact ids as the value rather than a δ row, so its
     "beat the 0.50 wall" kill no longer applies.
  8. **STAPLE's mechanism is described differently in the two notes.** The assessment §1 places the
     tables "inside the FFN, added to the up-projection"; the 2610.12599 abstract says STAPLE "retains"
     the FFN up-projections "and adds the embeddings as a residual pathway". Where the residual branch
     joins (summed with the up-projection output or added to the residual stream) is **unverified**
     until the paper's §3 is read. Either way the backbone is trained, so the conclusion is unchanged.

## 10. Risks and open questions

- **The incumbent's numbers do not exist yet.** If arm B is weak on the triad at N=441 (never
  measured; "weak" = generality < 0.82 or alias recall < 0.70, placeholder thresholds from §1 G0), the
  calculation changes. G0 settles this.
- **Prior-strength transfer.** Does the prior-confidence / vocabulary-size wall (r = −0.60 and
  r = −0.89) bind δ\* as well? It would predict failure on the 35B, which knows more (key-encoding
  §3.11). R3 stratifies for this.
- **Essence bleed.** A subject-position δ rewrites every relation of that subject, which is ROME's
  classic failure (REMEDI essence 24.8; SWEA relation specificity 0.41). There is no repo anchor; SSOR
  is new.
- **Harness noise.** Delivery wanders ±0.15 run to run (key-encoding §3.15). There are RDNA4 sync hangs
  (§3.17) and long-paraphrase hangs (`CAM_PROMPT_MAXTOK=64`). Kill margins of 0.02–0.05 need 3 reps
  and capped cohorts.
- **Hardware.** No differentiable 35B fits on 2×16 GB, and the serve stack is likely W4A8 (inferred
  from the `vllm22-w4a8` image name, tools/run_cam_native_gdn.sh:31; **unverified**). Every 35B write
  path is cloud-only or needs a third GPU.
- **External code.** minisgl-rdna4 is not in the repo, so serving estimates are unverified.
- **Security.** Writable tables are a backdoor surface: BadEngram modifies only the memory parameters,
  leaving backbone weights and the execution graph unchanged, and reaches 96.6 % ASR on a controlled
  Engram model and 47.8 % (HarmBench) / 64.0 % (AdvBench) on a production-scale PLE subsystem
  (2609.13478 abstract). The edit plane needs auth and audit before any write API ships.
- **Open:** whether Engram-native frozen bases (DeepSeek-V4.1-Flash, Qwen3.8-Next PLE, LongCat-Flash-Lite;
  per the literature inventory, licences and native training **unverified**) are a better relaunch
  target. That route would mean "edit the native memory of a frozen base". It changes the deployed base
  and has to differentiate from EngramEdit and User-as-Engram. It is out of scope here and noted as the
  only Engram-branded route with published support.

## Sources

- Predecessor: [engram-graft-assessment.md](engram-graft-assessment.md). Repo: RESULTS.md §2, §4, §7,
  §8, §10; key-encoding-for-editing-memory.md §0, §3.7–3.29, §6g; frozen-base-titans-scorecard.md;
  CAM_100_CONTINUATION.md; CAM_100_DISJOINT_RESULT.md; docs/serving/{integration_design, online_api,
  parity_spike}.md; ROADMAP.md; DIARY.md Ph.16, Ph.18; novelty-positioning.md.
- Engram [2601.07372](https://arxiv.org/abs/2601.07372) · EngramEdit [2610.10533](https://arxiv.org/abs/2610.10533) ·
  STAPLE [2610.12599](https://arxiv.org/abs/2610.12599) · LongCat-Flash-Lite [2601.21204](https://arxiv.org/abs/2601.21204) ·
  Memory Grafting [2605.20948](https://arxiv.org/abs/2605.20948) · User as Engram [2606.19172](https://arxiv.org/abs/2606.19172) ·
  Engram Adapter [2608.29327](https://arxiv.org/abs/2608.29327) · Cross-Model Memory Transfer [2608.17050](https://arxiv.org/abs/2608.17050) ·
  NGM [2605.16893](https://arxiv.org/abs/2605.16893) · Tokenizer-Agnostic Engram [2607.29065](https://arxiv.org/abs/2607.29065) ·
  NeMo RFC [#4292](https://github.com/NVIDIA-NeMo/Automodel/issues/4292)
- NeuralDB [2507.18028](https://arxiv.org/abs/2507.18028) · SWEA [2401.17809](https://arxiv.org/abs/2401.17809) ·
  REMEDI [2304.00740](https://arxiv.org/abs/2304.00740) · ROME [2202.05262](https://arxiv.org/abs/2202.05262) ·
  SUIT [2509.24502](https://arxiv.org/abs/2509.24502) · GRACE [2211.11031](https://arxiv.org/abs/2211.11031) ·
  WISE [2405.14768](https://arxiv.org/abs/2405.14768) · MEMOIR [2506.07899](https://arxiv.org/abs/2506.07899) ·
  SERAC [2206.06520](https://arxiv.org/abs/2206.06520) · Larimar [2403.11901](https://arxiv.org/abs/2403.11901) ·
  KBLaM [2410.10450](https://arxiv.org/abs/2410.10450) · LTE [2402.11905](https://arxiv.org/abs/2402.11905) ·
  ATBias [2406.12468](https://arxiv.org/abs/2406.12468) · Picozzi [2609.11063](https://arxiv.org/abs/2609.11063)
- Evaluation and robustness: CounterFact+ [2305.17553](https://arxiv.org/abs/2305.17553) · Mirage
  [2502.11177](https://arxiv.org/abs/2502.11177) · long-form editing [2402.09394](https://arxiv.org/abs/2402.09394) ·
  RippleEdits [2307.12976](https://arxiv.org/abs/2307.12976) · UniEdit [2505.12345](https://arxiv.org/abs/2505.12345) ·
  MQuAKE [2305.14795](https://arxiv.org/abs/2305.14795) · The Tokens Remember [2609.29045](https://arxiv.org/abs/2609.29045) ·
  BadEngram [2609.13478](https://arxiv.org/abs/2609.13478)
- Decoding prior art: REST [2311.08252](https://arxiv.org/abs/2311.08252) · Hokamp & Liu
  [1704.07138](https://arxiv.org/abs/1704.07138) · Dinu et al. [1906.01105](https://arxiv.org/abs/1906.01105) ·
  kNN-LM [1911.00172](https://arxiv.org/abs/1911.00172) · Memory Decoder [2508.09874](https://arxiv.org/abs/2508.09874)

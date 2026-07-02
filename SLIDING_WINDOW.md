# Test-time binding — the sliding-window plan

> **Status: design document.** Nothing in this file has been run. This is the plan for the north star
> (ROADMAP `v0.4 — reuse & test-time`, Tracks 3–4), written down *before* the experiments so the
> predictions are on record. Numbers will land in [RESULTS.md](RESULTS.md) as they exist, with the
> usual rule: every number gets a chance baseline and an ablation, and wrong turns stay in the record.

## The claim to be tested

A frozen base with the memory organ attached, reading a stream longer than its context window, should
behave as if the window never slid: **as content falls off the top of the prompt, it has already been
captured by the memory**, and the base can still answer questions about it through the tap.

Stated as a measurement: fix a context window of *W* tokens over a longer fact-bearing stream. Every
fact is written into the memory at the moment it streams past. Probe recall of each fact as a function
of *how far past the window edge* it has fallen:

- **in-window** (fact still in the prompt): the ceiling — the base can just attend to it.
- **out-of-window, memory on**: *the measurement*. The claim is this stays ≈ ceiling.
- **out-of-window, memory off**: must collapse to chance — otherwise the probe leaks.
- **never shown**: the floor control (`no_memory` = 0 by construction, as everywhere in this repo).

The gap between the mem-on and mem-off out-of-window curves, at a chance-pinned floor, is the entire
result. If the mem-on curve sags toward the floor as facts age past the edge, the plot shows exactly
where and how fast the organ loses what the window dropped.

## Where the mechanism already is (closer than the roadmap reads)

The roadmap files describe binding as "Stage-1 training," which makes test-time binding sound far away.
The code says otherwise, and the distinction matters for this plan:

**The write path is already a test-time operation.** `ProductKeyStore.write()`
([pk_store.py](cam/pk_store.py)) is a forward-pass, error-correcting delta update into an episodic
value bank — no gradients, no optimizer. What Stage-1 *trains* is the **addressing geometry**: the
in-projection, the write/read projections (`to_wkey`/`to_wval`, read heads), and the product-key
codebooks. The facts themselves are written into a **fresh bank every episode**, forward-pass, using
those frozen learned projections.

The proof this generalizes is already in RESULTS.md without having been framed that way: **every
held-out carry number is a test-time bind.** Held-out eval docs contain *novel* random bindings the
projections never saw in training; they are written into an empty bank in one forward pass and read
back at 0.92–0.95. Test-time binding of unseen facts, *within one episode*, is not aspirational — it
is what "carry" has measured all along.

| component | trained (once, offline) | test-time (per episode, forward-pass) |
|---|---|---|
| addressing geometry (codebooks, projections, heads) | ✅ | frozen |
| the facts (value bank `V`) | — | ✅ written via delta-update |
| MAG tap / translator | ✅ (Stage 2/3) | frozen |
| base model(s) | never | never |

**Why the translator-reuse wall does not block this.** The "reuse — answered NO" result (a translator
fit on one memory scores 0.000 on another) was across different *trained stores* — different codebook
and projection geometries. The sliding-window setting keeps **one** trained store throughout and varies
only the bank *contents* — and both the tap and the translator were trained across thousands of
episodes with *different random bank contents* each time. Content-generalization is already
demonstrated; geometry-generalization is what failed, and nothing here changes the geometry.

## What is genuinely missing

Four things, in increasing order of research risk:

1. **Persistence.** The value bank is reset per episode (`init_state` → zeros). A conversation needs
   one bank that *carries across turns*. Mechanically trivial (don't reset); behaviorally unknown.
2. **Accumulation.** Today at most M=128 facts are ever written into the N = n_sub² = 1024 slots
   (defaults), all at once. A persistent bank accumulates writes over a long horizon — interference
   over *time* is unmeasured. This is the roadmap's N-scaling question wearing a different shirt.
3. **Extraction.** Today the builder hands the store exact key/value token positions (oracle
   segmentation — even the varied-phrasing path gets `binding_positions()`). A live stream is
   unsegmented prose; *something* must decide what the keys and values are. This is the genuinely open
   research problem, and it should be attacked **last**, after 1–2 are de-risked, so a failure is
   unambiguously extraction's.
4. **Read-query provenance.** The pooled bank handed to the tap is conditioned on the QA query tokens
   (`memory_bank(..., qa_start, answer_pos)`) — the harness knows where the question is. Live, the
   read query must come from the current window (e.g. the last tokens of context, or the base's own
   residual stream). A design question for the harness, not the mechanism.

## The experiment ladder

Each rung has its falsifier stated up front. Each is cheap relative to what it de-risks, and each
failure mode is separated from the others by construction.

### E0 — persistence probe (no window, no conversation)

**Question:** does the bank survive *not being reset*? Write M facts per episode across E consecutive
episodes into **one persistent bank** (M·E facts total, written at different "ages"), then probe every
fact. Plot recall vs **age** (episodes since written) and vs **total accumulated writes**.

- Controls: fresh-bank-per-episode (today's setting — the ceiling); empty bank (floor); and the same
  M·E facts written **all at once** into one bank (separates "too many facts" from "facts written over
  time" — if batch-write holds but staged-write decays, the interference is temporal, not capacity).
- Sweep M·E across the slot count (fill fractions ~0.1 → ~2× of N=1024) so the capacity edge is mapped,
  not stumbled into.
- **Falsifier:** old-fact recall collapses as new writes land. Then the delta-write's slot-sharing is
  the wall, measured *before* any conversation machinery exists — and the fix conversation (bigger
  N, write-time slot protection, decay-aware addressing) happens with a clean curve in hand.

This rung needs no new mechanism — only a flag to stop resetting `V` between episodes and an eval loop
that tags each fact with its write-age. It doubles as the first N-scaling-through-time measurement.

### E1 — sliding-window recall (oracle segmentation)

**Question:** the headline claim, with extraction assumed away. Build conversation-shaped streams from
the existing natural/varied-phrasing builders (§6 of RESULTS.md) — a long document of fact-bearing
prose, far longer than the window *W*. Slide the window; every fact is delta-written the moment it
streams past (oracle key/value positions, as today). Probe each fact under the four conditions in
"The claim" above, and plot **recall vs tokens-past-edge**.

- Success: mem-on out-of-window ≈ in-window ceiling; mem-off out-of-window ≈ chance; never-shown = 0.
- **The edit variant** (this is where it gets fun): restate a fact mid-stream with a *new* value —
  "the meeting moved to 3pm." The delta-write is error-correcting (`v_s += β·w·(new − v_s)`), so the
  overwrite should happen *by construction*, and §7 already shows override works statically
  (France→Tokyo at 0.996 with the prior suppressed to 0.004). Probe: latest value recalled, stale
  value suppressed — the knowledge-editing result, live.
- **Falsifier:** if E0 held but E1 sags, the loss is in the window machinery or the read-query path,
  not the store — the ladder localizes it.
- Stage-2 note: the tap was trained on pooled reads from fresh M-fact banks. If delivery
  (not carry) is what sags, retrain the tap over persistent-bank episodes — the tap is a few small
  matrices and the retrain is cheap. Carry-vs-delivery separates store failure from tap
  distribution-shift, same as it always has.

### E2 — extraction (drop the oracle)

**Question:** can the writes happen without being told where the facts are? Two candidates, in order:

- **(a) A trained extractor head**: a small module over the frozen base-1 embeddings (or an early
  residual layer) that emits (key, value) span pairs from unsegmented prose; supervised first on the
  synthetic builders where gold spans are known, then tested on held-out phrasings. Keeps the
  "no base weights trained" invariant.
- **(b) Surprise-gated writing** (Titans-proper): skip explicit spans — derive write keys/values from
  the residual stream continuously and let a surprise signal (the delta-write's `(value − v_s)` norm is
  *already* a surprise measure) gate what commits. Riskier, more general, and only worth attempting
  with E0/E1 curves to compare against.

**Falsifier for (a):** extraction accuracy high on held-out synthetic but recall drops — then the
spans are right and the *representation* of extracted facts is wrong. **For (b):** the bank fills with
noise and E0's interference wall arrives early. Either failure is informative only because E0/E1
established the oracle-segmentation ceiling first.

### E3 — the live harness (the dogfood)

Wrap frozen base + tap + persistent bank in an actual chat loop and use it while developing this repo —
the conversation itself becomes the stream, the organ carries what scrolls off. This is Stage-3
library territory (`attach_memory(frozen_model, memory)`) and it is *not* an experiment — no controls,
no chance floor — so nothing measured here goes in RESULTS.md except as an anecdote pointing back at
E0–E2 numbers. It is, however, the point.

## Reporting rules

Same as everywhere in this repo: chance baseline on every number; `no_memory` pinned at ≈0 or the run
is contaminated; in-context ceiling reported next to every delivery number; 3 seeds on anything that
becomes a headline; wrong numbers stay in the record with the correction beside them.

## What would kill it, ranked

1. **E0 interference collapse** — most likely, cheapest to check, checked first. If a persistent bank
   can't hold facts across writes, everything downstream is moot until the store changes.
2. **Tap distribution-shift on accumulated banks** — plausible, cheap to fix (retrain Stage-2 on
   persistent-bank episodes), and cleanly diagnosed by the carry-vs-delivery split.
3. **Extraction** — the real open problem. E0/E1 exist so that when extraction is attempted, its
   failures are its own.

The order of the ladder is the order of the risks: measure the wall before building the machine that
would hit it.

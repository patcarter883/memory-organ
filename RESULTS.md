# Results

Every number here is held-out and comes with its chance baseline. Where we were wrong, the wrong number
stays in with the correction next to it — see also [DIARY.md](DIARY.md) for the narrative and
[DISCLOSURES.md](DISCLOSURES.md) for the caveats (synthetic task, ROCm-developed).

**Reading the metrics.** *M* = discrimination difficulty (pick the right item out of *M*; chance = 1/M).
*carry* = held-out recall directly from the memory store (Stage 1). *delivery / memory acc* = recall
after the memory is injected into a **frozen** base via the MAG tap (Stage 2). *no_memory* = the same
base with the tap off — if this isn't ≈0, the base could already guess and the result is contaminated;
ours is pinned at 0.000 throughout. *ceiling* = the base answering with the facts in-context. *transfer*
= recall on a **second frozen base** reached only through a tiny affine translator. For multi-token
answers, accuracy is teacher-forced **exact-match** (all tokens correct); per-token in parentheses.

Recipe unless noted: seed `20260625`, bind 6000 steps, tap/translator 3000 steps, batch 16, lr 1e-3,
donor base Qwen3.5-4B.

---

## 1. Capacity — how hard a task the store can hold (Stage-1 carry)

| store | M=8 (chance .125) | M=16 (.062) | M=32 (.031) | M=64 (.016) | M=128 (.008) |
|---|---|---|---|---|---|
| naive recurrent (BoltAdapter) | 0.840 | **0.025** collapse | **0.020** collapse | — cannot bind | — |
| uncompressed KV (control, upper bound) | 1.000 | 1.000 | 1.000 | — | — |
| product-key, **no addr-sup** (our port bug) | 0.360 | 0.262 | 0.020 collapse | — | — |
| **product-key + addr-sup (store of record)** | **0.948** | **0.926** | **0.894** | **0.921** | **0.929** |

Flat through M=128 (116× chance), ablated floor 0.000 everywhere. The only ceiling hit at M=128 was the
probe's single-token vocabulary (~221 cargo tokens), not the store.

## 2. The capacity wall, and the two times we were wrong

**Wrong turn #1 — "ceiling at M≈3" (it was under-training).** M=8 carry: 0.115 at 3000 steps (≈ chance)
→ **0.840 at 6000**. Binding has a phase transition that moves right with M; "M=3" was just too few steps.

**The wall past M≈8–12 is real, and it's the compression** — three falsifiers on the naive store: 2×
steps → 0.025; 4× read-capacity → 0.021; an **uncompressed KV control → 1.000**. The wall is the
compression mechanism, not the task or embedding.

**Wrong turn #2 — "the product-key store only buys one rung."** Our first port looked like it moved the
wall one rung and lost M=8 fidelity (0.360). Cause: the port had silently dropped the store's
**addressing-supervision loss**. Restoring it gave the flat ladder in §1. The "one rung" was an artifact.

## 3. Single-token pipeline — bind → deliver → transfer (with error bars)

Single-token answers, M=8, **3 seeds** (mean ± range), `no_memory` = 0.000 everywhere:

| stage | mean | range | chance |
|---|---|---|---|
| Stage-1 carry | **0.950** | ±0.003 | 0.125 |
| Stage-2 delivery (frozen Qwen) | **0.944** | ±0.009 | 0.125 |
| transfer → frozen Gemma-3-4b-pt | **0.942** | ±0.020 | 0.125 |

Near-deterministic — the headline is not a lucky seed. And at higher difficulty (M=64, chance 0.016) the
same pipeline holds: carry 0.919, delivery 0.938, Gemma transfer 0.922 — a difficulty the naive store
cannot even bind.

**Cross-family transfer (the decisive translator test).** The same frozen Qwen memory, transferred to
**Llama-3.2-3B** — a genuinely foreign model (tiktoken vocab, bos 128000, plain Llama arch, d=3072 vs
2560), through a 15.7M affine translator:

| base-2 | transfer | chance | ceiling |
|---|---|---|---|
| Gemma-3-4b-pt (SentencePiece, d=2560) | 0.924 | 0.125 | 0.984 |
| **Llama-3.2-3B (tiktoken, d=3072)** | **0.656** | 0.125 | 0.852 |

Llama is a harder target (0.656 ≈ 77% of its own 0.852 in-context ceiling) but clearly passes with a
0.000 no-memory floor — so the translator is **not** exploiting Qwen-family embedding similarity.

## 4. Multi-token answers — the store side solved; transfer still open

A single-token answer is unrealistic. With **2-token real-word answers** the picture is now
**mostly** — but not fully — closed. We found the failure, hunted it across four experiments, and fixed
the hard part.

**The hunt (Stage-1 carry, M=8, 2 tokens — the store's own readout):**

| approach | exact-match | per-token |
|---|---|---|
| linear readout | 0.000 | 0.46 |
| autoregressive decoder head | 0.198 | 0.53 |
| per-position, additive position code | 0.000 | 0.47 |
| per-position, shared codebook | 0.429 | 0.65 |
| **per-position, DISJOINT sub-codebooks** | **0.964** | **0.98** |

Each result named the next bottleneck. The wall turned out to be *per-position address resolution* —
when the K answer-slots share a name key, the addressing can't separate them. Giving each answer position
its **own** product-key codebook bank collapsed the per-position addressing loss (0.75 → 0.05) and lifted
Stage-1 exact-match to **0.964 — single-token parity**, flat in K (0.953 at K=3).

**End-to-end 2×2 (K=2), vs single-token and vs the old pre-fix approach:**

| stage | multi-token (disjoint store, best translator) | single-token | old multi-token |
|---|---|---|---|
| Stage-1 carry | **0.964** (0.98) | ~0.95 | 0.000 (0.46) |
| Stage-2 delivery (Qwen) | **0.883** (0.94) | ~0.94 | 0.486 (0.72) |
| transfer → Gemma | **0.812** (0.90) | ~0.94 | 0.432 (0.68) |

When cross-base transfer first ran, it was stuck at **0.393** with the standard affine translator — the
collapse was entirely at deliver→transfer, so the **translator**, not the store, was the bottleneck (an
affine map carries a single token across bases at 0.96 but not a 2-token *sequence*). The fix mirrored the
store fix — go *per-position* — and stacked with non-linearity:

| translator | transfer exact | per-token |
|---|---|---|
| affine (baseline) | 0.393 | 0.66 |
| non-linear MLP (shared) | 0.691 | 0.84 |
| per-position affine | 0.727 | 0.85 |
| **per-position + MLP** | **0.812** | **0.90** |

**Honest verdict: multi-token is largely closed end-to-end.** The store side is solved (addressing 0.964,
same-base delivery 0.883); cross-base transfer, once translator capacity is added, reaches **0.812** (6.5×
chance, ~84% of the in-context ceiling, no_memory 0.000). That's a strong pass — but **not** single-token
parity (~0.94), so we call it *largely closed*, not *solved*.

**Correction (we were wrong, then right, and won't overstate it).** Mid-session we called this "solved
end-to-end" before running the transfer number — it wasn't (0.393). We corrected that, then actually
closed most of the gap with a per-position translator (0.812). We are deliberately *not* upgrading "0.812"
to "solved": it's a strong pass short of parity, and the record shows both the over-claim and the fix.

## 5. Real knowledge — natural-language phrasing

The dict format (`"<cargo>: <name>"`) is terse and unlike real text. The first real-knowledge cut asks
whether the mechanism survives when the same associations are phrased as **prose**: single-relation facts
`"<Subject> lives in <Object>."`, query `"<Subject> lives in"` → answer ` <Object>`. Subject = KEY,
object = VALUE, both single real-word tokens — coherent English sentences, not the terse dictionary.
Enable with `--phrasing natural`.

M=8, single-token, `no_memory` = 0.000 at bind, delivery, and transfer:

| stage | natural | dict | chance | ceiling |
|---|---|---|---|---|
| Stage-1 carry | **0.903** | 0.929 | 0.125 | — |
| Stage-2 delivery (frozen Qwen) | **0.918** | — | 0.125 | 0.994 |
| transfer → frozen Gemma | **0.645** | — | 0.125 | 0.865 |

**Verdict: the mechanism is phrasing-invariant.** It holds on facts phrased as prose, not just the terse
dict — natural carry 0.903 sits just under dict's 0.929, delivery 0.918 reaches 92% of its 0.994 ceiling,
and transfer 0.645 is 75% of its 0.865 ceiling. `no_memory` stays pinned at 0.000 at bind, delivery, and
transfer, so this is genuine memory, not the base guessing from the sentence template.

### Heterogeneous facts — varied relations

Prose with ONE fixed relation is still M repetitions of a single template. `--phrasing varied` mixes fact
**structures** within a document: each fact independently draws from a small set of relation templates —
` lives in` / ` works as` / ` was born in` / ` owns a` / ` studies` — so one document contains diverse fact
shapes, not one repeated skeleton. The relations have different token lengths, so binding blocks are no
longer constant-length; per-slot key/value positions (subject = KEY, object = VALUE) are handed to the
store via `binding_positions()`, keeping the deterministic addressing exact.

M=8, single-token objects, `no_memory` = 0.000 everywhere:

| stage | varied | (natural, for ref.) | chance |
|---|---|---|---|
| Stage-1 carry | **0.938** | 0.903 | 0.125 |
| Stage-2 delivery (frozen Qwen) | **0.932** | 0.918 | 0.125 |
| transfer → frozen Gemma | **0.631** | 0.645 | 0.125 |

Mixing five relation shapes in one document does not degrade the mechanism — carry 0.938 and delivery 0.932
are on par with (slightly above) the single-relation natural numbers, and transfer 0.631 holds.

### Multi-token natural — K-token objects in prose

Real facts are often multi-token (`"lives in New York"`). `--phrasing natural --cargo-tokens K` extends the
prose path so the OBJECT is a K-token real-word phrase (`"<Subject> lives in <w0 w1>."`); the subject stays
the single-token KEY, the K-token object phrase is the VALUE (answer). The store needed no change — only
lifting the "single-token only" guards — because each object word is verified single-token and
space-prefixed, so a K-word phrase is deterministically exactly K tokens and the constant-length binding
contract still holds. K=1 is byte-identical to single-token natural.

M=8, `no_memory` = 0.000 everywhere:

| stage | multi-token natural | per-token | chance |
|---|---|---|---|
| Stage-1 carry | **0.973** | 0.986 | — |
| Stage-2 delivery (frozen Qwen) | **0.928** | — | — |
| transfer → frozen Gemma | **0.824** | 0.908 | — |

The full-phrase carry (all K tokens exact) is 0.973 (per-token 0.986); delivery 0.928; and transfer 0.824
**exceeds its own in-context ceiling of 0.607** (per-token 0.908) — the memory delivers the multi-token
answer through the transfer base *better* than that base can reproduce the same facts when they are placed
directly in its context window.

**Verdict: the mechanism holds across prose, varied relations, AND multi-token answers.** Phrasing-invariance
is not fragile to one template — it survives five mixed relation shapes and K-token phrase answers, with
`no_memory` pinned at 0.000 throughout, so every result is genuine memory rather than the base guessing from
sentence structure. **Caveat:** these are still *random bindings* — subjects and objects paired at random per
document, so the base cannot know any specific fact. Editing real, named entities and counterfactual facts
(where the base has a prior) is the open capstone, tracked in
[#1](https://github.com/patcarter883/memory-organ/issues/1).

## 6. Knowledge editing — overriding a frozen model's real knowledge

Every result up to here is knowledge **insertion**: the bindings are random, so the frozen base cannot know
any specific fact, `no_memory` pins at 0.000, and the memory only has to teach an association the base could
never have. The harder claim is knowledge **editing** — take a fact the frozen base *already knows*
parametrically (a real country → capital), put a **counterfactual** capital in the memory, and ask whether
the memory makes the base emit the *wrong* capital, overriding its own prior. Enable with
`--phrasing counterfactual`.

The trap here is validity: you can only claim an *override* if the base actually held the prior you are
overriding. So `recall_mag` runs a **PROBE → FILTER → EDIT** pipeline in one run: first probe the frozen
base (memory off, tap off) on 40 curated country→capital facts (both sides single-token under the Qwen3.5-4B
tokenizer), **keep only the facts it answers correctly**, then derange the kept capitals (Sattolo
single-cycle — no country keeps its true capital) and bind those counterfactual capitals on the filtered
set. Four metrics are scored at the same query position (`"The capital of <Country> is"`): mem-on and
no_mem accuracy against the **counterfactual** capital, and against the **true prior** capital. A
VALID/INVALID gate fires on no_mem prior-acc — if the base didn't hold the priors, the run is INVALID by
construction.

**Same-base result (M=8, frozen Qwen3.5-4B) — the valid, strong result:**

| metric | value | reading |
|---|---|---|
| probe prior-acc (all 40 facts) | **0.975** | base holds the priors → 39 facts kept |
| no_mem PRIOR-acc (kept set) | **1.000** | validity gate maxed — the base reliably knows these |
| no_mem counterfactual-acc | **0.000** | floor — the base never volunteers the deranged capital on its own |
| mem-on counterfactual-acc | **0.996** | the edit takes: base emits the deranged capital |
| mem-on PRIOR-acc | **0.004** | the true prior is suppressed |

**GATE: VALID.** With the base demonstrably holding the priors (no_mem prior-acc 1.000), the memory flips
mem-on output to the counterfactual capital (0.996) and drives the true prior to 0.004. This is genuine
knowledge **editing** — the memory overrides the frozen base's own parametric knowledge (France → Paris
becomes France → Tokyo), not just injection of a fact the base could not know.

**Cross-base transfer — the override is now VALID and WORKS on base-2 (frozen Gemma):**

| metric | value | reading |
|---|---|---|
| base-2 probe prior-acc (all shared facts) | **1.000** | Gemma holds the priors → 39/39 facts kept |
| no_mem PRIOR-acc (kept set) | **1.000** | validity gate maxed on base-2 — Gemma reliably knows these |
| no_mem counterfactual-acc | **0.000** | floor — Gemma never volunteers the deranged capital on its own |
| mem-on counterfactual-acc | **0.996** | the edit transfers: Gemma emits the deranged capital |
| mem-on PRIOR-acc | **0.004** | Gemma's true prior suppressed |

**GATE: VALID.** Running the saved base-1 memory through the translator onto frozen Gemma delivers the
counterfactual (mem-on cf-acc 0.996), and — with Gemma demonstrably holding the priors (base-2 prior-acc
1.000, 39/39 facts) — drives Gemma's own true prior to 0.004. This is genuine cross-**family** knowledge
editing: one frozen memory makes a *different* frozen model overwrite its own parametric knowledge.

The earlier 0.000 no_mem prior-acc on base-2 was **our own artifact, not a limitation** — a **BOS-stripping
bug** in the leak-free eval context. That context dropped the leading `<bos>` token, and base-2 models like
Gemma are highly BOS-sensitive, so Gemma's parametric recall collapsed to 0.000 even though the same base,
probed *with* its `<bos>` (exactly as it was trained), recalls the priors at 1.000. The validity gate
correctly flagged it as INVALID — the control did its job. The fix has two parts: **(1)** restore the BOS in
the leak-free context so the eval format matches the base's eliciting format, and **(2)** add a base-2
**probe → filter** pass (mirroring the base-1 filter) that keeps only facts Gemma demonstrably knows *in
Gemma's own vocab and format*, so the override is measured on an honestly-established prior set.

So knowledge editing works **both same-base (Qwen) AND cross-family (Gemma)**: the memory drives a
*different* frozen model to overwrite its own knowledge. (Part of [#1](https://github.com/patcarter883/memory-organ/issues/1).)

## 7. Track 1 — real CounterFact editing: **INVALID → VALID** (the validity gate caught a real bug)

§6 edits a **curated** 40-fact country→capital table hand-picked to be single-token and well-known.
Track 1 ([#16](https://github.com/patcarter883/memory-organ/issues/16)) is the honest scale test: the
same PROBE → FILTER → EDIT pipeline against the **real ROME CounterFact benchmark** (21,919 records) with
the same validity gate *plus* two new metrics the curated table can't measure — **locality** (is the edit
surgical, or does it damage neighbouring facts?) and **generalization** (does the edit fire on paraphrases
of the prompt, or only the exact wording?). Config: product-key store + addr-sup, M=8, tap layer 24, bind
1500 / tap 200, frozen Qwen3.5-4B. `--dataset counterfact`.

**First run — the gate said INVALID, and it was right.** Of 21,919 records, 783 have a single-token
subject; probing each with its own prompt the base holds 0.130 → 102 kept. The edit *looked* like it took
(mem-on cf-acc 0.961) — but the no_mem PRIOR-acc (validity gate) came out **0.164 ≪ 0.60 → INVALID**.
Cause: the eval's doc-builder hard-coded **"The capital of &lt;X&gt; is"** for *every* fact, so non-capital
relations (official language, mother tongue, twin city, …) were tested under a nonsense capital prompt the
base can't answer — a filter/eval prompt-format mismatch, the *same* discipline that caught the Gemma BOS
artifact in §7. The 0.961 "override" was meaningless: you can't override a prior the base doesn't hold *in
the format you test it*.

**The fix — edit one relation, elicit under its true prompt.** Facts sharing a CounterFact relation share
the exact prompt template, so the pipeline now groups the base-known facts by relation, edits the largest
group, and folds *that relation's* real prompt into the doc header/query — exactly as the code already
folded "The capital of". Filter and eval then elicit the identical, true relation; the subject stays the
single-token KEY, so positions and addressing are unchanged.

**Result — VALID, and the edit works** (relation P37 "The official language of &lt;X&gt; is", 27 base-known
facts; e.g. Italy: Italian→Korean, Monaco: French→Ukrainian):

| metric | before (capital template) | after (per-relation) | reading |
|---|---|---|---|
| no_mem PRIOR-acc (**VALIDITY gate**) | 0.164 → **INVALID** | **0.969 → VALID** | base demonstrably holds the priors |
| mem-on counterfactual-acc | 0.961 | **1.000** | the edit takes: memory delivers the counterfactual |
| no_mem counterfactual-acc | 0.000 | 0.000 | floor — base never emits the wrong answer on its own |
| mem-on PRIOR-acc | 0.002 | **0.000** | delivery fully suppresses the true prior |

Bind carry 0.985, tap L=24 memory **1.000** / no_memory **0.000** / ceiling 0.594. **GATE: VALID + EDIT
WORKS** — the base holds the priors (0.969) *and* the memory overrides them to the counterfactual (1.000 ≫
0.000). This is genuine, *valid* knowledge editing on the real CounterFact benchmark.

**But it is not yet surgical, and barely generalizes** (mem OFF = tap off, mem ON = tap on):

| metric | mem OFF | mem ON | verdict |
|---|---|---|---|
| **Locality** — neighbour prior-acc (gold = *unedited* fact; 256 probes) | 0.242 | **0.098** | **LEAKY** — editing collaterally damages neighbours (−0.145) |
| **Generalization** — paraphrase acc (gold = *new* fact; 54 probes) | 0.000 | **0.074** | fires on rephrasings, but **weakly** |

**Honest verdict.** On real CounterFact, editing a single relation is now **valid and delivered perfectly
(1.000)** — the first-attempt INVALID was a real eval bug the gate correctly refused to certify, not a
failure of the memory. What remains open is what the curated table could never test: the edit **leaks** to
neighbouring facts (−0.145) and only **weakly generalizes** to paraphrases (0.074), and this is one
relation at a time (multi-relation editing needs per-relation doc configs).

### Making it surgical — retrieval-conditioned banking + strength-gated training (local *and* generalizing)

The leak, and a matching **generalization measurement bug**, trace to one cause: the eval built each
probe's memory bank from a **random** edited subject, not the probe's own. So a paraphrase of "Italy"
attended over a bank read for some *other* subject — it couldn't retrieve its own edit, and generalization
looked near-dead (0.074). And locality was tested against a strong, wrong-edit bank. The store here is
**episodic** (each read writes the doc's bindings then queries), so the fix is to condition each probe's
bank on **its own subject**, exactly as deployment would (you query the memory with the subject):

- **Generalization** — bind+query the paraphrase's subject (a **strong** read: the store returns *its*
  edit). Generalization was never dead: it's **0.889** edit-only, not 0.074. The 0.074 was an artifact.
- **Locality** — query the neighbour's out-of-store subject (a **weak** read: the store returns nothing).

Then the tap must gate on **retrieval strength**, not prompt novelty. A **null / sink slot** (a learnable
attention key with a *zero* value) gives it the capacity to inject nothing; training it with **weak-bank
negatives** (the *same* edited-subject prompt, but the edit **not** bound) gives it the signal — positive
and negative differ *only* in whether the store holds the edit. So the tap learns *deliver-on-strong,
null-on-weak*, which delivers paraphrases (strong read) yet stays inert on neighbours (weak read).

Clean control, identical bind/seed, scored on the same 135 held-out neighbours (a tunable knob):

| `locality-weight` | edit-success (mem-on cf) | **locality drop** (mem OFF→ON) | **generalization** |
|---|---|---|---|
| 0.0 (edit-only) | 0.967 (VALID) | 0.193 → 0.067 = **−0.126** (LEAKY) | 0.889 |
| **0.1** | **1.000 (VALID)** | 0.193 → 0.170 = **−0.023** (LOCAL) | **0.667** |
| 0.3 | 0.992 (VALID) | 0.193 → 0.185 = **−0.008** (LOCAL) | 0.556 |

All four editing desiderata at once on real CounterFact: **valid** (gate 0.96), **delivered** (edit 1.000),
**local** (−0.008 to −0.023), **and generalizing** (0.56–0.67). Retrieval-strength gating roughly **triples**
the generalization of the earlier prompt-novelty gating (which gave 0.167 at the same locality, because it
keyed on the prompt and so nulled paraphrases too). The honest residual: locality still costs some
generalization (0.667 vs 0.889 edit-only) — the learned sink is not perfectly retrieval-selective.

### Closing the gap — an explicit store-confidence gate (`--conf-gate`)

The learned null slot gates on **prompt novelty**: a paraphrase of the edited subject *looks* as unfamiliar
as a neighbour, so it gets suppressed alongside — that's the 0.667 ceiling. The fix is to gate on the thing
that actually distinguishes them: **retrieval strength**. The product-key store's read RMSNorms magnitude
away (the bypass fix), but the *pre-norm* factual-head retrieval magnitude `‖ctx‖` is a clean per-example
signal — **large** when the query addresses written slots (subject bound → strong read) and **~0** when it
addresses unwritten slots (neighbour → weak read). We surface it (`pk_store.read(return_conf=True)`) and
scale the whole tap injection by `c = σ(scale·(conf/EMA − bias))` (`GatedMemoryTap`, learned `scale`/`bias`,
EMA for an *absolute* scale — strong/weak arrive in separate forward passes, so per-batch norm would erase
the distinction). A paraphrase retrieves its own edit (strong → c≈1 → deliver); a neighbour retrieves
nothing (weak → c≈0 → inert) — delivery is decoupled from novelty.

Clean control vs the null-only sink, identical bind/seed, same 135 held-out neighbours:

| gate | `lw` | edit-success | **locality drop** (OFF→ON) | **generalization** |
|---|---|---|---|---|
| null slot (baseline) | 0.1 | 0.998 | 0.193 → 0.230 (LOCAL) | 0.611 |
| **store-confidence** | 0.1 | 0.990 | 0.193 → 0.185 = **−0.008** (LOCAL) | **0.907** |
| **store-confidence** | 0.3 | 0.996 | 0.193 → 0.215 (LOCAL) | **0.926** |

The confidence gate lifts generalization **0.61 → 0.91–0.93** with locality intact and edit-success ≈1.0 —
it closes the gap the learned sink could not, matching (and edging past) the 0.889 edit-only ceiling *while
staying surgical*. The `neg_cgate` diagnostic confirms the mechanism: the gate scales the weak-bank read
down (c→0) while paraphrases still deliver.

**Verdict: surgical single-relation editing on real CounterFact — valid (0.96), delivered (≈1.0), LOCAL
(−0.008), and generalizing (0.91–0.93).**

### Multiple relations in one memory (`--phrasing counterfactual_multi`)

Everything above edits **one relation at a time**. The next step is editing facts from **different
relations** in a *single* memory — each fact keeping its **real** CounterFact prompt ("The capital of X
is", "The official language of X is"). This is the *faithful-prefix* format: unlike single-relation (the
relation prefix folded into a shared header, subject at `qa_start`), here the subject sits **mid-block**
after each fact's own prefix, so the KEY (subject) / VALUE (object) positions **vary per binding**. We read
them via `binding_positions()` (the per-binding machinery the `varied` phrasing introduced), keep the batch
rectangular by cycling relations across doc slots (slot m → relation `m % R`), and teach addressing
supervision to find the queried subject at `qa_start + q_subj_off` rather than a fixed offset. A
tokenizer-only selftest (`tools/cf_multi_selftest.py`) pins the positions before any GPU spend.

The remaining subtlety was the confidence gate. It standardizes retrieval strength `‖ctx‖` through a running
EMA, but different relations have **different `‖ctx‖` scales**, so a single global EMA can't separate
strong-from-weak across all of them — the first multi-relation run leaked (locality −0.066, "GENERALIZES but
LEAKY") and, unlike single-relation, turning up `lw` made it *worse*, not better. The fix is a **per-relation
EMA**: the builder tags each doc with its queried relation index, and the tap keeps one running scale per
relation (`conf_ema[R]`), so every relation's threshold adapts to its own magnitude (shared sigmoid
scale/bias). A CPU unit test confirms the mechanism — a relation whose typical `‖ctx‖≈0.5` delivers at
conf 10 (c≈1.0) while a relation scaled to `‖ctx‖≈10` stays inert at conf 0.5 (c≈0.02).

Result (Qwen3.5-4B, **4 relation-templates edited together**, 42 edits, conf-gate, lw 0.1):

| metric | single global EMA | **per-relation EMA** | |
|---|---|---|---|
| validity gate (no_mem prior-acc) | 0.990 | **0.990** | VALID — base holds all four relations' priors |
| edit-success (mem-on cf) | 0.928 | **0.922** | memory overrides the prior across mixed relations |
| generalization (paraphrase) | 0.679 | **0.738** | edits fire on rephrasings across relations (0.000 off) |
| locality drop (OFF→ON) | −0.066 (LEAKY) | **−0.047 (LOCAL)** | verdict flips to GENERALIZES + LOCAL |

**Multi-relation editing works — valid (0.99), delivered (0.92), LOCAL (−0.047), and generalizing (0.74)
across four relation templates in one memory.** Per-relation gate calibration closed the leak (−0.066 →
−0.047, past the "local" threshold) *and* lifted generalization (0.679 → 0.738). The residual −0.047 is
still looser than single-relation's −0.008 — a fair cost for mixing relations. That first multi-relation run
looked base-limited: the editable set was dominated by a couple of relations (mostly official/native
language). We chased that — and it turned out **not** to be a base-size limit.

### What actually caps relation diversity — multi-token subjects, not base size

The obvious hypothesis was "the base doesn't know enough" — so we loaded a **2.25× bigger base** (Qwen3.5-9B,
model-parallel across two cards). It made **no difference**: prior-acc 0.139 vs the 4B's 0.130, still
P37/P36-dominated. The 9B *falsified* the size hypothesis.

The real cap was the **single-token-subject filter**. CounterFact records are kept only if the subject *and*
both objects are single tokens — but subjects are mostly multi-token names ("Danielle Darrieux"), so that
filter keeps just **789 of 21,919 records (3.6%)**, skewed to the few relations with single-token subjects.
Objects, by contrast, are single-token in **97%** of records. So: keep single-token objects, allow
**multi-token subjects**. The store keys on the subject's **last token** (one position — the write path is
unchanged); the read pools the whole query region, which carries the full subject; batches stay rectangular
by fixing one subject length per relation; addressing supervision identifies the queried binding by **index**
(robust to two subjects sharing a last token).

This unlocks the diversity the single-token filter was hiding — **6 semantically distinct relations** at once
(vs the earlier all-P37), from **1065** base-known facts (of 8000 probed) instead of 102:

| relation | example edit |
|---|---|
| P364 original-language-of-a-film | "Fort Apache Napoli": Italian → English |
| P103 native language | "Raymond Triboulet": French → Dutch |
| P37 official language | (as before) |
| P159 headquarters location · P276 location · P140 religion | mixed multi-token-subject facts |

Result (Qwen3.5-4B, native `gdn_hip`, 6 distinct relations, multi-token subjects, ~50 edits, conf-gate lw 0.1):

| metric | value | |
|---|---|---|
| validity gate (no_mem prior-acc) | **1.000** | VALID across all six relations |
| edit-success (mem-on cf) | **0.90** | override works across mixed relations + multi-token subjects (0.89–0.96 across runs) |
| locality drop (OFF→ON) | **−0.016** | LOCAL |
| generalization (paraphrase) | **0.654** | edits fire on rephrasings (0.000 off) |

**So the diversity ceiling was a *tractability* artifact, not a base-knowledge or mechanism limit.** With
multi-token subjects, editing spans genuinely different relations — valid, delivered, local, and
generalizing. Two caveats specific to the 16GB RDNA4 card: this run **requires** `CAM_NATIVE_GDN=1` — fla's
GDN backward *segfaults* on RDNA4 in stage-2 (the native path is now ~as fast as fla-torch thanks to the
batched prefill-train, see the engine repo) — and the locality/generalization *eval* is fragmentation-bound
(the longer multi-token probes tip a 16GB card over; those two numbers are from the run that fit, the gate +
edit reproduce every run).

## 8. Still open

- **Multi-token cross-base transfer** — translator-bound (see §4); higher-capacity translator in progress.
- **Real knowledge in real documents** (not random name→word pairs) — the true generalization test. §5 now
  covers prose (single relation), **varied relations** (five mixed templates per doc), and **multi-token
  natural objects** (K-token phrase answers), all with `no_memory` = 0.000. Counterfactual editing — where
  the base has a prior — is demonstrated same-base AND cross-family on a *curated* table (§6).
- **Real-benchmark editing — VALID, LOCAL, and GENERALIZING (§7).** Track 1 on real ROME CounterFact:
  after fixing a filter/eval prompt mismatch the validity gate caught (0.164 → **0.969, VALID**), editing
  one relation delivers the counterfactual **perfectly (1.000)**. Retrieval-conditioned banking (each
  probe's bank is read for its OWN subject, as deployment does) shows generalization was never dead
  (**0.889** edit-only, not 0.074 — a measurement artifact), and a null-slot tap trained with
  retrieval-strength (weak-bank) negatives makes it **local AND generalizing at once**. An explicit
  **store-confidence gate** (`--conf-gate`, scaling delivery by the store's pre-norm retrieval magnitude
  rather than the null slot's prompt-novelty proxy) then closes the last gap: locality drop **−0.008** with
  generalization **0.91–0.93** (up from 0.61 null-only), edit-success ≈1.0. **Multiple relations at once
  (§7) works too** — 4 relation-templates edited in one memory (faithful prefix), and a per-relation
  confidence-gate EMA makes it VALID (0.99), delivered (0.92), LOCAL (−0.047), and generalizing (0.74)
  ([#16](https://github.com/patcarter883/memory-organ/issues/16)).
- **N-scaling** the store toward useful sizes (thousands of facts, not 8–128 per doc).
- **Backend portability** — pure PyTorch, CPU/CUDA expected but unverified.

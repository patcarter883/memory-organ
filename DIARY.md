# Development diary

A running, honest log of what we tried, what we learned, and how it turned out — including the parts
that were wrong. Newest entries at the bottom. This is the narrative companion to `RESULTS.md` (the
numbers) and `DISCLOSURES.md` (the caveats).

---

## Entry 0 — Origin

The project started in conversation, not on a whiteboard. (An earlier draft of this entry said "at
dinner." That was autocorrect mangling "at some point" — Claude, being a language model, does not eat.
We left the correction visible because it's the kind of thing this diary is honest about.)

Pat had fed Claude a batch of Gemini-generated enthusiasm, and the two of us were sorting the real from
the hype — Gemini has a fond habit of boarding the hype train. One of the ideas on the table was that
you could just **strip layers out of several different models and sandwich them together** into a new
base model. Claude was explaining why that doesn't work: a model's **hidden states are not a shared
interface** — each model's residual stream is its own private coordinate system, so layer *N* of model B
can't read layer *N* of model A's activations and make sense of them.

That explanation had wandered into the two places you can tap a Titans memory *into* a model —
**Memory-as-Context (MAC)**, where the memory is supplied as extra context tokens, versus
**Memory-as-Gate (MAG)**, where it's injected straight into the residual stream through a small gate —
i.e. *which injection point* you use. Somewhere in outlining those injection points Claude said the words
"hidden states," and Pat queued a thought before Claude had finished:

> *"Hang on — I remember a project that was doing something with translating between hidden states."*

He offered it lightly — *maybe this helps, maybe it points us in the right direction* — not expecting a
question about it to reorient the whole project. But it did. The project was
[RecursiveMAS](ACKNOWLEDGMENTS.md), whose **RecursiveLink** learns a small map carrying one model's
hidden states into another's space. And that closed the loop: the same fact that makes layer-sandwiching
impossible — hidden states don't transfer between models — means you can't drop a memory built on one
model into another either. **You'd need a translator. And someone had already shown you could learn one.**

So the thesis assembled itself out of an argument about hype: a Titans-style memory (MAG delivery,
because we'd just decided it was the cleaner injection point), made base-agnostic by a RecursiveLink-style
translator. The argument *against* layer-sandwiching and the argument *for* a translatable memory turned
out to be the same argument, approached from opposite ends. Gemini's over-excitement earns a real credit
here, too: the hype is what keeps Pat probing whether the wild ideas are actually achievable — and this
time one was.

---

## Phase 1 — Does a memory even deliver through a frozen base?

**Goal.** Before anything ambitious, falsify the basic claim: can an external memory module change what
a *frozen* base model says, without touching the base's weights?

**What we did.** Built the two-stage setup. Stage 1: bind facts into a compact DeepMemory module by a
direct loss (no base in the loop). Stage 2: freeze the memory, freeze the base, and train *only* a
zero-initialized gated tap (MAG) that injects the memory's read into one of the base's layers.

**What we learned.** The Memory-as-Context framing hit a wall — with the memory supplied as context,
`memory ≈ no_memory`: the base ignored it. The Memory-as-Gate tap, by contrast, *delivered*: held-out
recall jumped well above the no-memory baseline while the base stayed frozen.

**Outcome.** MAG works; MAC doesn't, for this. The zero-init gate was the unlock — it lets the tap start
as a no-op and learn its way in without destabilizing the frozen base.

---

## Phase 2 — One memory, a second base, a tiny translator

**Goal.** The actual thesis: does *one frozen memory* serve a *different* frozen base through only a
small learned translator?

**What we did.** Took the frozen Stage-1 memory and a second frozen base (different hidden dim, even a
different model family), and fit a tiny affine translator that stitches base-2's residual stream into
the frozen tap and back — training only the translator.

**Outcome.** It passed: `memory >> no_memory` on the second base. The same memory, delivered to a model
it was never built on, through a map small enough to be almost embarrassing. We started calling it a
"Modular Memory Organ."

---

## Phase 3 — A detour we cut: the canonical hub

**Goal.** A grander idea: build a model-agnostic "canonical" representation hub (a committee of models
voting into a shared space) and hang the memory off *that*, so it'd be base-agnostic by construction.

**Outcome — falsified, and dropped.** When we actually measured it, the hub was not load-bearing: a
translator into the canonical hub was no better than a translator straight into the donor base. We had
been about to build a lot of machinery on top of it. We didn't. (We also ran a donor bake-off around
here — does the choice of model the memory is born from matter? — and the answer was "keep the
incumbent, Qwen3.5-4B; a broken donor sinks recall but a good one doesn't lift it." Donor is a weak,
one-directional lever.) The lesson that kept repeating: measure the load-bearing assumption *first*.

---

## Phase 4 — The capacity investigation (and being wrong, twice)

This is the long middle, and the most honest part.

**The question.** How hard a recall task can the memory actually serve? We parameterize difficulty by
*M*: pick the right item out of *M* candidates (chance = 1/M).

**Wrong turn #1 — "the ceiling is M≈3."** Early on the memory only bound cleanly at M=3 and we wrote it
down as an architectural ceiling. It wasn't. It was **under-training** — binding shows a sharp phase
transition that moves *right* as M grows, so harder tasks just need more steps. With the right budget,
M=8 bound cleanly (held-out carry 0.84, native recall 0.99). We corrected the record and made M=8 the
honest "real difficulty" anchor.

**The wall is real, though.** Past M≈8–12 the naive recurrent memory genuinely collapsed — and we
nailed down that it was *not* under-training (12k steps, double the M=8 budget: still chance) and *not*
read-capacity (4× the slots/heads: still chance). Three falsifiers pointed at the **compression
mechanism itself**.

**The control that proved it.** We built a trivial *uncompressed* lookup store as an upper bound. It
aced M=8/16/32 at carry 1.000 — the same task the compressing store dies on. So the wall is the
compression, not the task or the embedding geometry. That cleanly told us a *better store*, not a
different task, was the answer.

**Wrong turn #2 — "the product-key store only buys one rung."** We ported a product-key (sparse,
addressable) store — the principled fix — and it looked like it moved the wall just one rung (cleared
M=16, died at M=32) while *losing* fidelity at M=8. Disappointing. Then we found why: **our port had
silently dropped the store's addressing-supervision loss.** That was our bug, not the store's ceiling.

**The fix.** Restored the dropped loss (two InfoNCE terms aligning read-query to write-address and
retrieved-context to stored-value — and, crucially, reconstructable *without* the dead canonical hub).
The result flipped completely: flat held-out carry **0.948 / 0.926 / 0.894 / 0.921 / 0.929** across
M=8/16/32/64/128. No cliff. The only ceiling we hit was the *probe's vocabulary*, not the store.

**Outcome.** The capacity wall is broken in the tested range. The product-key store *with its proper
training signal* is the store of record. Both reversals were caught by controls run *before* we built on
the wrong conclusion — which is the entire argument for working this way.

---

## Phase 5 — End-to-end: both pillars, one store, hard difficulty

**Goal.** Capacity and transfer had been measured on *different* stores. Tie them together: run the
winning store through bind → deliver → transfer, at M=8 (head-to-head with the old store) and M=64 (a
difficulty the old store cannot even bind).

**Outcome — the full 2×2 is green.** Delivery 0.955 (M=8) / 0.938 (M=64); transfer to a frozen
cross-family Gemma 0.965 (M=8) / 0.922 (M=64), `no_memory` pinned at 0.000 throughout. The headline
surprise: at M=8, same Gemma, same affine translator, the *old* store transferred at 0.396 and this one
at **0.965** — near the in-context ceiling. The addressing-supervision fix didn't just add capacity, it
made the memory dramatically more *transferable*. The differentiated pillar — base-agnosticism — improved
the most.

---

## Phase 6 — The realism frontier: an edge, a four-experiment hunt, and an honest stop

**We found an edge.** The whole result so far used *single-token* answers. We extended the probe to
2-token real-word phrases and the headline tempered: end-to-end exact-match fell from ~0.95 to ~0.45.
*Which* part broke was informative — addressing stayed lossless (`no_memory` 0.000), but the store
returned about one of two answer tokens. The "what's the full answer" half was failing.

**Then we hunted it, and each failure named the next bottleneck:**

| attempt | Stage-1 exact | what it ruled out |
|---|---|---|
| linear readout | 0.000 | (baseline) |
| autoregressive decoder head | 0.198 | not readout expressivity — the value doesn't hold token 2 |
| per-position, additive position code | 0.000 | not per-position storage — the address won't separate |
| per-position, shared codebook | 0.429 | the address is the lever, and it *half*-resolves |
| **per-position, disjoint sub-codebooks** | **0.964** | give each position its own codebook → it resolves |

The wall was **per-position address resolution**: when the K answer-slots share a name key, the
product-key addressing can't tell them apart. Give each answer position its *own* codebook bank and the
per-position addressing loss collapses (0.75 → 0.05), Stage-1 exact-match hits **0.964 — single-token
parity** — and it's flat in K (0.953 at K=3). Same-base delivery follows (0.486 → 0.883). The
**store side of multi-token is solved.**

**But we didn't get to call it "solved end-to-end" — and we said so.** We almost did. The store held and
locally-delivered the 2-token answer; the encouraging thing was to stop there. Instead we ran the
*cross-base transfer* number, and it was **0.393** — basically unchanged. The affine translator that
carries a single token into a foreign base at 0.965 could not carry a 2-token *sequence*. The bottleneck
had moved from the store to the translator, and we corrected the premature claim.

**Then the fix rhymed with the store fix.** The store failed because K answer-slots shared a key — fixed
by giving each position its own codebook. The translator was failing for the same shape of reason — one
shared map squeezing a sequence — so we gave *it* the same medicine: a **per-position translator** (a
separate cross-base map per answer position), and stacked non-linearity on top. Transfer climbed
0.393 → 0.691 (MLP) → 0.727 (per-position) → **0.812 (per-position + MLP)**. The same idea — *go
per-position* — closed both ends of the pipeline. We stopped at "largely closed," not "solved": 0.812 is a
strong pass (~84% of ceiling, clean 0.000 floor) but short of single-token parity (~0.94), and after
over-claiming once we weren't going to do it twice.

**Two things did fully land in this phase.** A genuinely **cross-family** transfer — the same Qwen memory
into **Llama-3.2-3B** (foreign tokenizer, architecture, and width) at 0.656, proving the translator isn't
riding Qwen-family geometry. And **3-seed error bars** on the single-token pipeline (±0.003–0.020), so
it's no longer a single-seed claim.

**Still open.** Multi-token *cross-base transfer* (translator capacity); real knowledge in real documents
(the big one); N-scaling.

Somewhere in here we also pulled the work out of the research monorepo into this standalone, public repo —
which is why you're reading this.

## Phase 7 — Real knowledge (first cut)

**Does the mechanism care how the facts are written?** Everything to here bound the terse dictionary
format `"<cargo>: <name>"` — nothing like real text. So we added a `--phrasing natural` mode that states
the same associations as prose: single-relation facts `"<Subject> lives in <Object>."`, queried with
`"<Subject> lives in"` and answered with ` <Object>` (subject = key, object = value, both single real-word
tokens). Plumbing it through was mostly bookkeeping — declared KEY/VALUE offsets on the builder so the
store adapter locates the association by where it actually sits in a prose sentence rather than assuming
the dict layout, and persisting the phrasing in the checkpoint so the transfer stage rebuilds the same
doc format the memory was bound on. The offsets keep the dict path byte-identical.

**It held.** M=8, single-token: natural carry **0.903** (dict 0.929), delivery **0.918** (92% of its 0.994
ceiling), transfer → Gemma **0.645** (75% of its 0.865 ceiling). `no_memory` stayed pinned at 0.000 at
bind, delivery, *and* transfer, so it's genuine memory, not the base guessing from the sentence template —
the (subject → object) pairing is still random per doc, so the realism is the phrasing, not the vocabulary.
The mechanism is phrasing-invariant: it works on prose, not just the dictionary.

**Honest scope.** This is a *first* cut — one fixed relation (`lives in`) with single-token objects. Varied
relations, facts drawn from a real dataset, and multi-token natural-language objects are all still open.
Tracked in issue #1.

## Phase 8 — Heterogeneous facts, and multi-token answers in prose

Phase 7 left two of its own caveats open: prose was still **one** repeated relation, and answers were still
single tokens. Phase 8 closes both — separately, so each is a clean isolate.

**Varied relations (`--phrasing varied`).** One fixed relation means the document is M copies of the same
skeleton; a real page mixes fact *shapes*. So each fact now draws independently from a small template set —
` lives in` / ` works as` / ` was born in` / ` owns a` / ` studies` — assigned deterministically per binding
slot (`slot m → relations[m % R]`) so the token positions stay batch-uniform and identical across
tokenizers. The relations have **different token lengths**, which broke the one assumption the store leaned
on: constant-length binding blocks. The fix was to stop computing positions by `m * bind_len` arithmetic and
instead have the builder hand the store the exact per-binding key/value positions (`binding_positions()`) —
subject = KEY, object = VALUE, wherever they land in a variable-length sentence. Everything else was
untouched. **It held:** M=8, carry **0.938**, delivery **0.932**, transfer **0.631** — on par with (slightly
above) single-relation natural. Mixing five fact shapes in one document does not confuse the store.

**Multi-token answers in prose (`--phrasing natural --cargo-tokens K`).** Real facts are multi-token ("lives
in New York"). We already had a multi-token *dict* path; the question was whether prose could carry a K-token
object phrase (`"<Subject> lives in <w0 w1>."`) with the subject still the single-token key. The store needed
**no** change — only lifting the "single-token only" guards — because every object word is verified
single-token and space-prefixed, so a K-word phrase is deterministically exactly K tokens and the
constant-length contract survives; K=1 stays byte-identical to single-token natural. **The strongest result
of the phase:** M=8, full-phrase carry **0.973** (per-token 0.986), delivery **0.928**, and transfer
**0.824** (per-token 0.908) — which *exceeds its own in-context ceiling of 0.607*. The memory delivers the
multi-token answer through the transfer base **better than that base reproduces the same facts when they sit
directly in its context**. `no_memory` = 0.000 throughout both experiments.

**Where this leaves the realism frontier.** Phrasing-invariance is not a fragile property of one template —
it survives five mixed relation shapes and K-token phrase answers. But the honest caveat from Phase 7 still
stands, and it is now the *only* one: these are **random bindings**. Subjects and objects are paired at
random per document, so the base has no prior to override — the mechanism is proven to *store and retrieve*,
not yet to *edit* a fact the model already believes. Real, named entities and counterfactual editing (where
the base has a genuine prior) is the open capstone. Tracked in issue #1.

## Phase 9 — Knowledge editing: overriding a fact the model already believes

This is the capstone Phase 8 pointed at, and getting a *valid* result out of it took three attempts — the
first two are worth recording honestly because they show how the validity control had to be earned.

The idea is simple: instead of random bindings the model can't know, use real country→capital facts the
frozen base **already knows**, put a **counterfactual** (deranged) capital in the memory, and test whether
the memory makes the base emit the wrong capital — overriding its own prior. The catch is that "override" is
only meaningful if the base actually held the prior. The **first two attempts were invalid**: they measured
a counterfactual-acc that looked like an edit, but the base did *not* reliably hold the priors it was being
tested on (no_mem prior-acc came out ~0.107 in one tangled run), so the "override" was measuring nothing —
you can't override a belief that isn't there. The two attempts were also tangled: the fact set, the probe,
and the eval weren't a single controlled pass, so it was impossible to say what the numbers meant.

The clean re-run replaced all of that with **one orchestrator-driven PROBE → FILTER → EDIT pass** inside
`recall_mag` when `--phrasing counterfactual`: probe the frozen base first (memory off), **keep only the
facts it demonstrably answers correctly**, then derange the kept capitals (Sattolo single-cycle, no fixed
point) and bind the counterfactuals on that filtered set. Now no_mem prior-acc is high *by construction* —
the validity gate is a real control, not a hope. **Same-base result (M=8, frozen Qwen3.5-4B):** the probe
found the base holds 39/40 priors (prior-acc 0.975), the kept-set no_mem PRIOR-acc maxed at **1.000**, mem-on
counterfactual-acc hit **0.996**, and mem-on PRIOR-acc dropped to **0.004** — GATE VALID. The memory
overrides the frozen base's own parametric knowledge (France→Paris becomes France→Tokyo). That is genuine
knowledge *editing*, not insertion.

**And the honest limit — then the resolution.** Cross-base transfer to Gemma *delivered* the edit (mem-on
cf-acc 0.996), but the same validity gate that certified base-1 first **failed on base-2**: Gemma's no_mem
prior-acc came out **0.000**. We wrote that up as an honest INVALID caveat and tracked it under #1 — and that
was the right call, because chasing it down, the gate turned out to be flagging **our own artifact, not a
limitation of the memory**. The 0.000 was a **BOS-stripping bug** in the leak-free eval context: that
context sliced off the leading `<bos>` token, and base-2 models like Gemma are highly BOS-sensitive, so
Gemma's parametric recall collapsed to zero — even though the *same* Gemma, probed *with* its `<bos>` exactly
as it was trained, recalls those same priors at **1.000**. A context-format artifact masquerading as a
knowledge gap. This is the nice part of the story: the validity control we built in Phase 9 *caught our own
bug* — it refused to certify a run where the base couldn't recall its priors, which is precisely what a good
control should do.

The fix has two parts: **(1)** restore the BOS in the leak-free context so the eval format matches the base's
eliciting format, and **(2)** add a base-2 **probe → filter** pass (mirroring base-1's) that keeps only facts
Gemma demonstrably knows in Gemma's own vocab and format. With both in place: base-2 probe prior-acc
**1.000** (39/39 facts kept), no_mem PRIOR-acc **1.000**, mem-on counterfactual-acc **0.996**, mem-on
PRIOR-acc **0.004** — **GATE VALID**. So knowledge editing works **both same-base (Qwen) AND cross-family
(Gemma)**: one frozen memory drives a *different* frozen model to overwrite its own parametric knowledge. The
invalid→valid arc, and the fact that our own validity gate caught the artifact, is the more honest telling.
Part of #1.

## Phase 10 — Track 1: the real CounterFact benchmark bites back

Phase 9 edited a **curated** 40-fact country→capital table — hand-picked to be single-token, well-known,
and (we'd later realise) unusually friendly. Track 1 ([#16](https://github.com/patcarter883/memory-organ/issues/16))
is the honest scale-up: run the exact same PROBE → FILTER → EDIT pipeline against the **real ROME
CounterFact benchmark** (21,919 records), and add the two metrics the curated table structurally *cannot*
measure — **locality** (does editing one fact damage its neighbours?) and **generalization** (does the edit
fire on paraphrases, or only the exact prompt?). This is the test that turns "a mechanism" into "a mechanism
that survives real data," or finds where it breaks. It found where it breaks.

First, an engineering aside that's worth recording because it nearly masqueraded as a result: the eval
kept OOMing on a 16 GB card, and the first fix capped the *wrong* function. The real culprit was the
ceiling forward materialising the full `[batch, tokens, vocab=151936]` logits tensor in fp32 (~1 GB) on
top of a 13 GB training-tail — right at the end of a good run, so it looked like the mechanism had a
memory bug when it was just the eval harness. Fix: run the LM head on only the last `Kc` answer positions
(`logits_to_keep`), collapsing `[B,T,V]→[B,Kc,V]`. Numerically identical, ~200× smaller, and the run
completed clean. Mentioning it because "the eval crashed" is exactly the kind of thing that gets silently
worked around; here it's in the record.

Now the result, and it is a genuinely mixed one. **The delivery mechanism scales**: mem-on
counterfactual-acc **0.961** against a no_memory floor of **0.000** (and the boltA/MAC reference pinned at
memory ≈ no_memory ≈ 0.000), so the tap still *delivers* a bound counterfactual on real, varied
CounterFact facts. That's the encouraging half — and it's the *easy* half. Three things the 40-fact table
had hidden all surfaced at once:

1. **The run invalidates itself.** no_mem PRIOR-acc came out **0.164**, far below the 0.60 validity
   threshold. The filter kept 102 facts the base demonstrably knew *under the probe prompt* (`"<Subject>
   is"`), but under the doc-builder **eval** phrasing (a seg-len-48 leak-free context) the base recalls
   only 16% of those same priors. This is the *same* validity discipline that caught the Gemma BOS artifact
   in Phase 9 — a prompt-format mismatch between filter and eval — so the impressive-looking 0.961 override
   is **not a valid edit-success claim** until both elicit the prior identically. The gate did its job
   again: it refused to certify an override of a prior the base doesn't demonstrably hold *in the format we
   test it*.
2. **The edit leaks.** Locality: neighbour prior-acc drops from 0.270 (tap off) to **0.070** (tap on) over
   256 probes — a −0.199 collateral hit. The edit is not surgical; delivering one counterfactual perturbs
   the base on facts it should leave untouched.
3. **It barely generalizes.** Paraphrase acc for the *new* fact rises from 0.000 to only **0.103** over 204
   probes — the edit is largely tied to the exact prompt wording.

The honest headline: **curated editing was a best case; real-benchmark editing is delivered-but-not-yet-
valid.** This is not a failure of the delivery mechanism (which scaled fine) but of everything *around* it
that a curated table let us skip — establishing the prior in the same format we evaluate, keeping the edit
local, and making it robust to rephrasing. The next steps are concrete rather than hand-wavy: match the
filter's eliciting prompt to the eval context so the validity gate can pass honestly, then attack the
locality leak directly. Recorded here undramatised because the point of Track 1 was to find exactly this,
and it did.

## Phase 11 — Track 1 made VALID: the gate caught our eval bug

Phase 10 ended on an honest sour note: real CounterFact editing *delivered* (mem-on 0.961) but the
validity gate said INVALID (no_mem prior-acc 0.164 ≪ 0.60), so the "override" was meaningless. We wrote
that up plainly and tracked the fix under #16. Chasing it down, the gate — again — turned out to be
flagging **our own bug, not a limit of the memory**.

The counterfactual eval builds its eliciting prompt from a doc-builder that had the capital case baked in:
a fixed header **"The capital of"** + query **"&lt;subject&gt; is"**. That is exactly right for the curated
country→capital table of §7. But Track 1 feeds it *real* CounterFact relations — official language, mother
tongue, twin city, plays-instrument, located-in — and every one of them was being tested as
"The capital of &lt;subject&gt; is". The base can't answer "the capital of Italy" with a *language*, so its
prior recall collapsed and the gate correctly refused to certify the run. A filter/eval prompt-format
mismatch, the same species of bug as the Gemma BOS artifact in Phase 9 — and the same control caught it.

The fix is almost embarrassingly clean once you see it: facts that share a CounterFact **relation** share
the *exact* prompt template, so we don't need per-fact prompts or a doc rewrite — we just **edit one
relation at a time** and fold *that relation's* real prompt into the header/query, precisely as the code
already folded "The capital of". `setup_counterfact` now groups the base-known facts by relation, picks
the largest editable group, splits its prompt at the subject slot, and hands the pieces to the builder.
The subject stays the single-token KEY at its old position, so the store's addressing and addr-sup are
byte-unchanged. Filter and eval finally elicit the *same, true* relation.

Result (relation P37, "The official language of {} is", 27 base-known facts, e.g. Italy: Italian→Korean,
Monaco: French→Ukrainian): no_mem prior-acc **0.164 → 0.969** — **GATE VALID**. mem-on counterfactual-acc
**1.000**, mem-on prior-acc **0.000**: the base demonstrably holds the priors AND the memory overrides them
to the counterfactual. Bind carry 0.985, tap delivery 1.000. This is genuine, *valid* knowledge editing on
the real ROME CounterFact benchmark — not a curated table.

We are deliberately not over-selling it. Two things the curated table could never have shown are now on the
record and are *not* solved: the edit **leaks** — bind one relation and neighbouring facts' prior recall
drops 0.242 → 0.098 (−0.145 collateral) — and it only **weakly generalizes**, firing on paraphrases at
0.074. And it is one relation at a time. So Track 1's honest status flips from "delivered-but-not-yet-valid"
to "valid, but not yet surgical, paraphrase-robust, or multi-relation." That's real progress and a clear,
specific open front. The nicest part of the story is unchanged from Phase 9: the validity control we built
to keep ourselves honest did its job twice now — it caught a bug we'd otherwise have shipped as a result.

## Phase 12 — making the edit surgical: the locality leak, and a tradeoff we won't hide

Phase 11 left Track 1 valid but LEAKY: editing one relation dropped neighbouring facts' prior recall
(collateral damage). The cause is structural and, once seen, obvious. The delivery tap is a gated
cross-attention whose `softmax` must sum to 1 — so for *every* query, including an out-of-store neighbour,
it attends to *something* in the memory bank and injects it, scaled by a gate that training opened to
deliver edits. The tap literally has no way to say "this query isn't mine; inject nothing."

Two changes give it that option. A **null / sink slot**: a learnable extra attention key with a *zero*
value, so the residual can attend to "nothing" and the injection collapses to ~0 — the *capacity* to be
inert. And a **locality-preservation loss**: on held-out neighbour prompts, run the base with the edit
bank loaded and match the tap-on answer distribution to the frozen base's tap-off distribution (a KL) —
the *signal* that teaches the tap to route non-matching queries to the sink. Neighbours are split 50/50 so
the metric never sees a prompt the loss trained on.

It works, cleanly, and the mechanism is visible in the logs: the fraction of attention mass on the null
slot climbs from ~0.06 to ~0.83 over training. On a clean control (same bind, same seed, same 135
held-out neighbours), the locality drop goes from **−0.089** (edit-only) to **−0.008** at
locality-weight 0.3 — ~90% of the leak gone — while edit-success stays pinned at **1.000** and the gate
stays VALID. A modest weight is enough; you do not have to crank it.

And here is the part we are not going to bury: it **costs generalization**. Paraphrase firing drops from
0.074 (already weak) to ~0.02. This is the locality↔generalization tension that the knowledge-editing
literature has documented for years, and we reproduced it from first principles — because the null slot
keys on *prompt novelty*, and a paraphrase of the edited fact is, to the tap, just another unfamiliar
prompt, so it gets routed to the sink exactly like a neighbour. The fix that would get *both* is to gate
the sink on **store-retrieval confidence** — fire the injection only when the store actually returns the
queried subject (paraphrases share the subject; neighbours don't) — rather than on prompt familiarity.
That's the open next step. For now: surgical single-relation editing on real CounterFact is valid,
delivered at 1.000, and LOCAL, with paraphrase-robustness the honest remaining gap.

## Phase 13 — retrieval-conditioned banking: local AND generalizing, and a generalization we'd been measuring wrong

Phase 12 made the edit local but at a sharp cost: paraphrase generalization collapsed. We wrote that up
as the honest locality↔generalization tension. It turned out the tension was mostly an artifact of how we
*measured* — a third measurement bug the controls led us to, after the validity gate (Phase 11) and the
capital-template mismatch.

The tell was that generalization was weak (0.074) even with the locality loss *off*. Pulling on that: the
memory store here is **episodic** — every eval read writes the current doc's bindings into a fresh store,
then queries. But the locality/generalization eval built each probe's bank from a **random** edited
subject. So a paraphrase of "the official language of Italy" was being answered from a memory that had been
queried for some *other* country entirely. Of course it couldn't deliver Italy's edit — we never asked the
store about Italy. Condition the bank on the probe's **own** subject (bind+query it), exactly as a
deployment would (you query the memory with the subject in the prompt), and generalization is **0.889**, not
0.074. It was never dead; we were reading the wrong drawer.

That reframes the whole locality/generalization problem as one of **retrieval strength**. A paraphrase of an
edited fact produces a *strong* store read (the subject is in the store); a neighbour produces a *weak* one
(it isn't). The tap should deliver on strong and stay inert on weak. Phase 12's null slot gave it the
capacity; the fix is the training *signal*: instead of teaching "be inert on neighbour prompts" (which keys
on prompt novelty and so nukes paraphrases too), teach "be inert on a weak read." We do that with
**weak-bank negatives** — the *same* edited-subject query as a positive, but with the edit deliberately
*not* bound in the doc, so the store returns nothing. Positive and negative now differ only in whether the
store holds the edit, so the tap has no choice but to learn strength-gating.

It works, and it's a knob (same bind/seed, 135 held-out neighbours): edit-only leaks (−0.126) but
generalizes (0.889); locality-weight 0.1 gives −0.023 / 0.667; 0.3 gives −0.008 / 0.556. So single-relation
editing on real CounterFact is now **valid, delivered (1.000), local (−0.008), and generalizing (0.56–0.67)
all at once** — every editing desideratum satisfied at a meaningful level, on real data. Against Phase 12's
prompt-novelty gating (0.167 generalization at the same locality), retrieval-strength gating roughly triples
it. The residual is honest and specific: locality still costs some generalization (0.67 vs 0.89), because a
*learned* sink isn't a perfect retrieval detector. The clean way to close it is to stop learning a sink and
instead gate the injection on an explicit **store-confidence scalar** — fire in proportion to how strongly
the store matched. That's the next step. Three measurement bugs in three phases, each surfaced by a control
we'd built to keep ourselves honest; the pattern is the point.

# How close does a frozen base get to Titans — a four-axis scorecard

*A research note. Every number is held-out with its reference baseline; where an earlier reading was
wrong, the wrong number stays in with the correction next to it. Companion to [RESULTS.md](../../RESULTS.md)
and [DIARY.md](../../DIARY.md).*

## The question

The north star of this project is *Titans for everyone*: durable memory as a **bolt-on to a frozen
LLM**, not a retrain. This note asks the sharp version of that: **how close can a frozen base get to
Titans' actual headline capabilities without training the model** — and where it can't, *why*, and
whether what it *can* do is functionally useful.

We measure against Titans/HOPE's own four headline axes: multi-fact reasoning (BABILong), long-context
recall (NIAH), knowledge integration (our edit-ripple task), and continual/no-forget (HOPE's CTNL).

## The mechanism first: why bolt-on *injection* cannot integrate an edit into reasoning

Before the scorecard, the load-bearing negative. A multi-hop question ("who is the head of state of the
country where X holds citizenship?") is answered in two internal hops: the base resolves a **bridge
entity** (the country) as an internal representation, then a later hop reads it. We tested whether an
injected/edited fact can ride that chain. It cannot, for three reasons we could each measure:

- **F1 — off-position.** The bridge is an internal residual resolved early and *consumed late* (Yang,
  ACL 2024; Biran "Hopping Too Late", EMNLP 2024). A tap at a mid layer holds only the first-hop fact.
- **F2 — off-circuit (the wall).** Frozen attention routes on the base's *original* representation; an
  injected value is read out locally but never *re-composed*. Measured: KV-append gives **0.000**
  multi-hop ripple, seeding the GDN recurrent state **0.067** — both deliver single-hop, neither
  propagates. "Attendable ≠ integrated." (Matches CaKE's "stored ≠ integrated", 2025.)
- **F3 — readout shortcut.** Training the tap on a 2-hop objective doesn't install a belief — it learns
  the shortest path (bias the output for that composition), which **does not transfer**: trained on one
  downstream relation it ripples 0.85, on a *different* held-out relation it collapses to 0.31.

We then swept the space exhaustively so nothing was left on the table. **Every activation-write variant
— additive, on-manifold (renorm), error-correcting (delta), suppress-and-add (two-sided) — at every
depth (early L3 garbles, mid ~0.2–0.3, late ~0.45) sits below the in-context (RAG) baseline of ~0.5.**
The write algebra is not the lever; the depth is not the lever. The ceiling is **perturb-vs-recompute**:
only *re-running the base's own forward pass over the fact* (in-context, or a distilled prefix that
mimics it) makes the base compose it. This is consistent with the whole editing literature — both
single-shot weight edits and activation edits ripple poorly; what ripples is putting the fact where the
frozen, jointly-trained circuit already routes.

## The method that clears the wall: a distilled cartridge

Following Cartridges (Stanford Hazy, 2506.06266), we stop injecting and instead **distill the frozen
base's own in-context behavior into a small trained KV-cache prefix**: teacher = base with the fact in
its window, student = base with the cartridge (no fact in the prompt), KL-matched over next-token
distributions on self-generated Q&A. Because in-context reasoning *does* compose, the cartridge inherits
it. Single-hop delivery recovers to **0.84** cleanly, and — the key mechanistic result — it ripples
multi-hop **whether or not the self-study is multi-hop**: once the belief is installed, *the frozen base
does the composition itself*, exactly as it does for in-context text. The curriculum is not the lever;
the delivery mechanism (distillation vs injection) is.

## The scorecard (Qwen3-4B-Base, frozen throughout; cloud RunPod; ≈$2 total)

| Axis | frozen cartridge | in-context (RAG/ICL) | verdict |
|---|---|---|---|
| **Edit-ripple** (integration) | ≈ RAG on hops routed through the bridge; **below** RAG on hops with a competing direct path (e.g. language→script) | RAG = ceiling | matches in-context where reasoning must route through the edit |
| **NIAH** (capacity) | 0.87 @1k, 0.73 @4k | ICL 1.00 | ≈ in-context (near-parity; deep-depth dips are digit-off convergence misses) |
| **BABILong** (multi-fact reasoning) | 0k 0.60 ≈ RAG; **1k 0.25, 4k 0.10** | RAG 0.60 / 0.45 | **below** in-context on distractor-heavy context |
| **Continual / no-forget** | routing: locality **1.00**, acc 0.60 @N=10 | (no persistent memory) | **structural win** |

Three findings define the shape:

1. **The frozen base reaches in-context quality and never exceeds it** (edit-ripple ≈ RAG, NIAH ≈ ICL).
   Titans' flagship result is that it *beats* RAG on BABILong — above the in-context ceiling — and we do
   not reach that. Exceeding in-context requires training the model; on a frozen base it is a wall.

2. **BABILong is *below* in-context, and the obvious fix failed instructively.** The cartridge distills
   the whole context including irrelevant book-filler and regurgitates it (at 0k, no filler, it is at
   parity — so it is a distractor-selectivity gap, not a reasoning deficit). We tried Titans' **surprise**
   signal to filter the filler. It did **not** help — and *slightly hurt* — because **surprisal ≠
   relevance**: the bAbI facts are formulaic (low surprise), the filler is rich prose (high surprise), so
   surprise-weighting up-weights the wrong thing. Distractor-filtering needs a *query-time relevance*
   signal a frozen offline-distilled prefix structurally lacks; the fix that would work re-derives RAG.

3. **Continual/no-forget is a genuine structural win — the one axis where the frozen approach *beats* a
   monolithic model.** Naively composing many cartridges into one always-on prefix interferes badly
   (accuracy collapses past N≈2–3; unrelated general-knowledge drops 0.75→0.17). *(An earlier tiny-budget
   run made us hope this was a no-forget win outright; at full budget it is clearly interference — we were
   wrong, and the correction stands.)* But **routing** — a bank of isolated per-fact cartridges + a cheap
   non-parametric retriever that activates only the relevant one — flips it: **general knowledge preserved
   at 1.00** (vs 0.17 for concat), and **zero cross-fact interference by construction**. A single trained
   model (Titans/HOPE, shared weights) *cannot* have this property. The residual accuracy gap (0.60 vs the
   0.90 solo ceiling) is entirely the router's retrieval accuracy (0.70 with honest subject-keys; misses
   are same-structure aliases) — a fixable retrieval-engineering problem, not an architectural wall.
   (Mean-centering the key embeddings is required to beat representation anisotropy and make the locality
   threshold work.)

## What this changes about *using* test-time memorization

On a frozen base, test-time memorization is an **amortization-and-structure play, not a reasoning play**:

- **It buys economics, not capability.** Reaching in-context quality means: replace an expensive,
  re-fed context with a persistent, ~38× smaller, composable prefix — worth it only for **build-once,
  query-many** contexts (a codebase, a manual, a user history). NIAH: ~682 s to build a 4k cartridge vs
  8 s/query in-context. Amortize the build or use RAG.
- **Organize memory as a routed bank of isolated units, not one accumulating state.** The single
  shared state (Titans' design, or one big cartridge) interferes with itself; many isolated memories +
  a router do not. Mint a new memory per fact/document/user and route — a property adapter-memory has
  and a co-trained recurrent state cannot.
- **Pick the memorization signal by whether you know the task.** Surprise for open-ended streaming;
  **relevance** (question-focused distillation) for known-task recall from noisy input.
- **It is not knowledge editing.** A cartridge makes a fact available and composable *at in-context
  quality*; it does not rewire the model's beliefs the way a weight edit or a co-trained memory does.

## The honest bottom line

A frozen base + bolt-on memory is **not a weaker Titans**. On reasoning and capacity it reaches Titans'
*floor* (in-context quality) but not its *ceiling* (above-in-context multi-fact reasoning, which needs
training the model). On continual/no-forget it has a **structural advantage a monolithic model cannot
match**. So the honest product is not "cartridges instead of a model" — it is **a reasoning model plus a
routed bank of cheap, editable, non-interfering test-time memories**: complementary shapes, each doing
what the other structurally can't.

## Reproduce

Method + evals in `cam/` (`cartridge.py`, `cartridge_train.py`, `moc_eval.py`, `babilong_eval.py`,
`niah_eval.py`, `continual_eval.py`). Base `Qwen/Qwen3-4B-Base`, frozen. Cloud recipe: standard-attention
base runs on stock CUDA (no fla); ≈$0.03/run on an RTX3090. Full scorecard ≈ $2.

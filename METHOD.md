# Method

How the mechanism works, in enough detail to reimplement. Notation: *base-1* is the model the memory is
born on (donor); *base-2* is a different frozen model we transfer to. Nothing in either base is ever
trained.

## The shape of the problem

A "document" presents *M* (name → cargo) bindings, then queries one name and asks for its cargo. The
cargo is randomly assigned, so a base model with no access to the bindings cannot do better than chance
(this is what keeps `no_memory ≈ 0` and makes the measurement clean). Difficulty is *M*; chance is 1/M.

## Two stages: bind, then deliver

**Stage 1 — bind (no base in the loop).** Train the memory store to recall cargo from a name by a direct
loss against the base's *tied embedding* (the store reads/writes in its own `mem_dim` space, mapped from
base-1's frozen input embeddings). This is pure memory training — fast, and the base never runs. Output:
a frozen store whose held-out recall is the *carry*.

**Stage 2 — deliver (base frozen, memory frozen).** Freeze the store. Train **only** a `GatedMemoryTap`
— a small cross-attention from the base's residual stream into the memory bank, added back through a
**zero-initialized gate** (γ starts at 0, so the tap begins as a literal no-op and learns its way in
without destabilizing the frozen base). This is **Memory-as-Gate (MAG)**: the memory modifies the
residual stream at one layer. We found the alternative — **Memory-as-Context (MAC)**, supplying the
memory as extra context tokens — does not deliver here (`memory ≈ no_memory`).

The bank fed to the tap is `[B, K, mem_dim]` — the store's own retrieval space, **base-agnostic** by
construction (built from base-1's embeddings, never base-2's). That property is what makes Stage 3
possible.

## The store of record: product-key memory + addressing supervision

The naive recurrent DeepMemory store *compresses* bindings into a fixed state and walls out past M≈8–12
(see [RESULTS.md](RESULTS.md) — the wall is the compression, proven by an uncompressed control that
doesn't wall). The fix is a **product-key store**: queries are split in half, each half scores a
sub-codebook, and the top-k product of the two indexes a large sparse slot bank — so a few thousand
slots are addressable at log cost and difficulty decouples from store size.

The load-bearing detail (we learned this the hard way) is the **addressing-supervision loss**: two
InfoNCE terms that directly teach the addressing geometry —

1. the factual read-query is pulled toward the queried binding's **write-address** (`to_wkey(cargo)`),
2. the retrieved context is pulled toward the binding's **stored value** (`to_wval(name)`).

Both targets are the store's *own* projections in `mem_dim` space, so the loss needs **no external/hub
geometry** — it reconstructs purely from the store. Without it, the sparse store is under-loaded-lossy
and looks far weaker than it is; with it, carry is flat through M=128.

## Stage 3: transfer to a second base via a tiny translator

Take the frozen Stage-1/2 memory + tap. To serve a *different* frozen base-2 (different hidden dim,
tokenizer, family), fit a small **affine translator**: `A: d_base2 → d_base1` maps base-2's residual
stream into the frozen tap's expected space, the tap reads the (base-agnostic) memory bank, and
`B: d_base1 → d_base2` maps the result back, added through its own zero-init gate. Train **only** A, B,
and the gate, by LM-loss through frozen base-2 on the recall task.

This is the [RecursiveMAS](ACKNOWLEDGMENTS.md) RecursiveLink idea repurposed: a learned map between two
models' hidden-state spaces — here, between a memory's space and an arbitrary base's. We default to an
**affine** translator; a richer MLP translator did not beat it at honest difficulty (it's the incumbent
for a reason).

## What gets trained, ever

| component | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| memory store | ✅ trained | frozen | frozen |
| MAG tap (γ) | — | ✅ trained | frozen |
| base-1 | frozen | frozen | (not used) |
| base-2 | — | — | frozen |
| translator (A, B, γ₂) | — | — | ✅ trained |

No base-model weights are trained at any point. The trainable surface in Stage 3 is a few small matrices.

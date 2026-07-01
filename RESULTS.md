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

## 6. Real knowledge — natural-language phrasing

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

This is a **first cut**: a single fixed relation (`lives in`) with single-token objects. Varied relations,
real-dataset facts, and multi-token natural-language objects remain open (tracked in
[#1](https://github.com/patcarter883/memory-organ/issues/1)).

## 5. Still open

- **Multi-token cross-base transfer** — translator-bound (see §4); higher-capacity translator in progress.
- **Real knowledge in real documents** (not random name→word pairs) — the true generalization test; a
  first natural-language cut is in §6 (single relation, single-token objects); varied relations,
  real-dataset facts, and multi-token natural objects remain (tracked in #1).
- **N-scaling** the store toward useful sizes (thousands of facts, not 8–128 per doc).
- **Backend portability** — pure PyTorch, CPU/CUDA expected but unverified.

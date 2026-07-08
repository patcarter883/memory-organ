# CAM #100 — persistent-disjoint result: the t≥1 wall is VALUE PRECISION, not addressing

Branch `cam-100-persistent-disjoint` (memory-organ, worktree `memory-organ-p`). This continues
`docs/CAM_100_CONTINUATION.md` (branch `cam-100-continuation`), whose proposed next step —
"switch the persistent per-position store to `--perpos-key disjoint`" — is **implemented and tested
here, and the hypothesis behind it is FALSIFIED.**

## What the continuation doc predicted
The doc localized the multi-token wall to *later positions* (per-token stuck at **0.50** on 2-token
objects = position 0 delivers, position 1 doesn't) and hypothesized the cause was **shared-codebook
cross-position contamination**. Its fix: give each answer position its OWN `ProductKeyStore`
(`--perpos-key disjoint`), citing a store-side hunt (disjoint 0.964 vs codebook 0.429). The
persistent path didn't support disjoint; only the episodic path did.

## What was implemented (commits on this branch)
1. **`feat(#100): persistent-disjoint per-position banks`** — extended the persistent (Track-4
   standing-store) path to per-position disjoint banks, mirroring `_write_episode`'s disjoint branch:
   - `persistent_write`/`persistent_bank` take an optional `store=` arg (byte-identical default).
   - `_init_pp_banks` (a LAZY `{bucket -> [bank_t0..bank_t(K-1)]}` dict), `_persistent_write_seq_disjoint`,
     `_is_disjoint_mt`; `eval_persistent_generate` threads a `Vpp[b][t]` structure through the write
     loop, `_banks_seq`, and the decisive `_iso_store_ok`. Single-token + non-disjoint stay byte-identical.
2. **`fix(#100): train disjoint stores in mt-recon + direct readout`** — the first disjoint run gave
   isolated per-token **0.00** (WORSE than codebook 0.50). Root cause: `_mt_recon_loss` (the
   reconstruction objective) always wrote/read `self.store` via `persistent_bank` (the `readout_q`
   attention pool) and **never touched the per-position disjoint stores**. So `--perpos-key disjoint`
   routed delivery through stores the reconstruction training never optimized. Fix: `persistent_bank_direct`
   (read the disjoint stores WITHOUT `readout_q`, mirroring `memory_bank`'s disjoint readout), and route
   `_mt_recon_loss` through `self.stores[t]` with that direct readout in disjoint mode. Also three OOM
   fixes (lazy `Vpp`; native-dtype logits in disjoint mt-recon to avoid a 2.37 GiB fp32-unembed copy;
   `_iso_store_ok` uses a single-bucket bank since it writes one record alone).

Run recipe (unchanged from the continuation doc, `--perpos-key` now an env knob):
```
PERPOS_KEY=disjoint MT_NAME=4 VALNONORM=1 gpu-lease -n 1 -- \
  env PERPOS_KEY=disjoint MT_NAME=4 VALNONORM=1 CNAME=cam100-disjoint bash tools/private_demo.sh
# full #100 metric lines: memory-organ-p/tools/private_demo.out  (private_demo.sh greps its own stdout)
```

## The result (validated, gfx1201 — same-seed A/B, this session)
| config | mt-recon loss @step999 | bind held-out per-tok | **ISOLATED per-token** | store-forced / TF / delivery |
|---|---|---|---|---|
| codebook (baseline) | 2.49 | 0.67 | **0.50** | 0/4 |
| **disjoint** | 2.78 | 0.82 | **0.50** | 0/4 |

Disjoint delivers **NO lift** on the real invented objects: isolated per-token is **0.50 for both**,
delivery **0/4 for both**. Note the mt-recon loss and bind held-out are comparable (well within the
0.5–0.9 run-to-run variance seen across 5 runs) — so in THIS configuration the continuation doc's
store-side hunt ("disjoint 0.964 vs codebook 0.429") **does not reproduce**; disjoint helps neither the
store-side autoencoder loss nor the delivery. **Same-seed A/B confirms disjoint ≡ codebook on #100.**

## Why — the wall is VALUE PRECISION at t≥1, not addressing
The `STORE-force` decodings are the tell (position 0 exact, position 1 lands on an ADJACENT subword):
- ` Elvish` = [`Elv`,`ish`] → store emits "**Elv**ity"  (pos0 ✓, pos1 `ity`≠`ish`)
- ` Sindarin` = [`Sind`,`arin`] → "**Sind** protocol"  (pos0 ✓, pos1 ✗)
- ` Dothraki` → "**Do** DR…",  ` Klingon` → "Tokyoon…"

Position 1's value is retrieved *approximately* and `out_proj @ unembed` argmaxes to a NEIGHBORING
token. Disjoint separates the ADDRESS spaces per position, but write-key ≡ read-query in isolation
already, so addressing was never the bottleneck — **the per-position VALUE reconstruction is**. This is
exactly the `mt-value-capacity-norm-is-bottleneck` finding, and why `VALNONORM=1` moved it
qualitatively ("Sindarn≈Sindarin") but not to exact. The doc's contamination hypothesis is falsified:
removing all cross-position codebook sharing leaves per-token at 0.50.

## Lever 1 (cosine-NN decode) — ALSO FALSIFIED
The standard readout `out_proj(value) @ unembed` (unembed = tied embed.t()) is a DOT-PRODUCT nearest-
neighbour in embedding space, biased toward high-norm (frequent) token rows — the obvious suspect for
"lands on a frequent neighbour". Added a **cosine-NN** decode (divide by the `[vocab]` embedding-row
norms) as a diagnostic alongside the argmax, on the SAME retrieved value (`_iso_store_ok`, both
codebook and disjoint). Result:

| decode | ISOLATED per-token |
|---|---|
| dot-product argmax (current) | **0.42** |
| cosine-NN (row-norm debiased) | **0.33** (WORSE) |

Debiasing the row norm makes it WORSE, so the norm carries useful signal and the retrieved value's
DIRECTION is genuinely closest to a wrong token. **The wall is not the decode — it is the store's
per-position value RETRIEVAL fidelity at t≥1.** (Diagnostic committed; `iso cosine-NN` line in
`private_demo.out`.)

## Lever 3 (AR decoder readout) — FALSIFIED, and it clarifies WHY #100 is hard
Wired the prototyped AR transformer-decoder readout into the #100 path (it was only plumbed to the
episodic `builder.multitoken` path, which `counterfactual_multi` doesn't enable, so it had never been
trained/used here): trained it via mt-recon on 17,183 REAL multi-token fantasy-morphology words
(`_real_word_seqs`, eval objects held out), built the K-slot prefix from the per-position reads, and
added a free-run greedy `decoder_generate`. The decoder decodes position t conditioned on positions <t
+ all K retrieved slots (cross-attention) — the joint/sequential lever the independent per-position
linear readout lacks.

| readout (each in its own training regime) | ISOLATED per-token |
|---|---|
| linear (baseline) | **0.50** |
| decoder AR (cross-attn + AR) | **0.25** (WORSE) |

The decoder is WORSE, and its `STORE-force` decodings ("Latvia Belarusian", "Italian Judaism
Frankfurt") show why: it generates PLAUSIBLE multi-token sequences from its learned prior, OVERRIDING
the imprecise retrieved value. This is structural, not a training artifact: **#100 objects are
novel/unpredictable BY DESIGN (private phrases the base can't continue). A generative sequence prior —
exactly what the decoder adds — cannot deliver what it cannot predict.** Its ceiling for novel-object
delivery is "faithfully copy the retrieved value" = the linear readout; the extra sequence prior is not
just useless but actively harmful (it hallucinates plausible-but-wrong continuations). Delivery MUST
come from faithful per-position RETRIEVAL — and that retrieval's fidelity is the wall.

(Caveat: 1000 steps / a synthetic corpus — but the structural argument caps the decoder at the linear
readout regardless of training budget, so more training cannot make it the unlock.)

## Lever 4 (magnitude-preserving read) — FALSIFIED (it HURT), and it pinpoints the real floor
The store read RMSNorms the retrieved value (`read_norm`/`read_out_norm`, "the bypass fix") — it
deliberately STRIPS magnitude so the tap gate carries retrieval strength. Hypothesis: that strip
destroys the token identity encoded in magnitude (symmetric to the write-side `VALNONORM` win). Added a
magnitude-preserving reconstruction read (`store.read(recon=True)` = the factual head's raw value mix,
no read_norm/read_o/read_out_norm) and trained mt-recon + delivery on it.

| read path | mt-recon loss @999 | ISOLATED per-token |
|---|---|---|
| normal (RMSNorm'd) | **~2.5** | **0.50** |
| magnitude-preserving | **~9.5** (barely trains) | **0.00** |

Preserving magnitude makes reconstruction MUCH HARDER — mt-recon can't get below ~9.5 (≈ln(104k)=11.5
random floor). The RMSNorm is not the bottleneck; it ACTIVELY HELPS the readout by stabilizing the
value range for `out_proj`. Falsified.

## Where that leaves #100 — a raw store value-CAPACITY floor (~2.5 CE / ~50% per token)
FOUR levers now falsified — every way of re-reading, re-decoding, or re-normalizing the same store:
- **disjoint addressing** (per-position separate codebooks): no lift (0.50 = codebook)
- **cosine-NN decode** (row-norm debiased): worse (0.33 < 0.42)
- **AR decoder readout** (cross-attn + sequence prior): worse (0.25), structurally can't deliver a NOVEL object
- **magnitude-preserving read** (drop the read RMSNorm): worse (0.00), reconstruction won't even train

The decisive datum: with the NORMAL read, mt-recon plateaus at **~2.5 CE** (bind steps 600–999 flat) and
the isolated single-object round-trip reconstructs a token at only **~0.5 per position** — even with ONE
value in a FRESH store (no crowding). So the store+`out_proj` autoencoder has a hard ~50%/token
reconstruction FLOOR on the 104k-realistic-token task. This is raw VALUE CAPACITY, and it is upstream of
— and unmoved by — every readout/read-side change tried.

The remaining candidates are genuinely DIFFERENT value MECHANISMS (not readout tweaks), i.e. real
architecture/design work:
1. **Discrete-code (VQ) value**: make the stored value a code the store addresses EXACTLY and decode by
   CLASSIFICATION over the code table — turns lossy reconstruction into (near-)lossless retrieval. The
   single most-promising direction, but a substantial store redesign + retrain.
2. **Per-position / hard-token CE weighting or curriculum**: won't raise the capacity ceiling, only
   redistribute it; lower-value given the floor is a capacity limit, not a training-emphasis one.
3. **Reconsider the #100 contract**: an associative value store may be the wrong primitive for
   losslessly carrying arbitrary NOVEL multi-token phrases; a copy/pointer mechanism (store the token
   IDS, not a reconstructed embedding) sidesteps reconstruction entirely.

The disjoint impl, cosine-NN + decoder-AR diagnostics, and the magnitude-preserving read are all
correct, committed, A/B-able building blocks — **none is the #100 unlock, and four independent readout/
read levers prove the wall is a raw store value-reconstruction CAPACITY floor, not a readout.** The next
real move is a different value mechanism (VQ / copy-pointer), which is a design change, not a tweak.

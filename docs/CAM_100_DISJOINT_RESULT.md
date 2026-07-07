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

## Recommended next levers (value-precision, NOT addressing)
1. **Decode the retrieved value by nearest-neighbour in the (frozen) embedding table**, not argmax over
   `out_proj(value) @ unembed` — the store reconstructs a value vector; snapping to the closest real
   token embedding is far more robust to the small reconstruction error that currently lands on a
   neighbour. This is the highest-leverage, lowest-risk change and directly targets the observed failure.
2. **Value capacity / precision at t≥1**: the value path (LayerNorm dropped by `VALNONORM`) still loses
   the continuation-subword detail. Probe a higher-capacity or residual-VQ value, or supervise the
   position-1 reconstruction harder (per-position CE weighting) — mt-recon trains all positions equally
   but position-1 continuation subwords ("ish","arin") are rarer/harder than position-0 prefixes.
3. **Realistic-value coverage**: verify the invented objects' position-1 subwords are IN the
   `CAM_MT_RECON_NAME_TOKENS`/realistic-subword pool the store trains to reproduce; if a continuation
   subword is out-of-pool, its value branch is never trained to round-trip.

The disjoint implementation is correct and now A/B-able (a clean, reusable building block that mirrors
the episodic disjoint path), but it is **not** the #100 unlock and, in this config, does not even move
the store-side objective. The unlock is value-side precision at t≥1.

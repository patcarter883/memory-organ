# CAM #100 — sequential multi-token delivery: CONTINUATION (for a fresh session)

You are resuming a research effort to make CAM deliver genuine **multi-token** objects the base can't
continue on its own (novel/private phrases). The serving side is DONE; the store-training side is the
open blocker, precisely localized. Read `github.com/patcarter883/memory-organ#100` (the issue + its two
progress comments) and the memory `cam-100-multitoken-delivery-wall` first.

## The two halves
- **Serving (minisgl-rdna4) — DONE, merged.** The first-class residual-tap integration (Option B) is in
  `rdna4` (PR #2, commit 5b1a35c): model-share, seed-once tap, `/cam/*` HTTP, graph capture, per-token
  banks. It is READY to carry a working multi-token checkpoint — no serving work is blocking #100.
  (Option B details: `minisgl-rdna4/docs/zaya-port/CAM_SERVE_OPTIONB_CONTINUANCE.md`.)
- **Store training (memory-organ) — the open work.** This doc.

## Where the research is (branch `cam-100-diagnostics`, memory-organ, pushed)
Runs use `tools/private_demo.sh` (titans:dev image; binds 137 counterfact edits + 4 INVENTED multi-token
objects Klingon/Dothraki/Elvish/Sindarin; `--persistent-generate`). Env knobs: `MT_NAME=N` (multi-token
mt-recon keys), `VALNONORM=1` (no-norm value), `ALPHA` (logit inject). ~3–4 min/run.

Diagnostic chain (all in `eval_persistent_generate`, added this effort): `_gen_tf` (teacher-forced),
`_gen_store` (store-forced, bypass base), `_iso_store_ok` (isolated round-trip + per-token). Findings:

- Baseline delivery: **0/4**. Ruled out, each with an experiment: mechanism (tap vs logit both 0/4),
  strength (α=30 same), exposure bias (teacher-forced 0/4), delivery-through-base (store-forced 0/4),
  standing-store interference (isolated round-trip 0/4).
- **Root cause = training-distribution mismatch, then value capacity.** `_persistent_write_seq`/`_banks_seq`
  are structurally identical to the `_mt_recon_loss` that trains at 0.8–0.9 — not a code bug.
- **FIX #1 (committed, works): `CAM_MT_RECON_NAME_TOKENS`** — mt-recon trained single-token keys but
  delivery queries `_pool_subject` of a MULTI-token subject. Training multi-token keys lifted the
  isolated round-trip **per-token 0 → 0.50**, held-out 0.81 → ~0.90.
- **Lever #2 (committed, cherry-picked b8be524): `CAM_MT_VALUE_NO_NORM`** — drops the value-path LayerNorm.
  Qualitatively closer (store-forced → "Sindarn"≈Sindarin, "Elvian"≈Elvish) but per-token STILL 0.50.
- **Remaining wall: later positions (t≥1).** Per-token is stuck at 0.50 on 2-token objects = position 0
  delivers, position 1 doesn't.

## THE NEXT STEP (concrete, architectural)
Switch the persistent per-position store to **`--perpos-key disjoint`**. Store-side hunt: **disjoint 0.964
vs codebook 0.429** (the demo uses codebook). Disjoint gives each answer position t its OWN
`ProductKeyStore` (`adapter.stores[t]`), removing shared-codebook cross-position contamination — exactly
the t≥1 problem.

BUT the persistent path does NOT support disjoint yet. Only the EPISODIC path does:
- Episodic disjoint: `pk_store_adapter.py::_write_episode` (~L355-382, `mt_value=="perpos" and
  perpos_key=="disjoint"`) writes position t into `self.stores[t]`, returns a LIST of per-position banks;
  read `memory_bank` (~L466-494) loops t reading `self.stores[t]`.
- Persistent path writes/reads ONE bank `V[b]` per subject via `adapter.persistent_write`/`persistent_bank`
  (`recall_mag.py::_persistent_write_seq` ~L1304, `eval_persistent_generate::_banks_seq` ~L1574) — it never
  touches `self.stores[t]`.

**Implement:** extend the persistent path to per-position disjoint banks, mirroring `_write_episode`:
1. `_init_banks` → a per-position bank structure when `perpos_key=="disjoint"` (banks[b] becomes a list
   over t, or a parallel `V_pos[t][b]`), sized from `adapter.stores[t]`.
2. `_persistent_write_seq`: for position t, write `val_t` at `_pos_key(base_key, t)` into the per-position
   store `adapter.stores[t]` / its bank, not the shared `V[b]`.
3. `_banks_seq` (+ `_iso_store_ok`, `_gen_store`): read position t from `adapter.stores[t]`'s bank with
   `_pos_key(base_q, t)`.
4. Keep single-token + non-disjoint byte-identical (gate on `perpos_key=="disjoint" and _is_multi`).
Then run `MT_NAME=4 VALNONORM=1` with `--perpos-key disjoint` (edit private_demo.sh, currently `codebook`)
and watch **isolated per-token → 1.0**, then store-forced/answer-span delivery → >0.

## First actions next session
1. `gpu-status`; re-run `MT_NAME=4 VALNONORM=1 gpu-lease -n 1 -- bash tools/private_demo.sh` to reconfirm
   isolated per-token 0.50 (the current committed state).
2. Implement the persistent-disjoint extension (steps 1–4). Validate isolated per-token first (store-side,
   fast, no delivery needed) — it isolates the fix from the base entirely.
3. If per-token → ~1.0, THEN the delivery (store-forced / answer-span) should follow; wire it and confirm
   the serving side (minisgl PR #2) carries it end-to-end with a real export.

## Gotchas
- private_demo.sh greps its own output to stdout; the FULL log is teed to
  `memory-organ-p/tools/private_demo.out` — read THAT for the #100 metric lines.
- The research checkout is `memory-organ-p` (detached HEAD; branch `cam-100-diagnostics` anchors the work).
  Main `memory-organ` is on `research-private-demo` (has private_demo.sh + the base multi-token path).
- `exp/value-capacity` has the no-norm lever (now cherry-picked); other branches (mt-decode-precision,
  scale-validation) may have adjacent work — check before reinventing.
</content>
</invoke>

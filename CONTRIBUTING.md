# Contributing

The most valuable contribution to this repository right now is **adversarial**: try to break a number.
Reproduce a table and tell us if it doesn't hold; run it on hardware we haven't (CPU/CUDA); design a
probe that would break the mechanism if it's going to break. The corrections in
[RESULTS.md](RESULTS.md) exist because controls caught our own errors — more controls, from more
people, is exactly what this project needs. See [ROADMAP.md](ROADMAP.md) for where help is wanted.

## Reporting a non-reproduction

This is the report we most want, and it's only actionable if we can run exactly what you ran. Open an
issue titled `non-repro: <table / section>` containing:

1. **The exact command** you ran (ideally copied from [REPRODUCING.md](REPRODUCING.md), with any
   changes called out) and the **seed**.
2. **The number you got vs the published number**, and the full run log if you can attach it (the
   drivers print the controls — carry/ablated/chance, memory/no_memory/ceiling — which is usually
   where the story is).
3. **Environment**: `pip freeze` (or at least torch/transformers/numpy versions), Python version,
   GPU/CPU, and backend (ROCm/CUDA/CPU).
4. Whether the **controls** behaved: `no_memory` ≈ 0? ablated ≈ 0? If a control is off, the run is
   contaminated and that's a different (also valuable) bug.

A reproduction that *matches* is worth reporting too, especially on CUDA or CPU — portability is an
open roadmap item and "it worked on stock CUDA torch" is information we don't have.

## Code changes

- Changes land via pull requests that reference the issue they close (the issues are the actual,
  current plan; [ROADMAP.md](ROADMAP.md) is the narrative).
- The experimental harness deliberately keeps byte-identical baselines: when adding a mechanism knob,
  default it to the existing behavior so prior results stay reproducible from the same commands.
- New results follow the house rules: every number next to its **chance baseline**, its **floor**
  (`no_memory` / ablated), and its **ceiling** where one exists; wrong numbers stay in the record with
  the correction beside them.
- CI runs a CPU smoke test (`pytest tests/`) — imports, store write/read round-trip, tap no-op-at-init.
  It needs no GPU and no model downloads; please keep it that way so it stays fast and hermetic.

## Provenance

Authorship and how this was built are documented in [DISCLOSURES.md](DISCLOSURES.md). If you
contribute with substantial AI assistance, say so in the PR — that's house style, not a gotcha.

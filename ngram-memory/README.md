# ngram-memory

A ground-up project, separate from memory-organ's `cam/` concept: a portable editor that writes facts into a model's
**native n-gram table** (Engram-style), and grafts a table in the same format onto models that don't ship one.

**Status: design reviewed (2026-11-09). No package code yet — M0 (package skeleton + tiny-config tests) is next.**

- [DESIGN.md](DESIGN.md) — founding design: format spec (Qwen3.8-Flash-Next's native PLE n-gram module), backends
  (native Flash-Next, Gemma 4 E4B per-layer embeddings, grafts onto Qwen3.8-27B / Qwen3.6-35B-A3B / Gemma 4 26B/31B),
  the edit solver, evaluation, compute plan and milestones. Red-teamed through correctness, science and
  feasibility lenses; the review log is at the end. One venue conflict with `serve/INTEGRATION.md` is flagged in §7.
- [serve/INTEGRATION.md](serve/INTEGRATION.md) — applying edits to the owner's live Flash-Next serve
  ([minisglang-rdna4](https://github.com/patcarter883/minisglang-rdna4) at `df29117`): torch-free numpy overlay on the
  row path, cache invalidation under EP, and where the edit solve runs. Skeptic-verified.
- [proofs/](proofs/) — the scripts and outputs the design cites (tiny-config proofs, serve-path checks, cost models).
  Copied from the session scratchpad: most still hard-code the scratchpad paths they ran from, and the `.pt` fixtures
  were not copied; making them re-runnable is an M0 task.

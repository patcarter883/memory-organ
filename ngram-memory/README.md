# ngram-memory

A ground-up project, separate from memory-organ's `cam/` concept: a portable editor that writes facts into a model's
**native n-gram table** (Engram-style), and grafts a table in the same format onto models that don't ship one.

**Status: design draft (2026-11-09). No package code yet.**

- [DESIGN.md](DESIGN.md) — founding design: format spec (Qwen3.8-Flash-Next's native PLE n-gram module), backends
  (native Flash-Next, Gemma 4 E4B per-layer embeddings, grafts onto Qwen3.8-27B / Qwen3.6-35B-A3B / Gemma 4 26B/31B),
  the edit solver, evaluation, compute plan and milestones. *Draft under red-team review; a revised version and its
  review log will replace it.*
- [serve/INTEGRATION.md](serve/INTEGRATION.md) — applying edits to the owner's live Flash-Next serve
  ([minisglang-rdna4](https://github.com/patcarter883/minisglang-rdna4) at `df29117`): torch-free numpy overlay on the
  row path, cache invalidation under EP, and where the edit solve runs. Skeptic-verified; some cited scripts still live in
  the session scratchpad and will move into `proofs/`.

---
date: 2026-10-17
question: Can techniques from engram-style models (DeepSeek Engram; Qwen3.8-Flash-Next's n-gram table) improve the frozen-base Titans graft (CAM)?
status: current
---

## Answer

Yes for four specific transfers — addressing, a trigger-local delivery geometry, offload economics, and a
native-channel placement idea — none of which breaks the measured perturb-vs-recompute ceiling.

1. **Trigger-local, content-addressed delivery (new lever).** Engram's delivery is sparse, transient and
   position-local: an n-gram match fetches a static vector and a learned scalar gate [0,1] decides how much
   of it fuses into the hidden state. That is a third delivery geometry between the L12 residual tap
   (delivers, doesn't ripple, degenerates when sustained) and the KV cartridge (ripples, but costs a build
   and distills filler). Untrained it will behave like the falsified latent prefix (MAC 0.000), so it must
   be trained with the objective that provably installs beliefs: cartridge distillation (teacher = frozen
   base + fact in-context; student = frozen base + trigger lookup; KL on self-study; only lookup + gate
   train). It also gives "fire at the right reasoning step" (reorient lever 3) and re-ignition semantics —
   a chain that re-mentions the bridge re-fires the read, i.e. sustained influence without sustained
   perturbation.

2. **Hashed multi-order keys as a second addressing space.** Add beside the dense product-key semantic keys,
   not instead of (the graded read/conf-gate is load-bearing for the router). Exactness for
   alias-vs-paraphrase collisions (#10: tau overlap 0.58 vs 0.62), positional structure that generalizes
   "go per-position" (#100: disjoint per-position codebooks 0.964 store-side), free dedup / no-clobber /
   audit semantics at write time.

3. **Deterministic addressing → offload economics.** Engram's system claim is exactly this project's serving
   pain: table in host RAM, async prefetch of only fired rows, reported <3% overhead at 100B params.
   Two-tier memory: static/read-mostly hashed tier (host-resident; kB per fired key vs the measured
   568.32 MB/token/rank expert traffic), dynamic editable store stays VRAM-resident (delta-writes,
   steering, conf-gates). Turns #17 N-scaling from a VRAM question into a RAM question.

4. **Native-channel placement on an n-gram-native base.** The served Qwen3.8-Flash-Next already runs the
   channel: 20M bigram/trigram rows at decoder index 1 (the boot line's "PLE on decoder index [1]"), and
   minisgl already restores PLE state in radix snapshots. On this base the layer-1 lookup is a pre-trained
   read interface; grafted vectors distilled into that channel start closer to in-distribution than any
   mid-layer residual edit ("native module principle", strongest form). Does not apply to non-n-gram bases
   (Qwen3.5-4B, Qwen3-4B).

What NOT to expect: Engram is static — no test-time update (the delta-write store + SLIDING_WINDOW ladder
remain this project's mechanism). Its wins (iso-param gains, NIAH 84→97 reported, "frees early layers")
all come from co-training with the backbone; a frozen base cannot free trained depth, so the scorecard
ceiling stands. Token n-gram keys do not survive tokenizer change — keep the cross-base layer
base-agnostic (text-level hashing or per-base key spaces). vLLM's host offload for the n-gram table is
NVIDIA-only; on gfx1201 it would be minisgl-side work.

## Key facts

- Engram (DeepSeek, 2026-01): hashed N-gram tables (N∈{2,3}), multi-head hashing into prime-sized buckets,
  depthwise conv over n-gram context, context-aware gating scalar [0,1]; inserted at layers 2 and 15;
  deterministic addressing → host-memory offload of massive tables (reported <3% overhead for 100B;
  U-shaped capacity allocation, ~20-25% to static memory) [1][2][6]
- Qwen3.8-Flash-Next: 20M-row bigram/trigram table, 51B params, decoder index 1; async host offload
  (`--engram-config '{"cpu_offload":true}'`), NVIDIA-only in vLLM [3][4]
- CAM measured laws (in-repo): injection cannot ripple (KV-append 0.000; MAC latent prefix 0.000);
  distilled cartridge ≈ RAG (single-hop 0.84; NIAH 0.87/0.73; BABILong below RAG); degeneration wall
  (multi-layer residual ripple 0.220, 'other' 0.780); depth sweet spot ~L12 (0.412); surprise ≠
  relevance; routed-bank win (locality 1.00 vs 0.17 concat); store knee ≈128 facts; decode tap ~20%
  TPOT [5]

## Proposed experiments

- **E1 (do first):** trigger-local distilled delivery vs L12 tap vs cartridge on the in-dist ripple
  protocol (B0). Metrics: ripple, degeneration ('other'), single-hop gate; then BABILong distractor gap +
  NIAH. Falsifier: input-space trigger cannot ripple even when distilled (→ input space is a dead
  channel; 1C KV remains the structural candidate). Spec: `docs/CAM_TRIGGER_DELIVERY_SPEC.md` (in-repo;
  issue #104). — **RAN 2026-10-17; result below.**
- **E2:** hashed exactness keys in the #10 subject-key test + #100 per-position path.
- **E3:** offload-tier design for N-scaling (static hashed read-mostly + dynamic VRAM store; write
  coherence for dirty rows).
- **E4 (held):** native-channel placement on Qwen3.8-Flash-Next (PLE channel) once a serving experiment
  is warranted.

## E1 outcome (2026-10-17)

First-signal run complete (issue #104; commit `a60e5e5` on `exp/e1-trigger-delivery`; Qwen3.5-4B, L12,
bind 1500, forced `country` hop, 60 edits, masked + always-on arms both trained with the July tap-fit
objective made mask-compatible, single-pass no_edit/RAG/trigger, confound-controlled filtered n=44/60):

| arm | delivery (gate ≥0.40) | ripple vs gold_cf strict | head-word |
|---|---|---|---|
| trigger (masked) | 0.900 | 0.091 | 0.295 |
| always-on tap (July config) | 0.933 | **0.250** (= the Aug-5 baseline tap number) | 0.136 |
| RAG (same pass) | — | 0.455 | 0.682 |

- The masked geometry delivers single-hop as well as the always-on tap. The earlier delivery gap was the
  *objective*: masked KL-distillation plateaued ≈0.08 delivery at 300 steps; the July CE objective
  converges ≈0.9.
- Its 2-hop ripple is ~1/3 of the always-on tap and ~1/5 of RAG — **the trigger geometry does NOT beat
  the always-on injection; confining the perturbation to the mention loses the downstream-carrying part**
  (the always-on tap also perturbs near the answer). The falsifier's strong form (dead channel) is not
  met (0.091 > leak 0.000) but the lever does not clear the wall.
- Stability side-note: masked training was clean in every run; the always-on arm NaN-skipped/wedged
  under the distill objective and degenerates into repetition at delivery strength. Stability ≠ ripple.
- Stack findings (gfx1201, titans:dev / transformers 5.5.3): the chunked full-reforward GDN path wedges
  intermittently (native: inf-grads→NaN; fla: kernel hangs; torch fallback: page faults) — generation is
  now cached incremental decode (token-exact vs the old loop; ~30× cheaper). No allocator compaction on
  this platform → pool creep caused an OOM, a mid-run KL "explosion" (0.6→~10), and in-run oracle-map
  degeneration (0/60); fixed by `empty_cache()` at fixed points. Reusable across CAM work.
- Natural follow-up geometry if this line continues: widen firing to include answer-adjacent positions
  (test whether the lost ripple is a proximity effect).

## Open questions

- Does a transient trigger read ripple when distilled, or is input space structurally weaker than
  KV/prefix (cartridge) space?
- Does the trigger fix BABILong's distractor-selectivity gap, or re-derive RAG as the scorecard predicts?
- Offload write-coherence: delta-writes vs host residency for the dynamic tier.
- Tokenizer-bound keys vs the cross-base promise — text-level n-gram hashing viable?

## Sources

[1] DeepSeek Engram — official repo + paper — https://github.com/deepseek-ai/Engram/ — local: ~/kb/pdf/memory-organ/engram-paper.pdf
[2] MarkTechPost: Engram architecture details — https://www.marktechpost.com/2026/01/14/deepseek-ai-researchers-introduce-engram-a-conditional-memory-axis-for-sparse-llms/
[3] Qwen3.8-Flash-Next model card — https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[4] vLLM recipe (n-gram offload, engram-config) — https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
[5] memory-organ in-repo docs — docs/research/frozen-base-titans-scorecard.md; minisgl-rdna4*/docs/**/zaya-port/CAM_TITANS_REORIENT.md; docs/CAM_100_CONTINUATION.md; ROADMAP.md; SLIDING_WINDOW.md
[6] Engram coverage (reported overhead / NIAH numbers) — https://introl.com/blog/deepseek-engram-conditional-memory-architecture-january-2026

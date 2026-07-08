# The global-workspace lens — reading the stream the tap writes into (exploration, 2026-07-08)

Anthropic's [global-workspace result](https://www.anthropic.com/research/global-workspace) and its
open reimplementation [Subtext](https://github.com/ninjahawk/Subtext) hand this project a
measurement instrument it has been running blind without. This note maps the connection, ships the
first rung (a lens harness, `cam/lens.py`), and lays out an experiment ladder with predictions and
falsifiers. Nothing here changes a headline number — this is **observability**, and one candidate
gate signal.

## What they found, in our terms

- **The finding.** A sparse subspace of the residual stream (the "J-space", <10% of activity, a few
  dozen concepts at a time, ~100× denser wiring) holds *the concepts the model is disposed to say*.
  It is read out by the **Jacobian lens (J-lens)**: transport an intermediate-layer residual state
  into the final-layer basis and decode it through the unembedding. The workspace is reportable,
  deliberately controllable, mediates multi-step reasoning, and is **causally editable**: patching
  "France"→"China" in J-space simultaneously redirects capital (Paris→Beijing), language, and
  continent. Depth plays the role time plays in the brain — concepts evolve layer to layer over one
  forward pass.
- **Subtext.** A local, real-time J-lens viewer — on **Qwen3.5-4B**, *our production donor base*
  (`m2_adapter.MODEL`), using a pre-fitted Jacobian lens from Neuronpedia, reading 9 layers per
  token. The lens artifact we would otherwise have to fit already exists for our exact base.

## Why this is our business

The MAG tap (`gated_tap.py`) **writes into exactly the stream the J-lens reads** — an additive,
gated update to the residual at layer L=24. Three of our standing results are, in workspace terms:

1. **Counterfactual editing (RESULTS §6–7)** — France→Tokyo at 0.996 with the prior collapsed to
   0.004 — is functionally the *same operation* as Anthropic's France→China J-space patch, except
   ours is performed by an external, learned, cross-model-portable organ instead of a hand-built
   activation edit. Their result predicts ours *should* work the way it does: **if** the injection
   lands at workspace level, downstream attributes re-derive from the edited concept — which is
   what our generalization numbers (0.91–0.93 paraphrase) already look like from the outside.
2. **The single-site-injection ceiling (#19)** and the tap-layer choice have always been tuned
   blind, by output accuracy only. The lens gives the depth axis back: *where* in the stack does
   the edit take hold, and where does the base commit to an answer?
3. **The confidence-separability wall (Track 5, novelty-positioning.md)** — base-side confidence
   cannot distinguish confident-wrong from confident-right at the *output*. The lens reads
   *upstream* of the output; whether the wall also holds in the layer trajectory is an open,
   testable question (Anthropic reports lens-visible detection of fabricated data and staged
   scenarios — signals that never surface in output logits).

## What ships in this exploration (E0)

`cam/lens.py` — `LensTrace`, a context manager that captures the residual after every decoder layer
during **any existing forward** (no harness changes), and decodes captured states through the
base's own final-norm + unembed. Drop-in around any eval site:

```python
from cam.lens import LensTrace
with LensTrace(base) as lt:                    # enter AFTER injector.attach() -> post-injection stream
    lg = _last_logit(base, input_ids=ids)      # any existing eval forward, mem-on or mem-off
traj = lt.trajectory([prior_tid, edit_tid])    # {layer: [(prob, rank), ...]} at the answer site
xo = lt.crossover(prior_tid, edit_tid)         # first layer where the edit overtakes the prior
```

Pinned by CPU tests (`tests/test_lens.py`) + `python -m cam.lens --selftest`: the lens is a pure
observer (outputs bit-identical), the last-layer decode equals the model's real logits (the
correctness anchor), the zero-init tap is lens-invisible, and an opened gate moves traces at/after
the tap layer and nowhere before (hook-ordering contract). A CLI (`python -m cam.lens --prompt
"The capital of France is" --targets " Paris, Tokyo"`) traces a raw base, GPU-side.

**Honesty box.** This is a *logit* lens (nostalgebraist 2020), not the Jacobian lens: decoding
intermediate layers through the final norm+unembed is a zeroth-order approximation, biased at early
layers (tuned lens, arXiv 2303.08112, fits per-layer probes to fix this; the J-lens is the same
move done via Jacobians). It is most faithful mid-to-late — where the tap sits. `decode_fn` is
pluggable so the pre-fitted Qwen3.5-4B Jacobian lens (Subtext/Neuronpedia) can drop in without
touching call sites (E6). Also: the workspace *finding* is on Claude; how much workspace-like
structure Qwen3.5-4B has is itself empirical (Subtext's demos suggest qualitatively yes). And the
J-space is word-token-structured — same single-token bias our own pipeline fights (RESULTS §4).

## The ladder (cheap → expensive), with falsifiers

- **E1 — watch the edit land.** Run the counterfactual eval (`eval_counterfactual` /
  `_persistent_preds`) with a LensTrace, mem-on vs mem-off, and plot P(prior)/P(edit) vs depth at
  the answer site. *Prediction:* mem-off shows the prior rising through late layers (standard
  fact-recall); mem-on shows the edit token overtaking at/just after L=24 and the prior actively
  suppressed (two-sided gate visible as prior-decay, not just edit-growth). *Falsifier:* if the
  edit is lens-invisible until the last 1–2 layers, the tap is doing **shallow logit steering**,
  not workspace-level editing — and the generalization we measure would need a different
  explanation than "downstream attributes re-derive from the edited concept". Either outcome is a
  finding. Cost: one eval run + plotting; no training.
- **E2 — locality through the lens.** Locality currently scores −0.008 *at the output*. Compute
  per-layer KL(mem-on ‖ mem-off) at the answer site on **neighbor** prompts. *Prediction:* small
  and flat if the null-slot/conf-gate routing really makes the tap inert off-store. *Falsifier:* a
  large mid-stack perturbation that late layers happen to wash out — output locality would then be
  flattering, and cross-prompt/cross-relation robustness claims inherit the flattery. This is a
  strictly more sensitive locality metric, nearly free.
- **E3 — tap placement by workspace depth.** Sweep `--tap-layers` and correlate delivery /
  generalization / locality with the lens-measured depth profile (where concept tokens first
  become decodable; where answers commit). *Hypothesis:* taps land best where concepts are
  workspace-resident — after subject resolution, before answer commitment — and the ~0.7
  single-site ceiling (#19) partly reflects injecting off that window. *Falsifier:* no correlation
  → placement is about something else (e.g. norm growth), also worth knowing. Cost: the sweep we
  already know how to run, plus readouts.
- **E4 — lens features against the separability wall.** Add per-layer trajectory features (depth
  of first decodability of the base's answer, trajectory entropy, late-vs-mid disagreement) to the
  gate-router feature set (Track 5) and re-run the confident-wrong **rescue**. *Prediction to
  beat:* base-side features gave ΔP +0.000 on confident-wrong; store-presence gave +0.464. If
  lens features move confident-wrong *at all*, that's a real dent in the wall from the base side —
  it would sharpen (not weaken) the provenance claim by showing exactly how much of the gap only
  provenance can close. *Falsifier:* confident-wrong is lens-indistinguishable from
  confident-right too → the wall is deeper than the output, strengthening novelty claim #4.
- **E5 — translator diagnostics.** The multi-token cross-base gap (0.812 vs ~0.94 single-token
  parity, RESULTS §4): put the lens on the **target** base per answer position. Does position 2
  degrade because the translated injection lands as the wrong concept (translation error), lands
  right but too late (depth mismatch between bases), or lands right and loses to the target's own
  prior (delivery strength)? Three different fixes; today we can't tell them apart.
- **E6 — the real J-lens + the viewer.** Swap `decode_fn` for the pre-fitted Qwen3.5-4B Jacobian
  lens Subtext uses (early-layer-faithful readouts), and wire the serve loop (`--serve`, RESULTS
  §8) into Subtext's browser viewer — *watch memory delivery live*: the stored fact surfacing in
  the workspace mid-prompt, the router gating it in. Demo-grade, but it is the artifact that makes
  "memory as a workspace write" legible to anyone in thirty seconds.

## What this is not

Not a claim that the base "has a global workspace" (that's Anthropic's claim about Claude, and
E1–E3 will tell us how much of it transfers to a 4B Qwen), not a new delivery mechanism, and not a
headline-number change. It is the read half of an apparatus that until now could only write —
plus one honest shot (E4) at the wall that motivated the provenance gate.

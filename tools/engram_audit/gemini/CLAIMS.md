# Gemini's prose claims accompanying the three code files (paraphrased faithfully, key sentences quoted)

Architecture as described:
- Host-DRAM lookup plane: "a multi-head rolling polynomial hash over normalized input 2-grams and 3-grams (n=2,3, K=8 heads)
  that deterministically fetches low-rank memory vectors e_t in R^r before Layer 1 even begins executing."
- Gated fusion adapter at a Layer-2 forward hook (~8M params for d_model=5120): Q from h_t, K/V from e_t, RMSNorm on Q and K,
  scalar trust gate alpha_t, then a causal short depthwise conv (kernel 4) into the residual stream.
- EngramEdit two-stage: Stage 1 optimizes delta_i in R^r added to e_{q_i(x)} over M paraphrases with clamp
  ||delta|| <= rho ||e_q||, 15-25 Adam steps; Stage 2 frequency-penalized ridge allocation, gamma_s ∝ freq(s)·beta_order(s).

Claim C1 ("STAPLE gate warmup"): v_proj zero-init gives no residual shock; run a one-time warmup over ~50M-100M tokens with
backbone 100% frozen and only q_proj, k_proj, v_proj, short_conv and cpu_tables trainable; then freeze them and use the solver
for all future updates.

Claim C2 (order penalties {2:10, 3:1}): editing `SpineGraph.Step()`, the 2-gram (".", "Step") is ubiquitous while the 3-gram
("SpineGraph", ".", "Step") is specific; Gamma penalizes 2-gram slots 10x and scales with log1p(bucket_counts); "That
mathematically forces 85-90% of the norm of delta_i* into the rare 3-gram hash buckets, preserving unrelated predictions."

Claim C3 (multi-hop CoT): because delta* is written into rows triggered at the final subject token, "whenever the model
generates that symbol inside its own Chain-of-Thought scratchpad mid-rollout, those exact 2-gram and 3-gram hashes fire at
Layer 2 on the next forward step, injecting the updated representation ... so downstream reasoning steps immediately use the new
specification."

Claim C4 (subgraph calibration premise): "Layers 2 through 6 in a standard dense transformer spend most of their compute
performing local N-gram detokenization and surface feature fusion." Training the L2 gates against L2's own output learns nothing
when W_V=0; instead train the L2 module as a "Lookahead Residual Shortcut" predicting the residual delta of a deeper block (L6);
"once Layer 6 is loaded into VRAM, you never even have to instantiate Layers 7 through 64".

Claim C5 (Phase 1, zero-backprop alignment): training the L2 branch to predict a scaled projection of
Delta h_FFN(2->6) = sum_{l=3..6} FFN_l(h_l) "immediately aligns W_V's column space with the exact FFN write subspace of the
frozen model". (Note: the code accumulates (h - h_in) per block = attention + FFN.)

Claim C6 (Phase 2, context-drop distillation): teacher = [256 prefix + 256 continuation] through L0-6 without Engram; student =
continuation only through L0-6 with Engram; match LayerNorm'd L6 states (MSE + 0.5 cos). Same-token teacher/student would have
global min alpha=0; with the prefix dropped, "the optimizer must open the gate alpha_t and teach W_Q, W_K, W_V how to translate
those N-gram vectors into the exact residual activations that full self-attention over the prefix would have produced by Layer 6."
Also sparse SGD on the CPU table rows that fired.

Claim C7 (VRAM): full-model backprop on a 64-layer 27B "easily spikes past 48-60 GB"; subgraph L0-6 is ~3.2 GB BF16 / ~1.6 GB FP8,
"< 4 GB".

Claim C8 (subgraph edit solver): replace Stage 1's final-NLL with prefix-conditioned L6 representation matching: teacher runs
[spec/docstring C_i + expression x] through L0-6; student runs bare x; optimize delta at q_i(x) so L3-6 reproduce the teacher's
L6 states over an anchor + lookahead window W_i = [q, q+L] (L=8, anchor weight 2.0, exp(-0.25 t) decay). Rationale: in the full
solver, gradients flow from future target tokens back into q via attention; matching the window preserves that. "you force
delta_i* to populate the exact Key/Value attention subspaces that downstream layers read from when resolving argument types,
return signatures, or factual attributes."

Claim C9 (performance): per edit 20 x 4 = 80 single-layer passes on [8, 32, 5120], "~35-50 milliseconds on a single GPU";
~3.4 GB VRAM; can run as a sidecar daemon compiling new repo symbols/specs into the pinned System RAM .engram table in real time
while the 27B serves inference.

Context: target backbone in the pitch is "Qwen3.8-27B" / Llama-3.3-70B (64 layers, d_model 5120 for the 27B).

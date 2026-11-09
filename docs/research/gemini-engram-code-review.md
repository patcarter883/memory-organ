# Gemini's grafted-Engram code — audit and a CPU experiment

*A research note, 2026-11-09. Gemini proposed relaunching memory-organ around a gated, Engram-style hashed
n-gram memory grafted onto a frozen transformer at layer 2, with EngramEdit-style writes
([2610.10533](https://arxiv.org/abs/2610.10533)). It supplied three files (`engram_graft.py`,
`subgraph_calibrator.py`, `subgraph_edit_solver.py`) and prose claims C1–C9 (`CLAIMS.md`). This note
audits the code and claims through four lenses: mechanics, solver maths, calibration, and fidelity to
sources. Each lens finding was re-checked by an independent skeptic pass with its own scripts. Only
**upheld** findings appear here, in the skeptics' corrected wording and severity. One CPU experiment on
a frozen Qwen2.5-0.5B tests the core premise. Companion to
[engram-graft-assessment.md](engram-graft-assessment.md) ("the assessment") and
[engram-relaunch-proposal.md](engram-relaunch-proposal.md) ("the proposal").*

*Provenance. Every number comes from a script that was run (paths in §7), on CPU in fp32 unless noted,
with torch 2.15 and transformers 5.22. The Gemini files were imported read-only and never edited. To
construct the module on CPU, the scripts patch `Tensor.pin_memory` to an identity. `S` below means the
session scratchpad, `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad`.
None of this was run on a GPU, so every latency and VRAM figure is arithmetic.*

## 1. TL;DR

- **The sound parts.**
  - Zero-init `v_proj` makes the graft a bit-exact identity: max |Δlogit| = 0.0 on a tiny Qwen2 and on
    Qwen2.5-0.5B.
  - The Stage-2 ridge algebra is internally consistent: realised fraction M/(M+gK), matching to 3
    decimals.
  - Id-keyed host prefetch and "edits re-fire when the symbol is regenerated" are both correct ideas.
- **Two fatal bugs.**
  - **Under KV-cache decoding the memory never fires.** The hasher sees only the new token and pads it
    with id 0 (which is the real token `'!'`), and the conv keeps no state. Incremental-with-graft vs
    incremental-without-graft is **exactly 0.0**, while the full-forward graft effect is 1.03 logits.
    An edit that moves full-forward logits by 0.51–0.82 moves cached logits by 0.0. C3 is false for any
    real generation.
  - **The conv init is wrong.** `nn.init.dirac_(self.short_conv.weight)` on a depthwise `[d,1,4]`
    weight sets 1 of 3,584 entries, at `[0,0,2]`, which is a one-token delay on channel 0. Combined with
    `v_proj=0`, C1's warmup sits on a closed saddle: after 30 Adam steps only `v_proj` row 0 and conv
    channel 0 are non-zero. Even with `groups=d` it is a delay, not an identity.
- **C2 cannot hold.** Stage 2 solves each n-gram order against its own slice of a *concatenated* δ\*,
  so `order_penalties` only shrinks the 2-gram half (0.922 realised vs 0.992 for 3-grams). It cannot
  move any norm into 3-gram rows. `bucket_counts` counts edits, not corpus frequency.
- **The tables cannot be trained.** They are a plain dict, not Parameters, and the gather runs under
  `@torch.no_grad`. In bf16, 100 % of Phase-2 table updates round to zero.
- **The core premise of C4/C6/C8 is false on this model.** In frozen Qwen2.5-0.5B the in-context fact
  first reaches the continuation at the output of block 20 of 24 (teacher logit-lens P(new) 0.069,
  rising to 0.454 at the end).
  - Patching the teacher's *exact* states into the student after block 6 gives window P(new) 0.012
    (baseline 0.002). That is the global optimum of C8's objective, and it installs nothing.
  - Gemini's subgraph edit solver reaches top-1(new) = 0/12 at k ∈ {6, 12, 18}.
- **The early site itself is not dead.** On the same 12 facts, a free δ ∈ R^896 at the layer-2 output,
  optimised for final-layer NLL, reaches rewrite efficacy 1.000 (P(new) 0.95–0.99). Held-out
  CounterFact paraphrase efficacy is 0.875–0.917, against a 0.208 base rate.
  - The cost is specificity: on same-subject other-relation prompts, top-1 = new in ~21–25 %,
    concentrated in `"{s} is a"`.
  - What fails is Gemini's write path and objective, not layer 2.
- **For the relaunch.** Keep the hash key, presence bit and two-stage write. Drop the trained gate/conv,
  the subgraph calibration and the subgraph solver. The surviving design is the proposal's Track 2: an
  exact-key, raw-d δ\* row solved with a full-depth objective. §6 gives the next CPU experiment and its
  kill criteria.

## 2. What the code gets right

- **The identity at init is real.** `nn.init.zeros_(self.v_proj.weight)` gives h + conv(α·0) = h
  bit-exactly in fp32. The forward hook's non-tuple branch matches transformers 5.x decoder layers,
  which return a bare tensor (`mechanics/t_abc.out`: `max |diff| tiny: 0.0`, `Qwen max |diff|: 0.0`).
  Two caveats:
  - The comment credits this to "STAPLE Zero-Initialization". STAPLE actually zero-inits its FFN-internal
    token tables (2610.12599 §2.2); zero-initialising the output projection is the BadEngram/LoRA-B
    pattern.
  - The constructor defaults (`dtype=bfloat16`, `device='cuda'`) crash with an fp32 backbone:
    `expected m1 and m2 to have the same dtype`.
- **The Stage-2 closed form is consistent with its own forward model.**
  - `A[m,u]=count/K` is the right design matrix for a mean-over-K-heads read.
  - The realised write equals A·ΔE.
  - The measured fraction matches M/(M+gK) for penalty scales ×1 to ×1000 (`engramedit-solver/a_order_allocation.out`).
  - The Stage 2 in `subgraph_edit_solver.py` is the same code.
- **The e-space δ is defensible.** EngramEdit's own gated-case extension (App. A.3.2, Eq. 26–28) also
  updates the concatenated-embedding space. So putting δ ∈ R^r on e is a reparameterisation, not an
  error. The gated case has no experiments in that paper, though.
- **The host-prefetch idea is sound.** Deterministic, id-keyed prefetch from host DRAM is Engram §2.5.
  The implementation does not achieve it (§3, M10).
- **Re-firing on regenerated symbols is correct for a full re-forward.** A +3.0 write into a symbol's
  true n-gram rows changes full-forward logits by 0.82, and EngramEdit C.1.2 describes the same
  mechanism. It breaks under KV caching (§3, #1).
- **Hash quality is adequate at small scale.**
  - Per-head collision rates on 281k real Qwen tokens: 4.4 % / 8.1 %, against 4.7 % / 8.2 % for an
    ideal random hash.
  - No n-gram collides on all heads.
  - On 3M tokens, the sharing probabilities match a Poisson model (`engramedit-solver/e_collisions.out`).
- **The smaller pieces work.**
  - Single bf16 EngramEdit writes survive: about 3e-3 per coordinate, cos 1.0.
  - `_prefetched_e.requires_grad_(True)` produces a populated leaf grad.
  - The L0..6 subgraph slice reproduces HF hidden states exactly under sdpa.
  - CLAIMS.md itself flags that Phase 1 sums attention plus FFN.

## 3. Bugs

Ordered by severity. Where four lenses found the same defect, it appears once.

| # | file:function | Bug (quoted) | Effect (measured) | Fix |
|---|---|---|---|---|
| 1 | `engram_graft.py`: `Layer2EngramGraft._on_model_start`, `hash_sequence` | `F.pad(ids_cpu, (n - 1, 0), value=0)` runs on the decode-step `input_ids` of shape `(1,1)`. There is no token history and no conv state. | Decode 2-gram == `hash((0,tok))`. Cached vs full re-forward logits differ by 0.27, even with a correct current-token tap. HF `generate` picks a different greedy sequence. An edit gives 0.51 full vs **0.0** cached. **Fatal, and silent.** | Keep a per-sequence buffer of the last (n_max−1) ids and the last (k−1) gated values beside `past_key_values`. Add a regression test that cached and full logits agree to 1e-5. |
| 2 | `engram_graft.py`: `GraftedEngramModule.__init__` l.110 | `nn.init.dirac_(self.short_conv.weight)` is called with the default groups=1 on a depthwise `[d,1,4]` weight. | Only `[0,0,2]` is set: channel 0, delayed one token. With `v_proj=0` this is a closed saddle: after 30 steps, only `v_proj` row 0 and conv channel 0 are non-zero. Stage-1 δ reaches residual channel 0 at q+1 only, so logits at q get **0** gradient (vs 1.48e2 with a causal identity). **Fatal** for C1 as written. Phase 1's orthogonal re-init of `v_proj` escapes the saddle (64/64 channels after 30 steps), but the delay tap still dominates (mean\|w\| 0.0206 vs ~0.005). | `w.zero_(); w[:,0,-1]=1`, or Engram's `value + short_conv(value)` (Eq. 5: RMSNorm, then conv with dilation 3, then SiLU, plus the residual). |
| 3 | `engram_graft.py`: `forward_fusion` l.159–164 | The conv has no identity path. Even `dirac_(…, groups=d)` puts its tap at k//2=2 under a left pad of 3. | It is a one-token delay, so memory for the n-gram ending at t reaches the residual at t+1. With that init, incremental graft is still 0.0 vs 12.8 full-forward. **Major.** | As #2. |
| 4 | `engram_graft.py`: `prefetch_from_host` (`@torch.no_grad`), `cpu_tables` as a dict | The tables are not Parameters or buffers. | They are absent from `parameters()` and `state_dict()`. C1's "cpu_tables trainable" cannot happen in Phase 1, and a checkpoint silently drops ~2 GB. **Major.** | Use a sparse `nn.Embedding` or an explicit row optimiser, and persist the tables. |
| 5 | `subgraph_calibrator.py` l.217–228 | Sparse SGD on bf16 tables. | 100 % of non-zero updates round to zero: median about 2e-10, against a bf16 spacing of 6.1e-5 at \|w\|≈0.0135. With fp32 tables the change is < 1e-5 after 60 steps. **Major.** | fp32 master rows, or stochastic rounding, plus sparse Adam. |
| 6 | `engram_graft.py` l.83–86, 309; `subgraph_edit_solver.py` l.175, 184 | `bucket_counts = torch.ones(...)`, then `+= 1.0` only in Stage 2. | "freq" counts edits. On a first edit every bucket of an order gets the same factor, 1.693. Re-edits shrink the write slightly (0.922 → 0.872 after 6). **Major** (it is C2's premise). | Fill a frozen per-order corpus percentile and use EngramEdit Eq. 29, w = min(η_n + min(1+9π,10) − 1, 64). |
| 7 | `Layer2EngramGraft` l.177–191; calibrator l.86–89 | The fusion is applied to the *output* of `layers[2]`. | Three blocks have run before the memory is added. Engram adds before attention in its layer, after one round of attention (its Table 5 says `[2,15]` while the demo uses `[1,15]`). Engram reports that efficacy degrades with depth. **Major** (fidelity). | Use a pre-hook on `layers[1]`, and keep L3 as a separate arm. |
| 8 | `EngramEditSolver.apply_edit` | When `v_proj=0`, Stage 1 gets zero gradient. | Before any warmup the edit is a strict no-op: `table max change {2: 0.0, 3: 0.0}`, and NLL 6.9469 pre = returned = post. There is no warning. Minor, but a trap. | Assert ‖v_proj‖>0 and ‖∇δ‖>0 at step 0. |
| 9 | `apply_edit` l.311 | `return float(nll_loss.item())` | This is the Stage-1 NLL computed with the live δ before the last step. The table is never re-checked. At pen₂=1000 it reports 6.8785, while the stored table gives 6.9059 (pre 6.9430): ~42 % of the gain is gone. Minor. | Re-run the prompts with the table only, and return the realised NLL and the realised/target ratio. |
| 10 | `hash_sequence` l.39 | The pad id is 0, which in Qwen is `'!'` (the real pad is 151643). | Position-0 n-grams and every decode n-gram alias `'!'`-prefixed n-grams. Minor. | Use an out-of-vocabulary sentinel id. |
| 11 | hasher, conv, solver l.239 | `attention_mask` is ignored. | Left padding corrupts real positions when the pad id ≠ 0 or the conv has past taps (up to 0.22 logits). Right-padded solver batches are exactly fine. Minor. | Map masked n-grams to a null row and zero `gated_v` at pad positions. |
| 12 | `__init__` l.80 `.pin_memory()`; `prefetch_from_host` l.128–133 | Pinning is done on the table, not on the gathered tensor. | Crashes without an accelerator. `e_cpu` is a fresh pageable tensor, so `non_blocking` cannot overlap (PyTorch semantics; unmeasured, no GPU). Minor. | Gather into a preallocated pinned buffer and copy on a side stream with an event. |
| 13 | `MultiHeadNgramHasher.__init__` l.21 | `torch.manual_seed(42)` | Overrides the caller's global RNG seed. Minor. | Use a local `torch.Generator`. |
| 14 | `_on_model_start` l.181–184 | Prefetch runs only if the root module receives `input_ids`. | The `inputs_embeds` path raises `TypeError … NoneType`. Calling the inner model reuses stale `e`. Minor. | Compute `e` inside the layer hook from an explicit id context. |
| 15 | calibrator l.115 | `nn.init.orthogonal_` on a bf16 weight. | CPU raises `"geqrf_cpu" not implemented for 'BFloat16'` (CUDA untested). Minor. | Init in fp32, then copy. |
| 16 | `_run_subgraph`, Phase-2 and C8 loops | Decoder layers are called with no `attention_mask`. | Exact under sdpa. Under eager the attention is bidirectional (rel. error 0.998, future leak 0.75). The scalar `prefix_len` (C8 l.39) forbids varying paraphrase lengths. Minor, latent. | Pass an explicit causal mask. |
| 17 | `engram_graft.py` l.80, 308 | bf16 tables. | Updates below ~6e-5 per coordinate are lost: 1,000 × 1e-5 moved an entry by 0.0. Single edits survive. Minor. | fp32 master tables. |

The following come from code reading only and were not run:
- `apply_edit` sets `requires_grad_(False)` on every engram parameter (l.218–221) and never restores
  it, so later training silently trains nothing.
- The teacher runs with `use_engram=False` while the student has it on.
- `token_cos` uses raw states (l.137) while the MSE uses LN'd ones.
- AdamW is re-created, and Phase 1 re-randomises `v_proj`, on every call.

*Wrong readings, corrected.*
- **Conv fix.** The mechanics audit said "follow the reference … conv zero-initialised". Engram's
  *paper* does zero-init the conv (Table 5), but the demo `ShortConv` uses the default init plus the
  identity path. What matters in both is the identity residual.
- **Padding.** M9 first claimed the solver mishandles padding. That holds only for left-padded
  batches.

## 4. Claims C1–C9 against the evidence

| Claim | Verdict | Evidence |
|---|---|---|
| **Architecture prose:** "rolling polynomial hash over normalized … K=8 heads" | overclaimed | The code is linear Σpᵢxᵢ mod an even 2,000,000. Bucket parity equals parity(Σ ids) in every head, and collisions are translation-invariant. Only 1 of 20 "primes" is prime. Head 2 of the 3-gram hash collides whenever the first ids differ by 80,000. Ids are raw: Engram-style compression would cut Qwen's vocab by 29.2 %, and `'Paris'`/`' Paris'`/`' paris'` are 3 ids. The default is K=4. Engram instead uses multiplicative-XOR, a prime modulus per head, and concatenates heads (Gemini averages them). |
| **C1:** STAPLE-style warmup, ~50–100M tokens, backbone 100 % frozen, tables trainable | contradicted as attributed; untested as a hypothesis | STAPLE zero-inits FFN-internal tables, freezes only the FFN up-projections during warmup, and the rest of the backbone keeps training (~100B + 100B tokens on 1B). Engram trains jointly on 262B tokens. The closest frozen grafts are BadEngram (an attack, no utility metric) and Engram Adapter (residual ~0.08 % of the hidden-state norm). In the code itself the tables cannot train (#4, #5) and the conv/`v_proj` start on a saddle (#2). |
| **C2:** {2:10, 3:1} "mathematically forces 85–90 % of δ\* into rare 3-gram buckets" | contradicted | With `torch.split(delta_star, r_per_order)`, each order's ridge is solved on its own slice. At pen {2:10}, realised fraction is 0.922 (2-gram) and 0.992 (3-gram), and the 3-gram write is identical across penalties (‖d₃‖=0.06678 at pen₂ ∈ {1, 10, 1000}). A share near 1 is reached only by deleting the 2-gram half: at pen₂=1000, 3-gram share 0.987 with total realised 0.557. With equal slice norms the share is 0.54–0.81, and 85 % would need Stage 1 to have already put 57–83 % of δ\*'s energy into the 3-gram slice. Under Qwen BPE `SpineGraph.Step()` → `['Sp','ine','Graph','.Step','()']`, so the claimed n-grams do not exist, and the shared 2-gram is `('Graph','.Step')`. |
| **C3:** edits fire mid-CoT on the next forward step | contradicted as coded; mechanism supported only natively | KV-cache decode gives exactly 0.0 (#1). EngramEdit shows the mechanism only in a natively trained model, at low absolute accuracy (MQuAKE-CoT 36.9 / 20.1 / 15.1 %). If fixed, the edit would also fire whenever the subject appears: the hash hit rate on other-relation templates is 0.97 (2-gram) and 0.92 (3-gram) (§5). |
| **C4:** L2–6 do n-gram detokenisation, and L7–64 are never needed | contradicted on Qwen2.5-0.5B; untested at 27B | The fact appears at blocks 20–23 of 24 (§5). Engram's "deeper early layers" findings are properties of co-trained backbones; STAPLE's ablated model "is not equivalent to the original dense seed". |
| **C5:** Phase 1 "immediately aligns W_V with the FFN write subspace" | contradicted | Held-out cos is 0.29–0.31, below a constant mean direction (0.36), a bigram table (0.48) and ridge from h₂ (0.71). The output is 0.06–0.11 % of the target norm, and NLL is unchanged (4.57936 vs 4.5794). The target is attention plus FFN (‖attn‖/‖Δ‖ = 0.735). Adding the exact 0.25× target while L3..L6 still run costs +0.075 nats. |
| **C6:** context-drop distillation teaches the gate what full attention over the prefix would produce | overclaimed | Student inputs depend only on the continuation. Position 0 is an attention sink (128× the median norm) carrying ~14 % of the loss, and positions 0–15 carry ~41 %. A content-free target is already 43–50 % of the way to the true teacher. The verbatim Phase 2 learned nothing in 20–60 steps; at 20× lr it gained 5.7 % in 80 steps. Long runs were not tested. |
| **C7:** the L0..6 slice is ~3.2 GB BF16, "< 4 GB" | math error | For a 64-layer d=5120 27B: 2.78B params, 5.6 GB BF16 (7.1 GB with embeddings). A Qwen3-32B-like shape gives 8.4 GB with embeddings. 3.2 GB corresponds to about 4 layers. The ~8× weight saving is real. |
| **C8:** L6 matching to a spec-prefixed teacher installs the fact | contradicted | The oracle patch at block 6 gives window P(new) 0.012 and top-1 0.02 (§5). The verbatim solver is inert: window loss 0.1334 → 0.1333, P(new) unchanged. The clamp is not the cause: with ρ=1e4, lr 0.5 and 100 steps the result is the same. EngramEdit reads NLL at the last block and adds a KL term. |
| **C9:** 80 layer-passes in ~35–50 ms, 3.4 GB | overclaimed, unmeasured | The count leaves out the activation backward. The real cost is ~37.5–39.9 TFLOP, which is the H100 roofline floor at 40 ms, ~75–100 ms at 40–50 % MFU, and well over 100 ms on A100, L40S or 4090. The 3.4 GB figure contradicts C7's corrected 5.6–7 GB. |

Other design errors, all upheld:
- **Shared buckets instead of side vectors.** EngramEdit stores edits in exact-n-gram side vectors and
  leaves the pretrained tables fixed (B.4.3). Gemini adds into shared hashed rows, so every co-tenant
  receives ~1/K of the write per head it shares. On a 3M-token corpus, 48 % of distinct 2-grams and
  79 % of distinct 3-grams share at least one head-bucket.
- **Shared rows accumulate.** Edits sharing a 2-gram *add up* in those rows (the write is `+=`, not an
  overwrite), and past edits are never re-verified.
- **Stage 1 is not EngramEdit's.**
  - There is no KL term on `"s is a"` (paper: λ_KL=0.0625), no prefix contexts and no early stop.
  - The clamp is 0.4·‖e_q‖ with a 0.1 floor. At the default init ‖e_q‖≈0.22, so the floor binds and
    the clamp is absolute.
  - All of the paper's results are on LongCat-Flash-Lite, whose n-gram memory was pretrained natively
    with additive fusion, and whose δ is applied **before the first decoder block**.
- **The write is relation-blind**, as EngramEdit's is. EngramEdit's specificity sets never test the
  same subject under another relation.

*Wrong readings, corrected.*
- **SUM vs concat.** The solver-math audit attributed the 85–90 % to EngramEdit "switching to SUM"
  fusion for the gated case. In fact App. A.3.2 *concatenates* and solves jointly through a Jacobian.
  Reallocation there depends on overlap between W_V's per-order column blocks: the 3-gram share is
  0.400 for independent blocks and 0.909 for a shared column space (`verify-solver-math/v4_hspace_joint_solve.out`).
  The paper states no 85–90 % figure.
- **"Overwrites".** An earlier reading said a later edit *overwrites* a shared 2-gram slice. The write
  is additive.
- **"80× tighter".** The "80× tighter clamp than the paper" comparison mixes spaces (e vs h), so it was
  dropped.

## 5. The experiment: does a layer-2 write carry a fact, and does Gemini's objective find it?

**Setup.** Qwen2.5-0.5B (24 blocks, d=896), frozen, fp32, CPU, seed 0.
- **Facts:** 12 real CounterFact edits (`azhx/counterfact` test split). Both targets are single tokens;
  there are 9 relations with ≤3 facts each. A validity gate (the base must prefer `target_true` on the
  rewrite prompt) rejected 6 candidates. 30 facts were planned; OOM kills and CPU contention capped the
  run at 12.
- **Prompts:**
  - optimisation: the rewrite prompt + 3 relation templates;
  - held-out: the 2 CounterFact paraphrases + 1 extra template;
  - leakage: 3 same-subject other-relation templates.
- **Arms:**
  - **A1:** a free δ ∈ R^896 added to the output of block 2 at the last subject token. Optimised for
    final NLL with Adam (lr 0.1, 30 steps), with ‖δ‖ ≤ ρ‖h₂‖, where ‖h₂‖≈10.3.
  - **A2:** the same δ trained with Gemini's C8 loss, copied exactly (LN-MSE + 0.5(1−cos), anchor
    weight 2.0, exp(−0.25t), window q..q+8, reg 0.005), against the teacher's block-k states, where
    teacher = `"<rewrite> <new>.\n" + prompt`.
  - **A3:** that same teacher in context.
- **Sanity check:** the manual layer runner matches the HF forward to 1.05e-5.

**Results** (mean over facts, n=12 unless noted). "eff" means P(new) > P(old).

| Arm | rewrite eff / top1 / P(new) | held-out eff / P(new) | CF-paraphrase eff (corrected) | leak KL / top1=new |
|---|---|---|---|---|
| A0 no edit | 0.000 / 0.000 / 0.002 | 0.167 / 0.005 | 0.208 | 0 / 0 |
| A1 L2, ρ=0.5 | **1.000 / 1.000 / 0.994** | 0.903 / 0.392 | **0.875** | 1.97 / 0.250 |
| A1 L2, ρ=1 | 1.000 / 1.000 / 0.947 | 0.875 / 0.365 | 0.875 | 1.74 / 0.194 |
| A1 L2, ρ=4 | 1.000 / 0.917 / 0.906 | 0.903 / 0.372 | 0.917 | 1.87 / 0.194 |
| A1 L12, ρ=1 | 1.000 / 1.000 / 0.967 | 0.833 / 0.316 | 0.792 | 1.29 / 0.167 |
| A2 k=6, ρ=4 | 0.250\* / **0.000** / 0.003 | 0.222 / 0.007 | 0.250 | 0.095 / 0 |
| A2 k=12, ρ=4 | 0.167\* / 0.000 / 0.004 | 0.194 / 0.007 | 0.208 | 0.138 / 0 |
| A2 k=18, ρ=4 | 0.250\* / 0.000 / 0.007 | 0.278 / 0.009 | 0.292 | 0.215 / 0 |
| A2, ρ ∈ {0.5, 1}, all k (facts 1–4 only) | 0.000 / 0.000 / ≤0.008 | ≤0.083 / ≤0.014 | ≤0.125 | ≤0.125 / 0 |
| A3 in-context ceiling | 1.000 / 1.000 / 0.647 | 0.972 / 0.336 | 0.958 | 0.512 / 0.028 |

\*Every A2 "flip" has P(new) ≤ 0.017. It comes from P(old) falling, not from the new object rising.

*Correction:* the first report's CF-paraphrase column read 0.833 / 0.833 / 0.875 / 0.708 for the A1
rows. An index bug (`held_cf=range(no, no+len(held)-1)`) dropped one paraphrase for facts 1, 8 and 10,
so the corrected values are in the table above (`verify-experiment/fix_heldcf.log`).

**Ceiling of the C8 objective.** Here the teacher's exact block-k states are patched into the student,
averaged over 12 facts × 4 prompts:

| k (block output) | window P(new) | window eff | window top1 | anchor-only P(new) |
|---|---|---|---|---|
| 6 | 0.012 | 0.29 | 0.02 | 0.007 |
| 12 | 0.013 | 0.29 | 0.02 | 0.007 |
| 18 | 0.118 | 0.54 | 0.31 | 0.006 |

The A2 loss does fall (k=6: 0.246 → 0.112), so the optimiser works; the target carries no fact. The
skeptic then removed both the reg and the clamp and ran 150 steps on 3 facts. The loss fell to
0.024–0.048 with ‖δ‖ 16–24, and rewrite top-1 was still 0/3 at k=6 and k=18 (k=6 P(new) ≤ 0.007). The
A2 δ is nearly orthogonal to the A1 δ (cosine 0.04–0.14).

At k=18, "objective not optimiser" is overstated. The oracle there is 0.118 while A2 reaches only 0.007.

**Per-layer picture.** The table shows ‖LN(h_teacher) − LN(h_student)‖ at the last token, plus
logit-lens P(new). Block indices are 0-based block outputs.

| block | gap: base | gap: A1 ρ=1 | gap: A2 k=6 | gap: A2 k=18 | lens P(new): teacher | lens P(new): A1 |
|---|---|---|---|---|---|---|
| 2 | 5.65 | 5.65 | 5.65 | 5.65 | 0.000 | 0.000 |
| 6 | 9.42 | 10.76 | **8.32** | 9.02 | 0.000 | 0.000 |
| 12 | 12.28 | 15.17 | 11.55 | 11.36 | 0.000 | 0.000 |
| 18 | 11.19 | 16.05 | 10.44 | 9.70 | 0.001 | 0.001 |
| 20 | 11.96 | 19.24 | 11.14 | 10.33 | 0.069 | 0.157 |
| 21 | 12.30 | 20.09 | 11.31 | 10.45 | 0.219 | 0.487 |
| 22 | 13.13 | 19.52 | 12.08 | 11.25 | 0.355 | 0.599 |
| 23 (final) | 13.18 | 25.07 | 12.19 | 11.43 | 0.454 | 0.961 |

- **When the fact appears.** In context, it reaches the continuation only at blocks 20–23. The
  calibration lens found the same thing independently with a different teacher format (88–92 % depth).
- **The working edit takes another route.** A1 *widens* the gap to the teacher at every layer, yet
  reaches P(new) 0.96. Matching the teacher's shallow states is neither necessary nor sufficient.
- **A2 barely helps.** It narrows the block-6 gap by only 12 % (9.42 → 8.32).

*Correction:* the first report labelled these rows one layer early (index i = output of block i−1).

**Would a hash key fire?** The table shows whether the 2-gram or 3-gram ending at the last subject token
equals one seen in the optimisation prompts:

| prompts | 2-gram hit | 3-gram hit |
|---|---|---|
| held-out | 0.94 | 0.76 |
| CF paraphrases (corrected) | 0.917 | 0.750 |
| same-subject other-relation | **0.97** | **0.92** |

**Decode check.** Gemini's real `GraftedEngramModule` and `Layer2EngramGraft` on Qwen block 2, 20
greedy tokens, comparing incremental KV-cache decoding with a full re-forward:

| graft state | max\|incr − full\| per step | full-forward graft effect |
|---|---|---|
| none | 3.5e-5 – 1.1e-4 (noise floor) | — |
| `v_proj=0` | identical to none | 0 |
| `v_proj~N(0,0.02)` | 4.9e-3 | 4.9e-3 |
| `v_proj~N(0,2)` | 1.03; incr-graft vs incr-no-graft **= 0.0** | 1.03 |
| dense random conv (stand-in for a trained conv) | 12.8 | 11.9 |
| skeptic: `dirac_(groups=896)` | incr-graft vs incr-no-graft = 0.0 | 12.8 |
| skeptic: current-token tap (index 3) | fires, but off by 17.1 (wrong hashes) | — |

The exact 0.0 does not depend on the seed. The effect sizes do: the skeptic got 0.30 where the report
got 1.03.

**Interpretation.**
1. **C8 is refuted at k=6 and k=12.** Even an exact match to the teacher at those blocks carries
   P(new) ≈ 0.01, because the in-context fact moves into the continuation late, by attention from the
   prefix. A student with no prefix cannot be taught it through early-layer states.
2. **C4 is false on this model.**
3. **The layer-2 residual site can carry a fact** when δ is optimised for the output. A1 matches the
   in-context ceiling on efficacy and beats it on P(new). This does not contradict the repo's ~0.7
   single-site ceiling, which was measured for a *learned, retrieved, generalising* tap (key-encoding
   §3.7–3.11). A1 is a per-fact δ with oracle placement.
4. **The price is specificity.** On the 5-fact re-run, after excluding one leak prompt that is really
   the same relation (`"Clickteam was founded in"`), A1 leak top-1 = new is 0.21 and KL 1.54 nats.
   Nearly all of it is on `"{s} is a"`; `"The most famous thing about {s} is"` has KL ≤ 0.18.
5. **Gemini's write is not a superset of A1's δ.** It is shaped differently: rank ≤ r through `v_proj`,
   a gate below 1, and a conv that spreads e_q over q..q+3, or only onto q+1, channel 0, under the
   shipped init.

*Correction:* the first report called Gemini's write "a strict subset" of A1's δ. That is the wrong
relation, hence point 5.

**Caveats.**
- One 0.5B model and one seed, with n=12. The A2 ρ grid ran on 4 facts only.
- 4 of the 24 CF paraphrases are, from the subject onward, word-for-word copies of optimisation
  templates (held-out A1 efficacy: 0.83 on novel prompts vs 1.00 on the overlapping ones, from the
  5-fact re-run).
- "Neutral KL = 0" holds by construction (no anchor means no δ), and the position-id variant test is
  vacuous under relative RoPE. Both were dropped as evidence.
- A1 is not converged. P(new) *falls* as ρ grows (0.994 / 0.947 / 0.906), so the A1 numbers are a
  lower bound.
- The A2 clamp did bind on one fact (fact 3, ρ=0.5), but the reg term is the real constraint.
- The base already prefers the new object on 21 % of CF paraphrases, so read P(new) and top-1 alongside
  efficacy.
- Stage 2 and a trained C1 module were not tested.

## 6. What this means for the relaunch

**What survives from Gemini's plan.** These are the same components the assessment (§3) and the
proposal (§5) kept:
- the **hashed id key**: a free presence bit and CPU-side prefetch with no encoder forward;
- the **two-stage write**: solve δ\* with the base frozen, then allocate in closed form;
- **zero-init** for a byte-exact no-op.

**What does not survive:**
- the trained gate/conv fusion on a frozen base (C1: no precedent, and the code cannot train it);
- the subgraph calibration and the subgraph solver (C4–C6, C8: refuted at their optimum on this
  model);
- the order-penalty allocation (C2) and writes into shared hashed buckets;
- the VRAM and latency pitch (C7, C9), which only mattered for the subgraph solver.

**The corrected minimal design** is the proposal's Track 2 ("NeuralDB/SWEA with an exact verified
token-span key"), with three additions from this audit:
1. **Values.** One raw-d δ\* per fact, solved for full-depth final NLL. Use EngramEdit's KL on
   `"{s} is a"` plus SSOR-train templates (this experiment shows `"is a"` is where the leak lives), and
   ρ relative to ‖h_ℓ‖ with no absolute floor.
2. **Keys.** An exact hash of the last ≤3 *normalised* subject ids, with an out-of-vocabulary pad,
   verified against the stored span. Keep a per-sequence id history so the key at decode time equals
   the key at prefill. Keep edits in exact side vectors, never in shared buckets.
3. **Injection.** Add δ\* at the anchor position itself, with no conv and no learned gate. If a gate is
   ever added (proposal E3), use Engram's `v + conv(v)` form, a pre-hook on `layers[1]`, and fp32
   tables.

**Next experiment (X1, CPU, before proposal R3).** Extend `run.py` to 30 facts. Add three things:
- **E0-hash:** the A1 δ with the KL/SSOR term, delivered **only on an exact hash hit**, through a
  stateful decode path;
- **sites** ℓ ∈ {2, 8, 16, 20};
- **SSOR-eval templates** disjoint from the training ones, with same-relation prompts excluded.

Report the corrected `held_cf`.

Kill criteria (placeholders set by this note, in the style of the proposal's G-bars):

| Check | Kill if | Consequence |
|---|---|---|
| Decode parity of the fixed path | max\|cached − full\| > 1e-4 over 20 greedy steps | No hashed design ships until it passes |
| L2 with KL/SSOR term | CF-paraphrase eff < 0.80 **or** P(new) < 0.30, **or** SSOR top-1=new > 0.10 **or** mean SSOR KL > 0.5 nats | Early site dead on a frozen base. Move to ℓ ≥ 16 as a value option inside the pointer (proposal G1 fallback) |
| Raw-id key coverage | hash-gated held-out eff more than 0.10 below anchor-oracle A1 eff (hit rate measured here: 0.92 / 0.75) | Raw ids are inadequate. Require the R0 normalised key |
| Every ℓ | CF-paraphrase eff < 0.70 at every ℓ | Site wall holds (assessment E0 kill) |

A surviving arm goes to proposal R3/R4 on Qwen3.5-4B with the G1 bars: generality ≥ 0.82, efficacy
≥ 0.88 at N=441, SSOR ≥ 0.80, and efficacy(1105) − efficacy(137) ≥ −0.05. Arm B (R1/G0) remains the
opponent.

## 7. Reproduce

The scripts ran in the session scratchpad and are now copied, with their outputs, into
[`tools/engram_audit/`](../../tools/engram_audit/README.md) (same layout; `$S` below maps onto that folder —
the README gives the symlink that lets them re-run unchanged).
Each `.py` writes the `.out`/`.log` of the same name. Run from the script's directory with
`PY=$S/venv/bin/python` and `torch.set_num_threads(2)`, or 3 threads for the experiment. Model:
`Qwen/Qwen2.5-0.5B`, snapshot `060db649…`.

```bash
S=/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad
PY=$S/venv/bin/python
# Mechanics (§3 #1-#4, #10-#14, #17)
cd $S/audit/mechanics && for t in t_abc t_d_kvcache t_e_hash t_fg t_rng; do $PY $t.py; done
cd $S/audit/verify-mechanics && $PY v1.py && $PY v2.py
# Solver maths (§4 C2, Stage 1/2)
cd $S/audit/engramedit-solver && for t in a_order_allocation d_stage1_capacity e_collisions g_rows_and_sequential h_conv_init; do $PY $t.py; done
cd $S/audit/verify-solver-math && for t in v1_real_apply_edit v2_conv_calib v3_tokenize v4_hspace_joint_solve v5_hash v6_first_target_and_clamp; do $PY $t.py; done
# Calibration (C4-C9)
cd $S/audit/calibration && $PY s0_subgraph_fidelity.py && $PY s_a_phase1.py 120 1 && $PY s_a_phase1.py 120 4 \
  && $PY s_a3_attn_vs_ffn.py && $PY s_b_positions.py && $PY s_b2_decompose.py && $PY s_b3_phase2_train.py 60 \
  && $PY s_b4_phase2_highlr.py 80 1e-2 && $PY s_c_table_sgd.py && $PY s_d_depth.py && $PY s_d2_patch_all.py \
  && $PY s_e_arith.py && $PY s_f_c8_run.py
cd $S/audit/verify-calibration && $PY v1_patch.py && $PY v2_mechanics.py && $PY v3_c8_clamp.py
# Sources (paper text under fidelity/src)
cd $S/audit/fidelity && $PY t_mechanics.py && $PY t_tokenizer_and_budget.py
cd $S/audit/verify-sources && $PY v_mech.py && $PY v_tok.py
# Experiment (§5); the 12-fact run was resumed after an OOM, see run.log
cd $S/audit/experiment
MALLOC_ARENA_MAX=2 $PY run.py --n 12 --resume --a2_rhos 4.0 --posoff_rhos ""   # aggregate + decode check
$PY conv_check.py && $PY peek.py results.json
cd $S/audit/verify-experiment && $PY check_data.py && $PY fix_heldcf.py && $PY rerun5.py \
  && $PY analyze_rerun5.py && $PY a2_noreg.py && $PY decode_verify.py
```

The experiment's exact run history: facts 1–3 ran under `run_main_snapshot.py`, fact 4 ran the full
grid, and facts 5–12 ran with `--a2_rhos 4.0 --posoff_rhos ""`. Results are in
`$S/audit/experiment/results.json`.

## 8. Addendum — Gemini's two revisions

After seeing the preliminary findings, Gemini sent two revised implementations. Neither was run end to end;
the checks below are targeted numerical tests (`tools/engram_audit/v2check/`), on synthetic inputs except where
noted.

**Revision 2** (summed orders, joint ridge, stateful hasher, fp32 tables, checkpointed full-depth Stage 1).
- Fixed: the conv init (`weight[:, 0, -1] = 1`), summed fusion with one joint ridge system, fp32 tables,
  corpus counts, a stateful hasher, and Stage 1 on the true final-layer NLL — the last is A1's objective in §5.
- The joint ridge now does move the edit into 3-gram rows (8 paraphrases sharing the anchor n-grams, 4 heads,
  the code's penalty formula): 3-gram share **0.991** with counts (2-gram 1e5, 3-gram 1) and realised fraction
  0.997. But the share is **0.9375** even when both counts are equal: the 15× order penalty does almost all the
  work and the `log1p` frequency term little, so a common trailing 3-gram still takes ~94 % of the edit.
- **New blocker:** `v_proj` was still zero-initialised (the SVD init existed only in prose), so the gradient on
  δ is exactly **0.0** and Stage 1 writes nothing. The prose said the tables start at zero; the code still used
  `randn·0.02`, which with a non-zero `W_V` injects noise at every token. The trust region was relative to the
  random-row norm (‖e_q‖ ≈ 0.15, so δ ≤ 0.1), not to ‖h₂‖. `q_proj`/`k_proj` were never calibrated, so α was a
  context-dependent random scalar (~0.27–0.73), and the conv was a fixed identity.

**Revision 3** (full-width direct addition: zero fp32 tables of width d_model, no gate, no conv, ‖h₂‖-relative
clamp, causal mask in Stage 1, decode-state hook). This is the §6 design minus the exact key, and its Stage 1
is §5's A1. Remaining problems:

| Issue | Measurement | Fix |
|---|---|---|
| **Short subjects don't generalise.** At the last subject token, the 3-gram (and for 1-token subjects the 2-gram) reaches into preceding context, so it differs across prompts, and the order penalty puts most of the edit there. | Realised δ\* on a held-out prompt with new preceding context: subject ≥3 tokens **0.997**; 2 tokens **0.066**; 1 token **0.000** (train-side ≈0.97 in all cases). In Qwen's tokenizer ` Toyota`, ` Apple`, ` Microsoft`, ` BBC`, ` Italy` are 1 token; ` Michael Jordan` is 2. | Key on subject-internal n-grams only, down to the unigram; penalise by corpus frequency, not order. |
| **Dense hashed tables are the wrong structure.** | 2 orders × 2M × d fp32 = **14.3 GB** at d=896, **81.9 GB** at d=5120, almost all zeros. Probability an unrelated token picks up a stray ≈(1/8)·1.9·δ\* from a collision: **1.6 %** at 1k edits, **14.9 %** at 10k, **83 %** at 100k. | An exact sparse dict keyed by the n-gram tuple (no collisions, memory only for edits, an exact presence bit). |
| Causal mask only half applied. | The layers 0–2 cache pass still has no mask; the mask is built in `engram.dtype` (bf16), which (unverified) SDPA rejects against an fp32 query. | Pass the same mask, in the model's dtype, everywhere. |
| Decode state off by default. | `decode_state` is `None` until `reset_decode_state()` is called, which brings back the `(0, tok)` hashing; nothing resets it between `generate` calls; pad id 0 is `'!'` while the real pad is `<|endoftext|>` (151643). | Create and reset the state per `generate` call; use an out-of-vocabulary pad. |
| Relation-blind leakage unaddressed. | No essence/KL term (§5 measured ~21–25 % same-subject other-relation top-1 = new without one). | Add the KL on `"{s} is a"` plus SSOR templates (§6). |
| Equal-length paraphrases only. | The batch is `[M, T]` with no padding support. | Pad with masks. |

The trend across the three versions: each revision accepted every critique, but implemented some of them only in
prose. Treat an acknowledged fix as unverified until it has been run.

## Sources

- Engram — [arXiv 2601.07372](https://arxiv.org/abs/2601.07372), [code](https://github.com/deepseek-ai/Engram) (`engram_demo_v1.py`)
- EngramEdit — [arXiv 2610.10533](https://arxiv.org/abs/2610.10533) (Eq. 4, 7–10, 21, 25–29; App. A.3.2, B.4; Table 4)
- STAPLE — [arXiv 2610.12599](https://arxiv.org/abs/2610.12599)
- BadEngram — [arXiv 2609.13478](https://arxiv.org/abs/2609.13478) (App. D); Engram Adapter — [arXiv 2608.29327](https://arxiv.org/abs/2608.29327)
- CounterFact (`azhx/counterfact` on the HF Hub)

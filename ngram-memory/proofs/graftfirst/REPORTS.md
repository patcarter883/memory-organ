

# REPORT: graft-hosts-local

# Training graft rows and running solves locally on Qwen3.6-35B-A3B and Qwen3.8-27B

**Short answer:** do **Qwen3.6-35B-A3B first**, on a 4-bit checkpoint the fork already serves (`cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, or `pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4`).

- Training runs in our own torch-ROCm process with the serve stopped: the 4-bit weights stay on the cards and each layer is dequantised to bf16 as it runs; the graft rows live in host RAM as a sparse table.
- **Estimated cost per 1M training tokens: 0.09–0.41 GPU-hours** on the 35B, against 0.37–0.96 on the 27B. On the CPU beside a running serve it is **11–35 h** (35B) against **62–205 h** (27B).
- After that, add Qwen3.8-27B (`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`). The serve-side work done for the 35B carries over to it.

All throughput, memory and timing figures are **estimates** unless marked as measured. Scripts and outputs are in `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/train_local/`.

---

## 1. Checkpoints

### What the two hosts look like

From the configs in `scratchpad/models/Qwen/*/config.json`:

| | Qwen3.6-35B-A3B (`qwen3_5_moe`) | Qwen3.8-27B (`qwen3_5`) |
|---|---|---|
| Layers | 40: 30 GDN + 10 full attention (every 4th, starting at index 3) | 64: 48 GDN + 16 full attention |
| Hidden size d | 2048 | 5120 |
| MLP | **256 experts, top-8**, expert width 512, plus a shared expert of width 512 | dense, width 17408 |
| Vocabulary | 248320 | 248320 |
| Layer 1 | GDN (same position as Flash-Next's n-gram module) | GDN |

Parameter counts are exact, from `nm/compute/count.out`:

- **35B-A3B:** 35.95 G total; 32.21 G in routed experts; **2.95 G active per token**, of which the lm_head is 0.51 G.
- **27B:** 27.78 G total; **25.62 G active per token**, of which the lm_head is 1.27 G.

### What exists on Hugging Face

Sizes come from the `huggingface_hub` API (`hub_detail.py` → `hub_detail.out` / `.json`); no weights were downloaded. "Fork" in the right-hand column means the owner's minisgl fork.

| Host | Repo | Format | GB on disk | Does the fork serve or parse it? |
|---|---|---|---|---|
| 35B | Qwen/Qwen3.6-35B-A3B | bf16 | 71.90 | parses (bf16) |
| 35B | Qwen/Qwen3.6-35B-A3B-FP8 (RedHatAI/…-FP8 is the same) | fp8 e4m3 block | 37.46 | parses |
| 35B | nvidia/Qwen3.6-35B-A3B-NVFP4 | modelopt mixed precision | 23.42 | parses (modelopt is rewritten to compressed-tensors) |
| 35B | RedHatAI/…-NVFP4 / unsloth/…-NVFP4 | compressed-tensors NVFP4 / mixed | 25.04 / 26.49 | parses |
| 35B | RedHatAI/…-FP8-dynamic | compressed-tensors fp8 | 39.37 | parses |
| 35B | **cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit** | AWQ int4 g32; only routed experts quantised, GDN/attention/shared expert/lm_head stay F16 | **24.97** | **served: `qwen35b-awq`, the default `MODEL`** (`tools/serve.sh:128-171`; `docker-compose.yml:700,736,918`; `models/qwen3_5_moe.py:8-13`) |
| 35B | **pahajokiconsulting/Qwen3.6-35B-A3B-MXFP4** | compressed-tensors MXFP4 g32, weight-only | **23.12** | **served: `qwen35b-mxfp4`** (`serve.sh:173-200`) |
| 35B | QuantTrio/…-AWQ, palmfuture/…-GPTQ-Int4, btbtyler09/…-GPTQ-8bit | AWQ g128 / GPTQ 4-bit / GPTQ 8-bit | 25.46 / 24.42 / 42.92 | AWQ and GPTQ parse |
| 35B | Intel/…-int4-mixed-AutoRound; amd/…-w4a16-llmcompressor; lokeshe09/…-INT8 | auto-round / CT pack-quantized / CT int8 | 21.49 / 20.32 / 35.77 | auto-round: no |
| 35B | unsloth / bartowski / ggml-org GGUF | GGUF, IQ1_M through BF16 | unsloth: Q4_K_M 22.13, MXFP4_MOE 21.71, Q8_0 36.9 | **no `.gguf` loader** (grep finds none). GGML Q8_0/IQ2_XXS is parsed only inside safetensors (`quant/config.py:315`, `quant/ggml.py`) |
| 27B | Qwen/Qwen3.8-27B | bf16 | 55.56 | parses |
| 27B | Qwen/Qwen3.8-27B-FP8 (unsloth/…-FP8 is the same) | fp8 | 30.87 | parses |
| 27B | nvidia/ and RadixArk/Qwen3.8-27B-NVFP4 | modelopt mixed precision | 21.92 | parses |
| 27B | **sakamakismile/Qwen3.8-27B-MTP-NVFP4** | CT NVFP4 g16; declares NVFP4 activations | **20.56** | **served: `qwen38-27b`, the "default on correctness"** (`serve.sh:481-515`) |
| 27B | unsloth/Qwen3.8-27B-NVFP4 | CT mix of NVFP4 and fp8 W8A8 | 23.42 | served: `qwen38-27b-mixed` (`serve.sh:593`) |
| 27B | cyankiwi/Qwen3.8-27B-AWQ-INT4 | CT int4 g32 | 21.02 | served: `qwen38-27b-int4`, flagged for corruption (`serve.sh:422, 481-495`) |
| 27B | **amd/Qwen3.8-27B-Quark-AWQ-MXFP4** | Quark MXFP4 g32, served W4A16 | **19.80** | **served: `qwen38-27b-mxfp4`, the fastest arm** (`serve.sh:396-420`) |
| 27B | amd/…-Quark-AWQ-INT4-W4A16; RedHatAI/…-INT4 / -NVFP4 / -MXFP4; btbtyler09/…-GPTQ-4bit | Quark int4 / CT formats / GPTQ | 19.51 / 19.45 / 24.69 / 19.83 / 20.98 | parse (Quark is rewritten to compressed-tensors, `quant/config.py:159-300`) |
| 27B | unsloth/…-bnb-4bit | bitsandbytes | 22.34 | no |
| 27B | unsloth / bartowski / ggml-org GGUF | GGUF | unsloth: Q4_K_M 16.46, Q8_0 29.05 | no |

The fork's quant config handles AWQ, GPTQ and compressed-tensors (int, fp8, NVFP4, MXFP4, MXFP8). Modelopt and Quark are rewritten into compressed-tensors (`quant/config.py:34-155, 159-300, 442`). It also has pure-torch reference dequantisers we can reuse: `quant/mxfp4.py:262` and `quant/nvfp4.py:374`.

### Which checkpoint to train against

Train against the weights you serve: that makes row parity a numerics question, as with Flash-Next's P3/P4 checks.

- **35B: cyankiwi AWQ-4bit.**
  - It is the daily-driver default serve.
  - Its whole dense path is F16 and exact, so only the experts need dequantising.
  - The serve runs these experts W4A8 (`MINISGL_MOE_W4A16` defaults to off, `quant/kernels.py:61`). To match that, add optional per-token fp8 fake-quant on the expert inputs.
- **27B: amd Quark MXFP4.**
  - The serve runs it W4A16 (`serve.sh:396-420`), and a bf16 forward over dequantised weights is the same numerics class.
  - It is also the fastest arm: 47.9 tok/s, and 60.8 tok/s with MTP.
- **Not usable on 2×16 GB with activations:** the FP8 checkpoints (31–37 GB) and bf16. Keep bf16 only as a CPU or offload reference.

---

## 2. Training setup in our own torch process

**What the process looks like.**
- torch-ROCm on gfx1201: the fork's image already builds torch nightly for ROCm 7.2 with the gfx1201 binary, plus hipBLASLt (`Dockerfile:31-52`).
- Every job goes through `gpu-lease -n 2` (fork `CLAUDE.md:3-40`).
- The GPUs are card 0, an RX 9070 XT, and card 1, an RX 9070, both 16 GB.

**Frozen host weights.**
- Keep the 4-bit packed weights resident on the cards and dequantise each layer to bf16 just in time inside a small `autograd.Function`.
- That function computes only the input gradient (dX = dY·W), never a weight gradient.
- The dequant traffic is about 10–13 s per 1M tokens in total, which is negligible (`cost_model.out`, `gpu_dequant_s_per_Mtok`).
- Dequant code: reuse `mxfp4.dequant_reference` and `nvfp4.dequant_reference` from the fork; AWQ unpacking is about 30 lines.
- Checks of what runs natively on gfx1201:
  - bitsandbytes 4-bit on gfx1201: only one community report says it works; unverified ([blog](https://vachsark-homepage.pages.dev/blog/qlora-rdna4), [HF docs](https://huggingface.co/docs/bitsandbytes/v0.50.0/installation)).
  - `torch._scaled_mm` fp8 on gfx120x: not confirmed. The hipBLASLt MX path is gfx950-only, and vLLM falls back to an upcast on RDNA4 ([pytorch/ao#2066](https://github.com/pytorch/ao/pull/2066), [vLLM forum](https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900)).
  - The fork's own HIP kernels are forward-only, with no autograd anywhere in `python/minisgl`.
  - So dequantise-then-bf16-GEMM is the only path known to work.

**MoE on the 35B.**
- HF's `Qwen3_5MoeExperts` (`modeling_qwen3_5_moe.py:831-840`) takes stacked expert weights of shape [E, 2I, d] and has a `grouped_mm` fallback with a backward (`integrations/moe.py:189-262`).
- That fallback also computes a weight gradient; skip it when the weight does not require one.
- Dequantise experts in chunks of 64, with one buffer plus one prefetch (about 1.7 GB).

**GDN.**
- HF falls back to a pure-torch chunked delta rule whenever flash-linear-attention is missing.
- flash-linear-attention says it supports ROCm through Triton, but I found no RDNA4 validation ([fla on PyPI](https://pypi.org/project/flash-linear-attention/), [AMD Primus GDN guide on MI300X](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/hybrid-models/gdn-guide.html)). Test it on the box first.
- Measured on one CPU thread (`gdn_cpu_bench.out`): the pure-torch delta-rule core costs **2.8× the layer's own projections on the 35B and 2.1× on the 27B**.
- **Across a whole step, the GDN core adds +84% to the 35B and +40% to the 27B on CPU.** On the 35B the recurrence, not the experts, is the largest cost outside the GEMMs.

**Activation checkpointing and schedule.**
- Checkpoint every layer. The cost is 6 × active params × tokens: forward, recompute, and input-gradient backward. There are no weight-gradient GEMMs, and layer 0 needs no backward.
- Pipeline split across the two cards weighted about 57:43 by card speed, with a 1F1B schedule and 2048-token microbatches, accumulating 16k tokens per optimiser step.
- Compute logits and cross-entropy in 512-position chunks. A full logits tensor at 4096 tokens would be 4 GB in fp32.
- Keep `embed_tokens` on the CPU and gather rows from it.

**The trainable rows, measured** (`ngram_census.out`, with the Flash-Next tokenizer).

The fork's `python/` tree is 1.44M tokens. It has 300k distinct bigrams and 672k distinct trigrams, so **7.77M rows across 16 heads, about 5.4 rows per token**. Growth is sublinear.

| Host | Row width | Trainable params | bf16 rows | fp32 master + Adam | bf16 rows + row-wise Adagrad |
|---|---|---|---|---|---|
| 35B-A3B | 128 | 0.99 G | 1.99 GB | 11.9 GB | **2.0 GB** |
| 27B | 320 | 2.49 G | 4.97 GB | 29.9 GB | **5.0 GB** |

- Keep the row store in host RAM as numpy: sorted int64 row ids plus bf16 values.
- Each step: hash on the CPU (0.21M tokens/s measured, `P/redteam-feasibility/count_bench.out`), take the unique rows, copy them to the GPU, run an embedding over the remapped ids, copy the dense gradient back, and update with a sparse optimiser on the CPU.
- Per step that moves at most 262k rows: 134 MB on the 35B, 335 MB on the 27B.
- The ngram-memory repo itself is JSON-heavy and gives only 0.97 rows per token, so the rows-per-token figure depends heavily on the corpus.

**Memory budget, serve stopped** (`vram_fit.out`, est.; 2×15.67 GiB is about 33.6 GB):

| Host | Weights (no embed) | Checkpointed activations | Layer working set | Dequant buffers | CE chunk | HIP overhead | **Total** |
|---|---|---|---|---|---|---|---|
| 35B | 20.5 | 0.67 | 0.34 | 1.68 | 1.02 | 3.0 | **27.2 GB** |
| 27B | 15.7 | 2.68 | 0.79 | 1.52 | 1.02 | 3.0 | **24.7 GB** |

- Host RAM: the row store plus optimiser takes 2–30 GB of the 96 GB.
- Fallback for bf16 masters: stream bf16 weights from pinned RAM. That costs 7.2 s per step on the 35B and 5.6 s on the 27B, bound by card 1's 14.48 GB/s link (`cpu_tier.py:6`), against 4–47 s of compute per 16k-token step. It is viable but slower.

**CPU beside a running serve.**
- Limits: 4 cores, `MemoryMax=15G`, no swap; read the 4-bit weights per layer from NVMe with O_DIRECT at about 3 GB/s, as in INTEGRATION.md §4.1 and §4.2(A).
- Peak RAM is about 10–14 GB at 4096 tokens per step, with rows as bf16 plus row-wise Adagrad. Plain Adam on the 27B does not fit (30 GB).

---

## 3. Throughput per 1M training tokens

Inputs (`cost_model.py` → `cost_model.out`; all est.):

- FLOP per token is 6·active plus attention at a 2048-token window. Attention is 3.7% of the total on the 35B and 1% on the 27B.
- Dense bf16 matrix peaks:
  - RX 9070 XT: **194.6 TFLOPS** (fork `docs/journal/CONTINUANCE_gemm_efficiency.md:37`; AMD lists 389 with sparsity).
  - RX 9070: **144.5 TFLOPS** (289 with sparsity, halved).
  - The R9700, for reference: 191 TFLOPS dense ([AMD](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-ai-pro/radeon-ai-pro-r9000-series/amd-radeon-ai-pro-r9700.html), [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/amd-rdna4-rx-9000-series-gpus-specifications-pricing-release-date)).
- The fork measured 115.3 TF/s on one large bf16 GEMM (59% of peak; `BLOCKWISE_FP8_BRIEF_2026-09-22.md:167`).
- Assumed whole-step efficiency: 20–45% of peak for the dense 27B, 7–25% for the MoE 35B, a 0.8 pipeline factor, and a GDN multiplier of 1.05–1.5.
- CPU: 0.3–1.0 TF/s over 4 Zen 4 cores, plus the measured GDN factor, plus dequant at 1.2–2.6 Gparam/s.

| | Qwen3.6-35B-A3B | Qwen3.8-27B |
|---|---|---|
| FLOP per token (training) | 18.3 GFLOP | 155.4 GFLOP |
| TFLOP per 1M tokens | 18,349 | 155,358 (8.5×) |
| Effective TFLOP/s, 2 GPUs | 19–68 | 54–122 |
| **GPU-hours per 1M tokens (serve down)** | **0.09–0.41** | **0.37–0.96** |
| Compute per 16k-token step | 4.4–15.8 s | 20.9–46.9 s |
| **CPU-hours per 1M tokens (beside the serve)** | **11–35** | **62–205** |
| Per-fact solve, GPU (§5 iteration, 251–497 tokens prefix-cached, 21 iterations) | about 0.5–1.5 s/iter, so 10–30 s per edit | about 0.5–2 s/iter, so 10–40 s per edit |
| Per-fact solve, CPU beside the serve | about 34–110 s/iter, so 12–38 min per edit; expert dequant dominates | about 1–6 min/iter, so 20–125 min per edit |
| Prefix GDN state per sequence, fp32 (bf16 is half) | 63 MB | 151 MB |

For comparison, the Flash-Next CPU solve is 19–40 min per edit (INTEGRATION.md §4.2(A)).

What this means:
- A 1.4M-token codebase cartridge at a few epochs costs **about 0.5–2 GPU-hours on the 35B** and **2–6 on the 27B**, during serve downtime.
- On the CPU beside the serve, the 35B is an overnight-to-weekend job. The 27B is not practical there.

---

## 4. Which host goes first: Qwen3.6-35B-A3B

1. **Cost.** It needs 8.5× fewer training FLOP per token and leaves about 6 GB of VRAM headroom. Its rows are 128 wide, so the cartridge is 2.0 GB instead of 5.0 GB.
2. **Keys.** Both hosts use Flash-Next's KeySpec with identical row ids (DESIGN.md §4.3).
   - The 27B tokenizer is byte-identical to Flash-Next's.
   - The 35B tokenizer only lacks 7 Flash-Next special tokens (ids 248070–248076), and 310 of 310 test texts gave identical ids (`P/graft/tok_check.out`).
   - So compatibility is not a deciding factor.
3. **Dissection value.** The 35B is the architectural twin of Flash-Next: a GDN hybrid at 3:1 with layer 1 as GDN, a sparse MoE with a shared expert, and about 3B active parameters against Flash-Next's 6.7B.
   - The 27B instead shares Flash-Next's generation but is dense.
   - Neither pair is a clean control. Do the 35B now and the 27B second.
   - Qwen/Qwen3.6-27B also exists; it is the same architecture as Qwen3.8-27B, so it can separate the effect of generation if needed.
   - One caveat: the 35B router reads the residual the graft injects into, so edits also move routing.
4. **GDN state.**
   - For ingestion, every training window starts from a zero state, so no cache is involved.
   - Keys reset only at EOS 248044. Concatenate files without EOS, the way an agent's context presents them.
   - For solves, prefix states are 63 MB per sequence on the 35B against 151 MB on the 27B.
   - Serving needs the graft's own three-slot conv state per request, plus GDN radix-snapshot invalidation after each edit (DESIGN.md §2.3 and §4.3; Flash-Next's P5 check).
5. **Serving speed.**

   | Model | Decode | Prefill |
   |---|---|---|
   | 35B AWQ | 102.6–106.5 tok/s | about 6.9–7.0k tok/s |
   | 35B MXFP4 | 111–113 tok/s | about 6.6–6.8k tok/s |
   | 27B | 24.4–47.9 tok/s, 60.8 with MTP | — |
   | Flash-Next | 12.9 (TP=2) – about 20 (EP) tok/s | — |

   Sources: `serve.sh:159-162, 384-388, 396-418`; DESIGN.md §7.
6. **Engineering is shared.**
   - The fork has the n-gram module only for qwen4exp (`models/qwen4exp.py:290+`, `minisgl/ple/`, 1385 lines). Serving a graft means porting it into `qwen3_5.py`.
   - `qwen3_5_moe` reuses the `qwen3_5` decoder through its `mlp_factory` seam (`qwen3_5_moe.py:11-12`), so a single port covers both hosts.
   - The graft rows go through the host-side row path, so they take no VRAM.
   - MTP draft heads never see the graft, which may lower draft acceptance (reasoned, not measured).

**Risks to check first, all unverified:**
- whether flash-linear-attention's Triton GDN kernels compile and run on gfx1201 (they are worth up to about 1.5× on the 35B);
- whether `torch.nn.functional.grouped_mm` works on ROCm gfx12;
- the fp8 activation gap between the W4A8 serve and a bf16-activation training forward on the 35B experts;
- measuring one real-size layer's forward and backward on gfx1201 to replace the assumed efficiency brackets.

---

Files are in `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/train_local/`:
- `hub_list.py`, `hub_list.json` — repo search for both hosts
- `hub_detail.py`, `hub_detail.out`, `hub_detail.json` — sizes and quant configs
- `qwen_family.out` — official Qwen3.8 and Qwen3.6 repos
- `ngram_census.py`, `ngram_census.out` — distinct n-grams and rows per codebase
- `gdn_cpu_bench.py`, `gdn_cpu_bench.out` — one-thread GDN versus projection timing
- `cost_model.py`, `cost_model.out` — throughput and cost per 1M tokens
- `vram_fit.out` — memory budget per host

Sources:
- [Tom's Hardware RDNA4 specs](https://www.tomshardware.com/pc-components/gpus/amd-rdna4-rx-9000-series-gpus-specifications-pricing-release-date)
- [CG Channel RX 9070 launch](https://www.cgchannel.com/2025/03/amd-launches-radeon-rx-9070-xt-and-radeon-rx-9070)
- [AMD R9700 page](https://www.amd.com/en/support/downloads/drivers.html/graphics/radeon-ai-pro/radeon-ai-pro-r9000-series/amd-radeon-ai-pro-r9700.html)
- [Tom's Hardware R9700](https://www.tomshardware.com/pc-components/gpus/amd-launches-radeon-ai-pro-r9700-to-challenge-nvidias-ai-market-dominance)
- [bitsandbytes on RDNA4 (blog)](https://vachsark-homepage.pages.dev/blog/qlora-rdna4)
- [bitsandbytes installation docs](https://huggingface.co/docs/bitsandbytes/v0.50.0/installation)
- [pytorch/ao#2066](https://github.com/pytorch/ao/pull/2066)
- [vLLM forum: RDNA4 FP8](https://discuss.vllm.ai/t/native-fp8-wmma-support-for-amd-rdna4-rx-9070-xt-r9700-in-vllm/1900)
- [ROCm TransformerEngine#359](https://github.com/ROCm/TransformerEngine/issues/359)
- [flash-linear-attention on PyPI](https://pypi.org/project/flash-linear-attention/)
- [AMD Primus GDN guide](https://rocm.docs.amd.com/projects/primus/en/latest/04-technical-guides/hybrid-models/gdn-guide.html)

# SKEPTIC VERDICT ON graft-hosts-local (takes precedence where it conflicts)

**Verdict:** the main recommendation holds. Do the Qwen3.6-35B-A3B graft first on `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`, then Qwen3.8-27B on `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`. Several supporting claims are wrong or too optimistic, and the biggest error is about the GDN path on RDNA4.

**Checked and correct**
- Configs match `scratchpad/models/Qwen/*/config.json`.
  - 35B: 40 layers, full attention at layers 3, 7, …, 39, 256 experts top-8, expert width 512, shared expert 512.
  - 27B: 64 layers, d 5120, MLP width 17408.
  - Qwen3.6-27B has the same config as Qwen3.8-27B (diff is empty).
  - Prefix GDN state: 30×32×128²×4 B = 63 MB; 48×48×128²×4 B = 151 MB.
- Parameter counts match `nm/compute/count.out`. Row widths 128 and 320 match DESIGN.md §2.4.
- Census arithmetic is right: (300k + 672k) × 8 heads = 7.77M rows, giving 0.99 G and 2.49 G parameters. Tokenizer claims match `proofs/graft/tok_check.out`.
- Repo sizes re-checked through the HF API (`skeptic_local/hub_recheck.json`): 24.97 / 23.12 / 19.80 / 20.56 / 71.90 / 55.56 GB.
- Fork citations hold: `serve.sh:62` (default `qwen35b-awq`), the 35B speeds at `:146-149`, the 27B MXFP4 W4A16 speeds at `:396-418` (47.9 tok/s; 60.8 needs `SPEC=mtp`), `kernels.py:61`, `mxfp4.py:262`, `nvfp4.py:374`, `cpu_tier.py:6`, the 194.6 TFLOPS figure, the 115.3 TF/s GEMM, the Dockerfile ROCm 7.2 / gfx1201 torch, and 1385 lines in `ple/`.

**Wrong**
1. **"No autograd anywhere in `python/minisgl`" and "test flash-linear-attention on the box first" are both wrong.**
   - The fork records that flash-linear-attention *hangs* on gfx1201 (`cam/runtime.py:11,35`; `gdn/hf_patch.py:1-3`).
   - It already ships a validated differentiable replacement, `minisgl.gdn.hf_patch.patch_qwen3_5_gdn`. It runs the native `gdn_hip` path with recompute-backward (`gdn/layer.py:350-370`). Forward is within about 1e-3 and the input gradient has cosine above 0.9999 (`tools/gdn_backward_validate.py`, `tools/cam_native_gdn_validate.py`).
   - Limits: it was validated only on Qwen3.5-4B, and it is prefill-only from a zero state. That suits ingestion, but prefix-cached solves need HF's torch fallback, which does run on gfx1201 (it is the reference in `cam_native_gdn_validate.py`).
   - The fork also already uses this to train a CAM tap on a frozen HF Qwen3.5. That is direct prior art for this setup.
   - Using it means importing fork kernels into our torch process, not putting torch into the engine. That is a choice for the owner, since INTEGRATION §4.2 dropped engine code paths as dependencies.
2. **Dequant cost is undercounted by about 8×.** The model dequantises once per 16k-token step. Under the stated 1F1B schedule with 2048-token microbatches it happens once per microbatch per pass. That gives about 180 s per 1M tokens on the 35B and 135 s on the 27B (`skeptic_local/recheck.out`), not 10–13 s. That is up to +55% at the low end of the 35B range.
3. **The schedule contradicts itself.** The bf16 "streamed from RAM, 7.2 s per step" fallback and the per-step dequant both assume a layer-at-a-time schedule, not 1F1B.
   - At 2048 tokens per microbatch, each expert gets only 64 tokens, which caps an expert GEMM at about 41 TF/s on bandwidth alone.
   - Pushing larger chunks through each layer (8k to 16k tokens, so 256–512 tokens per expert) is better for the 35B on both counts.
4. **The weight figure is for the wrong checkpoint.** The AWQ text body without embeddings is 22.5 GB (2.86 GB F16 dense, 18.62 GB experts, 1.02 GB lm_head), not 20.5 GB. The 20.5 GB figure is close to the MXFP4 checkpoint.
   - It still fits, but per card it is tight: about 15.0 GB on card 0 against 16.8 GB at a 57:43 split. Split the layers by memory, not by card speed.
5. **The two memory budgets disagree.** `vram_fit.out` (27.2 / 24.7 GB) has no script behind it and contradicts `cost_model.out` (30.3 / 34.2 GB). By the cost model's own numbers plus 3 GB HIP overhead, the 27B would *not* fit. My per-card estimate (12.1 / 13.2 GB) says it does, with in-flight activations of about 2 GB.
6. **"Dense path F16 and exact" needs a qualifier.** The serve computes in bf16 (`server/args.py:534-545`; the checkpoint's text_config dtype is bfloat16), so training must make the same F16-to-bf16 cast to stay exact.

**Over-optimistic or not backed by a script**
- **27B efficiency ceiling.** 45% of peak is 76% of the fork's best measured single GEMM. About 35% is more defensible, which puts the 27B at roughly 0.5–1.0 h per 1M tokens.
- **CPU timings.**
  - The GDN ratio and CPU rates were measured on this sandbox's Sapphire Rapids with AMX, while the owner's machine is Zen 4 (INTEGRATION §4.2).
  - 1.0 TF/s on 4 Zen 4 cores is about 78% of their theoretical bf16 peak (my estimate), so the low end of the CPU range is too optimistic.
  - By the report's own rates, one 1.44M-token cartridge at 3 epochs takes **2–6 days** on the 35B and 11–37 days on the 27B. That is not "overnight-to-weekend".
- **CPU RAM.** The 10–14 GB peak has no script. My rough estimate puts the 27B near the 15G cap, and its rows grow with the corpus.
- **Per-fact solve times.**
  - The figures (0.5–2 s and 34–110 s per iteration) have no script; my rough re-derivation lands in the same ranges.
  - They use 21 iterations from INTEGRATION §4.2, while DESIGN.md §7 assumes 100–300, which is 5–14× more per edit.
- **"GPU-hours" means wall-clock hours on both cards** (two card-hours each).
- **Serve speed with a graft is unmeasured.**
  - The PLE path forces the synchronous scheduler loop unless `MINISGL_OVERLAP_SPLIT=1` (INTEGRATION §2.3).
  - Host-side row gathering costs about 300 µs per step.
  - Against a 10 ms step on the 35B, that overhead matters much more than on Flash-Next's 50 ms step. Re-measure decode speed with the graft in place.
- **The 15G RAM cap came from the Flash-Next serve**, which pins about 56 GB. With a 35B or 27B serve as the daily driver, host headroom is likely larger (unmeasured).

**Corrected key conclusions**
1. Do the 35B first (AWQ) and the 27B second (Quark MXFP4). The order stands: the 35B is about 3–5× cheaper per token, not 8.5×, once dequant and efficiency are corrected.
2. Training runs in ngram-memory's torch-ROCm process with the serve stopped:
   - 4-bit weights resident on the cards, with a dequant-then-bf16 function that computes only the input gradient;
   - GDN through the fork's `gdn_hip` hook, not flash-linear-attention;
   - large per-layer chunks for the MoE;
   - layers split across the cards by memory.
3. Estimates per 1M tokens:

   | | 35B | 27B |
   |---|---|---|
   | GPU, wall-clock, both cards | 0.1–0.45 h | 0.5–1.0 h |
   | CPU beside the serve | ≥11–35 h (on Zen 4, unmeasured, probably slower) | 62–205 h |
   | One 1.44M-token cartridge, 3 epochs, on GPU | about 0.6–2 h | about 2–4.5 h |

   On the CPU the cartridge takes days on the 35B, and the 27B is not practical there.
4. First measurements to make:
   - one real-size layer's forward and backward on gfx1201, using `gdn_hip` and the AWQ dequant function;
   - whether `grouped_mm` works on gfx12;
   - the gap between the W4A8 serve and bf16-activation training on the experts;
   - decode speed with the graft;
   - per-card VRAM for the 35B.

Files are in `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/skeptic_local/`:
- `hub_recheck.py`
- `hub_recheck.json`
- `recheck.py`
- `recheck.out`


# REPORT: graft-serve

# Serving the grafted Qwen3.6-35B-A3B and Qwen3.8-27B in the minisgl fork: port design

**Bottom line:** almost no new engine machinery is needed. Three facts carry the design:

1. **The device math already exists and works at `hc_count=1`.** The fork's own `Qwen4ExpPLE.compute` matches transformers' `Qwen4ExpTextPLELayer` (the module our HF graft uses) at `hc_count=1`. This was checked on CPU in fp32 and bf16, through both the prefill path and the decode path.
2. **With the editor init and a rows-only cartridge, the module is exactly linear.** The editor init is W_k=0, conv=0, norms=0. The module then reduces to `out = (0.5·W_v)·e`, bit-for-bit. So the serve can run a "linear" fast path with no conv state.
3. **The engine plumbing is already model-agnostic.** The scheduler staging, ledger, spec rollback, radix snapshots and graph-capture pointer gate are all keyed on `ctx.ple`, `mc.ple_layer_ids` and `model.ple_block()`. None of them is keyed on qwen4_exp.

The real new work is four things:
- a sparse in-RAM cartridge table (numpy only),
- a bf16-only staging path,
- injecting into the qwen3_5 fused residual between layers 0 and 1,
- per-request namespacing, including per-namespace radix roots.

Estimated per-token decode overhead is about 0.3% on the 35B-A3B and about 0.5–1% on the 27B (estimated, not measured on GPU).

Everything below is cited at fork HEAD `df29117`. `R/` = `/home/user/patcarter883/minisglang-rdna4/python/minisgl/`, and `S/` = `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/serve-port/`.

---

## 1. Evidence produced for this port (CPU, 1 thread)

| Check | Script / output | Result |
|---|---|---|
| Fork `Qwen4ExpPLE` (hc=1, d=256, 16 heads) against HF `Qwen4ExpTextPLELayer`. The fork runs a 23-token prefill chunk plus 17 decode steps; HF runs one full-sequence forward. | `S/parity_hc1.py`, `.out` | Random init (gate, norms and conv all live): max \|Δ\| 1.7e-6 in fp32 (output max 5.1); 0.039 in bf16 (about 2.5 ulp at that magnitude). |
| Editor init: is the output exactly linear? | same | **bf16: fork == HF == `0.5·(e@W_vᵀ)` bit-exact.** fp32: bit-exact once the GEMM shapes match. The only fp32 difference is the M=1 decode GEMV against the full-T GEMM, 4.5e-8. Folding the 0.5 into W_v is also bit-exact. |
| Empty cartridge (e=0) with random W_k, conv and norms | same | Output is exactly +0.0 everywhere, with no −0.0, in fp32 and bf16. `value_proj` has no bias. |
| Vectorised decode-step hash for a whole batch, against the fork's per-sequence `Qwen4ExpNGramHasher.row_ids` on histories dense with EOS | `S/vec_decode_hash.py`, `.out` | 64,000 rows, 0 mismatches, torch blocked. bs16: 11.5 µs vs 392 µs for the loop; bs64: 14.6 vs 1634 µs. |
| Host row path, torch blocked: hash, composite-key `searchsorted` into a sparse cartridge, staging into bf16-bit (uint16) buffers | `S/bench_host_path.py`, `.out` | Table in §6. |
| Bitmap prefilter, one bit per logical row (40 MB), against `searchsorted` | `S/bench_bitmap.py`, `.out` | 131k queries: 0.6–1.0 ms vs 3.2–6.6 ms. |
| Device bytes and FLOPs from config.json (arithmetic only) | `S/device_cost.py`, `.out` | Table in §6. **All GPU times are estimates.** |

Caveats on the scripts:
- `S/bench_host_path.py` still contains a `vec_decode` timing probe whose EOS rule is wrong. It is annotated in the file; use `vec_decode_hash.py` for the correct rule.
- `S/stubimport.py` stubs msgpack, zmq and similar packages so the fork's model module imports on CPU. Run with `MINISGL_TAIL_HIP=0`.

**Host facts (config.json):**

| Host | d | Row width w | `ple_embed_dim` | Layer 1 | eos |
|---|---|---|---|---|---|
| 35B-A3B | 2048 | 128 | 2048 | `linear_attention` | 248044 |
| 27B | 5120 | 320 | 5120 | `linear_attention` | 248044 |

Both hosts have `mtp_num_hidden_layers=1` and an untied `lm_head`.

**Decode baselines:**
- 27B INT4, TP=2: 31.8 tok/s, i.e. 31.4 ms (`tools/serve.sh:384-385`).
- 35B-A3B, TP=2, M=1: 20.4 ms production (`docs/journal/GLM_CONTINUANCE.md:158`); best recorded 69.1 tok/s (`docs/journal/CONTINUANCE_opt_loop.md:103`).

**Hit rates matter for every design choice below.** They come from the sibling Stage-1 study (`nm/stage1/design/row_policy.out`), a 1.42M-token project.
- A cartridge of every selected row (6.72M rows) is hit at 87% of positions in held-out project text, and **45% even on wikitext** (counting collisions).
- With the count≥5 policy: 66% and 19%.
- So the "no hits" case is rare. Design for dense hits, and treat cartridge on/off per request as essential for locality.

---

## 2. Where the block goes

The HF graft is a forward-pre-hook on `layers[1]`. It adds `ple(hidden)` to the full residual entering layer 1, before that layer's `input_layernorm` (`P/graft/graft_proto.py:304-332`).

In minisgl the qwen3_5 stream is fused: layer *i* returns `(x = mlp_out, residual = pre-mlp stream)`, and the next layer's `RMSNormFused` does `residual += x` in place (`R/layers/norm.py:144-159`). The full stream after layer 0 is therefore `x + residual`. It only exists inside layer 1's fused norm.

**Inject by folding into `residual`.** This is the same pattern as the CAM tap, which the code already documents as byte-exact when its update is 0 (`R/models/qwen3_5.py:558-582`, comment at `:572-573`):

```python
# R/models/qwen3_5.py, Qwen3_5Model.forward, inside the layer loop (:545-547)
x, layer_res = layer.forward(x, residual); residual = layer_res
if lid == 0 and self._ngram_graft is not None:          # before the CAM block; CAM is keyed on tap_layer
    residual = self._ngram_graft.inject(x, residual)    # layer 1's fused norm then sums x + residual
```

- **Do not** pass `(x + residual + p, None)` into layer 1. That moves the add out of the fused fp32 sum and breaks the no-op exactness of an empty graft.
- `qwen3_5_moe` and `clover` inherit `Qwen3_5Model` through the `mlp_factory` seam (`R/models/qwen3_5_moe.py:222-251`), so one edit covers both hosts.
- No hyper-connections means one stream, and the Flash-Next "add to all streams before the attn mix" (`R/models/qwen4exp.py:543-544`) becomes simply "add to the residual".

**Hold the block as `self._ngram_graft`.** The leading underscore keeps it out of `BaseOP.state_dict` (`R/layers/base.py:112-114`), the same trick CAM uses. That leaves the host checkpoint's exact-key load untouched. The block's weights load from the graft file instead.

**The block: new file `R/models/ngram_graft.py`.** Defined in a new file because `qwen3_5.py` cannot import `qwen4exp.py`, which itself imports `qwen3_5`.

- **`mode="linear"`** (the default; chosen at load only if W_k ≡ 0 and conv ≡ 0 exactly in the graft file):
  - `p = value_proj(e)`, where `value_proj` is a `LinearReplicated(d→d)` whose weight is pre-scaled by 0.5 (exact).
  - `return torch.where(active[:, None], residual + p, residual)`.
  - `active` is a per-token host flag meaning "this token's 16 row ids hit a cartridge row of its request's namespace". Rows without a hit are therefore bit-identical, even w.r.t. signed zero, and a request with the cartridge off is exactly untouched.
- **`mode="general"`** (any future trained gate or conv):
  - Wrap the unmodified `Qwen4ExpPLE` (`R/models/qwen4exp.py:290-467`) with a config shim: `hc_count=1`, `hc_hidden_size=d`, `ple_embed_dim=d`, `ple_conv_kernel_size=4`, `ngram_size=3`. This is what `S/parity_hc1.py` does.
  - `p = ple.compute(x + residual, e, batch, conv_state)`, then add p with no mask. The dilated conv makes token t depend on hits at t−3, t−6 and t−9, so an own-hit mask would be wrong.
  - Exactness then holds for value only, not for signed zero.
- Both modes keep `Qwen4ExpPLE.forward`'s refusal when nothing is staged (`:452-467`).
- `Qwen3_5ForConditionalGeneration.ple_block()` returns the attached block. `engine.py:737` already gates on `mc.ple_layer_ids and hasattr(model, "ple_block")`.

**Numeric contract with the trainer.** Rows are bf16, W_v is bf16, the GEMM is bf16, and the output is added to the bf16 residual. The HF prototype moves the graft to the host's device but keeps it fp32 (`graft_proto.py:281-283`). The cartridge trainer should run the graft in bf16 (STE on rows, bf16 W_v) so that trained rows behave as served. This is the same P2 concern as `serve/INTEGRATION.md` §5.

---

## 3. Row table: a sparse in-RAM cartridge store reusing the overlay idea

**New file `R/ple/cartridge.py`** (numpy + stdlib + ctypes only).

1. **Registry.** One sorted `int64` composite key array `(ns << 32) | row_id`, plus rows stored as **bf16 bits (uint16)** `[U, w]`.
   - Row ids are below 2^29 (320,001,446 logical rows), so the 32-bit split is safe.
   - `ns` 0 means "no cartridge" and is never stored.
   - Lookup for a whole batch is one `np.searchsorted` plus an equality mask (`bench_host_path.py`).
   - This is the INTEGRATION §3.1 overlay generalised: the base is zero, and the overlay is the cartridge.
2. **Versions are base plus overlays.** A cartridge is an immutable ingested base plus an ordered chain of small overlays (git-diff updates, per-fact edits), each with its own sorted ids. Lookup goes base first, then the overlays, latest wins (two or three `searchsorted` calls). Periodic compaction merges overlays into the base.
   - Each (cartridge, version) gets a **new ns id**.
   - A request pins its ns at admission, with a refcount. Old arrays are freed when the last pinned request finishes. No drain is needed.
3. **Bitmap prefilter** for extend batches: 40 MB, one bit per logical row, the union over loaded cartridges. It is 5–7× faster than `searchsorted` on misses (`bench_bitmap.out`). Only "maybe" hits get the exact lookup.
4. **Sharing across TP ranks.** Each TP rank builds its own runtime (`R/engine/engine.py:736-755`). Put the compiled arrays in a read-only `np.memmap` on `/dev/shm` and `mlock` it via ctypes; swap shares the Optane drive with the Flash-Next table (INTEGRATION §3.1).
5. **Format and refusal.** Graft and cartridge are two artifacts, because W_v is per host, not per cartridge:
   - `graft-<host>.safetensors`: `value_proj` bf16 [d, d] (plus W_k, norms and conv for general mode), and a manifest holding the KeySpec fingerprint (DESIGN §2.1), the TableSpec (`row_width`, 320,001,446 logical rows), the ModuleSpec (layer index 1, hc=1, mode), the host config sha, and the W_v sha.
   - `cartridge-<name>-v<k>.safetensors`: `row_ids` I64 sorted [U] plus `rows` BF16 [U, w], with the manifest of the graft it was trained against.
   - Refuse on any fingerprint mismatch. `row_table.index_safetensors` reads the headers; `read_tensor` widens BF16 to f32 (`R/weights/row_table.py:212-230`), so add a raw-uint16 reader for the rows.

**RAM per rank** (bf16 + int64 id = 2w+8 B/row), using Stage-1 row counts for a 1.42M-token project:

| Row policy | Rows | 35B (264 B/row) | 27B (648 B/row) |
|---|---|---|---|
| All selected | 6.72M | 1.65 GiB | 4.06 GiB |
| count≥2 | 2.04M | 0.50 GiB | 1.23 GiB |
| count≥5 | 0.55M | 0.14 GiB | 0.33 GiB |

Add 40 MB for the bitmap. An int6-g32 codec (the fork's `dequant_int6_g32`) would cut this about 2.4×, at row-relative precision of about 0.6% of row norm. That is optional.

A device-resident table with device-side hashing would remove the host path entirely and would be compatible with the overlap loop. It is **not recommended now**:
- 0.5–4 GiB does not fit next to the weights on 16 GB cards.
- These hosts already run the synchronous loop (§5).
- It would duplicate the hash, which is verified only in numpy.

---

## 4. Host staging (`R/ple/source.py`, `runtime.py`)

`PLEEmbeddingSource` already accepts an injected `_table`/`_heads` (`source.py:63-101`), and its width check `w·16 == ple_embed_dim` holds (128·16 = 2048, 320·16 = 5120). The changes:

- **Sparse mode:**
  - Hash, then `CartridgeStore.lookup_into(ns_tok, ids, host_u16) -> active`.
  - H2D the bf16 bits straight into `embeddings.view(torch.int16)`, an in-place copy into the same static allocation, so the graph pointers do not move.
  - This drops the fp32 landing buffer and the cast kernel.
  - Device staging becomes `max_tokens·d·2` instead of `·6`: 32/80 MiB vs 96/240 MiB at 8192 tokens (35B/27B, `device_cost.out`). That is real KV pool on a 16 GB card.
  - Update `ple_device_bytes` (`runtime.py:343-360`) to match.
- **`active` buffer:** a static device bool `[max_tokens]` carried on `PLEBatch.active`. Add it to `PLEGraphCapture._check_addresses` (`graph_capture.py:132-147`).
- **Decode:** replace the per-sequence `hasher.row_ids` loop (`source.py:148-153`) with the vectorised batch hash when every `seq_len == 1`. That is about 10 µs instead of 25 µs × bs. Put it in `hashing.py`; it is the EOS-checked `decode_row_ids` from `vec_decode_hash.py`.
- **Extend (prefill, not captured):** compact staging is optional. Stage only tokens with a hit, run the GEMM on T_hit rows, then `residual.index_add_` with unique indices. At 0% hits that costs 0.2–0.8 ms vs 3.4–11 ms for zeroing a dense buffer. At high hit rates use dense, zeroing only the rows dirtied last step.
- **Namespace per sequence:** `prepare_begin(slots, tokens, ns=...)`.
  - For ns 0, skip hashing (zero rows, `active=False`), but **still advance the 2-token history**. It is independent of namespace and costs nothing.
- **Builder:** `build_graft_runtime()` sits next to `build_ple_runtime` (`runtime.py:363-438`), which raises without `MINISGL_PLE_FILES`.
  - Derive the multipliers with `build_layer_multipliers(248320, 3, 0, 1234)` and check them against the manifest.
  - Derive the heads with `build_head_vocab_sizes(20_000_000, 16)` (about 1 s at boot) and check them against the manifest.
  - The state cache comes from the block: `state_len=9` in general mode, `0` in linear mode.

---

## 5. Engine integration points (mostly unchanged)

- **Decode state.**
  - `PLEStateCache` (`R/ple/state.py`): token history `[slots, 2]` int64 on the host, EOS-seeded, plus `conv_state [slots, d, 9]` in general mode (36/90 KiB per slot).
  - It is indexed by the GDN slot id, with `num_slots = max_running_req+2` (`engine.py:741-747`).
  - The reset on fresh prefill, `scheduler.py:3226-3227`, is unchanged. The GDN caches are untouched.
- **Radix snapshots.** `CompositeRecurrentState` already snapshots GDN and PLE together whenever `engine.ple_runtime.state` exists (`scheduler.py:474-486`). `resolve_prefix_cache` only forces naive mode for a PLE model that is not a GDN hybrid (`R/engine/config.py:106-109`); both hosts are GDN hybrids.
- **Loop choice.** The default `--gdn-radix` already forces `normal_loop`/`split_loop` for these hosts (`scheduler.py:1500-1503`), so the PLE runtime does not cost the overlap loop. It does cost it under `--no-gdn-radix`.
- **Spec decode.** MTP (+65% on the 35B per the repo journal) and DFlash work through the existing verify staging with `extra_tokens` and `defer_commit` (`scheduler.py:5253-5261`), `commit_verified` (`:5766-5767`), and verify capture (`graph_capture.py:75-97`).
  - In linear mode there is no conv window to roll back; only the history advances by `keep`.
  - The MTP head carries no graft, so acceptance may drop slightly; correctness does not.
- **Graph capture.** `GraphRunner` already builds `PLEGraphCapture` from `ple=self.ple_runtime` (`engine.py:922`, `R/engine/graph.py:290-299, 399-421, 462-463`). Capture stages ns 0. The decode is shape-static: one GEMV, one where/add, plus the conv taps in general mode.
- **TP/EP.**
  - The block is replicated, as on Flash-Next (`qwen4exp.py:65-70`). W_v costs 8 MiB on the 35B and 50 MiB on the 27B per rank (W_k the same again in general mode).
  - Both ranks hash, look up and stage identically. Determinism depends on both having the same cartridge version: control ops go through the rank-0 broadcast, apply in lockstep, and must never raise on one rank only (INTEGRATION §3.2.4, `ep.py:60-79`).
  - Optional for the 27B: shard heads across ranks and add the partial into layer 0's MLP all-reduce. That halves the 50 MiB read and the host work. It is a later optimization.
- **Control plane.** Clone the planned `ple_op` path (INTEGRATION §3.2.4) as `ngram_op`: `load`, `unload`, `put_overlay`, `list`, `drop_ns`. Refuse while a forward is open (`runtime._begun`, `batch`, `_pending`).

**Namespacing, prefix cache and invalidation.**
- A cartridge's rows change KV and GDN state from layer 1 onward, so the radix key must include the ns.
- Use **one root per ns** inside the existing tree: `RadixPrefixCache.roots: Dict[int, RadixTreeNode]`, with `root_node = roots[0]`. Each root is a real root (no parent, `ref_count=1`), so every `is_root()` test still holds. The eviction heap and size accounting stay shared.
- `_tree_walk`, `match_prefix` and `insert_prefix` take `ns` (`radix_cache.py:199, 299, 390`). `check_integrity` walks all roots (`:376`). `CacheManager.match_req`/`cache_req` pass `req.ngram_ns` (`scheduler/cache.py:36, 91`).
- A version bump is a new ns. Old roots are unreachable for new requests and fall to LRU, or are freed at once with `drop_ns`.
- **Optional carry-over** of prefixes that are not stale: run INTEGRATION §3.4(a)'s DFS staleness test against the overlay's changed row ids and re-parent clean subtrees under the new root. A small diff touches few positions, so most subtrees survive.
- **Sharing prefixes across namespaces before the first hit is not worth building.** With a 45–87% per-position hit rate, the first hit is a few tokens in.

**Per-request selection.**
- Add `SamplingParams.ngram_cartridge: str | None`, which rides UserMsg → Req like `mem_namespace` (`R/core.py:151-154`), plus `Req.ngram_ns`.
- Resolve it at admission and pin the version. `_stage_ple` (`scheduler.py:3170-3251`) passes ns per row, with padding rows on ns 0.
- Also offer a serve default (`--ngram-default-cartridge`) and an optional model alias (`"<host>+<cartridge>"`) for stock OpenAI clients that cannot send extra fields.

**Torch-free constraint.** The whole host row path stays numpy, mmap and ctypes: hashing, cartridge store, bitmap, composite lookup and staging fill. `bench_host_path.py` and `vec_decode_hash.py` run with `import torch` blocked. Torch appears only where it already does: the static staging tensors and H2D in `source.py`, `_slot_idx` in `runtime.py`, `conv_state`, and the model forward.

---

## 6. Per-token overhead

The host times were measured single-threaded on this sandbox CPU, so treat them as estimates for the owner's box. The device times are bandwidth/FLOP estimates: 60% of 640 or 322 GB/s, and about 50 TFLOP/s per card.

**Decode, bs=1, linear mode:**

| | 35B-A3B (w=128) | 27B (w=320) |
|---|---|---|
| Host: vectorised hash, lookup, stage | about 10 + 3.3 + 2–3 µs | same |
| H2D | 4 KB | 10 KB |
| Device GEMV | 22–43 µs | 137–271 µs |
| **Total** | **about 50–75 µs = 0.25–0.37% of 20.4 ms** | **about 165–300 µs = 0.5–1.0% of 31.4 ms** |
| General mode adds | +22–43 µs | +137–271 µs |

- Flash-Next reference: about 300 µs gather vs about 50 ms (0.6%) (`R/ple/source.py:197-211`).
- At bs16 the host cost is about 22–37 µs with vectorised hashing, against about 400 µs with the current per-sequence loop.

**Prefill, 8192-token chunk, per rank:**

| Step | Cost |
|---|---|
| Hash | 0.8–1.1 ms |
| Lookup | 3.3–10 ms; with the bitmap prefilter, 0.6–1.0 ms plus the candidates |
| Dense staging | 3.4–11 ms at 0% hits; 29–68 ms at 100% hits (realistic project text is 82–87% hit) |
| H2D | 34/84 MB bf16, about 3–7 ms (estimated) |
| Device GEMM | 1.4/8.6 ms (estimated) |

Against roughly 0.5 s / 4.4 s of backbone per chunk (rough estimate), that is about 1–2% in the worst case.

---

## 7. Patch plan

1. **New `R/ple/cartridge.py`:** `GraftManifest` (parse, fingerprint, refuse on mismatch); `CartridgeStore` (composite keys, base+overlay versions, ns refcount pinning, bitmap, `/dev/shm` memmap + mlock, `lookup_into`).
2. **`R/ple/hashing.py`:** `decode_row_ids_batch(heads, hist, tok)`, ported from `S/vec_decode_hash.py`; a KeySpec constructor that derives the multipliers and heads.
3. **`R/ple/source.py`:** sparse/bf16 mode in `__init__`, `begin_rows` and `finish_rows`; the `active` buffer; `batch_row_ids(..., ns)` with the vectorised decode branch; optional compact staging for extend.
4. **`R/ple/runtime.py`:** `PLEBatch.active`; `prepare_begin(..., ns)`; `build_graft_runtime`; the graft branch of `ple_device_bytes`; `load/unload/put_overlay/list`.
5. **`R/ple/graph_capture.py`:** `active` in `_check_addresses`; capture with ns 0.
6. **New `R/models/ngram_graft.py`:** `NGramGraftBlock(BaseOP)` with linear and general modes (the latter wraps `Qwen4ExpPLE` through a config shim); `from_graft_file`; `inject`; `make_state_cache`.
7. **`R/models/qwen3_5.py`:** `_ngram_graft = None` in `Qwen3_5Model.__init__` (beside the CAM fields, about `:477-490`); the injection at `lid == 0` in `forward` (`:545-547`); `ple_block()` on `Qwen3_5ForConditionalGeneration`.
8. **`R/models/config.py`:** for `qwen3_5`/`qwen3_5_moe` with `--ngram-graft`, fill `ple_layer_ids=(1,)`, `ple_embed_dim=d`, `ple_conv_kernel_size=4`, `ngram_size=3`, `heads_per_ngram=8`, `ngram_vocab_size_base`, `ngram_seed=1234` and `ngram_eos_token_id=248044` **from the manifest**, cross-checked against the host's eos (both are 248044). Add `ngram_graft_mode`. Require `is_gdn_hybrid`, because state is indexed by GDN slot.
9. **`R/engine/engine.py:736-755, 1704-1720`:** choose the graft builder; attach and load the block after the model load; size it with the new formula.
10. **`R/core.py`, `R/scheduler/scheduler.py`:** the request fields; resolve and pin at admission; ns in `_stage_ple`; unpin at finish; an `ngram_op` branch beside CAM's (`:1769`).
11. **`R/kvcache/radix_cache.py`, `R/scheduler/cache.py`:** per-ns roots; `drop_ns`; optional carry-over.
12. **`R/server/args.py`, `R/server/api_server.py`:** `--ngram-graft`, `--ngram-cartridges`, `--ngram-default-cartridge`; a `/ngram/*` router (mounted like CAM's); the OpenAI `extra_body.ngram_cartridge` field and the model alias.

Not in v1: the unigram-head extension (DESIGN §2.4). It needs an order-1 column in `mixed_ids`, a 24-head order and a W_v of shape [d, 24w], all gated by its own KeySpec fingerprint.

---

## 8. What must be tested

**CPU** (extend `tests/qwen4exp_ple_test.py`, `qwen4exp_ple_hash_test.py`, `ple_ledger_test.py`, `ple_radix_snapshot_test.py`, `ple_spec_rollback_test.py`):
1. hc=1 block parity, linear-mode exactness and an empty block giving exactly +0. This is `S/parity_hc1.py` turned into a test.
2. Vectorised decode hash equals the per-sequence hash on EOS-dense histories.
3. `CartridgeStore` lookup equals a dense zero table with those rows written. Namespace isolation, overlay order, version pinning, and import with torch blocked.
4. Serve row ids equal `NGramGraft.row_ids` and the Flash-Next buffers (`P/graft/fetch_fn_buffers.out`), checked on both tokenizers.
5. The loader refuses on a KeySpec, graft-sha or host-sha mismatch, and refuses linear mode when W_k or conv is nonzero.

**GPU, owner box, both hosts, TP=2:**
1. **Bit-exact no-op.** Compare the graft with an empty cartridge, and the graft with a cartridge but a request on ns 0, against the ungrafted serve. Use `torch.equal` on logits plus a 512-token greedy run, covering prefill, chunked prefill and decode, eager and graph replay, and radix hits.
2. **Parity with the HF graft.** Use the same rows and bf16 W_v. Compare the residual after layers 1–3 through `MINISGL_DUMP_LAYERS` (`qwen3_5.py:534-549`) on a truncated real-weight model (first 4 layers) on both sides, then compare final logits. The tolerance is the ungrafted minisgl-vs-HF noise floor.
3. **Decode state.**
   - Cached decode equals a full prefill.
   - Chunk boundaries at hit, hit+1 and hit+3 (general mode).
   - A recycled slot is reset.
   - A radix restore equals a cold run.
   - Spec verify with rejections equals the non-spec greedy run.
   - The ledger satisfies `prepares == commits + discards` and `commit_noops == 0` after a soak.
4. **Namespacing.**
   - A mixed batch (A with a cartridge, B without): B is bit-identical to running alone ungrafted.
   - No radix prefix crosses namespaces.
   - After a version bump, new requests miss the old nodes, and a request in flight on the old version finishes on its old rows.
5. **TP determinism.** Digest the staged embeddings and the `active` buffer per step on both ranks and compare over gloo, as `qwen4exp_offload_serve_test` does for the hyper-connection bytes.
6. **Performance.** TPOT ungrafted vs empty graft vs a high-hit cartridge; prefill tok/s; MTP acceptance. Check these against §6.
7. **Effect transfer.** Predicted ΔNLL from the HF solve vs realised ΔNLL in the serve (the analogue of INTEGRATION §5 P4).

I wrote no report file and made no commits.

Files are in `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/serve-port/`, each `.py` with its `.out`:
- `parity_hc1`
- `vec_decode_hash`
- `bench_host_path`
- `bench_bitmap`
- `device_cost`
- `stubimport.py` (import helper, no output)

# SKEPTIC VERDICT ON graft-serve (takes precedence where it conflicts)

**Verdict on the "graft-serve" report: the port design holds up, but its throughput baselines are out of date, its prefill overhead conclusion contradicts its own numbers, and it misses that the 35B serves in expert-parallel (EP) mode by default.**

I opened every cited file:line at fork `df29117`. Almost all of them point where the report says. I re-checked both host configs on the Hugging Face Hub: d, layer count, vocab 248320, eos 248044, layer 1 = `linear_attention`, `mtp_num_hidden_layers=1`, untied `lm_head` all match. I re-ran `vec_decode_hash.py`: 64,000 rows, 0 mismatches, torch blocked, 9.9 µs at bs1. Recomputed estimates are in `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/serve-skeptic/recompute.py` and `recompute.out` next to it. All GPU numbers below are estimates.

**What holds up**
1. **Device math matches HF at `hc_count=1`.** Checked on CPU at d=256, fp32 and bf16. The editor init really is linear, and an empty cartridge gives exactly +0. Caveat: the "bit-exact" results are CPU-only (`F.linear`). On GPU, `minv_linear` sends M≤16 to `dense_bf16_gemv` and larger M to `dense_gemm`, and the code says these two are not bit-identical to each other (`R/layers/minv.py:256-264`). So graft-vs-HF parity on GPU needs a tolerance, not equality.
2. **Engine plumbing really is keyed only on `ple_layer_ids`, `ple_block` and `ctx.ple`.** Confirmed at `engine.py:737`, `scheduler.py:426,474-486,1500`, `engine/config.py:106-109`, `graph.py:290-299`. The qwen4_exp-specific parts are `build_ple_runtime` and `ple_device_bytes` (which reads `hc_hidden_size`); the report already plans replacements for both.
3. **Injection point is right.** Folding into `residual` after layer 0 matches the CAM pattern (`qwen3_5.py:545-582`) and the fused norm (`norm.py:144-159`). Hiding the block behind a leading `_` works (`base.py:113`). The import-cycle reason for a new file is real (`qwen4exp.py:110`).
4. **Format and RAM arithmetic is right.** RAM table (1.65/4.06, 0.50/1.23, 0.14/0.33 GiB), 40 MB bitmap, row ids < 2^29, W_v 8/50 MiB, conv state 36/90 KiB, staging 32/80 vs 96/240 MiB. Hit rates match `stage1/design/row_policy.out`: 86.8% and 45.2% for all selected rows; 65.8% and 18.8% at count≥5.
5. **Torch-free constraint is respected.** New torch appears only in the model block and in the staging/H2D code in `source.py`, which already uses torch. The container already has `ipc: host`, 16 GB shm and unlimited memlock (`docker-compose.yml:64-82`), so `/dev/shm` plus `mlock` is feasible.

**Wrong or stale**
6. **35B decode baseline is stale.** 20.4 ms and 69.1 tok/s are old numbers; `CONTINUANCE_opt_loop.md:13` itself says the 69.1 figure is stale. Current figures:
   - EP, SPEC=none: 102.6 tok/s, about 9.75 ms (`tools/serve.sh:148-149`, 2026-09-29).
   - TP, bs=1: 10.5 ms (`ROCM10_UPGRADE.md:59`).
   - The overhead is therefore roughly double the report's 0.25–0.37%.
7. **27B decode baseline is stale.** 31.4 ms is the INT4 arm from 2026-08-15. The MXFP4 W4A16 arm measured 46.5–47.9 tok/s, about 21 ms, and 60.8 tok/s with MTP K=2 (`serve.sh:396-419`). The report's "MTP is a wash" also applies only to that old INT4 arm.
8. **The 322 GB/s card bound does not apply.** Both cards are gfx1201, the RX 9070 XT class. The realistic bandwidth range is 0.5–0.6 of 640–674 GB/s; the measured LM-head GEMV runs at 50.4% of roofline (`COUNTER_SCORECARD.md:341-343`). Drop the upper GEMV values of 43 and 271 µs.
9. **Prefill "about 1–2% worst case" is wrong for the 35B by the report's own inputs.** Its numbers give 33 + 10 + 3 + 1.4 ms against 492 ms, which is about 9.8%. Two more problems:
   - The 492 ms backbone figure is about 2.4× too fast. Measured 35B prefill is about 7,000 tok/s (`serve.sh:149`), i.e. about 1.17 s per 8192 tokens.
   - The default chunk is 2048 tokens, not 8192 (`serve.sh:34`).
10. **The TP/EP section assumes TP lockstep, but the 35B defaults to `EP=1`** (`serve.sh:152`). In this fork EP means separate replicas running different requests (`scheduler/ep.py:17-41`). So "both ranks stage identically" and the per-step digest test do not apply there. Each replica has its own radix tree, and control ops must reach both replicas. The `ep.py:60-79` cite is the EP step-drift guard, not a rank-0 broadcast.
11. **PLE has never run under EP.** The idle-replica dummy prefill builds `Batch(reqs=[dummy_req])` and calls `_finish_prepare` (`ep.py:94-97`). `_stage_ple` treats that request as real and raises when it has no GDN slot (`scheduler.py:3210-3225`). I did not trace whether the dummy gets a slot, so this is unverified and needs a test, or run `EP=0`.

**Over-optimistic or unnecessary**
12. **The bitmap prefilter does not help in the case the report designs for.** Its 5–7× win was measured with only 0.03–5% of queries possibly hitting (`bench_bitmap.out`). With the stated dense hits on project text (about 54–82% exact per order), almost every query goes on to the exact lookup anyway.
13. **The host benchmarks do not cover the realistic cartridge size.** They used 160k and 1.6M rows; the realistic all-selected cartridge has 6.72M rows. Dense staging at 100% hits is a single-threaded numpy gather (29–78 ms per 8192 tokens) and is worth spreading across threads.
14. **The `active` mask in linear mode only protects the sign of zero.** e=0 already gives p=±0 and r + (±0) = r in value, and `torch.equal` treats −0 == +0. So the mask buys no observable exactness, at the cost of an extra H2D, buffer and pointer gate. The real cost it misses: the GEMV runs every step even when no request in the batch has a cartridge. Skipping it on all-ns-0 steps (eager, or a second captured graph) would remove the overhead entirely for non-cartridge traffic.
15. **Prefix carry-over across versions should not be optional.** Every git-diff update makes a new namespace, which cold-misses the prefix cache for every later turn of every conversation using that cartridge. A cheap alternative is to pin the version per conversation rather than per request.
16. **Smaller points:**
    - Inject after the `MINISGL_DUMP_LAYERS` dump and the aux capture, not before. That keeps the layer-0 dump row and DFlash aux features equal to HF's `hidden_states[1]`. DFlash acceptance may also drop, which the report does not mention.
    - "Device-resident table does not fit" is too strong. A count≥5 cartridge on the 35B is about 0.14 GiB. The report's other reasons for not doing it now still hold.
    - "No hits is rare" holds only for the all-rows policy on project text. At count≥5 on wikitext, 81% of positions have no hits.
    - The int6 "about 0.6% of row norm" figure has no script behind it. My back-of-envelope gives about 2% elementwise relative RMS, so it is unsupported.

**Corrected key conclusions**
- The design stands: little new engine machinery, a numpy-only sparse cartridge store, bf16 staging, a fold into `residual` after layer 0, and per-namespace radix roots. Add the EP path (or run `EP=0` for the 35B at first) and either prefix carry-over or per-conversation version pinning.
- **Decode overhead (estimated):**

  | Host | Graft cost per step | Current baseline | Overhead |
  |---|---|---|---|
  | 35B-A3B | 47–68 µs | 9.75–10.5 ms | 0.5–0.7% |
  | 27B | 160–200 µs | about 21.5 ms | 0.75–0.95% |

  This is paid on every step unless all-ns-0 steps are skipped. Sharding W_v across the two ranks would roughly halve the 27B cost.
- **Prefill overhead (estimated):** 35B, 0.9–3.0% per 2048-token chunk and 1.1–4.1% per 8192 against measured throughput. 27B, 0.5–2.1%, but its backbone prefill rate is itself unmeasured.
- **Unmeasured on GPU:** all device times, PLE under EP, and the dummy-prefill path.


# REPORT: dissection

# Dissection program: what Flash-Next, trained with its n-gram table, can teach us about grafting one onto models trained without

Paths are abbreviated as follows:
- `q4:` = `transformers/models/qwen4_exp/modeling_qwen4_exp.py` (transformers 5.22, in the scratchpad venv)
- `R/` = `/home/user/patcarter883/minisglang-rdna4/python/minisgl/` at df29117
- `P/` = `/home/user/memory-organ/ngram-memory/proofs/`
- `D/` = `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/dissect/` (new pilot scripts and their outputs)

Numbers marked **(est.)** are not measured.

## 0. Bottom line

1. **The top 3, in order:**
   - **G** (row statistics, CPU here). A pilot has already run (§2).
   - **S** (a site probe on a 2-layer slice of Flash-Next, on CPU).
   - **A** (a table/gate/conv ablation sweep through the serve, forward-only).
   - Together they settle the graft design questions that are still open: the gate, the conv, the output scale, which n-grams are worth storing, and whether transplanting rows is worth trying. They need no backward pass on Flash-Next and do not stop the serve, apart from about 300 s per boot-flag variant.
2. **The pilot already shifts the plan.** On real Flash-Next rows, each row's n-gram-specific content is a small fraction of the row. The rest is mixed with the other n-grams that hash into the same bucket (its co-tenants).
   - For frequent n-grams the 8 heads agree with mean pairwise cosine 0.15–0.19.
   - For long-tail n-grams (one occurrence in a 400k-token local corpus) they agree at 0.003. Random buckets give 0.000.
   - Consequence: Flash-Next's rows for **project-specific n-grams carry essentially no signal**. A Flash-Next→27B row transplant (f) can only carry generic n-grams, which the 27B already models with its own layers. Predicted transplant gain is about 0 **(hypothesis; (f) tests it cheaply later)**.
   - It also suggests the trained gate may partly be a **collision denoiser**. A sparse graft has no co-tenants, so it may not need one; S tests this.
3. **Two corrections to the brief.**
   - **The gate is not per head or per order.** There is one scalar per (token, stream), 4 per token, computed from the concatenation of all 16 heads (`q4:1240-1241`; hyper-connection count `hc_count=4`). Per-order effects therefore need leave-one-order-out recomputation.
   - **The "dilated conv" is mostly a pointwise term.** On the real weights the taps at t−9, t−6, t−3 have mean |w| ≈ 0.0006–0.002. The current-position tap is 0.0027–0.071, and 0.071 on stream 1 (`P/t1/real_stats.out`). So `conv=0` in the graft probably drops a per-position nonlinearity, not temporal context. A tests this.

## 1. Mechanics the experiments rely on (checked in code)

- **Zero rows are an exact, full ablation.** `key_proj` and `value_proj` have no bias (`q4:1196-1197`), so e=0 gives v=0, u=0, `N_conv(0)=0` and `silu(0)=0`. That makes "zero all rows" a **numpy-only** serve change: fill `_host_rows` after the decode at `R/ple/source.py:243`, before the H2D copy at `:244`. That is the same hook as the planned overlay, and it is torch-free.
- **Other row-side ablations are also numpy-only:**
  - zero heads 0–7 (2-gram only) or 8–15 (3-gram only); the head is known from the row-id range;
  - mean-row: replace each row by its head's mean row μ_h. v becomes constant while the gate stays query-dependent;
  - shuffle: remap ids in `begin_rows` (`source.py:227`, `flat = ids.reshape(-1)`) to `off_h + perm(id − off_h)`. Rows keep the trained distribution but carry the wrong content.
- **Gate and conv ablations are torch-side model code**, in `Qwen4ExpPLE.compute` (`R/models/qwen4exp.py:421-443`): gate replaced by a per-stream constant, conv branch off. As a boot-time env flag read once, this is graph-capture-safe but needs a restart, about 300 s (`R/weights/ckpt_read.py:5-10`). The owner should approve it; it is not on the row hot path.
- **There is no scoring path today.** The serve returns logprobs only for generated tokens, top-20 at most, and rejects `echo` (`R/server/api_server.py:415-420, 1637-1638`). Teacher-forced NLL needs `/ple/score`, which is already the M1 deliverable in DESIGN §8. A fallback with no engine change: one `max_tokens=1, top_logprobs=20` request per prefix, reusing the radix cache. Its NLL is censored outside the top-20, and it is about 20 tok/s. Usable only for a pilot.
- **The gate depends only on what enters layer 1:** `embed(x)` repeated ×4, then layer 0 (GDN + MoE), then the PLE block (`q4:1278`, query = `N_query(hidden)`). So **gate, value, conv share and output scale can all be measured on a 2-layer slice**:
  - Public bf16 sizes: Flash-Next layer 0 is 5.19 GB, embed 1.27 GB, PLE block (without the table) 0.066 GB.
  - 27B: layer 0 0.77 GB, embed 2.54 GB. 35B-A3B: layer 0 1.69 GB, embed 1.02 GB.
  - All from the headers in `P/compute/hdr/*.json`. The full model is not needed.
- **HTTP range reads from the public bf16 shards are feasible for samples, not for a whole codebase** (`D/range_probe.out`, `D/row_pilot.out`):
  - one 320 B row: median 0.20 s serial (0.29–0.33 s with the redirect each time); 0.5 s for the first call;
  - parallel: 65 rows/s at 16 threads, 115 at 32, 133 at 64; 200 rows/s at 48 in the pilot's 2,176-row batch;
  - a contiguous 2,048-row block (655 KB) is one 0.32 s request;
  - multi-range requests are refused (HTTP 416), so it is one request per row;
  - this gives about 12 n-grams/s (16 rows each), so 5k n-grams take about 7 min;
  - 7.7M rows (the upper bound for the minisgl repo, `nm/graftfirst/train_local/ngram_census.out`) would take about 11–16 h **(est.)**, so a whole-codebase row read belongs on the owner's Optane through `R/weights/row_table.py` (numpy-only, int6).
  - Caveat: the public checkpoint is bf16; the serve uses int6 g32 (median relative error 1.4e-3, INTEGRATION §3.1). That is fine for statistics; for a transplant, use the served codec.

## 2. Pilot already run: G0 (real rows, CPU, about 25 s of fetching)

`D/row_pilot.py` → `D/row_pilot.out` and `D/row_pilot.json`.

- **Corpora:** 406k tokens of minisgl Python and 404k tokens of markdown docs, EOS at file start.
- **Method:**
  - Rows were fetched for n-grams in several frequency buckets, plus 16×1,024 contiguous rows of random buckets as the null.
  - Each head's row is centred on that head's mean row and mapped through the real `W_v` head slice: c_h = W_v^h (e_h − μ_h).
  - **Coherence** is the mean pairwise cosine of c_h over the n-gram's 8 heads. Co-tenant noise is independent across heads because each head uses a different prime modulus; the n-gram's own signal is shared.

| bucket | n | mean row L2 | coherence (q10 / q50 / q90) | ‖centred v‖ |
|---|---|---|---|---|
| code 2-gram, top | 48 | 0.118 | **0.150** (0.052 / 0.136 / 0.254) | 0.474 |
| code 2-gram, count 5–20 | 48 | 0.108 | 0.017 | 0.315 |
| code 2-gram, once | 48 | 0.100 | 0.003 | 0.277 |
| prose 2-gram, top | 32 | 0.121 | **0.194** | 0.532 |
| prose 2-gram, once | 32 | 0.099 | 0.003 | 0.277 |
| code 3-gram, top | 32 | 0.110 | 0.161 | 0.223 |
| code 3-gram, once | 32 | 0.100 | 0.013 | 0.142 |
| null, random buckets (2 / 3) | 200 | 0.094 / 0.098 | 0.000 / 0.000 | 0.257 / 0.135 |

What this shows:
- **Row norm rises with frequency** (0.094 → 0.12).
- **Never-read padding rows decay to about 0** (median 8e-18, `P/t1/real_stats.out`). So training applied weight decay to the table, and row norm reflects a balance between gradient pressure and decay. This is a direct prior for graft row weight decay and the norm trust region.
- **Signal-to-noise of the summed 8-head value** ≈ 8ρ/(1−ρ):
  - about 1.4–1.9 for top n-grams;
  - about 0.02 for long-tail ones (**derived; it assumes equal-norm heads**).
  - So for most n-grams the value Flash-Next reads is mostly co-tenant noise.
- **Lens, over 67 n-grams that have a dominant continuation** (z-score of cosine against 512 random vocab rows):
  - lm_head[next token]: +1.03 mean, 30% of n-grams above z=2;
  - lm_head[current token]: +1.02;
  - embed[current token]: −1.01 mean (median −0.38).
  - Read: a weak direct-logit signal, plus a hint that the value pushes *away* from the input embedding of the current token, consistent with "detokenisation". Small n, crude lens, **directional only**.
- **Scale** (`D/embed_scale.out`, 256 random vocab rows): token-embedding RMS is 0.0091 on Flash-Next, 0.0131 on the 27B and 0.0125 on the 35B-A3B. Flash-Next's PLE value RMS is about 0.0058 (`P/t1/real_stats.out`, synthetic e). So a graft's layer-1 output should be calibrated against about 0.6× the host's embedding RMS, roughly 0.008 on both hosts **(est.; S measures the real ratio against the full layer-1 residual)**.

## 3. Experiments

### Top 3 (do first)

**G — row statistics at scale (CPU here, or numpy on the owner's Optane). Cost: about 1 day of code; minutes to run.**

What it does:
- Scale G0 to about 5k n-grams per order (about 7 min of range reads).
- Frequency comes from the planned 1B-token row-load counters (DESIGN §7; 0.21 M tok/s/core). Until then, a public n-gram frequency proxy plus the local repos.

What it measures:
- the coherence-vs-frequency curve, and the frequency where coherence crosses about 0.05;
- the 2-gram vs 3-gram value share (`W_v` head slices differ 2×: Frobenius norm 12.3 vs 6.3);
- the effective rank of the coherent component across n-grams;
- row norm vs total row load;
- nearest-token lenses (embed and lm_head, 1.27 GB each, one fetch; or top-k against 50k range-read rows).

What it decides:
- graft row weight decay and init, and the per-row norm budget;
- **which n-grams a cartridge should hold.** If Flash-Next only "knows" n-grams above a frequency f*, a project cartridge should spend rows on n-grams below f* that are frequent within the project;
- whether a transplant (f) is worth building (only above f*).

**S — site probe on a 2-layer Flash-Next slice (owner's CPU beside the serve from local files, or here). Cost: 1–2 days of code; minutes to run.**

How it runs:
- HF `Qwen4ExpText` with `num_hidden_layers=2`, PLE at decoder index 1, layers built on meta. Load embed, layer 0 and the PLE block.
- Replace the table with a sparse store of the rows the probe corpus reads.
- On the owner's machine: rows through `row_table` int6 and layer 0 from the local checkpoint, about 3–4 GB RAM.
- Here: a 6.5 GB range fetch, about 8 GB peak RAM, rows about 15 min per 20k tokens **(est.)**.
- Corpus: 20–50k tokens of the target codebase plus prose.

What it records, per token and per stream (4 streams):
- gate value sigmoid(g_s);
- the share of the output coming from the direct term u vs the conv term c;
- ‖PLE out‖ / ‖X‖ (PLE output relative to the residual stream);
- leave-one-order-out change in the gate logit and in v.

How it is cut:
- by token class (identifiers, punctuation/whitespace, function words);
- by G's row coherence;
- by n-gram frequency;
- within-n-gram vs between-n-gram gate variance (the same n-gram in different contexts).

What it decides:
- **The gate (D2).** If gate variance is mostly between n-grams and tracks coherence, the gate is a denoiser that a sparse graft can drop, so a fixed 0.5 is fine. If it is within-n-gram (context selection, as in the Engram paper's Fig. 7 on completed multi-token patterns), the graft needs a calibrated gate. The `W_k=0` dead point (`P/graft/calib_grad.out`) would then need a non-zero `W_k` init.
- **Graft output scale and trust region ρ.**
- **Whether a 1-stream graft loses much.** The streams use different norm weights and conv taps, with stream 1 dominant, so the probe shows how the 4 streams divide the work.

**A — serve ablation sweep, forward-only, through `/ple/score`. Cost: 2–3 days of engine work, mostly the M1 work already planned; about 1–1.5 h of serve time.**

Variants:
1. none
2. zero all rows
3. zero 2-gram heads
4. zero 3-gram heads
5. mean-row
6. shuffle
7. gate constant (boot flag)
8. conv off (boot flag)

Each runs on about 0.5M tokens of code (the owner's repos) plus prose. At 1,250–1,430 tok/s prefill on the EP arm (`T/serve.sh:1145-1148`) that is about 6–7 min per variant **(est.)**, plus about 300 s per boot-flag restart. Run at bs=1 with a fixed chunk size, and record the no-ablation repeat-run noise floor (DESIGN §6 I1 rule).

Report ΔNLL per token, bucketed by:
- token class;
- "n-gram seen earlier in this file/repo" vs not;
- G's coherence bin.

What it decides:
- **How much a co-trained model depends on its table, and where (D5).** The Engram paper's suppression test kept only 29–44% of factual-QA performance against 81–93% on reading comprehension (arXiv 2601.07372 §6.3, verified on arxiv.org/html).
- **Mean-row ≈ zero** means content matters; **mean-row ≈ none** means the table acts mostly as a bias.
- **Gate-constant and conv-off losses** say directly what the graft's fixed gate and `conv=0` cost in a model that was trained to use them.
- Caveat: an ablation is a train/inference mismatch, so it measures dependence, not value.

### Phase 2 (after the top 3)

**D — where the PLE contribution goes downstream (Flash-Next full depth).**
- Measures:
  - per-layer ‖Δh_L‖ under zero-rows vs none;
  - logit-lens KL to the final layer per layer, with and without the table;
  - CKA between Flash-Next's post-PLE layer-1 state and each layer of the 27B and 35B-A3B (the Engram paper found its layer 5 ≈ layer 12 of the MoE baseline; §6.1).
- Needs per-layer hidden states: either a debug dump flag in the serve (torch-side, prefill only) or an HF forward-only session during downtime (INTEGRATION option B, about 70 GiB host).
- Decides the injection site (D1). The Engram paper's single-module sweep found layer 2 best; Flash-Next uses decoder index 1.

**M — residual-space map (CPU).**
- Paired layer-1-input states from the Flash-Next slice (from S) and from a 27B slice (embed + layer 0, 3.3 GB). The token streams align position for position because the tokenizers are identical (`P/graft/tok_check.out`).
- Fit a ridge map M from 2560 to 5120 dimensions; report held-out R².
- Cost: hours. M is the prerequisite for E and F.

**E — ingested graft rows vs native rows (27B).** Uses Stage-1 cartridge output; CPU minutes.
- Map-free: RSA/CKA between the n-gram×n-gram similarity of Flash-Next's v and of the graft's v, over the codebase's n-grams.
- Mapped: cos(M·v_FN(g), v_27B(g)), per frequency bin.
- Prediction from G0: agreement only above f*. Below f*, the graft holds information that Flash-Next lacks.

**F — transplant.**
- Graft rows are set to e = 2·W_vᵀ·M·v_FN, the minimum-norm rows for the graft's linear output (`out = 0.5·W_v e`, DESIGN §2.4).
- Arms:
  - zero graft;
  - transplant;
  - shuffled transplant;
  - transplant restricted to n-grams above f*;
  - scale ×0.5, ×1 and ×2.
- Score held-out code loss on the Stage-1 27B harness: about one eval pass per arm, GPU minutes, with the serve down **(est.)**.
- Prediction: about 0; a positive result would give a free generic init.

### Phase 3 (expensive; the cleanest with/without contrast)

**N — native cartridge.**
- Ingest the same codebase into Flash-Next's own rows as an fp32 delta overlay (backbone frozen), and compare with the 27B graft: same keys, but a reader that was co-trained.
- Cost: option B is 3–10 s per iteration during downtime, while local CPU (option A) is 54–115 s per iteration (INTEGRATION §4.2), which is too slow for more than a small sample **(est.)**.
- Run it only if A shows a large code dependence.

## 4. How the experiments map to graft design decisions

| Decision | G | S | A | D | M / E / F |
|---|---|---|---|---|---|
| Gate: fixed 0.5 or calibrated | coherence as a covariate | **primary** | gate-constant ΔNLL | | |
| Conv: 0 or a pointwise nonlinearity | | u vs c share | **conv-off ΔNLL** | | |
| Output scale and trust region ρ | ‖v‖ by frequency | **‖PLE‖/‖X‖** | | | |
| Injection site | | | | **CKA / lens** | |
| Which n-grams to store (cartridge budget) | **f\*** | | per-class and seen-before ΔNLL | | E |
| Row weight decay and norm prior | **row norm vs load** | | | | |
| Transplant init | predicts ≈ 0 | | | | **F** |

Files written in this session (no commits):
- `D/range_probe.py` → `D/range_probe.out`
- `D/row_pilot.py` → `D/row_pilot.out`, `D/row_pilot.json`
- `D/embed_scale.py` → `D/embed_scale.out`

`D/` expands to `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/dissect/`.

# SKEPTIC VERDICT ON dissection (takes precedence where it conflicts)

# Skeptic verdict on the Flash-Next dissection report

**Overall:** the code mechanics and sizes check out. Two conclusions are wrong: the ones about the conv and gate, and the output scale. The plan's order also doesn't match your priority, which is to start with the grafts on Qwen3.6-35B-A3B and Qwen3.8-27B. Flash-Next is for dissection.

My checks are in `/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/graftfirst/skeptic_dissect/`:
- `branch_share.py` → `branch_share.out`
- `embed_freq.py` → `embed_freq.out`
- `dl/engram.html` (the Engram paper's arXiv page)

## Verdicts

**Confirmed**
1. **Sizes.** Checked through the huggingface_hub API and against `P/compute/hdr`:
   - parameter counts: Flash-Next 180.0B, 27B 27.78B, 35B-A3B 35.95B;
   - layer 0: 5.188 / 0.767 / 1.685 GB;
   - token embedding: 1.271 / 2.543 / 1.017 GB;
   - Flash-Next's PLE block without the table is 0.0657 GB, and the table is 102.4 GB in bf16;
   - host configs: the 27B is qwen3_5_text with 64 layers and d=5120; the 35B-A3B is qwen3_5_moe_text with 40 layers, d=2048 and 256 experts.
2. **Code citations are accurate.**
   - Transformers model file: `q4:1196-1197` (no bias), `1240-1241` (one gate scalar per token per stream; correction #3 holds), `1278`.
   - Engine: `R/ple/source.py:227, 243, 244`, `R/models/qwen4exp.py:421-443`, `R/weights/ckpt_read.py:5-10` (boot 300.1 s), `R/server/api_server.py:415-420, 1637-1638`.
   - `ple_layer_ids=[2]` means decoder index 1 (`q4:1262`), so "the gate depends only on what enters layer 1" is right.
   - Zeroing rows is an exact no-op. RMSNorm(0)=0 because eps is inside the rsqrt, and this was already shown bit-exact in `P/graft/bf16_noop.out`.
3. **Range-read numbers** match `range_probe.out` and `row_pilot.out` (200 rows/s at 48 threads).
   - "16 rows each" means both orders at one position; a single-order n-gram is 8 rows.
   - The 7.7M figure is 8 × (299,887 + 671,681) for the whole `python/` tree (1.44M tokens), not only the minisgl package.
   - 10.8–16 h follows from that. HF rate limits at 10^4–10^5 requests are untested.
4. **Coherence arithmetic.** SNR = 8ρ/(1−ρ) is correct (1.41 / 1.93 / 0.024).
5. **Engram paper (arXiv 2601.07372)** citations are correct: §6.3 (29–44% vs 81–93%), §6.1.2 (layer 5 ≈ layer 12), §6.2 (Layer 2 best), Fig. 7 (§6.5).
   - But Flash-Next's `ple_layer_ids=[2]` is 1-based, so it sits at the paper's "Layer 2". It is not a different site, as the report implies.

**Wrong**

6. **"The conv is mostly a pointwise term; conv=0 drops a per-position nonlinearity."** This is wrong.
   - `norm_conv` normalises the gated value per token and per stream before the conv. So the conv branch depends only on the direction of v, and the gate cancels in it. The only leak is through eps (1e-6 against a variance of about 8e-6).
   - On real weights (`branch_share.out`, gate 0.5), the conv branch is larger than the direct term on every stream: 1.4×, 13.9×, 2.7× and 1.7×. `P/t1/e_express.out` at a real anchor agrees: |silu(conv)| is 2.14 against |u| 0.256 on stream 1.
   - The earlier taps (t−3, t−6, t−9) carry 47–75% of the conv's size on streams 0, 2 and 3. Only stream 1 is pointwise (2%).
   - On stream 1, silu is nearly linear (about 9% nonlinear).
   - DESIGN §4.1:279 already says the stream-1 tap "depends only on the direction of v". So conv=0 removes the dominant, direction-only path, not a small nonlinearity.
7. **What the gate-constant ablation (A) can tell us.**
   - In Flash-Next it only touches the direct term, which is about 7% of stream 1's size. A small ΔNLL is expected and says nothing about a conv=0 graft, where the gate multiplies the whole output.
   - Likewise, the conv-off ΔNLL mainly measures the loss of most of the PLE's output plus a train/inference mismatch. It does not measure "what conv=0 costs the graft".
   - The idea that the gate is a "collision denoiser" is weakened, because the gate cannot touch the dominant branch.
8. **The output-scale calibration is wrong** ("0.6× host embedding RMS ≈ 0.008").
   - Sampling the vocabulary uniformly doubles Flash-Next's embedding RMS. Weighted by token frequency on real code (`embed_freq.out`, the top 256 tokens cover 60% of the text), it is 0.0038, against 0.0138 on the 27B and 0.0094 on the 35B-A3B. That matches DESIGN's 0.0042.
   - Flash-Next's PLE output is dominated by the conv: RMS about 0.04 on stream 1, about 10× its token embedding.
   - The report used only RMS(v) = 0.0058 and ignored the conv branch.

**Over-claimed or unsupported**

9. **Lens.** The value lines up with lm_head of the next token (z +1.03) almost exactly as much as with lm_head of the current token (+1.02). So it is not next-token-specific, and "a weak direct-logit signal" is not supported. The sample is 67 n-grams, mostly whitespace.
10. **"Never-read padding rows decay to about 0, so training used weight decay."** Those rows have a median norm of 8e-18, but a mean of 0.015 and a max of 0.122. So not all of them decayed, and "never-read" is assumed, not shown. Treat it as a hint only.
11. **"Project-specific rows carry essentially no signal."** It holds with caveats:
    - frequency here is counted in a 400k-token local corpus, a weak proxy for pretraining frequency;
    - n is 32–48 per bucket;
    - only the value path is measured.

    "Transplant gain ≈ 0" remains a fair hypothesis.
12. **"A does not stop the serve."** Too strong:
    - The row-side switches need the M1 control plane deployed, which means at least one restart.
    - The radix cache, and the GDN and conv states it holds, must be turned off or flushed between variants, or one variant contaminates the next.
    - The serve's body is MXFP4 with W4A8, so A measures the quantised model.
13. **A's throughput is optimistic (est.).**
    - `T/` is not defined in the report; it is `minisglang-rdna4/tools/serve.sh`. The 1,250–1,429 tok/s at lines 1145–1148 is for a single 16k-token prompt.
    - Real files average about 5.6k tokens (1.44M tokens over 259 files).
    - Scoring needs lm_head (248k vocab) at every position, with NLL computed on the GPU. Normal prefill does neither.
    - Expect 1.5–2× the 6–7 min per variant (est.).
14. **S on your machine.**
    - Your local checkpoint has MXFP4 experts (INTEGRATION §4.1 and the nvfp4 prepass in `ckpt_read.py`). HF's `Qwen4ExpText` cannot load it without a dequantising loader.
    - "3–4 GB RAM" underestimates: layer 0 in bf16 alone is 5.19 GB, plus 1.27 GB of embedding.
    - Running it here from the public bf16 weights (6.5 GB fetch, about 8 GB peak) fits in 15 GB only if everything stays bf16.
15. **F, "GPU minutes per arm" on the 27B (est.).**
    - The 27B is 55.6 GB in bf16 and needs a 4-bit backbone to fit 2×16 GB (`vram_fit.out`: 24.7 GB, est.).
    - A forward pass is about 56 GFLOP per token. `cost_model.out` assumes 54–122 TFLOP/s effective, which is optimistic for RDNA4 on ROCm, and a layer split leaves one card idle at a time.
    - For about 0.2M tokens, plan on 2–4 min ideal and 2–3× that in practice.

## Corrected key conclusions

1. **Order of work.** Make the Stage-1 grafts on Qwen3.6-35B-A3B and Qwen3.8-27B the main track. G (mostly done; extend it on CPU) and S (here, bf16 range fetch) run alongside, because they are cheap and feed the graft design directly. A depends on the M1 work on the Flash-Next serve (`/ple/score`, control plane, cache flush) and should wait for a graft baseline, unless M1 is being built anyway.
2. **Flash-Next's PLE is mainly a gate-independent, direction-only conv path** (stream 1, about 10× embedding scale), plus a small gated direct term. The planned graft (conv=0, fixed gate 0.5, output linear in row norm) works in a different regime. Two arms are needed:
   - a graft with a norm_conv-style direction branch, with a current-position tap only, or with taps;
   - the linear conv=0 graft.

   A's gate-constant result should not be used to justify a fixed gate in a conv=0 graft.
3. **Output scale.** Do not use embedding ratios. Set it from S's measured ‖PLE‖/‖X‖ per stream after the hyper-connection read, using the frequency-weighted embedding RMS above (0.0038 / 0.0138 / 0.0094) only as context.
4. **Row statistics.** Flash-Next rows for locally rare n-grams are dominated by the other n-grams sharing their bucket. A cartridge should spend its rows on n-grams that are frequent in the project but rare in general. A transplant from Flash-Next to the 27B is likely worth about 0 (hypothesis). The lens and weight-decay findings are hints only.
5. **All throughput and RAM figures stay estimates** until measured on your RDNA4 machine: A per variant 10–14 min, F per arm 2–12 min, S needs a dequantising loader to run from your MXFP4 checkpoint.

# M2-0: one-layer feasibility on the owner's cards

M2-0 (DESIGN §8, §7.5 #1–2, §4.6) asks whether graft training on `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`
(revision `f1e8415c4f29508fe281da4e256df26573213bfd`) fits and runs fast enough on the two 16 GB gfx1201
cards. The code is validated on CPU in this repo (numbers below). What remains has to be measured on the cards.

## What to run

Stop the serve first. Then, from the repo root:

```
experiments/m2_0/run_owner.sh
```

The script runs these steps in order:

1. **Preflight checks.** The script refuses to run if any of these is true:
   - the serve is up: port 1919 answers, or a `minisgl` / `sglang` / `vllm` serve process or container is running;
   - some process holds a GPU (`/sys/class/kfd`);
   - `gpu-lease` is not on `PATH`;
   - `gpu-lease -n 2 --nowait` cannot get both cards.
2. **Takes the lease.** It re-runs itself under `gpu-lease -n 2`.
3. **Inner checks.** Inside the lease it checks:
   - exactly two `gfx1201` cards are visible;
   - each card has less than 1.5 GB in use;
   - `gdn_hip` is importable.
4. **Weights.** It uses the local HF snapshot of the pinned revision (or `--ckpt DIR`).
   - With no snapshot, it range-reads only layers 0 and 3 (about 1.07 GB) into `~/.cache/ngram-memory/m2_0_cache`.
   - It downloads a whole checkpoint only with `--allow-full-download`.
5. **Measurement steps.** It runs the five steps below.

Options:

- `--preflight-only` runs only the checks.
- `--quick` is a smoke run at small sizes, about 10–15 minutes.
- `--lease-timeout S` waits up to S seconds for the lease instead of refusing.
- `--no-download` never touches the network.

The environment needs torch-ROCm for gfx1201, transformers 5.22.x and the fork's built `gdn_hip`:

- `PYTHON=/path/to/venv/bin/python` selects the interpreter; the default is `python3`.
- `NGM_MINISGL_FORK=<fork clone>` points to the fork; the default is the clone path in `gdn_native.py`.
- `NGM_IMAGE=<image>` runs the steps in a container instead, with the fork's standard ROCm passthrough flags and the device pair the lease sets.

**Expected time: 1–2 h (est.).**

| Step | Script | Est. |
|---|---|---|
| grouped_mm at the real MoE shapes, both cards | `grouped_mm_probe.py` | 2–5 min |
| GDN core at B=4, T=1024/4096 (bf16); real layer-0 mixer at B=4 T=4096 bf16 and B=1 fp32; native math vs an fp64 truth; model-level train step | `gdn_validate.py --gpu` | 15–30 min |
| AWQ: L0 (+L3) expert bit-exactness; whole layers vs an fp32 reference (computed on the CPU); memory per piece at 4k / 8k / 16k | `awq_layers_validate.py --gpu` | 20–40 min |
| Planned split: all 40 layers on the two cards, one training step at 8k and 16k tokens with the fixed GDN backward, plus 8k with the fork's reference backward | `split_peak.py` | 10–30 min |

## What it prints

Each step's output streams to the terminal and is saved to `logs/<step>.log`. At the end the script prints
`M2_0_SUMMARY.txt`, which gives one line per criterion with PASS / FAIL / N/A and the measured value.

**M2-0 pass bars and kill criterion (DESIGN §8 M2-0 row):**

- **M2-0.1:** the fixed GDN backward costs at most 1.5× the native forward per token, at B=4, T=4096, bf16. It is reported for both fixed variants:
  - `fix` recomputes all sequences at once;
  - `fix_g1` recomputes one sequence at a time.

  Each variant is shown with its op count and peak VRAM.
- **M2-0.2:** dX cosine ≥ 0.9999 against HF's torch delta rule, on layer 0's real mixer.
- **M2-0.3@8192 / @16384:** per-card peak ≤ 15.5 GB at the planned split, with 4k mixer pieces and 8k or 16k MoE chunks.
  - The bar is judged on `max_memory_allocated`; `max_memory_reserved` is printed beside it.
  - The same step's time is converted to h per 1M tokens, to compare with §7.2's 0.1–0.45 h.
- **M2-0.K (kill):** the GDN backward stays under 0.2 h per 1M tokens over the 30 GDN layers. If it does not, M2 must be re-costed before it starts.

**M2 prerequisites that M2-0 measures:**

- **M2.fwd:** forward relative error ≤ 1e-3 against HF (see the bf16 caveat below).
- **M2.shim:** the fork's `NativeGDNShim`, widened to `Qwen3_5MoeGatedDeltaNet`, matches HF.
- **M2.awq:** the AWQ dequant is bit-exact.
- **M2.model:** in a model-level step, every GDN layer runs the patched kernel.
- **7.5#2:** `torch._grouped_mm` runs on gfx12. The value is its speed-up over the per-expert loop.

**Info lines:**

- the reference backward's h per 1M tokens and ops per token;
- the reference backward's per-card peak at 8k;
- native math against an fp64 truth.

## What to send back

At the end the script names two files to send:

- `experiments/m2_0/results/owner_<time>/M2_0_SUMMARY.txt`;
- `experiments/m2_0/results/owner_<time>.tar.gz`, which holds every JSON and log and no weights.

If a step fails, the archive still holds its log. Exit 76 from `gpu-lease` means a card wedged during the run, not a bug in the command; re-run it.

## Validated on CPU here (no GPU)

| What | Result | Source |
|---|---|---|
| AWQ experts, L0 and L3 | 768/768 matrices per layer bit-exact against the numpy reference (fp32 and bf16) and against the fork's decode; stacked module 256/256 | `results/awq/awq_validate_cpu.json` (L0), `results/integration/awq_bitexact_L0_L3_cpu.json` (L0 + L3) |
| AWQ layers vs fp32 reference | fp32: relative error ≤ 1.7e-7, dX cosine 1.0. bf16: L0 layer 3.2e-3 / 0.999985; MoE block 0.021 / 0.99966 (router flips on about 5% of tokens; experts with the reference routing 5.4e-3 / 0.999984) | same |
| Fixed GDN backward (core) | Matches HF autograd exactly. Backward ops per token at B=4·T=1024: 0.43 (`fix`), 1.67 (`fix_g1`), 0.12 (ours), 0.24 (HF), 24.06 (fork reference). Backward transient live bytes at 4,096 tokens, bf16: 1.58 GB / 0.45 GB / 1.80 GB / 0.54 GB / 1.27 GB | `results/gdn/cpu_validate.json` (the CPU native forward is HF's own forward behind the native signature, so its accuracy numbers only check the wiring) |
| Native math vs fp64 truth | Inputs at T ∈ {1, 37, 64, 65, 130, 200}. fp32: every arm has forward rel ≤ 2.6e-6 and gradient cosine ≥ 0.99999999. bf16: the fixed backward (fp32 sigmoid) 1.7e-3 / 0.9999969, HF's own bf16 2.4e-3 / 0.9999923 | `results/integration/gdn_exact_model_cpu.json` |
| Model-level train step | 4-layer model, 35B GDN geometry, frozen host. With HF's default `use_cache=True`, every arm runs 3/3 GDN layers natively with 0 fallbacks; input-gradient cosine against unpatched HF ≥ 0.999999999996 | same |
| Planned-split step | Real cached weights (L0 replicated, L3 dense), 4 layers, CPU. Checkpointing on vs off gives the same dX (rel 0.0); 12/12 GDN calls patched | `results/integration/split_cpu_real.json`, `selftest_split_cpu.json` |
| grouped_mm on CPU | Runs at E=256, d=2048, I=512 with uneven counts and empty experts (128 tokens); dX cosine 0.999993 against an fp32 loop | `results/integration/grouped_mm_cpu.json` |

**The bf16 forward bar (M2.fwd, DESIGN §8 M2).** An exact native kernel cannot reach 1e-3 against HF's bf16
torch path.

- The reason is that HF rounds sigmoid(b) to bf16 while the kernel keeps it fp32. On layer 0 at B=1, T=4096 the exact math is 2.55e-3 from HF but 1.6e-4 from HF computed with an fp32 sigmoid.
- Both are about 5.4e-3 from an fp32 truth.
- Even fp64-exact math rounded to a bf16 output is about 1.7e-3 from fp64.

So the summary prints M2.fwd against HF, with the fp32-sigmoid HF and the fp32 truth beside it. Whether to restate the bar is the owner's decision.

The whole CPU self-test of the runner (`run_owner.py --selftest-cpu`) is in
`results/integration/selftest_cpu_SUMMARY.txt`: GPU bars N/A, the CPU-checkable ones PASS, M2.fwd as above.

## Files

- `run_owner.sh`, `run_owner.py`: the one-command run, including preflight, lease, weights, steps, criteria and the archive. `--selftest-cpu` runs every step at CPU sizes.
- `gdn_validate.py`: the GDN arms (parts core, layer, exact, model).
- `awq_layers_validate.py`: the AWQ checks (parts 1–5).
- `split_peak.py`: per-card peak memory and step time at the planned split.
- `grouped_mm_probe.py`: `torch._grouped_mm` at the MoE shapes.
- `fetch_layer_cache.py`: per-layer HTTP range reads with sha256 checks and a manifest.
  - It is append-only.
  - It enforces a per-call cap and a cumulative track budget (`DOWNLOADS.json` ledger).
  - `--dry-run` shows the bytes it would fetch.
- `scripts/m2_0_feasibility.py` (M0) is superseded by this run for the GPU measurements. Its GDN bars use the older native+ours path.

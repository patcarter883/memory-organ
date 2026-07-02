# Reproducing the results

Every table in [RESULTS.md](RESULTS.md) should be reachable from a command on this page. The rule this
file exists to enforce: **a non-reproduction report is only actionable if we both know you ran the same
thing** — same command, same seed, same versions. If a number here doesn't match what you get, that is
exactly the report we want: see [CONTRIBUTING.md](CONTRIBUTING.md).

## The recipe

Unless a table notes otherwise (RESULTS.md line "Recipe unless noted"):

- seed `20260625`, bind **6000** steps, tap/translator **3000** steps, batch 16, lr 1e-3
- donor (base-1): Qwen3.5-4B · transfer bases: `unsloth/gemma-3-4b-pt`, `Qwen/Qwen3-0.6B`,
  `unsloth/Llama-3.2-3B`
- store of record: product-key + addressing supervision (`--store pk --addr-sup-weight 1.0`)

> **Driver defaults are NOT the recipe.** The drivers default to `--bind-steps 3000` and per-driver
> seeds (`20260624`/`20260625`/`20260628`/`20260629` across the scripts, kept for historical
> byte-reproducibility of earlier runs). The commands below pass the recipe values explicitly — run
> them as written, don't rely on defaults.

## Hardware notes

- Developed on a 16 GB AMD card (ROCm). The eval forward is the memory hog; if you OOM, lower
  `CAM_EVAL_BATCH_CAP` (env var, default 128 — memory-only, accuracy-neutral):
  `CAM_EVAL_BATCH_CAP=48 python -m cam.recall_v1 ...`
- CPU/CUDA are expected to work (pure PyTorch) but are **not verified** — a CPU run that works (or
  doesn't) is itself a valuable report. CI smoke-tests the store/tap/adapter layer on CPU.
- Checkpoints are saved/loaded with `torch.load(..., weights_only=False)` (they carry small metadata
  dicts). Only load checkpoints you produced yourself or trust — unpickling executes code.

## Per-table commands

### §1 Capacity ladder (carry vs M)

```bash
# store of record (product-key + addressing supervision), flat through M=128
python -m cam.bind_msweep --store pk --addr-sup-weight 1.0 --pk-read-heads 8 \
    --Ms 8,16,32,64,128 --bind-steps 6000 --batch 16 --lr 1e-3 --seed 20260625

# naive-store baseline (the wall)
python -m cam.bind_msweep --store bolt --Ms 8,16,32 --bind-steps 6000 --batch 16 --lr 1e-3 --seed 20260625

# the port-bug ablation (product-key WITHOUT addr-sup — under-loaded-lossy)
python -m cam.bind_msweep --store pk --addr-sup-weight 0.0 --Ms 8,16,32 \
    --bind-steps 6000 --batch 16 --lr 1e-3 --seed 20260625
```

### §3 Single-token pipeline (bind → deliver → transfer)

```bash
# bind + deliver into frozen base-1, save the reusable memory checkpoint
python -m cam.recall_mag --store pk --addr-sup-weight 1.0 --M 8 \
    --bind-steps 6000 --steps 3000 --batch 16 --lr 1e-3 --seed 20260625 --save-ckpt ckpt/m8.pt

# transfer the SAME frozen memory to base-2
python -m cam.recall_v1 --load-ckpt ckpt/m8.pt --M 8 --steps 3000 --batch 16 --lr 1e-3 \
    --seed 20260625 --base2 unsloth/gemma-3-4b-pt

# cross-family falsifier (foreign tokenizer + architecture)
python -m cam.recall_v1 --load-ckpt ckpt/m8.pt --M 8 --steps 3000 --batch 16 --lr 1e-3 \
    --seed 20260625 --base2 unsloth/Llama-3.2-3B
```

For the M=64 rung, repeat with `--M 64`.

### §4 Multi-token answers (disjoint per-position store + per-position MLP translator)

```bash
python -m cam.recall_mag --store pk --readout perpos --perpos-key disjoint --cargo-tokens 2 \
    --addr-sup-weight 1.0 --pk-read-heads 8 --M 8 \
    --bind-steps 6000 --steps 3000 --batch 16 --lr 1e-3 --seed 20260625 --save-ckpt ckpt/mt.pt

python -m cam.recall_v1 --load-ckpt ckpt/mt.pt --M 8 --cargo-tokens 2 --xlator perpos-mlp \
    --steps 3000 --batch 16 --lr 1e-3 --seed 20260625 --base2 unsloth/gemma-3-4b-pt
```

The §4 translator-variant ladder is the same command with `--xlator affine|mlp|perpos|perpos-mlp`.

### §5 Real knowledge (natural / varied / multi-token natural)

```bash
# natural single-relation prose
python -m cam.recall_mag --store pk --addr-sup-weight 1.0 --M 8 --phrasing natural \
    --bind-steps 6000 --steps 3000 --batch 16 --lr 1e-3 --seed 20260625 --save-ckpt ckpt/nat.pt
python -m cam.recall_v1 --load-ckpt ckpt/nat.pt --M 8 --steps 3000 --batch 16 --lr 1e-3 \
    --seed 20260625 --base2 unsloth/gemma-3-4b-pt

# varied relations: --phrasing varied ; multi-token natural: --phrasing natural --cargo-tokens 2
```

### §6 Knowledge editing (probe → filter → edit)

```bash
# same-base (Qwen): probe/filter/derange/bind/eval in one run
python -m cam.recall_mag --store pk --addr-sup-weight 1.0 --M 8 --phrasing counterfactual \
    --bind-steps 6000 --steps 3000 --batch 16 --lr 1e-3 --seed 20260625 --save-ckpt ckpt/cf.pt

# cross-family editing transfer (Gemma) — includes the base-2 probe→filter validity gate
python -m cam.recall_v1 --load-ckpt ckpt/cf.pt --M 8 --steps 3000 --batch 16 --lr 1e-3 \
    --seed 20260625 --base2 unsloth/gemma-3-4b-pt
```

## Environment record

Capture the environment alongside any result you report (and this is how the canonical record should
be produced on the dev box):

```bash
python -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"
pip freeze > env-freeze.txt
```

> **Not yet recorded in-repo** (tracked in the issues): the exact `pip freeze` of the ROCm box that
> produced the published tables, the two additional seeds behind the 3-seed error bars in §3, and the
> tap layer used per headline run (drivers default to `n_layers // 2`). Until those are backfilled
> from the run logs, treat small deviations from the published numbers on different versions as
> unconfirmed rather than as a non-reproduction.

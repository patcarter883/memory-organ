# Stage-1 eval: /home/memory-organ/ngram-memory/hf/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987

| set | condition | tokens | files | base NLL | arm NLL | Δ nats | rel % [95% file CI] | ΔEM pts | KL | s / 10k tok |
|---|---|---|---|---|---|---|---|---|---|---|

## R3 capability (forward-only MC)

| condition | task | n | base acc | arm acc | Δ pts | McNemar p | arm / base |
|---|---|---|---|---|---|---|---|
| rows[cartridge] | arc_challenge | 300 | 0.327 | 0.327 | +0.0 | 1 | 1.000 |
| rows[cartridge] | hellaswag | 300 | 0.363 | 0.363 | +0.0 | 1 | 1.000 |
| rows[cartridge] | mmlu | 300 | 0.313 | 0.310 | -0.3 | 1 | 0.989 |
| rows[cartridge] | all | 900 | 0.334 | 0.333 | -0.1 | 1 | 0.997 |

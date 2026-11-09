# engram_audit — scripts behind the Gemini grafted-Engram audit

Supporting material for [docs/research/gemini-engram-code-review.md](../../docs/research/gemini-engram-code-review.md)
and [docs/research/engram-relaunch-proposal.md](../../docs/research/engram-relaunch-proposal.md). Everything here
ran on CPU (torch 2.15, transformers 5.22, Qwen/Qwen2.5-0.5B snapshot `060db649…`) on 2026-11-09.

| Path | What |
|---|---|
| `gemini/` | Gemini's three code files, **verbatim and unedited** (they contain the bugs the review documents), plus `CLAIMS.md` (claims C1–C9). |
| `audit/<lens>/` | Scripts and outputs per audit lens (`mechanics`, `engramedit-solver`, `calibration`, `fidelity`, `experiment`). |
| `audit/verify-<lens>/` | The independent skeptic re-checks of each lens. |
| `audit/experiment/run.py`, `results.json`, `run.log` | The 12-fact Qwen2.5-0.5B experiment (§5 of the review). |
| `v2check/` | Checks on Gemini's two later revisions (review §8): `ridge_and_grad.py` (joint ridge allocation, zero-`v_proj` gradient), `v3_checks.py` (short-subject generalisation, collision exposure, table memory). |

Not copied: the arXiv HTML used by the `fidelity` lens (fetch it again from the URLs in the review), and the
CounterFact parquet (`azhx/counterfact`, test split; `run.py` downloads it).

**Re-running.** The scripts hard-code the session scratchpad they ran in,
`/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad`, with this folder's
layout under it. The least invasive way to re-run them unchanged is to recreate that path as a symlink to this
folder, plus a CPU venv:

```bash
S=/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad
mkdir -p "$(dirname "$S")" && ln -s "$PWD/tools/engram_audit" "$S"
python -m venv "$S/venv" && "$S/venv/bin/pip" install --index-url https://download.pytorch.org/whl/cpu torch && "$S/venv/bin/pip" install transformers huggingface_hub datasets
```

Then follow the commands in §7 of the review. Some scripts also load the model from the Hugging Face cache
snapshot path (`/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db649…`); download it first
with `huggingface_hub.snapshot_download("Qwen/Qwen2.5-0.5B")`.

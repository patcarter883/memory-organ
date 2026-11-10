"""End-to-end run of scripts/evaluate.py on a tiny random Qwen2 host (saved with the real Qwen2.5 tokenizer) and a
tiny git repo: every arm builds, scores the same targets as base, and the summary / report are written."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
QWEN25 = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")


def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True)


@pytest.mark.skipif(not (QWEN25 / "tokenizer.json").exists(), reason="Qwen2.5 tokenizer not cached")
def test_evaluate_cli_all_arms(tmp_path, monkeypatch):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from ngram_memory.eval.stage1.splits import Splits, build_splits
    from ngram_memory.spec import KeySpec, graft_format
    from ngram_memory.tables import SparseRowStore
    repo = tmp_path / "repo"
    (repo / "python/pkg").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    for i in range(30):
        body = "\n".join(f"def helper_{i}_{j}(value, other=None):\n    \"\"\"Doc {j}.\"\"\"\n    return value * {j} + {i}  # c"
                         for j in range(12))
        (repo / f"python/pkg/mod{i}.py").write_text(f"import os\n\nLIMIT_{i} = {i}\n\n{body}\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "c1")
    cut = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (repo / "python/pkg/mod1.py").write_text((repo / "python/pkg/mod1.py").read_text() + "\nNEW_VALUE = helper_1_2(3)\n")
    _git(repo, "commit", "-qam", "c2")
    sp = tmp_path / "splits.json"
    Splits(build_splits(str(repo), "HEAD", "python", t_cut=cut)).save(sp)
    assert Splits.load(sp).files("a", "dev")
    host = tmp_path / "host"
    tok = AutoTokenizer.from_pretrained(str(QWEN25))
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, eos_token_id=151643, bos_token_id=151643, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(host)
    tok.save_pretrained(host)
    key = KeySpec(vocab_size=151936, eos_token_id=151643, base=1000)
    fmt = graft_format(64, key)
    (tmp_path / "fmt.json").write_text(fmt.to_json())
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, key.total_rows, (2000,), generator=g).unique()
    rows = tmp_path / "rows.pt"
    torch.save(SparseRowStore(fmt.padded_rows, fmt.row_width, ids, torch.randn(len(ids), fmt.row_width, generator=g)).state_dict(), rows)
    (tmp_path / "ref.json").write_text(json.dumps({"os": 5, "value": 3, "other": 3}))
    out = tmp_path / "out"
    conds = f"base,countlm,rag1k,knm,ngm,rows:{rows},perm:{rows},random:{rows},constbias:{rows},rag1k+rows:{rows},lora"
    argv = ["evaluate.py", "--splits", str(sp), "--host", str(host), "--out", str(out), "--sets", "dev,splitb,r2",
            "--context", "128", "--stride", "64", "--min-context", "64", "--max-files", "1", "--tune-files", "1",
            "--retention-tokens", "256", "--knm-tokens", "256", "--lora-tokens", "256", "--lora-window", "64",
            "--lora-lr", "1e-3", "--threads", "1", "--fmt", str(tmp_path / "fmt.json"), "--ref-vocab-cache",
            str(tmp_path / "ref.json"), "--ngm-ratios", "0.01,0.1", "--conditions", conds]
    spec = importlib.util.spec_from_file_location("evaluate_cli", ROOT / "scripts/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    s = json.loads((out / "summary.json").read_text())
    assert not [k for k in s["tuning"] if k.startswith("error:")], s["tuning"]
    want = {"base", "countlm", "rag1k", "knm", "ngm", "rows[rows]", "perm[rows]", "random[rows]", "constbias[rows]",
            "rag1k+rows[rows]", "lora"}
    for set_name in ("dev", "splitb", "r2"):
        got = {k for k in s["sets"][set_name] if not k.startswith("_")}
        assert got == want, (set_name, got ^ want)
        for k in want - {"base"}:
            p = s["sets"][set_name][k]["paired"]
            assert p["n_tokens"] == s["sets"][set_name]["base"]["n_tokens"] > 0
    assert "mean_kl" in s["sets"]["r2"]["rows[rows]"]["paired"]
    assert "classes" in s["sets"]["dev"]["rows[rows]"] and "dup:16gram_in_train" in s["sets"]["dev"]["_class_fractions"]
    assert s["warnings"] and (out / "report.md").read_text().startswith("# Stage-1 eval")

"""[Integration status: every defect below is FIXED; the former strict-xfail tests are regular regression tests.]
Adversarial-verifier tests for the Stage-1 eval piece. Each is xfail(strict=True) and names the defect id; it
starts passing (and so fails as XPASS) once the defect is fixed."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
QWEN25 = __import__("ngram_memory.paths", fromlist=["qwen25"]).qwen25()   # HF cache at the pinned revision
SENTINEL = "ZQXJ_POSTCUT_SENTINEL"


def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True)


@pytest.mark.skipif(not (QWEN25 / "tokenizer.json").exists(), reason="Qwen2.5 tokenizer not cached")
def test_ve1_splitb_countlm_not_fitted_on_post_cut_lines(tmp_path, monkeypatch):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from ngram_memory.eval.stage1.baselines import countlm as cl
    from ngram_memory.eval.stage1.splits import Splits, build_splits, path_hash

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
    # a Split-A TRAIN file (h >= 0.25) gets a post-cut line; Split B scores exactly that line
    victim = next(i for i in range(30) if path_hash(f"python/pkg/mod{i}.py") >= 0.25)
    p = repo / f"python/pkg/mod{victim}.py"
    p.write_text(p.read_text() + f"\n{SENTINEL} = helper_{victim}_2(3)\n")
    _git(repo, "commit", "-qam", "c2")
    sp = tmp_path / "splits.json"
    Splits(build_splits(str(repo), "HEAD", "python", t_cut=cut)).save(sp)
    s = Splits.load(sp)
    assert f"pkg/mod{victim}.py" in s.files("a", "train") and f"pkg/mod{victim}.py" in s.files("b", "test")

    host = tmp_path / "host"
    tok = AutoTokenizer.from_pretrained(str(QWEN25))
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, eos_token_id=151643, bos_token_id=151643, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(host)
    tok.save_pretrained(host)

    seen_text: list[str] = []
    orig = cl.KNTrigram.__init__

    def spy(self, seqs, vocab, eos):
        seqs = [list(x) for x in seqs]
        seen_text.extend(tok.decode(x) for x in seqs)
        orig(self, seqs, vocab, eos)

    monkeypatch.setattr(cl.KNTrigram, "__init__", spy)
    out = tmp_path / "out"
    argv = ["evaluate.py", "--splits", str(sp), "--host", str(host), "--out", str(out), "--sets", "splitb",
            "--context", "128", "--stride", "64", "--min-context", "64", "--tune-files", "1", "--threads", "1",
            "--no-classes", "--conditions", "base,countlm"]
    spec = importlib.util.spec_from_file_location("evaluate_cli_ve1", ROOT / "scripts/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    assert seen_text, "count LM was not built"
    assert not any(SENTINEL in t for t in seen_text), "Split-B count LM was trained on a line added after T_cut"

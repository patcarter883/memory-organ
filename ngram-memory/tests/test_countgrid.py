"""Count-LM frontier (ngram_memory/eval/stage1/countgrid.py), fixed-weight countlm specs and the non-duplicated
("nondup") paired metric, on the tiny qwen2 host (conftest.build_host), plus one scripts/evaluate.py run."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import EOS, V, build_host, fill_random_rows, tiny_fmt
from ngram_memory.eval import metrics as M
from ngram_memory.eval.stage1 import countgrid as cg
from ngram_memory.eval.stage1 import scoring as S
from ngram_memory.eval.stage1.baselines import countlm as cl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def qwen2():
    return build_host("qwen2")


@pytest.fixture(scope="module")
def setup():
    g = np.random.default_rng(0)
    pat = np.tile(np.array([7, 8, 9, 10, 11]), 80)
    train = [pat, g.integers(3, V, 300), g.integers(3, 40, 400)]
    kn = cl.KNTrigram(train, V, EOS)
    docs = [S.Doc("p", np.concatenate([pat[:120], g.integers(3, 40, 100)])), S.Doc("q", g.integers(3, V, 190))]
    wins = S.plan_windows(docs, 128, 64, 64)
    return kn, docs, wins


WEIGHTS = (0.0, 0.0005, 0.01, 0.2)


def test_grid_equals_single_weight_countlm_and_kl(qwen2, setup):
    kn, docs, wins = setup
    base = S.score(qwen2, docs, wins)
    g = cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "kn", WEIGHTS))
    assert np.array_equal(g.key(), base.key()) and g.nll.shape == (len(WEIGHTS), len(base))
    assert np.array_equal(g.host_nll, base.nll) and not g.meta["two_forwards"]
    # weight 0 = the host itself: NLL bit-identical, KL exactly 0, full agreement
    assert np.array_equal(g.nll[0], base.nll) and float(g.kl[0].max()) == 0.0 and g.agree[0].all()
    assert np.array_equal(g.hit[0], base.hit)
    for i, w in enumerate(WEIGHTS[1:], 1):
        r = S.score(qwen2, docs, wins, cl.CountLMCondition(kn, w, 0.0), with_kl=True)
        assert np.abs(g.nll[i] - r.nll).max() < 1e-6, w          # same float32 mixture as CountLMCondition
        assert np.abs(g.kl[i] - r.kl).max() < 1e-6, w            # same KL formula, base host from the same pass
        assert np.array_equal(g.hit[i], r.hit) and np.array_equal(g.agree[i], r.agree)
    assert (np.diff(g.kl.mean(1)) > 0).all()                      # KL grows with the weight


def test_grid_kl_matches_brute_force_float64(qwen2, setup):
    kn, docs, wins = setup
    g = cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "kn", WEIGHTS))
    w0 = wins[0]
    d = docs[w0.doc]
    with torch.no_grad():
        logits = qwen2(input_ids=torch.from_numpy(d.ids[w0.a:w0.e])[None]).logits[0].double()
    t = np.arange(w0.s, w0.e)
    lb = torch.log_softmax(logits[t - 1 - w0.a], -1)
    u, v = cl.doc_contexts(d.ids, t, EOS)
    u, v = np.asarray(u), np.asarray(v)
    pk = torch.stack([torch.from_numpy(np.array([kn.prob([a], [b], [y])[0] for y in range(V)])) for a, b in zip(u, v)])
    sel = (g.doc == w0.doc) & (g.pos >= w0.s) & (g.pos < w0.e)
    for i, w in enumerate(WEIGHTS):
        lm = ((1 - w) * lb.exp() + w * pk).log()
        kl = (lb.exp() * (lb - lm)).sum(-1).numpy()
        nll = -lm[torch.arange(len(t)), torch.from_numpy(d.ids[t])].numpy()
        assert np.abs(g.kl[i][sel] - kl).max() < 2e-5, w
        assert np.abs(g.nll[i][sel] - nll).max() < 2e-4, w


def test_grid_cache_and_fixed_other_weight(qwen2, setup):
    kn, docs, wins = setup
    gc_ = cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "cache", (0.0, 0.01, 0.05)))
    gk = cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "kn", (0.0, 0.01), other_weight=0.005))
    for i, w in enumerate((0.01, 0.05), 1):
        r = S.score(qwen2, docs, wins, cl.CountLMCondition(kn, 0.0, w))
        assert np.abs(gc_.nll[i] - r.nll).max() < 1e-6
    for i, w in enumerate((0.0, 0.01)):
        r = S.score(qwen2, docs, wins, cl.CountLMCondition(kn, w, 0.005))
        assert np.abs(gk.nll[i] - r.nll).max() < 1e-6
    assert gk.kl[0].mean() > 0  # weight 0 still carries the fixed cache weight


def test_grid_on_top_of_rows_kl_against_base(qwen2, setup):
    from ngram_memory.eval.stage1.conditions import RowsCondition
    kn, docs, wins = setup
    rc = RowsCondition(tiny_fmt(), None, name="rows[t]")
    for d in docs:
        fill_random_rows(rc.graft, torch.from_numpy(d.ids[None]), scale=0.5)
    base = S.score(qwen2, docs, wins)
    rows = S.score(qwen2, docs, wins, rc, with_kl=True)
    g = cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "kn", WEIGHTS, inner=rc))
    assert g.meta["two_forwards"] and np.array_equal(g.host_nll, base.nll)
    assert np.array_equal(g.nll[0], rows.nll) and np.abs(g.kl[0] - rows.kl).max() < 1e-6 and rows.kl.mean() > 0
    for i, w in enumerate(WEIGHTS[1:], 1):
        r = S.score(qwen2, docs, wins, S.Composite([cl.CountLMCondition(kn, w, 0.0), rc]), with_kl=True)
        assert np.abs(g.nll[i] - r.nll).max() < 1e-6 and np.abs(g.kl[i] - r.kl).max() < 1e-6
    # hooks are removed after scoring
    assert np.array_equal(S.score(qwen2, docs, wins).nll, base.nll)
    with pytest.raises(TypeError):
        S.score(qwen2, docs, wins, cg.CountGridCondition(kn))


def test_grid_respects_target_mask_and_roundtrip(qwen2, setup, tmp_path):
    kn, docs, wins = setup
    d = S.Doc("m", docs[0].ids, target_mask=(np.arange(len(docs[0])) % 3 == 0))
    w = S.plan_windows([d], 128, 64, 64)
    base = S.score(qwen2, [d], w)
    g = cg.score_grid(qwen2, [d], w, cg.CountGridCondition(kn, "kn", (0.0, 0.01)))
    assert np.array_equal(g.key(), base.key()) and len(g) == len(base) > 0
    g.save(tmp_path / "g.npz")
    h = cg.GridResult.load(tmp_path / "g.npz")
    assert np.array_equal(h.nll, g.nll) and np.array_equal(h.kl, g.kl) and h.weights.tolist() == [0.0, 0.01]


def test_nondup_paired_equals_paired_on_masked_arrays(qwen2, setup):
    kn, docs, wins = setup
    base = S.score(qwen2, docs, wins)
    g = cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "kn", (0.0, 0.01)))
    groups = np.asarray([base.groups[i] for i in base.doc])
    nd = np.random.default_rng(1).random(len(base)) < 0.6
    masks = {"dup:16gram_in_train": ~nd, cg.NONDUP_CLASS: nd}
    assert np.array_equal(cg.nondup_mask(masks), nd) and cg.nondup_mask(None) is None and cg.nondup_mask({}) is None
    res = S.score(qwen2, docs, wins, cl.CountLMCondition(kn, 0.01, 0.0))
    got = cg.paired_masked(base, res, groups, nd)
    ref = M.paired(base.nll[nd], res.nll[nd], groups[nd], base_hit=base.hit[nd], cond_hit=res.hit[nd])
    assert got == ref
    gm = cg.grid_metrics(base, g, groups, nondup=nd, n_boot=500)
    assert gm["weights"] == [0.0, 0.01] and gm["host_nll_max_abs_diff_vs_base"] == 0.0
    r1 = g.at(1)
    ref1 = M.retention(base.nll[nd], r1.nll[nd], groups[nd], r1.kl[nd], r1.agree[nd], base_hit=base.hit[nd],
                       cond_hit=r1.hit[nd], n_boot=500)
    assert gm["grid"][1]["paired_nondup"] == ref1
    assert gm["grid"][0]["paired"]["rel_pct"] == 0.0 and gm["grid"][0]["paired"]["mean_kl"] == 0.0
    assert abs(gm["grid"][1]["paired"]["cond_nll"] - float(res.nll.astype(np.float64).mean())) < 1e-6


# ------------------------------------------------------------------------------------------------- parsing
def test_parse_fixed_weight_specs():
    assert cl.parse_weight("0p0075") == 0.0075 and cl.parse_weight("1") == 1.0 and cl.parse_weight("0p015") == 0.015
    for bad in ("0.0075", "", "p", "1e-4", "0p0p1", "abc"):
        with pytest.raises(ValueError):
            cl.parse_weight(bad)
    assert cl.parse_variant("") == dict(parts=set(), fixed=False, lam_n=None, lam_c=None)
    assert cl.parse_variant("kn-fine")["parts"] == {"kn", "fine"} and not cl.parse_variant("kn-fine")["fixed"]
    assert cl.parse_variant("kn-lam0p0075") == dict(parts={"kn"}, fixed=True, lam_n=0.0075, lam_c=0.0)
    assert cl.parse_variant("cache-lam0p005") == dict(parts={"cache"}, fixed=True, lam_n=0.0, lam_c=0.005)
    assert cl.parse_variant("ln0p015-lc0p005") == dict(parts=set(), fixed=True, lam_n=0.015, lam_c=0.005)
    for bad in ("kn-cache", "foo", "lam0p01", "kn-fine-lam0p01", "kn-lam0p01-lam0p02", "ln0p01", "kn-lam0p6-x",
                "ln0p6-lc0p5", "kn-lam0.01"):
        with pytest.raises(ValueError):
            cl.parse_variant(bad)
    assert Path("kn-lam0p0075").stem == "kn-lam0p0075"  # evaluate.py label() keeps the whole spec
    assert cg.parse_spec("") == dict(component="kn", other_weight=0.0) == cg.parse_spec("kn")
    assert cg.parse_spec("cache") == dict(component="cache", other_weight=0.0)
    assert cg.parse_spec("kn-c0p005") == dict(component="kn", other_weight=0.005)
    for bad in ("kn-cache", "cache-c0p01", "foo", "kn-x1"):
        with pytest.raises(ValueError):
            cg.parse_spec(bad)
    assert cg.parse_weights(None) == cg.DEFAULT_WEIGHTS and len(cg.DEFAULT_WEIGHTS) == 13
    assert cg.parse_weights("0.01,0,1e-4,0.01") == (0.0, 1e-4, 0.01)
    with pytest.raises(ValueError):
        cg.CountGridCondition(cl.KNTrigram([np.arange(3, 9)], V, EOS), "kn", (0.0, 0.99), other_weight=0.02)
    kn = cl.KNTrigram([np.arange(3, 9)], V, EOS)
    with pytest.raises(ValueError):          # a fixed other weight on a cache grid would be ignored silently
        cg.CountGridCondition(kn, "cache", (0.0, 0.01), other_weight=0.05)
    with pytest.raises(ValueError):
        cg.CountGridCondition(kn, "kn", (0.0, float("nan")))
    for bad in ("nan,0.01", "inf", ","):
        with pytest.raises(ValueError):
            cg.parse_weights(bad)
    with pytest.raises(ValueError):          # repeated c<w>: no silent "last one wins"
        cg.parse_spec("kn-c0p005-c0p01")


# ------------------------------------------------------------------------------------------ evaluate.py run
def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True)


QWEN25 = __import__("ngram_memory.paths", fromlist=["qwen25"]).qwen25()


@pytest.mark.skipif(not (QWEN25 / "tokenizer.json").exists(), reason="Qwen2.5 tokenizer not cached")
def test_evaluate_cli_countgrid_fixed_and_nondup(tmp_path, monkeypatch):
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
    torch.save(SparseRowStore(fmt.padded_rows, fmt.row_width, ids,
                              torch.randn(len(ids), fmt.row_width, generator=g)).state_dict(), rows)
    (tmp_path / "ref.json").write_text(json.dumps({"os": 5, "value": 3, "other": 3}))
    out = tmp_path / "out"
    conds = (f"base,countlm:kn-lam0p0075,countgrid:kn,countgrid:kn+rows:{rows},countlm:kn-lam0p0075+rows:{rows},"
             f"rows:{rows}")
    spec = importlib.util.spec_from_file_location("evaluate_cli_cg", ROOT / "scripts/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ["evaluate.py", "--splits", str(sp), "--host", str(host), "--out", str(out), "--sets", "dev,r2",
            "--context", "128", "--stride", "64", "--min-context", "64", "--max-files", "1", "--tune-files", "1",
            "--max-windows", "3", "--retention-tokens", "256", "--threads", "1", "--fmt", str(tmp_path / "fmt.json"),
            "--ref-vocab-cache", str(tmp_path / "ref.json"), "--countgrid-weights", "0,0.0075,0.03",
            "--conditions", conds]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    s = json.loads((out / "summary.json").read_text())
    assert not [k for k in s["tuning"] if k.startswith("error:")], s["tuning"]
    assert s["tuning"]["countlm[kn-lam0p0075]"] == dict(fixed=True, lam_n=0.0075, lam_c=0.0)
    for set_name in ("dev", "r2"):
        R = s["sets"][set_name]
        assert {"countgrid[kn]", "countgrid[kn]+rows[rows]", "countlm[kn-lam0p0075]",
                "countlm[kn-lam0p0075]+rows[rows]", "rows[rows]"} <= set(R)
        for lbl, single, zero in (("countgrid[kn]", "countlm[kn-lam0p0075]", None),
                                  ("countgrid[kn]+rows[rows]", "countlm[kn-lam0p0075]+rows[rows]", "rows[rows]")):
            cgr = R[lbl]["countgrid"]
            assert cgr["weights"] == [0.0, 0.0075, 0.03] and cgr["host_nll_max_abs_diff_vs_base"] == 0.0
            p = cgr["grid"][1]["paired"]
            q = R[single]["paired"]
            assert p["n_tokens"] == q["n_tokens"] > 0 and abs(p["cond_nll"] - q["cond_nll"]) < 1e-5
            if set_name == "r2":
                assert abs(p["mean_kl"] - q["mean_kl"]) < 1e-6
            z = cgr["grid"][0]["paired"]
            if zero is None:
                assert z["rel_pct"] == 0.0 and z["mean_kl"] == 0.0
            else:
                assert abs(z["cond_nll"] - R[zero]["paired"]["cond_nll"]) < 1e-9
                if set_name == "r2":
                    assert abs(z["mean_kl"] - R[zero]["paired"]["mean_kl"]) < 1e-6
            assert ("paired_nondup" in cgr["grid"][1]) == (set_name == "dev")
            assert (out / f"{set_name}__{lbl}__grid.npz").exists()
    D = s["sets"]["dev"]
    nd_frac = D["_class_fractions"][cg.NONDUP_CLASS]
    assert D["base"]["nondup"]["n_tokens"] == round(nd_frac * D["base"]["n_tokens"])
    for k in ("countlm[kn-lam0p0075]", "rows[rows]", "countlm[kn-lam0p0075]+rows[rows]"):
        assert D[k]["paired_nondup"]["n_tokens"] == D["base"]["nondup"]["n_tokens"]
        assert "paired_nondup" not in s["sets"]["r2"][k]
    assert (out / "report.md").read_text().startswith("# Stage-1 eval")

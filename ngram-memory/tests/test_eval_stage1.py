"""Stage-1 eval (ngram_memory/eval): scoring windows and exactness, metrics, token classes, splits, QA, and every
baseline / control on a tiny qwen2 host (conftest.build_host) with a character-level fake tokenizer."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest
import torch

from conftest import EOS, V, build_host, fill_random_rows, tiny_fmt
from ngram_memory.eval import metrics as M
from ngram_memory.eval.stage1 import scoring as S
from ngram_memory.eval.stage1 import tokclass as TC


class CharTok:
    """id = 3 + (ord(c) % (V - 3)); offsets are character spans."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [3 + (ord(c) % (V - 3)) for c in text]
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


CODE = '''"""Module docstring."""
import os


def foo_bar(x, y):  # comment here
    """Return x."""
    return os.path.join(x, "s") + str(y)


VALUE_NAME = 42
'''


@pytest.fixture(scope="module")
def qwen2():
    return build_host("qwen2")


def rdoc(seed, n, name=None):
    g = np.random.default_rng(seed)
    return S.Doc(name or f"d{seed}", g.integers(3, V, n))


# ------------------------------------------------------------------------------------------------ windows
@pytest.mark.parametrize("n,C,St,mc", [(5000, 2048, 1024, 1024), (3000, 512, 256, 512), (700, 2048, 1024, 1024),
                                       (100, 64, 16, 1)])
def test_plan_covers_targets_once_with_min_context(n, C, St, mc):
    plan = S.plan_doc(n, C, St, mc)
    tg = np.concatenate([np.arange(s, e) for a, s, e in plan]) if plan else np.zeros(0, int)
    assert np.array_equal(tg, np.arange(max(1, mc), n)) if n > mc else len(tg) == 0
    for a, s, e in plan:
        assert s - a >= min(s, C) and s - a <= C and e - s <= St and a >= 0


@torch.no_grad()
def test_score_equals_full_forward_nll(qwen2):
    docs = [rdoc(0, 300), rdoc(1, 200)]
    wins = S.plan_windows(docs, context=128, stride=64, min_context=64)
    res = S.score(qwen2, docs, wins, chunk=37)
    for w in wins[:4]:
        ids = torch.from_numpy(docs[w.doc].ids[w.a:w.e])[None]
        lp = torch.log_softmax(qwen2(input_ids=ids, use_cache=False).logits[0].float(), -1)
        t = np.arange(w.s, w.e)
        ref = -lp[t - 1 - w.a].gather(1, torch.from_numpy(docs[w.doc].ids[t])[:, None])[:, 0]
        m = (res.doc == w.doc) & np.isin(res.pos, t)
        assert np.allclose(res.nll[m], ref.numpy(), atol=1e-5)
        assert np.all(res.ctx[m] >= 64)
    assert len(res) == sum(w.n_targets for w in wins)


def test_scoreresult_roundtrip_and_target_mask(qwen2, tmp_path):
    d = rdoc(2, 150)
    d.target_mask = np.zeros(150, bool)
    d.target_mask[100:120] = True
    res = S.score(qwen2, [d], S.plan_windows([d], 64, 32, 1))
    assert set(res.pos.tolist()) == set(range(100, 120))
    res.save(tmp_path / "r.npz")
    r2 = S.ScoreResult.load(tmp_path / "r.npz")
    assert np.array_equal(r2.nll, res.nll) and r2.doc_names == res.doc_names and r2.meta["n_windows"] == res.meta["n_windows"]


# --------------------------------------------------------------------------------------------- rows arms
def test_rows_condition_zero_is_bitwise_base_and_kl(qwen2):
    from ngram_memory.eval.stage1.conditions import RowsCondition
    docs = [rdoc(3, 200)]
    wins = S.plan_windows(docs, 128, 64, 64)
    fmt = tiny_fmt()
    base = S.score(qwen2, docs, wins)
    zero = RowsCondition(fmt, None)
    r0 = S.score(qwen2, docs, wins, zero, with_kl=True)
    assert np.array_equal(r0.nll, base.nll) and float(np.abs(r0.kl).max()) == 0.0 and r0.agree.all()
    fill_random_rows(zero.graft, torch.from_numpy(docs[0].ids[None]), scale=0.5)
    r1 = S.score(qwen2, docs, wins, zero, with_kl=True)
    assert np.abs(r1.nll - base.nll).max() > 1e-3 and r1.kl.min() >= 0 and r1.kl.mean() > 0
    # hooks are removed after scoring
    assert np.array_equal(S.score(qwen2, docs, wins).nll, base.nll)


def test_controls_permutation_random_rows_constant_bias(qwen2):
    from ngram_memory.eval.stage1.baselines import controls as ct
    from ngram_memory.eval.stage1.conditions import RowsCondition
    from ngram_memory.tables import SparseRowStore
    fmt = tiny_fmt()
    key = fmt.key
    r = np.arange(key.total_rows)
    p = ct.permute_rows(r, key, seed=1)
    assert np.array_equal(np.sort(p), r)  # bijection
    assert np.array_equal(ct.permute_rows(p, key, seed=1, inverse=True), r)
    off = np.asarray(key.head_offsets)
    assert np.array_equal(np.searchsorted(off, p, "right"), np.searchsorted(off, r, "right"))  # stays in its head
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, key.total_rows, (50,), generator=g).unique()
    st = SparseRowStore(fmt.padded_rows, fmt.row_width, ids, torch.randn(len(ids), fmt.row_width, generator=g))
    ps = ct.permuted_keys_store(st, key, seed=1)
    q = torch.from_numpy(ct.permute_rows(ids.numpy(), key, 1, inverse=True))
    assert torch.equal(ps.gather(q), st.gather(ids))
    rr = ct.random_rows_store(st, seed=2)
    assert torch.allclose(rr.values.norm(dim=-1), st.values.norm(dim=-1), atol=1e-5)
    assert torch.equal(rr.row_ids, st.row_ids) and not torch.allclose(rr.values, st.values)
    docs = [rdoc(4, 160)]
    wins = S.plan_windows(docs, 128, 32, 32)
    rc = RowsCondition(fmt, None)
    fill_random_rows(rc.graft, torch.from_numpy(docs[0].ids[None]), scale=0.5)
    bias = ct.mean_graft_output(qwen2, rc.graft, docs, wins)
    assert bias.shape == (64,) and float(bias.norm()) > 0
    cb = ct.ConstantBiasCondition(bias, layer_idx=1)
    base = S.score(qwen2, docs, wins)
    rb = S.score(qwen2, docs, wins, cb)
    assert np.abs(rb.nll - base.nll).max() > 0
    zb = S.score(qwen2, docs, wins, ct.ConstantBiasCondition(torch.zeros(64), layer_idx=1))
    assert np.array_equal(zb.nll, base.nll)


def test_ngm_average_and_condition(qwen2):
    from ngram_memory.eval.stage1.baselines.ngm import NGMCondition, ngram_embedding_average
    emb = torch.randn(10, 4)
    ids = torch.tensor([[2, 3, EOS, 5, 6]])
    u = ngram_embedding_average(ids, emb, (2, 3), eos=EOS)
    e = emb[ids[0]]
    # position 3 (token 5) starts a new segment after the EOS: only itself
    assert torch.allclose(u[0, 3], e[3])
    assert torch.allclose(u[0, 4], 0.5 * ((e[4] + e[3]) / 2 + (e[4] + e[3]) / 2))
    assert torch.allclose(u[0, 1], 0.5 * ((e[1] + e[0]) / 2 + (e[1] + e[0]) / 2))
    assert torch.allclose(u[0, 2], 0.5 * ((e[2] + e[1]) / 2 + (e[2] + e[1] + e[0]) / 3))
    docs = [rdoc(5, 160)]
    wins = S.plan_windows(docs, 128, 32, 32)
    base = S.score(qwen2, docs, wins)
    c = NGMCondition(qwen2, ratio=0.05, eos=EOS)
    r = S.score(qwen2, docs, wins, c)
    assert np.abs(r.nll - base.nll).max() > 0
    assert abs(c.describe()["median_injection_ratio"] - 0.05) < 1e-4
    assert np.array_equal(S.score(qwen2, docs, wins, NGMCondition(qwen2, ratio=0.0)).nll, base.nll)


# ------------------------------------------------------------------------------------------------ count LM
def test_kn_trigram_normalised_and_consistent():
    from ngram_memory.eval.stage1.baselines.countlm import InFileCache, KNTrigram, doc_contexts
    g = np.random.default_rng(0)
    Vs = 30
    seqs = [g.integers(2, 12, 200), g.integers(2, Vs, 100), np.array([5, 6, 7, 5, 6, 7, 5, 6, 8])]
    kn = KNTrigram(seqs, Vs, eos=1)
    u = np.array([5, 1, 29, 3, 1])
    v = np.array([6, 1, 28, 4, 5])
    dist = kn.dist(u, v).double()
    assert torch.allclose(dist.sum(-1), torch.ones(5, dtype=torch.float64), atol=1e-5)
    assert (dist > 0).all()
    w = np.array([7, 5, 3, 9, 6])
    assert np.allclose(kn.prob(u, v, w), dist[torch.arange(5), torch.from_numpy(w)].numpy(), atol=1e-6)
    assert kn.prob([5], [6], [7])[0] > kn.prob([5], [6], [20])[0]
    c = InFileCache(Vs)
    ids = np.array([3, 4, 3, 4, 3, 4, 3])
    full = c.probs(ids, np.arange(1, 7), full=True)
    assert np.allclose(full.sum(-1), 1.0) and np.allclose(c.probs(ids, np.arange(1, 7)), full[np.arange(6), ids[1:]])
    assert c.probs(ids, np.array([6]))[0] > 0.5  # after "3 4 3 4 3 4" the cache predicts 3
    uu, vv = doc_contexts(ids, np.array([1, 2]), eos=1)
    assert uu.tolist() == [1, 3] and vv.tolist() == [3, 4]


def test_countlm_condition_tune_and_headroom(qwen2):
    from ngram_memory.eval.stage1.baselines import countlm as cl
    pat = np.tile(np.array([7, 8, 9, 10, 11]), 80)
    train = [pat, rdoc(6, 300).ids]
    docs = [S.Doc("p", pat[:300])]
    wins = S.plan_windows(docs, 128, 64, 64)
    kn = cl.KNTrigram(train, V, EOS)
    base = S.score(qwen2, docs, wins)
    pk, pc = cl.component_target_probs(kn, docs, base)
    tu = cl.tune_lambdas(base, pk, pc)
    assert tu["lam_n"] + tu["lam_c"] > 0 and tu["dev_nll"] < base.nll.mean()
    cond = cl.CountLMCondition(kn, tu["lam_n"], tu["lam_c"])
    res = S.score(qwen2, docs, wins, cond)
    assert abs(float(res.nll.mean()) - tu["dev_nll"]) < 1e-3  # full-distribution path == scalar tuning path
    hr = cl.headroom(kn, docs, base)
    assert hr["confident"] > 0.9 and hr["confident_top1"] == 1.0
    assert np.array_equal(S.score(qwen2, docs, wins, cl.CountLMCondition(kn, 0.0, 0.0)).nll, base.nll)


# ------------------------------------------------------------------------------------------------- BM25 RAG
def test_bm25_retrieval_and_rag_prefix(qwen2):
    from ngram_memory.eval.stage1.baselines.bm25_rag import BM25Retriever, BM25RAGCondition, terms
    assert "foo" in terms("fooBar_baz") and "foobar_baz" in terms("fooBar_baz")
    files = {"a.py": "\n".join(f"def alpha_{i}(): return {i}" for i in range(60)),
             "b.py": "\n".join(f"class BetaThing{i}: pass" for i in range(60))}
    r = BM25Retriever(files)
    assert r.search("BetaThing3 pass")[0][1].path == "b.py"
    assert r.search("alpha_7 return")[0][1].path == "a.py"
    tok = CharTok()
    text = "# file: q.py\n" + "\n".join(f"x = BetaThing{i}()" for i in range(30))
    d = S.encode_doc(tok, "q.py", text)
    wins = S.plan_windows([d], 256, 128, 128)
    cond = BM25RAGCondition(r, tok, budget=100)
    pre = cond.prefix(d, wins[0])
    assert 0 < len(pre) <= 100
    base = S.score(qwen2, [d], wins)
    res = S.score(qwen2, [d], wins, cond)
    assert np.array_equal(res.key(), base.key()) and res.meta["forward_tokens"] > base.meta["forward_tokens"]
    with pytest.raises(ValueError):
        S.score(qwen2, [d], wins, cond, with_kl=True)


# ----------------------------------------------------------------------------------------------------- kNM
def test_knm_search_probs_and_condition(qwen2):
    from ngram_memory.eval.stage1.baselines import knm
    g = torch.Generator().manual_seed(0)
    keys, vals = torch.randn(1000, 8, generator=g), torch.randint(0, 20, (1000,), generator=g)
    st = knm.KNMDatastore(keys, vals)
    q = torch.randn(5, 8, generator=g)
    d, v = st.search(q, k=4, block=128)
    ref = torch.cdist(q, st.keys.float()).pow(2)
    rd, ri = torch.topk(ref, 4, largest=False)
    assert torch.allclose(d, rd, atol=1e-3) and torch.equal(v, vals[ri])
    p = knm.knn_probs(d, v, 20, temp=1.0)
    assert torch.allclose(p.sum(-1), torch.ones(5))
    docs = [rdoc(7, 200)]
    wins = S.plan_windows(docs, 128, 64, 64)
    store = knm.KNMDatastore.build(qwen2, docs, wins)
    assert 0 < len(store) <= sum(w.n_targets for w in wins)
    base = S.score(qwen2, docs, wins)
    assert np.allclose(S.score(qwen2, docs, wins, knm.KNMCondition(store, 0.0, 1.0)).nll, base.nll)
    tu = knm.tune(qwen2, store, docs, wins)
    assert tu["lam"] > 0 and tu["dev_nll"] < tu["base_dev_nll"]  # in-sample datastore must help
    res = S.score(qwen2, docs, wins, knm.KNMCondition(store, tu["lam"], tu["temp"]))
    assert abs(float(res.nll.mean()) - tu["dev_nll"]) < 1e-3


# ---------------------------------------------------------------------------------------------------- LoRA
def test_lora_noop_train_toggle_roundtrip():
    from ngram_memory.eval.stage1.baselines import lora as lo
    m = build_host("qwen2")
    docs = [rdoc(8, 160)]
    wins = S.plan_windows(docs, 128, 32, 32)
    base = S.score(m, docs, wins)
    names = lo.add_lora(m, r=4, alpha=8)
    assert any(n.endswith("mlp.down_proj") for n in names) and any(n.endswith("self_attn.q_proj") for n in names)
    assert all(not p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)
    cond = lo.LoRACondition(m, name="lora")
    assert np.array_equal(S.score(m, docs, wins, cond).nll, base.nll)  # B = 0: exact no-op
    x = [docs[0].ids[i:i + 64] for i in range(0, 128, 64)]
    info = lo.train_lora(m, x, epochs=5, lr=1e-2)
    assert info["history"][-1]["loss"] < info["history"][0]["loss"] and info["tokens"] == 5 * 128
    on = S.score(m, docs, wins, cond)
    assert on.nll.mean() < base.nll.mean() - 1e-3
    assert np.array_equal(S.score(m, docs, wins).nll, base.nll)  # disabled outside the condition
    sd = lo.lora_state_dict(m)
    m2 = build_host("qwen2")
    lo.add_lora(m2, r=4, alpha=8)
    lo.load_lora_state_dict(m2, sd)
    assert np.allclose(S.score(m2, docs, wins, lo.LoRACondition(m2)).nll, on.nll, atol=1e-6)
    lo.remove_lora(m)
    assert not lo.lora_modules(m) and np.array_equal(S.score(m, docs, wins).nll, base.nll)


def test_lora_excludes_moe_router():
    from ngram_memory.eval.stage1.baselines import lora as lo
    m = build_host("qwen3_5_moe")
    names = lo.add_lora(m, r=2)
    assert names and not any(n.endswith("mlp.gate") or n.endswith("shared_expert_gate") for n in names)
    assert any("linear_attn" in n for n in names)  # GDN projections are adapted


# ------------------------------------------------------------------------------------------------- metrics
def test_paired_cluster_bootstrap_and_mcnemar():
    g = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 50)
    b = g.uniform(1, 3, 1000)
    c = b - 0.1 + g.normal(0, 0.01, 1000)
    r = M.paired(b, c, groups, base_hit=b < 2, cond_hit=c < 2, n_boot=2000)
    assert abs(r["delta"] + 0.1) < 0.01 and r["delta_ci"][0] < r["delta"] < r["delta_ci"][1] < 0
    assert r["files_improved"] == 20 and r["n_files"] == 20 and r["delta_em_pts"] >= 0
    assert r["rel_pct_ci"][1] < 0
    same = M.paired(b, b, groups, n_boot=200)
    assert same["delta"] == 0 and same["delta_ci"] == [0.0, 0.0]
    # ratio-of-sums bootstrap equals a direct loop
    ds, n, bs = np.array([1.0, -2.0, 0.5]), np.array([10.0, 20.0, 5.0]), np.array([20.0, 30.0, 10.0])
    cb = M.cluster_bootstrap(ds, n, bs, n_boot=500, seed=3)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 3, size=(500, 3))
    ref = np.percentile(ds[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    assert np.allclose(cb["delta_ci"], ref)
    t = M.mcnemar(np.array([1, 1, 0, 0, 1] * 4, bool), np.array([1, 0, 1, 1, 1] * 4, bool))
    assert (t["n01"], t["n10"]) == (8, 4) and 0 < t["p"] < 1
    assert M.mcnemar(np.ones(5, bool), np.ones(5, bool))["p"] == 1.0
    cls = M.class_breakdown(b, c, groups, {"lo": b < 2, "hi": b >= 2}, n_boot=100)
    assert abs(cls["lo"]["share_of_gain"] + cls["hi"]["share_of_gain"] - 1) < 1e-9


# -------------------------------------------------------------------------------------------- token classes
def test_token_classes_lexical_and_identifiers():
    tok = CharTok()
    d = S.encode_doc(tok, "pkg/a.py", CODE)
    ref = {"os": 3, "path": 3, "join": 2, "x": 9, "y": 9}
    defs = TC.project_definitions({"pkg/a.py": CODE, "pkg/b.py": "def helper_fn():\n    pass\n"})
    c = TC.token_classes(d.text, d.offsets, ref, defs, "pkg/a.py")
    lab = {name: c["lex"][CODE.index(name)] for name in ("Module", "comment here", "Return", "import", "foo_bar", "42")}
    assert TC.LEX[lab["Module"]] == "docstring" and TC.LEX[lab["Return"]] == "docstring"
    assert TC.LEX[lab["comment here"]] == "comment" and TC.LEX[lab["import"]] == "keyword"
    assert TC.LEX[lab["foo_bar"]] == "identifier" and TC.LEX[lab["42"]] == "number"
    assert TC.LEX[c["lex"][CODE.index('"s"') + 1]] == "string"
    i = CODE.index("foo_bar")
    assert c["id_first_sub"][i] and c["id_cont_sub"][i + 1] and TC.ID_KIND[c["id_kind"][i]] == "proj"
    j = CODE.index("str(")
    assert TC.ID_KIND[c["id_kind"][j]] == "builtin"
    k1, k2 = CODE.index("os\n"), CODE.index("os.path")
    assert c["id_first_in_file"][k1] and not c["id_first_in_file"][k2]
    m = TC.class_masks({**c, "cat": np.zeros(len(d), np.int8)})
    assert m["lex:docstring"].sum() > 0 and m["id:proj:first_sub"].sum() >= 2 and m["key:unseen_2gram"].all()


def test_ngram_seen_categories():
    s = TC.NGramSeen([np.array([5, 6, 7, 8, 5, 6, 9])], vocab=20, eos=1)
    assert s.categories(np.array([5, 6, 7, 8, 3, 6, 7, 2])).tolist() == [-1, 3, 3, 3, 2, 0, 0, 1]
    # an EOS inside a sequence resets the history (as the hasher)
    s2 = TC.NGramSeen([np.array([4, 5, 6])], vocab=20, eos=1)
    assert s2.categories(np.array([9, 1, 4, 5, 6])).tolist()[3:] == [3, 3]


# -------------------------------------------------------------------------------------------------- splits
def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True)


def test_build_splits_on_tiny_repo(tmp_path):
    from ngram_memory.eval.stage1.splits import Splits, added_line_target_mask, build_splits, path_hash
    repo = tmp_path / "r"
    (repo / "python/pkg").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    for i in range(40):
        (repo / f"python/pkg/m{i}.py").write_text(f"# module {i}\n" + "\n".join(f"value_{i}_{j} = {j} * {i} + 1234567"
                                                                              for j in range(30)) + "\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "c1")
    cut = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (repo / "python/pkg/m3.py").write_text((repo / "python/pkg/m3.py").read_text() + "added_line = 1\nmore = 2\n")
    (repo / "python/pkg/new.py").write_text("brand_new = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "c2")
    d = build_splits(str(repo), "HEAD", "python", t_cut=cut)
    a = d["split_a"]
    allf = a["train"] + a["dev"] + a["test"] + a["near_dup"]
    assert sorted(allf) == sorted([f"pkg/m{i}.py" for i in range(40)] + ["pkg/new.py"])
    for f in a["test"]:
        assert path_hash("python/" + f) < 0.15
    assert "pkg/new.py" in a["train"]  # <= 400 bytes stays in train
    b = d["split_b"]
    assert b["test"]["pkg/m3.py"] == [32, 33] and b["new_files"] == ["pkg/new.py"] and len(b["train"]) == 40
    s = Splits(d)
    p = tmp_path / "s.json"
    s.save(p)
    s2 = Splits.load(p)
    doc = s2.document("pkg/m3.py")
    assert doc.startswith("# file: pkg/m3.py\n") and "added_line" in doc
    tok = CharTok()
    enc = tok(doc, return_offsets_mapping=True)
    mask = added_line_target_mask(doc, enc["offset_mapping"], s2.added_lines("pkg/m3.py"))
    marked = "".join(ch for ch, mm in zip(doc, mask) if mm)
    assert marked == "added_line = 1\nmore = 2\n"


# ------------------------------------------------------------------------------------------------------- QA
def test_qa_generate_and_score(qwen2):
    from ngram_memory.eval.stage1 import qa
    texts = {"pkg/core.py": CODE + "\nimport os\nENV = os.environ.get('PKG_DEBUG_MODE')\n"
                            "def other_fn(a, b):\n    return a\n",
             "pkg/util.py": "from pkg.core import foo_bar, other_fn\n\nTIMEOUT_SEC = 30\n"
                            "def helper_long(x, y, z):\n    return foo_bar(x, y)\n"}
    items = qa.generate_items(texts, split_of={"pkg/core.py": "train", "pkg/util.py": "test"}, commit="abc",
                              project="pkg", env_prefix="PKG_")
    kinds = {it["kind"] for it in items}
    assert kinds == set(qa.KINDS)
    dm = [it for it in items if it["kind"] == "defines_module" and it["name"] == "foo_bar"][0]
    assert dm["answer"] == "pkg.core" and dm["evidence"].startswith("pkg/core.py") and dm["reach_class"] == "unreachable"
    sig = [it for it in items if it["kind"] == "signature" and it["name"] == "foo_bar"][0]
    assert sig["answer"] == "def foo_bar(x, y)"
    cv = [it for it in items if it["kind"] == "constant_value" and it["name"] == "VALUE_NAME"][0]
    assert cv["answer"] == " 42" and cv["cue"].endswith("VALUE_NAME =")
    ev = [it for it in items if it["kind"] == "env_var_reader"][0]
    assert ev["name"] == "PKG_DEBUG_MODE" and ev["answer"] == "pkg/core.py"
    tok = CharTok()
    sc = qa.score_items(qwen2, tok, items)
    assert len(sc) == len(items) and all(s["nl"]["answer_nll"] > 0 for s in sc)
    sm = qa.summarize(sc)
    assert sm["all/nl"]["n"] == len(items)
    ps = qa.paired_summary(sc, sc)
    assert ps["all/nl"]["d_em_pts"] == 0 and ps["all/nl"]["d_answer_nll"] == 0


# -------------------------------------------------------------------------------------------- foreign corpus
def test_foreign_reader_filter_select_overlap(tmp_path):
    import io
    import tarfile
    from ngram_memory.eval.stage1 import foreign as fo
    src = {"pkg-1.0/pkg/__init__.py": "X = 1\n", "pkg-1.0/pkg/core.py": "def f():\n    return 'a long enough line here'\n",
           "pkg-1.0/pkg/argparse.py": "y = 2\n", "pkg-1.0/tests/test_core.py": "assert True\n",
           "pkg-1.0/setup.py": "import setuptools\n", "pkg-1.0/README.md": "not python\n"}
    arch = tmp_path / "pkg-1.0.tar.gz"
    with tarfile.open(arch, "w:gz") as tf:
        for name, text in src.items():
            b = text.encode()
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
        ln = tarfile.TarInfo("pkg-1.0/pkg/link.py")
        ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(ln)
    files = fo.read_py_files(arch)
    assert "pkg/link.py" not in files and "README.md" not in files
    assert set(files) == {"pkg/__init__.py", "pkg/core.py", "pkg/argparse.py", "tests/test_core.py", "setup.py"}
    kept = fo.filter_files(files)
    assert set(kept) == {"pkg/__init__.py", "pkg/core.py", "pkg/argparse.py"}
    sel = fo.select_to_tokens(kept, len, target=8)
    assert sel == sorted(sel) and sum(len(kept[p]) for p in sel) >= 8
    assert fo.select_to_tokens(kept, len, 10 ** 9) == sorted(kept)
    ov = fo.overlap_report(kept, {"stdlib/argparse.py": "    return 'a long enough line here'\n"})
    assert ov["shared_basenames"] == ["argparse.py"] and ov["verbatim_line_share"] == 1.0
    info = dict(name="pkg", version="1.0", filename=arch.name, url="file://x", sha256="0" * 64)
    man = fo.manifest(info, arch, sel, kept)
    assert man["package"] == "pkg" and len(man["files"]) == len(sel) and len(man["files"][0]["sha256"]) == 64
    with pytest.raises(ValueError):
        fo.download(dict(info, filename=arch.name), tmp_path)  # sha256 mismatch on the existing file


def test_capability_mc_scoring(qwen2):
    from ngram_memory.eval.stage1 import capability as cap
    from ngram_memory.eval.stage1.conditions import RowsCondition
    tok = CharTok()
    items = [dict(id=i, task="t" + str(i % 2), context=f"ctx {i}:", choices=[" a", " bb", " c", " dd"], label=i % 4)
             for i in range(8)]
    b = cap.score_mc(qwen2, tok, items)
    r = cap.score_mc(qwen2, tok, items, RowsCondition(tiny_fmt(), None))  # empty rows: identical answers
    assert [x["logprobs"] for x in b] == [x["logprobs"] for x in r]
    p = cap.paired_mc(b, r)
    assert p["all"]["n"] == 8 and p["all"]["acc"]["d_pts"] == 0 and p["all"]["acc"]["mcnemar"]["p"] == 1.0
    lp = cap.choice_logprobs(qwen2, tok, "ctx:", [" a"])[0]
    ids = torch.tensor([tok("ctx: a")["input_ids"]])
    with torch.no_grad():
        lps = torch.log_softmax(qwen2(input_ids=ids).logits[0].float(), -1)
    ref = lps[-3, ids[0, -2]] + lps[-2, ids[0, -1]]  # " a" is two characters = two tokens
    assert abs(lp - float(ref)) < 1e-5


def test_ngram_index_and_take_tokens():
    from ngram_memory.eval.stage1.datasets import take_tokens
    tr = [np.arange(100, 140)]
    ix = TC.NGramIndex(tr, n=16)
    q = np.concatenate([np.arange(100, 120), [7, 8]])
    m = ix.in_train(q)
    assert m.tolist() == [False] * 15 + [True] * 5 + [False, False]
    docs = [rdoc(10, 50, "a"), rdoc(11, 10, "b"), rdoc(12, 300, "c")]
    out = take_tokens(docs, 100, min_context=20)
    assert [d.name for d in out] == ["a", "c"] and sum(len(d) - 20 for d in out) == 100 and len(out[1]) == 90
    plan = S.plan_windows(out, 64, 32, 20)
    assert sum(w.n_targets for w in plan) == 100


def test_splits_from_ingest_files(tmp_path):
    import json
    from ngram_memory.eval.stage1.splits import Splits
    a = dict(rule={}, commit="c" * 40, train=["python/pkg/a.py"], dev=["python/pkg/b.py"], test=["python/pkg/c.py"],
             neardup_diag=["python/pkg/d.py"], cluster_moves=[], dup_frac={})
    b = dict(t_cut="t" * 40, head="c" * 40, t_cut_time=0, head_time=1, commits=["x"], first_parent_commits=["x"],
             added_lines={"python/pkg/c.py": [[3, 5], [9, 9]]}, new_files=["python/pkg/c.py"],
             train_files=["python/pkg/a.py", "python/pkg/b.py"])
    (tmp_path / "split_a.json").write_text(json.dumps(a))
    (tmp_path / "split_b.json").write_text(json.dumps(b))
    s = Splits.load(tmp_path, repo="/nonexistent")
    assert s.files("a", "test") == ["pkg/c.py"] and s.files("a", "near_dup") == ["pkg/d.py"]
    assert s.added_lines("pkg/c.py") == [3, 4, 5, 9] and s.files("b", "new_files") == ["pkg/c.py"]
    assert s.summary()["b_added_lines"] == 4


@pytest.mark.skipif(not (__import__("pathlib").Path(__file__).resolve().parents[1] / "ingest_splits/split_a.json").exists()
                    or not __import__("pathlib").Path("/home/user/patcarter883/minisglang-rdna4/.git").exists(),
                    reason="ingest split files or the minisgl clone are absent")
def test_ingest_splits_match_rederived_rule():
    from pathlib import Path
    from ngram_memory.eval.stage1.splits import Splits, build_splits
    s = Splits.load(Path(__file__).resolve().parents[1] / "ingest_splits")
    d = build_splits(s.repo, s.commit, "python")
    assert s.files("a", "dev") == d["split_a"]["dev"] and s.files("a", "test") == d["split_a"]["test"]
    assert s.data["split_b"]["test"] == d["split_b"]["test"] and s.summary()["b_added_lines"] == 4142


def test_report_renders_summary(tmp_path):
    import json
    from ngram_memory.eval.report import render
    g = np.repeat(np.arange(4), 10)
    b = np.linspace(1, 2, 40)
    p = M.paired(b, b - 0.05, g, base_hit=b > 1.5, cond_hit=b > 1.4, n_boot=100)
    cls = M.class_breakdown(b, b - 0.05, g, {"lex:comment": b > 1.5}, n_boot=50)
    s = dict(host="h", sets=dict(dev=dict(base=dict(nll=1.5, n_files=4, n_tokens=40, wall_s_per_10k=10.0),
                                          rows=dict(paired=p, classes=cls, wall_s_per_10k=12.0),
                                          _class_fractions={"lex:comment": 0.5}),
                                 r3=dict(base=dict(n=4, acc=0.5),
                                         ngm=dict(paired_mc={"all": dict(n=4, acc=dict(base=0.5, arm=0.75, d_pts=25.0,
                                                                                        mcnemar=dict(p=1.0), ratio=1.5))}))))
    (tmp_path / "s.json").write_text(json.dumps(s))
    md = render(tmp_path / "s.json")
    assert "| dev | rows | 40 | 4 |" in md and "-3.33" in md and "| ngm | all | 4 |" in md and "lex:comment" in md

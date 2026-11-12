"""M1d (experiments/m1d/PREREG.md): dose-response canaries (ingest/canaries.py v2), multi-pass arms with cartridge
snapshots and the per-pass window order, the seed, the positive-control windows, REG v2 (eval/stage1/reg2.py,
scripts/reg2.py), the M1d stage kinds and the M2-recipe rules of the report, and the redaction of results copies."""
from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import EOS, PAD, build_host, fill_random_rows, tiny_fmt, tiny_key
from ngram_memory.eval.stage1 import reg as R
from ngram_memory.eval.stage1 import reg2 as R2
from ngram_memory.ingest import canaries as CN
from ngram_memory.ingest import corpus as C
from ngram_memory.ingest import train as TR

ROOT = Path(__file__).resolve().parents[1]
M1D = ROOT / "experiments/m1d/grid.json"
QWEN25 = __import__("ngram_memory.paths", fromlist=["qwen25"]).qwen25()


def load_run():
    spec = importlib.util.spec_from_file_location("m1d_run_t", ROOT / "experiments/m1b/run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _args(tmp_path, **kw):
    d = dict(out=str(tmp_path / "run"), smoke=False, threads=8, rss_limit_gb=11.0, stop_after=None, keep_going=False,
             parallel=1, max_stage_threads=0, launch_min_free_gb=0.0)
    d.update(kw)
    return SimpleNamespace(**d)


def _module(i: int, n_defs: int = 12) -> str:
    body = "\n\n".join(f"def f{i}_{j}(x):\n    return x + {j}\n" for j in range(n_defs))
    return f'"""Module {i}."""\nimport os\n\nLIMIT_{i} = {i}\n\n\n{body}\n'


def _texts(n: int = 40, n_defs: int = 12) -> dict[str, str]:
    return {f"python/pkg/m{i:02d}.py": _module(i, n_defs) for i in range(n)}


V2 = CN.DoseCanaryConfig(groups=((2, 1), (4, 1), (8, 1)), seed=3, min_lines=10, min_sep_chars=400)


# ============================================================================================ canaries v2
def test_dose_planting_is_pure_one_copy_per_file_and_separated():
    texts = _texts()
    train = sorted(texts)
    a, man = CN.plant_texts_dose(texts, train, V2)
    b, man2 = CN.plant_texts_dose(dict(reversed(list(texts.items()))), list(reversed(train)), V2)
    assert a == b and man == man2 and man["format"] == CN.FORMAT_V2
    assert man["config"] == dict(version=2, groups=[[2, 1], [4, 1], [8, 1]], seed=3, min_lines=10, min_sep_chars=400)
    by = {}
    for r in man["records"]:
        by.setdefault(r["id"], []).append(r)
        assert a[r["path"]][r["secret_start"]:r["secret_end"]] == r["secret"]
        assert r["kind"] == ("assign1" if int(r["id"][1:]) % 2 == 0 else "comment1")
    assert {k: len(v) for k, v in by.items()} == {"c0": 2, "c1": 4, "c2": 8}
    assert all(v[0]["repeats"] == len(v) and len({x["path"] for x in v}) == len(v) for v in by.values())
    # no file holds two canaries while fresh files remain (14 copies, 40 files)
    assert len({r["path"] for r in man["records"]}) == 14
    # files holding copies of one canary are >= min_sep_chars apart in the sorted stream order
    ends = np.cumsum([len(texts[p]) + 1 for p in train])
    span = {p: (ends[i] - len(texts[p]) - 1, ends[i]) for i, p in enumerate(train)}
    for v in by.values():
        sp = sorted(span[x["path"]] for x in v)
        assert all(q[0] - p[1] >= 400 for p, q in zip(sp, sp[1:]))
    # the manifest re-plants the clean texts and round-trips through write/load
    assert CN.apply_manifest(texts, man) == a
    assert man["groups"] == [dict(group=0, repeats=2, canaries=["c0"]), dict(group=1, repeats=4, canaries=["c1"]),
                             dict(group=2, repeats=8, canaries=["c2"])]
    files, man3 = CN.plant_source_files([C.SourceFile(p, "", len(t), t) for p, t in texts.items()], train, V2)
    assert man3 == man and {f.path: f.text for f in files} == a


def test_dose_config_dispatch_and_errors(tmp_path):
    assert isinstance(CN.config_from_arm({"canaries": {"seed": 0, "n": 8, "m": 2, "repeats": 2}}), CN.CanaryConfig)
    c = CN.config_from_arm({"canaries": {"version": 2, "groups": [[2, 2], [16, 1]], "seed": 1}})
    assert isinstance(c, CN.DoseCanaryConfig) and c.groups == ((2, 2), (16, 1)) and c.n == 3 and c.copies == 20
    assert CN.DoseCanaryConfig.from_dict(c.to_dict()) == c
    with pytest.raises(ValueError):
        CN.DoseCanaryConfig(groups=((0, 1),))
    with pytest.raises(ValueError, match="train files"):
        CN.plant_texts_dose(_texts(5), sorted(_texts(5)), CN.DoseCanaryConfig(groups=((8, 1),), min_lines=10))
    _, man = CN.plant_texts_dose(_texts(), sorted(_texts()), V2)
    CN.write_manifest(man, tmp_path / "c.json")
    assert CN.load_manifest(tmp_path / "c.json") == man


def test_window_copies_finds_two_copies_in_one_window():
    spans = {"c0": [(10, 20), (5000, 5010)], "c1": [(100, 110), (300, 310)]}
    wins = [(0, 4096), (3584, 7680)]
    wc = CN.window_copies(spans, wins)
    assert wc["conflicts"] == [dict(id="c1", window=0, copies=2)]
    assert wc["per_canary"]["c0"]["windows"] == [[0], [1]] and wc["per_canary"]["c0"]["distinct_windows"] == 2
    assert wc["canary_windows"] == [0, 1]
    assert CN.window_copies({"c0": [(3600, 3610)]}, wins)["per_canary"]["c0"]["windows"] == [[0, 1]]   # overlap


# ================================================================================== passes, snapshots, seeds
def test_arm_steps_epochs_and_snapshot_steps():
    R_ = load_run()
    B = dict(steps=64)
    assert R_.arm_steps({"steps": "epoch"}, B, 310) == 310
    assert R_.arm_steps({"steps": "epoch", "epochs": 3}, B, 310) == 930
    assert R_.arm_steps({"steps": "epoch", "epochs": 3}, dict(B, epoch_steps_cap=3), 310) == 9
    assert R_.snapshot_steps({"steps": "epoch", "epochs": 3}, B, 310, 930) == []
    cfg = {"steps": "epoch", "epochs": 3, "snapshots": "epochs"}
    assert R_.snapshot_steps(cfg, B, 310, 930) == [(1, 310), (2, 620)]
    assert R_.snapshot_steps(cfg, B, 310, 400) == [(1, 310)]               # --max-steps cut
    with pytest.raises(SystemExit):
        R_.snapshot_steps({"steps": 100, "snapshots": "epochs"}, B, 310, 100)


def test_pass_orders_and_what_the_seed_changes():
    n = 7
    host_free = SimpleNamespace(train_windows=list(range(n)), cfg=TR.TrainConfig(seed=0, anchors_per_step=1))
    order = [TR.RowTrainer.train_index(host_free, k) for k in range(3 * n)]
    one_pass = [TR.RowTrainer.train_index(host_free, k) for k in range(n)]
    assert order[:n] == one_pass                                         # pass 1 = the one-pass order
    for e in range(3):
        assert [w for ep, w in order[e * n:(e + 1) * n]] == np.random.default_rng([0, e]).permutation(n).tolist()
        assert {ep for ep, _ in order[e * n:(e + 1) * n]} == {e}
    assert sorted(w for _, w in order[n:2 * n]) == list(range(n))        # every pass visits every window once
    s1 = SimpleNamespace(train_windows=list(range(n)), cfg=TR.TrainConfig(seed=1, anchors_per_step=1))
    assert [TR.RowTrainer.train_index(s1, k) for k in range(n)] != one_pass
    a0 = [TR.RowTrainer.anchor_indices(host_free, 0, 24, k) for k in range(5)]
    a1 = [TR.RowTrainer.anchor_indices(s1, 0, 24, k) for k in range(5)]
    assert a0 != a1                                                      # the seed also moves the anchor windows
    assert TR.visited_windows(n, 0, 3 * n) == list(range(n))
    # the canary planting does not read the training seed
    assert CN.config_from_arm({"seed": 1, "canaries": {"version": 2, "groups": [[2, 1]]}}) == \
        CN.config_from_arm({"seed": 0, "canaries": {"version": 2, "groups": [[2, 1]]}})


def test_snapshot_segments_equal_one_run_and_pass1_equals_one_pass(tmp_path):
    from test_ingest_train import _setup
    R_ = load_run()
    host, fmt, store, ws, anc, cfg, ix = _setup(max_steps=0)
    n = len(ws)
    mk = lambda steps: TR.RowTrainer(host, fmt, store.copy(), ws, TR.TrainConfig(**{**cfg.to_dict(),  # noqa: E731
                                                                                 "max_steps": steps}), anchors=anc)
    full = mk(2 * n)
    full.train()
    v_full = full.inner.values.data.clone()
    full.close()
    seg = mk(2 * n)
    snaps = R_.train_with_snapshots(seg, [(1, n)], tmp_path, tmp_path / "ck.pt", None,
                                    lambda c, p: torch.save(c.state_dict(), p))
    seg.train(None, tmp_path / "ck.pt")
    assert torch.equal(seg.inner.values.data, v_full)                   # segmenting does not change training
    seg.close()
    one = mk(n)
    one.train()
    snap = torch.load(tmp_path / "cartridge_e1.pt", weights_only=True)
    ref = one.cartridge().state_dict()
    assert torch.equal(snap["row_ids"], ref["row_ids"]) and torch.equal(snap["values"], ref["values"])
    assert snaps == [dict(pass_no=1, step=n, rows=int(ref["row_ids"].numel()), path=str(tmp_path / "cartridge_e1.pt"))]
    one.close()


class _FakeTrainer:
    def __init__(self, step=0):
        self.step_idx, self.calls = step, []

    def train(self, n, ckpt, cb):
        self.calls.append(n)
        self.step_idx += n

    def cartridge(self):
        return [0] * (10 + self.step_idx)


def test_train_with_snapshots_resume_rules(tmp_path):
    R_ = load_run()
    saved = []
    save = lambda c, p: (saved.append(p), Path(p).write_text("x"))  # noqa: E731
    t = _FakeTrainer()
    out = R_.train_with_snapshots(t, [(1, 5), (2, 10)], tmp_path, None, None, save)
    assert t.calls == [5, 5] and [o["step"] for o in out] == [5, 10] and len(saved) == 2
    t2 = _FakeTrainer(step=12)                                           # resumed past both: both kept, not re-made
    assert R_.train_with_snapshots(t2, [(1, 5), (2, 10)], tmp_path, None, None, save) == out and len(saved) == 2
    (tmp_path / "cartridge_e2.pt").unlink()
    with pytest.raises(SystemExit, match="missing"):
        R_.train_with_snapshots(_FakeTrainer(step=12), [(1, 5), (2, 10)], tmp_path, None, None, save)
    t3 = _FakeTrainer(step=10)                                           # resumed exactly at the boundary: made
    R_.train_with_snapshots(t3, [(1, 5), (2, 10)], tmp_path, None, None, save)
    assert t3.calls == [] and (tmp_path / "cartridge_e2.pt").exists()


def test_positive_control_windows():
    R_ = load_run()
    stream = np.arange(2000, dtype=np.int64) + 5
    spans = {"c0": [(1500, 1520), (100, 120)], "c1": [(30, 40)]}
    ws = R_.canary_windows(stream, spans, 512, 32, 2, EOS)
    assert [w.start for w in ws] == [0, 0, 1040]
    for w, (lo, hi) in zip(ws, [(30, 40), (100, 120), (1500, 1520)]):
        assert w.start <= lo and hi + 32 == w.start + w.ids.size or w.start == 0
        assert np.array_equal(w.ids, stream[w.start:w.start + w.ids.size])
        assert (w.labels[:-1] == stream[w.start + 1:w.start + w.ids.size]).all()
    assert ws[2].ids.size == 512 and ws[2].context.tolist() == stream[1038:1040].tolist()
    assert ws[0].context.tolist() == [EOS, EOS]
    end = R_.canary_windows(stream, {"c": [(1990, 1999)]}, 512, 32, 2, EOS)[0]
    assert end.start + end.ids.size == 2000 and end.labels[-1] == C.IGNORE


@pytest.mark.skipif(not (QWEN25 / "tokenizer.json").exists(), reason="Qwen2.5 tokenizer not cached")
def test_canary_stream_spans_locate_every_secret():
    R_ = load_run()
    tok = C.TokenizerJSON(QWEN25 / "tokenizer.json")
    texts = _texts(30)
    files = [C.SourceFile(p, "", len(t), t) for p, t in sorted(texts.items())]
    files, man = CN.plant_source_files(files, sorted(texts), V2)
    docs = C.tokenize_files(tok, files, "bare")
    stream = C.pack_stream(docs, EOS)
    spans = R_.canary_stream_spans(tok, files, docs, man)
    assert {k: len(v) for k, v in spans.items()} == {"c0": 2, "c1": 4, "c2": 8}
    sec = {r["id"]: r["secret"] for r in man["records"]}
    for cid, sp in spans.items():
        for lo, hi in sp:
            txt = tok.tok.decode(stream[lo:hi].tolist())
            assert sec[cid] in txt and len(txt) < len(sec[cid]) + 8


# ================================================================================================= REG v2
@pytest.mark.parametrize("seed", range(6))
def test_excluded_longest_is_exact_against_brute_force(seed):
    g = np.random.default_rng(seed)
    T = g.integers(0, 6, 400)
    spans, qs = [], []
    for i in range(12):
        a = int(g.integers(0, 380))
        b = a + int(g.integers(4, 16))
        spans.append((a, b))
        kind = i % 3
        if kind == 0:
            q = T[a:b].copy()                                      # loops over its own prompt
        elif kind == 1:
            q = T[(a + 37) % 300:(a + 37) % 300 + 12].copy()        # copies other train text
        else:
            q = g.integers(0, 6, 10)
        qs.append(q)
    got = R2.excluded_longest(T, qs, spans)
    for q, (a, b), x in zip(qs, spans, got["longest"]):
        Tm = T.copy()
        Tm[a:b] = -5                                                    # brute force with the span removed
        parts = [Tm[:a], Tm[b:]]
        ref = max(R.brute_longest(p, q) if p.size else 0 for p in parts)
        assert x == ref, (q, a, b, x, ref)
    assert got["recomputed"] >= 1
    assert R2.excluded_longest(T, qs, spans, full=R.VerbatimIndex(T).longest(qs)) == got
    # a continuation that copies another prompt's text (and its own) is found through the neighbourhood search
    a0, b0 = spans[0]
    a1, b1 = spans[1]
    q = np.concatenate([T[a1:b1], T[b1:b1 + 3]])
    x = R2.excluded_longest(T, [q, qs[1]], [(a0, b0), (a1, b1)])["longest"][0]
    if b0 <= a1 or a0 >= b1 + 3:                                        # its own span does not cover the copy
        assert x >= len(q)


def test_true_lcp_and_true_stats():
    D = type("D", (), {"__init__": lambda s, ids: setattr(s, "ids", ids), "__len__": lambda s: len(s.ids)})
    docs = [D(np.arange(100))]
    pre = [R.Prefix(0, 10, 5), R.Prefix(0, 50, 5), R.Prefix(0, 96, 5)]
    conts = [np.arange(10, 18), np.array([50, 51, 0, 0]), np.arange(96, 100)]
    lcp, avail = R2.true_lcp(docs, pre, conts)
    assert lcp == [8, 2, 4] and avail == [90, 50, 4]
    st = R2.true_stats(lcp, avail, conts, max_new=8, min_run=4)
    assert st["n_exact"] == 2 and st["n_ge_min_run"] == 2 and st["true_lcp"]["max"] == 8   # p0 full, p2 hits EOF
    pm = R2.paired_mean_diff([3, 4, 5], [1, 1, 1], n_boot=500)
    assert pm["mean"] == pytest.approx(3.0) and pm["ci"][0] <= 3.0 <= pm["ci"][1]


class CharTok:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        d = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            d["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return d

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i >= 2)


class FakeSplits:
    def __init__(self, texts):
        self.texts, self.data = texts, {"doc_header": "# file: {path}\n"}

    def files(self, split, part):
        return sorted(self.texts) if part == "train" else []

    def text(self, p, at=None):
        return self.texts[p]


def test_masked_prompts_hide_other_canaries_only():
    texts = {f"pkg/m{i}.py": _module(i, 4) for i in range(2)}
    _, man = CN.plant_texts(texts, sorted(texts), CN.CanaryConfig(n=3, m=1, seed=2, min_lines=5))
    docs, sites = R.load_train_docs(FakeSplits(texts), CharTok(), man)
    plain = {p.site.id: p for p in R.canary_prompts(docs, sites, context=4096)}
    got = R2.masked_canary_prompts(docs, sites, CharTok(), context=4096)
    secrets = {s.id: s.secret for s in sites}
    n_masked = 0
    for p in got:
        txt = CharTok().decode(p.prompt)
        assert all(secrets[o] not in txt for o in secrets if o != p.site.id)
        if p.masked:
            n_masked += 1
            assert R2.MASK_TEXT in txt and p.tail == plain[p.site.id].tail
        else:
            assert np.array_equal(p.prompt, plain[p.site.id].prompt)
    assert n_masked >= 1                                                # three canaries in one file: some contexts


def test_score_tails_cached_equals_full_sequence_scoring():
    from ngram_memory.backends.graft.module import NGramGraft
    from ngram_memory.eval.stage1.conditions import RowsCondition
    model = build_host("qwen2")
    g = NGramGraft(tiny_fmt(), table="sparse", init="random")
    gen = torch.Generator().manual_seed(7)
    prompt = torch.randint(3, 512, (11,), generator=gen).numpy()
    tails = [torch.randint(3, 512, (n,), generator=gen).numpy() for n in (5, 1, 7, 3, 6)]
    fill_random_rows(g, torch.randint(3, 512, (4, 64), generator=gen))
    cond = RowsCondition(g.fmt, None, graft=g)
    for c in (None, cond):
        ref = R.sequence_logprobs(model, prompt, tails, batch_size=2) if c is None else None
        if c is not None:
            with c.activate(model):
                ref = R.sequence_logprobs(model, prompt, tails, batch_size=2)
        got = R2.score_tails_cached(model, prompt, tails, c, batch_size=2)
        assert np.allclose(got, ref, atol=2e-4), (got, ref)
    assert not np.allclose(R2.score_tails_cached(model, prompt, tails, None),
                           R2.score_tails_cached(model, prompt, tails, cond), atol=1e-3)   # the graft is live


def test_canary_rows_counts_rows_in_the_cartridge():
    key = tiny_key()
    texts = {f"pkg/m{i}.py": _module(i, 4) for i in range(2)}
    _, man = CN.plant_texts(texts, sorted(texts), CN.CanaryConfig(n=2, m=1, seed=2, min_lines=5))
    docs, sites = R.load_train_docs(FakeSplits(texts), CharTok(), man)
    from ngram_memory.hashing import hasher_for
    first = min((s for s in sites if s.id == "c0"), key=lambda s: (s.doc, s.secret_start))
    d = docs[first.doc]
    off = np.asarray(d.offsets)
    lo = int(np.nonzero(off[:, 0] >= first.secret_start)[0][0])         # first token of the secret
    rows_all = hasher_for(key).rows_for(np.asarray(d.ids), context=None)
    some = np.unique(rows_all[lo + 6:lo + 9].reshape(-1))[:5]
    got = R2.canary_rows(docs, sites, key, some, np.full(some.size, 2.0))
    c0 = next(x for x in got if x["id"] == "c0")
    assert c0["nonzero"] and c0["rows_in_cartridge"] == some.size and c0["norm_max"] == 2.0
    none = R2.canary_rows(docs, sites, key, np.array([10 ** 9], np.int64), np.ones(1))
    assert not any(x["nonzero"] for x in none) and all(x["rows_read"] > 0 for x in none)
    clean = R2.clean_rows(docs, sites, key)
    every = np.unique(rows_all.reshape(-1))
    spec = R2.canary_rows(docs, sites, key, every, np.ones(every.size), clean)
    for x in spec:                     # random hex: most of a secret's 3-grams occur nowhere else in this corpus
        assert 0 < x["specific_rows"] < x["rows_read"] and x["specific_in_cartridge"] <= x["specific_rows"]
    assert spec[0]["specific_in_cartridge"] == spec[0]["specific_rows"] and spec[0]["specific_nonzero"]
    assert not set(np.unique(rows_all[first.tok_lo:first.tok_hi + 1].reshape(-1)).tolist()) <= set(clean.tolist())


def _cond(n_runs=0, n_true=0, n_exact=0, canaries=(), hits=()):
    return dict(runs_excl=dict(n_ge_min_run=n_runs), true=dict(n_ge_min_run=n_true, n_exact=n_exact),
                canaries=[dict(id=i, exact=e, secret_anywhere=False, rank=r) for i, e, r in canaries],
                secret_hits=list(hits))


def test_verdict_v2_and_positive_control():
    base = _cond(1, 1, 0, [("c0", False, 40), ("c1", True, 1)])
    ok = R2.verdict_v2(base, _cond(2, 2, 0, [("c0", False, 9), ("c1", True, 1)]))
    assert ok["passed"] and ok["canaries_extracted"] == [] and ok["canaries_rank1"] == []
    assert not R2.verdict_v2(base, _cond(3, 0, 0, [("c0", False, 9), ("c1", True, 1)]))["parts"]["runs"]
    assert not R2.verdict_v2(base, _cond(0, 0, 1, [("c0", False, 9), ("c1", True, 1)]))["parts"]["true"]
    v = R2.verdict_v2(base, _cond(0, 0, 0, [("c0", False, 1), ("c1", True, 1)]))
    assert v["canaries_rank1"] == ["c0"] and not v["passed"] and R2.positive_control_flagged(v)
    v = R2.verdict_v2(base, _cond(0, 0, 0, [("c0", True, 5), ("c1", True, 1)]))
    assert v["canaries_extracted"] == ["c0"] and R2.positive_control_flagged(v)
    assert not R2.positive_control_flagged(ok)


def test_canary_delta_and_dose_response():
    g = np.random.default_rng(0)
    refs_b = g.normal(-90, 3, 1000)
    base = dict(logp_true=-90.0, _logp_refs=refs_b.tolist(), exposure_bits=1.0, rank=500)
    cond = dict(logp_true=-80.0, _logp_refs=(refs_b + 0.5).tolist(), exposure_bits=5.0, rank=30)
    d = R2.canary_delta(base, cond, n_boot=500)
    assert d["dd_nats"] == pytest.approx(9.5) and d["ci"][0] <= 9.5 <= d["ci"][1] and d["ci"][1] - d["ci"][0] < 1e-6
    rows = [dict(id=f"c{i}", dd_nats=float(r), exposure=1.0, rank=10) for i, r in enumerate([0.1, 0.2, 0.4, 0.8])]
    dr = R2.dose_response(rows, {"c0": 2, "c1": 4, "c2": 8, "c3": 16})
    assert [x["repeats"] for x in dr["groups"]] == [2, 4, 8, 16] and dr["slope_nats_per_doubling"] > 0


@pytest.mark.skipif(not (QWEN25 / "tokenizer.json").exists(), reason="Qwen2.5 tokenizer not cached")
def test_reg2_cli_end_to_end_tiny_host(tmp_path):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from test_reg import _repo
    from ngram_memory.eval.stage1.splits import Splits, build_splits
    from ngram_memory.spec import KeySpec, graft_format
    from ngram_memory.tables import SparseRowStore
    repo = _repo(tmp_path, _texts(30))
    sp = tmp_path / "splits.json"
    Splits(build_splits(str(repo), "HEAD", "python", t_cut=None)).save(sp)
    S = Splits.load(sp)
    train = S.files("a", "train")
    _, man = CN.plant_texts_dose({p: S.text(p) for p in train}, train,
                                 CN.DoseCanaryConfig(groups=((2, 1), (4, 1)), seed=0, min_lines=10, min_sep_chars=200))
    CN.write_manifest(man, tmp_path / "canaries.json")
    host = tmp_path / "host"
    tok = AutoTokenizer.from_pretrained(str(QWEN25))
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, eos_token_id=151643, bos_token_id=151643,
                      tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(host)
    tok.save_pretrained(host)
    fmt = graft_format(64, KeySpec(vocab_size=151936, eos_token_id=151643, base=1000))
    (tmp_path / "fmt.json").write_text(fmt.to_json())
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, fmt.key.total_rows, (2000,), generator=g).unique()
    rows = tmp_path / "rows.pt"
    torch.save(SparseRowStore(fmt.padded_rows, fmt.row_width, ids,
                              torch.randn(len(ids), fmt.row_width, generator=g)).state_dict(), rows)
    spec = importlib.util.spec_from_file_location("reg2_cli", ROOT / "scripts/reg2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    s = mod.main(["--splits", str(sp), "--host", str(host), "--rows", str(rows), "--fmt", str(tmp_path / "fmt.json"),
                  "--label", "X", "--rows", str(rows), "--fmt", str(tmp_path / "fmt.json"), "--label", "PC",
                  "--positive-control", "PC", "--canaries", str(tmp_path / "canaries.json"), "--threads", "1",
                  "--n-prefixes", "5", "--min-prefix", "8", "--max-prefix", "16", "--max-new", "12", "--min-run", "8",
                  "--batch-size", "4", "--score-batch", "7", "--canary-context", "32", "--canary-max-new", "20",
                  "--n-random", "15", "--n-boot", "100", "--out", str(out)])
    assert set(s["conditions"]) == {"base", "X", "PC"} and s["canaries"]["n_records"] == 6
    assert s["canaries"]["repeats"] == {"c0": 2, "c1": 4}
    for k, c in s["conditions"].items():
        assert c["true"]["n"] == 5 and c["runs_excl"]["longest"]["n"] == 5
        assert all(x["n_ref"] == 15 and 1 <= x["rank"] <= 16 for x in c["canaries"])
        assert len((out / f"continuations__{k}.jsonl").read_text().splitlines()) == 5
        if k != "base":
            assert len(c["canary_rows"]) == 2
    assert set(s["verdict"]) == {"X", "PC"} and s["verdict"]["PC"]["positive_control"]
    assert s["validated"] is False and set(s["reg_pass"]) == {"X"}            # random rows memorise nothing
    vb = s["versus_base"]["X"]
    assert len(vb["canary_delta"]) == 2 and vb["dose_response"]["n"] == 2
    text = (out / "summary.json").read_text()
    assert not any(r["secret"] in text for r in man["records"])            # no planted secret in the summary


# ============================================================================================ the runner
def test_m1d_grid_loads_and_is_well_formed(tmp_path):
    R_ = load_run()
    run = R_.Runner(_args(tmp_path, grid=str(M1D)))
    g = run.grid
    names = [s["name"] for s in g["stages"]]
    for s in g["stages"]:
        assert s["kind"] in run.kinds() and set(s.get("after", [])) <= set(names[:names.index(s["name"])])
    assert run.soft == {"PC_memorise", "REG2"}
    assert g["reg2_flags"]["n_random"] >= 1000 and not any(s["kind"].endswith("test") for s in g["stages"])
    arms = {n: R_.resolve_arm(g, n, Path(tmp_path)) for n in g["arms"]}
    f1 = json.loads((ROOT / "experiments/m1c/results/runner-2026-11-11/F1_fullpass/arm.json").read_text())
    same = {k for k in f1 if k not in ("canaries", "name")}
    for n, a in arms.items():
        diff = {k for k in same if a.get(k) != f1[k]}
        expect = {"P3_seed0": set(), "S1_seed1": {"seed"}, "R01_rho0p1": {"rho"}, "DF_distill": {"objective"},
                  "PC_memorise": {"lr", "lambda_l1_rel", "lambda_l2"}}[n]
        assert diff == expect, (n, diff)
        assert isinstance(R_.canary_config(a), CN.DoseCanaryConfig)
    assert arms["P3_seed0"]["epochs"] == 3 and arms["P3_seed0"]["snapshots"] == "epochs"
    assert g["budgets"]["smoke"]["train_windows_cap"] == 3 and "train_windows_cap" not in g["budgets"]["full"]


def test_m1d_dag_parallel_order(tmp_path, monkeypatch):
    R_ = load_run()
    a = _args(tmp_path, grid=str(M1D), parallel=2, threads=32, max_stage_threads=16)
    run = R_.Runner(a)
    order, lock = [], threading.Lock()

    def stage(st):
        with lock:
            for d in st.get("after", []):
                assert (Path(a.out) / d / "result.json").exists(), f"{st['name']} started before {d}"
            order.append((st["name"], run.threads()))
        return dict(ok=True)
    monkeypatch.setattr(run, "kinds", lambda: {k: stage for k in R_.Runner.kinds(run)})
    assert run.run() == 0
    names = [n for n, _ in order]
    assert set(names[:2]) == {"B0_base_dev", "P3_seed0"} and names[-1] == "Z_report"
    assert all(t == 16 for _, t in order)


def _row(stage, e1, a, share, within=True, bar_a=True, bar_b=True, rho=0.3, obj="nll", seed=0):
    return dict(stage=stage, config=dict(rho=rho, objective=obj, seed=seed, steps="epoch"),
                x=dict(e1=e1, e1_ci=[e1 - 0.1, e1 + 0.1], e1_all=e1, r1=0.0, kl=0.0, r2=0.0), x_share=share,
                x_within=within, lambda_x=dict(weight=0.0075 if a is not None else None),
                bar_a=dict(diff=None if a is None else dict(pts=a, ci=[a - 0.1, a + 0.1]), holds=bar_a),
                bar_a_at_lstar=dict(diff=dict(pts=a or 0.0)), counts_matched=dict(weight=3e-4),
                bar_b=dict(ratio=3.0, diff=dict(ci=[0.1, 0.2]), holds=bar_b), any_bar=bar_a or bar_b)


def _report(tmp_path, monkeypatch, vals):
    R_ = load_run()
    run = R_.Runner(_args(tmp_path, grid=str(M1D)))
    st = next(s for s in run.grid["stages"] if s["name"] == "Z_report")
    rows = [_row(st["arms"][k], *v[:3], **(v[3] if len(v) > 3 else {})) for k, v in vals.items()]
    ctx = dict(lambda_star=dict(weight=0.0075, e1=2.0))
    monkeypatch.setattr(run, "_m1c_rows", lambda names, c, w: (ctx, [r for r in rows if r["stage"] in names]))
    return run.stage_m1d_report(st)


BASE = {"P3_e1": (1.3, 1.10, 0.03), "P3_e2": (1.6, 1.40, 0.04), "P3_e3": (1.7, 1.45, 0.05),
        "S1_seed1": (1.25, 1.05, 0.03, dict(seed=1)), "R01_rho0p1": (1.3, 1.05, 0.02, dict(rho=0.1)),
        "DF_distill": (1.0, 0.8, 0.05, dict(obj="distill"))}


def test_m1d_report_rules(tmp_path, monkeypatch):
    out = _report(tmp_path, monkeypatch, BASE)
    d = out["decisions"]
    assert d["epochs"]["value"] == 2                     # pass 2 adds 0.30, pass 3 adds 0.05 < 0.1
    assert d["rho"]["value"] == 0.1                      # R01 within 0.1 of pass 1
    assert d["distill"]["value"] == "drop"
    rep = d["replicate"]
    assert rep["abs_diff_e1"] == pytest.approx(0.05) and rep["abs_diff_bar_a"] == pytest.approx(0.05)
    assert rep["seed1_holds_both"] and not rep["m1c_pass_fragile"]
    assert out["m2_recipe"] == dict(objective="next-token", epochs=2, rho=0.1)
    c = out["m1c_comparison"]
    assert c["delta"]["bar_a_diff"] == pytest.approx(1.10 - 1.138484416686527)
    v = dict(BASE, P3_e2=(1.35, 1.15, 0.03))             # pass 2 adds 0.05: stop at 1
    assert _report(tmp_path / "a", monkeypatch, v)["decisions"]["epochs"]["value"] == 1
    v = dict(BASE, P3_e2=(1.6, 1.40, 0.09))              # share +0.06: stop at 1
    assert _report(tmp_path / "b", monkeypatch, v)["decisions"]["epochs"]["value"] == 1
    v = dict(BASE, P3_e3=(1.9, 1.60, 0.05))              # still gaining at pass 3
    e = _report(tmp_path / "c", monkeypatch, v)["decisions"]["epochs"]
    assert e["value"] == 3 and e["still_gaining_at_last"]
    v = dict(BASE, R01_rho0p1=(1.1, 0.95, 0.02, dict(rho=0.1)), DF_distill=(1.4, 1.2, 0.04, dict(obj="distill")),
             S1_seed1=(1.0, 0.8, 0.03, dict(seed=1, bar_a=False)))
    o = _report(tmp_path / "d", monkeypatch, v)["decisions"]
    assert o["rho"]["value"] == 0.3 and o["distill"]["value"] == "keep"
    assert o["replicate"]["m1c_pass_fragile"] and o["rho"]["within_seed_noise"] is True
    v = dict(BASE, R01_rho0p1=(1.1, None, 0.3, dict(rho=0.1)))            # R01 has no lambda_X
    assert _report(tmp_path / "e", monkeypatch, v)["decisions"]["rho"]["value"] == 0.3


def test_rows_eval_links_the_snapshot_and_reports_its_step(tmp_path, monkeypatch):
    R_ = load_run()
    run = R_.Runner(_args(tmp_path, grid=str(M1D)))
    out = Path(run.out)
    sd = out / "P3_seed0"
    sd.mkdir(parents=True)
    (sd / "cartridge_e1.pt").write_text("e1")
    (sd / "fmt.json").write_text("{}")
    R_.jdump(sd / "result.json", dict(config=dict(rho=0.3, seed=0, name="P3_seed0"), train=dict(steps=930),
                                      rows={}, cartridge=dict(rows=9),
                                      snapshots=[dict(pass_no=1, step=310, rows=7, path=str(sd / "cartridge_e1.pt"))]))
    seen = {}
    monkeypatch.setattr(run, "evaluate", lambda name, sets, conds, **kw: seen.update(conds=conds, kw=kw) or
                        dict(summary={}, run=dict(wall_s=1.0, peak_rss_mb=1.0)))
    monkeypatch.setattr(R_, "dev_metrics", lambda s, lab, prim: dict(label=lab))
    st = next(s for s in run.grid["stages"] if s["name"] == "P3_e1_dev")
    r = run.stage_rows_eval(st)
    link = out / "P3_e1_dev" / "cartridge.pt"
    assert link.is_symlink() and link.read_text() == "e1"
    assert seen["conds"] == f"rows:{link},countgrid:kn+rows:{link}"
    assert r["snapshot_step"] == 310 and r["cartridge"]["rows"] == 7 and r["config"]["snapshot"] == 1
    with pytest.raises(R_.StageFailed):
        run.stage_rows_eval(dict(st, name="X", snapshot=2))


def test_reg2_stage_runs_without_a_failed_positive_control(tmp_path, monkeypatch):
    R_ = load_run()
    run = R_.Runner(_args(tmp_path, grid=str(M1D)))
    out = Path(run.out)
    for n in ("P3_seed0", "S1_seed1"):
        sd = out / n
        sd.mkdir(parents=True)
        R_.jdump(sd / "canaries.json", dict(sha256="abc"))
        R_.jdump(sd / "result.json", dict(train=dict(steps=930 if n == "P3_seed0" else 310), snapshots=[
            dict(pass_no=1, step=310, rows=1, path=str(sd / "cartridge_e1.pt"))]))
    cmds = []

    def fake(cmd, log, lim, th):
        cmds.append(cmd)
        d = Path(cmd[cmd.index("--out") + 1])
        d.mkdir(parents=True, exist_ok=True)
        R_.jdump(d / "summary.json", dict(verdict={}, reg_pass={}, validated=False, versus_base={}, conditions={},
                                          canaries={}, probes=[]))
        return dict(returncode=0, aborted=None, wall_s=1.0, peak_rss_mb=1.0)
    monkeypatch.setattr(R_, "run_guarded", fake)
    st = next(s for s in run.grid["stages"] if s["name"] == "REG2")
    r = run.stage_reg2(st)
    c = cmds[0]
    assert c.count("--rows") == 3 and "--positive-control" not in c and r["skipped"][0]["label"] == "PC"
    assert [c[i + 1] for i, x in enumerate(c) if x == "--label"] == ["P3_e1", "P3_e3", "S1_seed1"]
    R_.jdump(out / "S1_seed1" / "canaries.json", dict(sha256="other"))
    with pytest.raises(R_.StageFailed) as e:
        run.stage_reg2(st)
    assert "different canary manifests" in e.value.info["error"]


def test_redaction_removes_every_planted_secret(tmp_path):
    spec = importlib.util.spec_from_file_location("m1d_redact", ROOT / "experiments/m1d/redact.py")
    red = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(red)
    _, man = CN.plant_texts_dose(_texts(), sorted(_texts()), V2)
    run = tmp_path / "run"
    (run / "P3_seed0" / "eval").mkdir(parents=True)
    CN.write_manifest(man, run / "P3_seed0" / "canaries.json")
    sec = man["records"][0]["secret"]
    (run / "P3_seed0" / "train.json").write_text(json.dumps(dict(canaries=dict(manifest=man), note=f"x {sec} y")))
    (run / "P3_seed0" / "eval" / "summary.json").write_text(json.dumps({"a": [sec]}))
    (run / "P3_seed0" / "continuations__x.jsonl").write_text(sec)
    (run / "progress.json").write_text("{}")
    r = red.redact(run, tmp_path / "out")
    assert r == dict(files=4, secrets=3)
    got = "".join(p.read_text() for p in (tmp_path / "out").rglob("*") if p.is_file())
    assert sec not in got and red.MASK in got and not (tmp_path / "out/P3_seed0/continuations__x.jsonl").exists()
    assert red.leaks("zz" + sec[3:20] + "zz", [sec])


# ================================================================================ countgrid without KL (M1d)
def test_countgrid_without_kl_gives_the_same_nll_and_hits():
    from test_countgrid import WEIGHTS
    from ngram_memory.eval import metrics as M
    from ngram_memory.eval.stage1 import countgrid as cg
    from ngram_memory.eval.stage1 import scoring as S
    from ngram_memory.eval.stage1.baselines import countlm as cl
    from ngram_memory.eval.stage1.conditions import RowsCondition
    from conftest import V
    qwen2 = build_host("qwen2")
    g = np.random.default_rng(0)
    pat = np.tile(np.array([7, 8, 9, 10, 11]), 80)
    kn = cl.KNTrigram([pat, g.integers(3, V, 300), g.integers(3, 40, 400)], V, EOS)
    docs = [S.Doc("p", np.concatenate([pat[:120], g.integers(3, 40, 100)])), S.Doc("q", g.integers(3, V, 190))]
    wins = S.plan_windows(docs, 128, 64, 64)
    rc = RowsCondition(tiny_fmt(), None, name="rows[t]")
    for d in docs:
        fill_random_rows(rc.graft, torch.from_numpy(d.ids[None]), scale=0.5)
    base = S.score(qwen2, docs, wins)
    groups = np.asarray([base.groups[i] for i in base.doc])
    for inner in (None, rc):
        for other in (0.0, 0.003):
            cond = cg.CountGridCondition(kn, "kn", WEIGHTS, other_weight=other, inner=inner)
            full = cg.score_grid(qwen2, docs, wins, cond)
            lite = cg.score_grid(qwen2, docs, wins, cond, kl=False)
            assert np.array_equal(full.nll, lite.nll) and np.array_equal(full.hit, lite.hit)
            assert np.isnan(lite.kl).all() and not lite.agree.any() and lite.meta["kl"] is False
            assert not lite.meta["two_forwards"] and "kl" not in full.meta
            assert lite.meta["forward_tokens"] * (2 if inner is not None else 1) == full.meta["forward_tokens"]
            if inner is None:
                assert np.array_equal(lite.host_nll, base.nll)
            else:
                assert np.isnan(lite.host_nll).all()
            mf, ml = cg.grid_metrics(base, full, groups), cg.grid_metrics(base, lite, groups)
            for a, b in zip(mf["grid"], ml["grid"]):
                assert "mean_kl" in a["paired"] and "mean_kl" not in b["paired"]
                assert {k: v for k, v in a["paired"].items() if k not in ("mean_kl", "top1_agreement")} == b["paired"]
            if inner is not None:
                assert ml["host_nll_max_abs_diff_vs_base"] is None
    # the default path is untouched: same arrays as before the flag existed (kl=True is the default)
    assert np.array_equal(cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "kn", WEIGHTS)).kl,
                          cg.score_grid(qwen2, docs, wins, cg.CountGridCondition(kn, "kn", WEIGHTS), kl=True).kl)
    assert M is not None


def test_eta_simulation_respects_the_dag_and_thread_policy():
    spec = importlib.util.spec_from_file_location("m1d_eta", ROOT / "experiments/m1d/eta.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    g = json.loads(M1D.read_text())
    r = E.simulate(g, 2, 32, 16)
    end = {s["stage"]: s for s in r["stages"]}
    assert set(end) == {s["name"] for s in g["stages"]}
    for s in g["stages"]:
        for d in s.get("after", []):
            assert end[d]["end_h"] <= end[s["name"]]["start_h"] + 1e-9
    assert all(s["threads"] == 16 for s in r["stages"]) and 8 < r["hours"] < 14
    g2 = json.loads(M1D.read_text())
    g2["eval"]["full"].pop("countgrid_kl_sets")
    assert E.simulate(g2, 2, 32, 16)["hours"] > r["hours"]
    # 4 x 8 threads: training steps run at M1c's measured 4 x 8 rate
    assert 49.0 < E.STEP_NLL * (16 / 8) * (1 + E.BETA_TRAIN * (32 - 16) / 32) < 50.0


def test_seed_does_not_change_the_graft_or_the_initial_rows():
    from test_ingest_train import _setup
    host, fmt, store, ws, anc, cfg, ix = _setup(max_steps=2)
    trs = []
    for seed in (0, 1):
        torch.manual_seed(seed)                                         # the runner seeds torch from the arm seed
        t = TR.RowTrainer(host, fmt, store.copy(), ws, TR.TrainConfig(**{**cfg.to_dict(), "seed": seed}), anchors=anc)
        trs.append((t.graft.ple.value_proj.weight.detach().clone(), t.inner.values.data.clone(),
                    [t.train_index(k) for k in range(len(ws))]))
        t.close()
    (w0, v0, o0), (w1, v1, o1) = trs
    assert torch.equal(w0, w1) and torch.equal(v0, v1) and o0 != o1

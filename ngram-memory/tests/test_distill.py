"""Stage-1b distillation-with-context (ingest/teacher.py + TrainConfig.objective in ingest/train.py) on tiny hosts:
objective='nll' is bit-identical to the M1b trainer, the distill loss vanishes when student == teacher, gradients
reach only the rows read, retrieval never returns the window's own files, the teacher cache round-trips under its
lock, and resume is bit-exact with distill/mix."""
from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import EOS, V, build_host, tiny_fmt
from ngram_memory.ingest import corpus as C
from ngram_memory.ingest import kltargets as KL
from ngram_memory.ingest import rowpolicy as RP
from ngram_memory.ingest import teacher as TT
from ngram_memory.ingest import train as TR
from ngram_memory.tables import SparseRowStore

REPO = Path(__file__).resolve().parents[2]
M1B_COMMIT = "94a1b62"            # last commit before the objective switch (M1b's trainer)
TIMING = ("fwd_bwd_s", "step_s", "rss_mb", "peak_rss_mb")


# ---------------------------------------------------------------------------------------------- fixtures
def encode(text: str) -> np.ndarray:
    out = []
    for w in text.split():
        m = re.fullmatch(r"t(\d+)", w)
        out.append(int(m.group(1)) if m and 3 <= int(m.group(1)) < V else 3 + zlib.crc32(w.encode()) % (V - 3))
    return np.asarray(out, np.int64)


def decode(ids) -> str:
    return " ".join(f"t{int(i)}" for i in ids)


def _repo(n_files=8, lines=10, words=6, seed=0):
    """Files i and i + n/2 share most lines (so BM25 finds the partner file); each has a private vocabulary."""
    rng = np.random.default_rng(seed)
    half = n_files // 2
    base = [[rng.integers(3, 200, words) for _ in range(lines)] for _ in range(half)]
    files = {}
    for i in range(n_files):
        ls = []
        for j, ln in enumerate(base[i % half]):
            ln = ln.copy()
            if (j + i) % 3 == 0:
                ln[0] = 200 + 30 * (i % half) + rng.integers(0, 30)      # private-ish token
            ls.append(" ".join(f"t{t}" for t in ln) + "\n")
        files[f"pkg/f{i}.py"] = "".join(ls)
    return files


def _data(window=48, stride=40):
    files = _repo()
    docs = [C.Document(p, "raw", encode(files[p])) for p in sorted(files)]
    stream = C.pack_stream(docs, EOS)
    ws = C.make_windows(stream, window, stride, 2, EOS)
    return files, docs, stream, ws


def _tcfg(**kw):
    d = dict(ctx_tokens=40, k=16, chunk_lines=3, chunk_stride=2, prefill_chunk=None, logits_chunk=16)
    d.update(kw)
    return TT.TeacherConfig(**d)


def _setup(arm="G-lin", anchors=True, kind="qwen2", **cfg_kw):
    host = build_host(kind)
    fmt = tiny_fmt("G-dir" if arm == "G-dir" else "G-lin")
    files, docs, stream, ws = _data()
    ix = RP.NGramIndex(fmt.key, [stream])
    rows = RP.select_rows(ix, RP.RowPolicy("count", 2))
    store = SparseRowStore.reserved(fmt.padded_rows, fmt.row_width, torch.from_numpy(rows))
    lw, _ = RP.ngram_loss_weights(ix, pct=99.0)
    win_w = [lw[w.start:w.start + w.ids.size] for w in ws]
    anc = {}
    if anchors:
        rng = np.random.default_rng(5)
        a_stream = C.pack_stream([rng.integers(3, 60, 40) for _ in range(4)], EOS)
        aw = C.make_windows(a_stream, 40, 40, 2, EOS)
        anc = {"near": (aw, KL.compute_kl_targets(host, aw, k=16))}
    kw = dict(arm=arm, lr=0.05, anchor_lambdas={"near": 1.0}, loss_chunk=16, max_steps=8)
    kw.update(cfg_kw)
    return host, fmt, store, ws, win_w, anc, kw, (files, docs)


def _teacher(host, ws, files, docs, **kw):
    return TT.compute_teacher_targets(host, ws, docs, files, encode, decode, _tcfg(**kw))


def _strip(recs):
    return [{k: v for k, v in r.items() if k not in TIMING} for r in recs]


def _m1b_train_module():
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{M1B_COMMIT}:ngram-memory/ngram_memory/ingest/train.py"],
                             check=True, capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"M1b trainer source ({M1B_COMMIT}) not available in this checkout")
    name = "ngram_memory.ingest._train_m1b_ref"
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "ngram_memory.ingest"
    sys.modules[name] = mod
    exec(compile(src, f"<{M1B_COMMIT}:train.py>", "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------------------------- nll == M1b trainer
@pytest.mark.parametrize("arm", ["G-lin", "key-shuffle"])
def test_nll_objective_is_bit_identical_to_the_m1b_trainer(arm, tmp_path):
    ref = _m1b_train_module()
    kw = dict(lambda_l2=1e-3, lambda_l1_rel=0.1, kappa=5.0, acc_init_steps=3, max_steps=6)
    out = {}
    for tag, mod in (("ref", ref), ("new", TR)):
        host, fmt, store, ws, win_w, anc, ckw, _ = _setup(arm=arm, **kw)
        cfg = mod.TrainConfig(**ckw)
        tr = mod.RowTrainer(host, fmt, store, ws, cfg, anchors=anc, loss_weights=win_w,
                            log_path=tmp_path / f"{tag}.jsonl")
        recs = tr.train(ckpt_path=tmp_path / f"{tag}.pt")
        out[tag] = (_strip(recs), tr.inner.values.data.clone(), tr.opt.acc.clone(), tr.eval_nll(ws)["nll"])
        tr.close()
    (ra, va, aa, na), (rb, vb, ab, nb) = out["ref"], out["new"]
    assert ra == rb
    assert torch.equal(va, vb) and torch.equal(torch.nan_to_num(aa, -1.0), torch.nan_to_num(ab, -1.0)) and na == nb
    assert TR.TrainConfig(**{k: v for k, v in kw.items()}).objective == "nll"
    # an M1b checkpoint (no objective fields) resumes in the new trainer and finishes bit-identically
    host, fmt, store, ws, win_w, anc, ckw, _ = _setup(arm=arm, **kw)
    b = ref.RowTrainer(host, fmt, store, ws, ref.TrainConfig(**ckw), anchors=anc, loss_weights=win_w)
    b.train(3, ckpt_path=tmp_path / "m1b_ck.pt")
    b.close()
    assert "objective" not in torch.load(tmp_path / "m1b_ck.pt", weights_only=False)["cfg"]
    c = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**ckw), anchors=anc, loss_weights=win_w)
    c.load_checkpoint(tmp_path / "m1b_ck.pt")
    c.train()
    assert torch.equal(c.inner.values.data, va)
    c.close()


def test_visited_windows_follow_the_trainer_order():
    host, fmt, store, ws, win_w, anc, ckw, _ = _setup(anchors=False)
    n = len(ws)
    for steps in (1, 3, n, n + 2, 2 * n + 1):
        tr = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**dict(ckw, max_steps=steps)))
        assert TR.visited_windows(n, ckw.get("seed", 0), steps) == sorted({tr.train_index(k)[1] for k in range(steps)})
        tr.close()


# ------------------------------------------------------------------------------------------------ retrieval
def test_retrieval_excludes_the_windows_own_files_and_is_deterministic():
    files, docs, stream, ws = _data()
    spans = TT.doc_spans(docs)
    assert spans[-1].end == stream.size
    for s in spans:          # every span ends on the EOS pack_stream appended
        assert stream[s.end - 1] == EOS
    r = TT.TeacherRetriever(files, _tcfg(ctx_tokens=200), encode)
    n_multi = 0
    for w in ws:
        own = TT.window_files(w, spans)
        # the inputs' files plus the file of the last target (stream[w.start + L])
        assert own and own == sorted({sp.path for sp in spans if sp.start < w.start + w.ids.size + 1 and sp.end > w.start})
        n_multi += len(own) > 1
        q = decode(w.ids)
        got = r.retrieve(q, own)
        assert got.stats["excluded"] == own
        assert got.stats["files"] and not set(got.stats["files"]) & set(own)
        assert got.stats["tokens"] == got.prefix.size <= 200
        hits = r.hits(q, own)
        assert all(r.chunks[i].path not in own for _, i in hits)
        assert hits == r.hits(q, own)
        # without the exclusion the window's own file is retrievable (the exclusion is doing the work)
        assert set(r.chunks[i].path for _, i in r.hits(q, [])) & set(own)
    assert n_multi > 0       # some packed windows span two files: all of them are excluded


def test_retrieval_ties_break_by_path_and_budget_truncates():
    txt = "t5 t6 t7\nt8 t9 t10\n"
    files = {"b/x.py": txt, "a/x.py": txt, "c/y.py": "t50 t51\n"}
    r = TT.TeacherRetriever(files, _tcfg(ctx_tokens=7, chunk_lines=2, chunk_stride=2), encode)
    h = r.hits("t5 t9", [])
    assert [r.chunks[i].path for _, i in h[:2]] == ["a/x.py", "b/x.py"] and h[0][0] == h[1][0] > h[2][0]
    got = r.retrieve("t5 t9", [])
    ids_a = list(r.chunk_ids(h[0][1]))
    assert got.prefix.size == 7 and got.stats["chunks"] == 1 and list(got.prefix) == ids_a[:7]
    got = TT.TeacherRetriever(files, _tcfg(ctx_tokens=100, chunk_lines=2, chunk_stride=2), encode).retrieve("t5", [])
    # best chunk nearest the window: the prefix ends with the best hit's tokens
    assert list(got.prefix[-len(ids_a):]) == ids_a and got.stats["files"] == ["a/x.py", "b/x.py", "c/y.py"]
    assert TT.TeacherRetriever(files, _tcfg(ctx_tokens=0), encode).retrieve("t5", []).prefix.size == 0


# --------------------------------------------------------------------------------------------- teacher
def test_teacher_targets_are_the_host_on_context_plus_window():
    host = build_host("qwen2")
    files, docs, stream, ws = _data()
    tg = _teacher(host, ws[:3], files, docs)
    assert tg.meta["window_index"] == [0, 1, 2] and len(tg.meta["retrieval"]) == 3
    assert tg.meta["window_sha"] == [KL.window_digest(w) for w in ws[:3]]
    spans = TT.doc_spans(docs)
    r = TT.TeacherRetriever(files, _tcfg(), encode)
    for j, w in enumerate(ws[:3]):
        pre = r.retrieve(decode(w.ids), TT.window_files(w, spans)).prefix
        assert pre.size > 0 and tg.meta["retrieval"][j]["tokens"] == pre.size
        with torch.no_grad():
            x = torch.from_numpy(np.concatenate([pre, w.ids]))[None]
            z = host(input_ids=x).logits[0, pre.size:].float()
        ref = KL.topk_targets_from_logits(z, 16)
        torch.testing.assert_close(tg[j].logp, ref.logp, atol=1e-5, rtol=0)
        torch.testing.assert_close(tg[j].rest, ref.rest, atol=1e-5, rtol=0)
        assert (tg[j].idx[:, 0] == ref.idx[:, 0]).float().mean() > 0.99
        # context changes the targets (else distill == self-distillation)
        assert not torch.allclose(tg[j].logp, KL.compute_kl_targets(host, [w], k=16)[0].logp, atol=1e-3)
    # KV-cached prefill chunks agree with one forward
    tc = _teacher(host, ws[:2], files, docs, prefill_chunk=16)
    for j in range(2):
        torch.testing.assert_close(tc[j].logp, tg[j].logp, atol=1e-4, rtol=0)
    # graft-on hosts are refused
    hk = host.register_forward_pre_hook(lambda m, a: None)
    with pytest.raises(RuntimeError):
        _teacher(host, ws[:1], files, docs)
    hk.remove()


def test_teacher_cache_round_trip_and_lock(tmp_path):
    host = build_host("qwen2")
    files, docs, stream, ws = _data()
    calls = []

    def builder(*a, **kw):
        calls.append(1)
        return TT.compute_teacher_targets(*a, **kw)

    args = (tmp_path, host, "tiny/qwen2", ws, docs, files, encode, decode, _tcfg())
    res = []
    th = [threading.Thread(target=lambda: res.append(TT.load_or_build_teacher(*args, window_index=[1, 3],
                                                                                builder=builder)))
          for _ in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert len(calls) == 1 and len(res) == 2            # one computes under the lock, the other loads it
    (a, pa), (b, pb) = res
    assert pa == pb and pa.exists() and Path(str(pa) + ".lock").exists()
    assert not list(tmp_path.glob("*.part*"))
    c, pc = TT.load_or_build_teacher(*args, window_index=[1, 3], builder=builder)
    assert len(calls) == 1 and pc == pa
    for x in (a, b):
        for j in range(2):
            assert torch.equal(x[j].idx, c[j].idx) and torch.equal(x[j].logp, c[j].logp) and torch.equal(x[j].rest, c[j].rest)
    assert c.meta["window_index"] == [1, 3] and c.meta["cache_key"] == a.meta["cache_key"]
    assert c.meta["retrieval"] == a.meta["retrieval"]
    # any key ingredient changes the file
    _, p2 = TT.load_or_build_teacher(*args[:-1], _tcfg(ctx_tokens=24), window_index=[1, 3], builder=builder)
    _, p3 = TT.load_or_build_teacher(*args, window_index=[1, 2], builder=builder)
    f2 = dict(files, **{"pkg/f0.py": files["pkg/f0.py"] + "t9\n"})
    _, p4 = TT.load_or_build_teacher(tmp_path, host, "tiny/qwen2", ws, docs, f2, encode, decode, _tcfg(),
                                     window_index=[1, 3], builder=builder)
    assert len({pa, p2, p3, p4}) == 4 and len(calls) == 4
    assert TT.retrieval_summary(c)["windows"] == 2


# ------------------------------------------------------------------------------------------------ trainer
def test_distill_loss_is_zero_when_student_equals_teacher():
    host, fmt, store, ws, win_w, anc, ckw, _ = _setup(anchors=False, max_steps=1)
    tg = KL.compute_kl_targets(host, ws, k=16)             # teacher = the host on the window = the zero-row student
    tr = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**dict(ckw, objective="distill")), loss_weights=win_w,
                       teacher=tg)
    with torch.no_grad():                                   # every window: KL(host || host + zero rows) = 0
        for i, w in enumerate(ws):
            assert abs(tr._pass(w, None, 1.0, win_w[i], False, teacher=tg[i])["kl_teacher"]) < 1e-6
    r = tr.step()
    tr.close()
    nl = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**ckw), loss_weights=win_w)
    r0 = nl.step()
    nl.close()
    assert abs(r["kl_teacher"]) < 1e-6 and r["train_loss"] == r["kl_teacher"]
    assert r["grad_norm"] < 1e-3 * r0["grad_norm"]
    assert r["nll"] == pytest.approx(r0["nll"], rel=1e-6)   # the logged NLL is the same quantity as 'nll'
    v = KL.topk_kl(torch.randn(5, V).repeat(2, 1), KL.topk_targets_from_logits(torch.randn(5, V).repeat(2, 1), 8))
    assert (v > 0).all()


@pytest.mark.parametrize("objective,kind", [("distill", "qwen2"), ("mix", "qwen2"), ("distill", "qwen3_5")])
def test_distill_trains_only_read_rows_and_reduces_teacher_kl(objective, kind):
    host, fmt, store, ws, win_w, anc, ckw, (files, docs) = _setup(kind=kind, max_steps=12)
    fp0 = TR.host_fingerprint(host)
    tg = _teacher(host, ws, files, docs)
    cfg = TR.TrainConfig(**dict(ckw, objective=objective, distill_alpha=0.7))
    tr = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc, loss_weights=win_w, teacher=tg)

    def mean_kl():
        with torch.no_grad():
            return float(np.mean([tr._pass(w, None, 1.0, win_w[i], False, teacher=tg[i])["kl_teacher"]
                                  for i, w in enumerate(ws)]))
    before = mean_kl()
    read_union = set()
    for _ in range(cfg.max_steps):
        k = tr.step_idx
        _, wi = tr.train_index(k)
        read_union |= set(tr.inner.row_ids[torch.from_numpy(tr.read_rows(ws[wi]))].tolist())
        for set_no, name in enumerate(sorted(anc)):
            for j in tr.anchor_indices(set_no, len(anc[name][0]), k):
                read_union |= set(tr.inner.row_ids[torch.from_numpy(tr.read_rows(anc[name][0][j]))].tolist())
        r = tr.step()
        assert r["objective"] == objective and r["alpha"] == (1.0 if objective == "distill" else 0.7)
        want = r["kl_teacher"] if objective == "distill" else 0.7 * r["kl_teacher"] + 0.3 * r["nll"]
        assert r["train_loss"] == pytest.approx(want, rel=1e-5, abs=1e-7)
    after = mean_kl()
    if objective == "distill" and kind == "qwen2":   # (the tiny qwen3_5 host barely uses context: KL ~ 1e-5)
        assert after < before - 1e-4, (before, after)
    changed = set(tr.inner.row_ids[(tr.inner.values.data != 0).any(1)].tolist())
    assert changed and changed <= read_union <= set(store.row_ids.tolist())
    tr.close()
    assert TR.host_fingerprint(host) == fp0
    assert not any(p.requires_grad for p in host.parameters())


def test_distill_grad_reaches_only_rows_read_by_the_window():
    """No anchors: after one distill step the rows with a nonzero update are exactly rows the train window read."""
    host, fmt, store, ws, win_w, anc, ckw, (files, docs) = _setup(anchors=False, max_steps=1)
    tg = _teacher(host, ws, files, docs)
    tr = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**dict(ckw, objective="distill")), teacher=tg)
    _, wi = tr.train_index(0)
    read = set(tr.inner.row_ids[torch.from_numpy(tr.read_rows(ws[wi]))].tolist())
    tr.step()
    changed = set(tr.inner.row_ids[(tr.inner.values.data != 0).any(1)].tolist())
    assert changed and changed <= read
    tr.close()


@pytest.mark.parametrize("objective", ["distill", "mix"])
def test_resume_is_bit_exact_with_distill(objective, tmp_path):
    kw = dict(lambda_l2=1e-3, lambda_l1_rel=0.1, kappa=5.0, acc_init_steps=3, max_steps=6, objective=objective)
    host, fmt, store, ws, win_w, anc, ckw, (files, docs) = _setup(**kw)
    tg, _ = TT.load_or_build_teacher(tmp_path / "cache", host, "tiny", ws, docs, files, encode, decode, _tcfg(),
                                     window_index=TR.visited_windows(len(ws), 0, 6))
    cfg = TR.TrainConfig(**ckw)
    a = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc, loss_weights=win_w, teacher=tg)
    ra = a.train()
    va, acc_a = a.inner.values.data.clone(), a.opt.acc.clone()
    a.close()
    b = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc, loss_weights=win_w, teacher=tg)
    b.train(3, ckpt_path=tmp_path / "ck.pt")
    b.close()
    tg2 = KL.KLTargets.load(next((tmp_path / "cache").glob("teacher_*.safetensors")))   # reloaded from the cache
    c = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc, loss_weights=win_w, teacher=tg2)
    c.load_checkpoint(tmp_path / "ck.pt")
    assert c.step_idx == 3
    rc = c.train()
    assert torch.equal(c.inner.values.data, va)
    assert torch.equal(torch.nan_to_num(c.opt.acc, -1.0), torch.nan_to_num(acc_a, -1.0))
    assert _strip(rc) == _strip(ra[3:])
    c.close()
    # a distill checkpoint does not resume as nll, nor against other teacher targets
    d = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**dict(ckw, objective="nll")), anchors=anc)
    with pytest.raises(ValueError):
        d.load_checkpoint(tmp_path / "ck.pt")
    d.close()
    other = KL.KLTargets(tg.k, tg.windows, dict(tg.meta, identity="different"))
    e = TR.RowTrainer(host, fmt, store, ws, cfg, anchors=anc, loss_weights=win_w, teacher=other)
    with pytest.raises(ValueError):
        e.load_checkpoint(tmp_path / "ck.pt")
    e.close()


def test_teacher_alignment_is_checked():
    host, fmt, store, ws, win_w, anc, ckw, (files, docs) = _setup(anchors=False, max_steps=len(_data()[3]))
    n = len(ws)
    tg = _teacher(host, ws, files, docs)
    cfg = TR.TrainConfig(**dict(ckw, objective="distill"))
    with pytest.raises(ValueError):          # nll with a teacher / distill without one
        TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**ckw), teacher=tg)
    with pytest.raises(ValueError):
        TR.RowTrainer(host, fmt, store, ws, cfg)
    with pytest.raises(ValueError):
        TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**dict(ckw, objective="kl")), teacher=tg)
    sub = KL.KLTargets(tg.k, tg.windows[:-1], dict(tg.meta, window_index=tg.meta["window_index"][:-1],
                                                   window_sha=tg.meta["window_sha"][:-1]))
    with pytest.raises(ValueError, match="visited"):   # a full epoch visits every window
        TR.RowTrainer(host, fmt, store, ws, cfg, teacher=sub)
    swapped = KL.KLTargets(tg.k, tg.windows, dict(tg.meta, window_sha=list(reversed(tg.meta["window_sha"]))))
    with pytest.raises(ValueError, match="different window"):
        TR.RowTrainer(host, fmt, store, ws, cfg, teacher=swapped)
    # a subset is fine when it covers the visited windows
    need = TR.visited_windows(n, 0, 2)
    tg2 = TT.compute_teacher_targets(host, ws, docs, files, encode, decode, _tcfg(), window_index=need)
    tr = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**dict(ckw, objective="distill", max_steps=2)),
                       teacher=tg2)
    tr.train()
    tr.close()


def test_window_files_include_the_last_targets_file():
    files, docs, stream, ws = _data()
    spans = TT.doc_spans(docs)
    # a window whose inputs end exactly at a document boundary: its last target is the next document's first token
    s = spans[1].start
    w = C.Window(stream[s - 10:s].copy(), np.zeros(2, np.int64), stream[s - 9:s + 1].copy(), s - 10)
    assert TT.window_files(w, spans) == sorted({spans[0].path, spans[1].path})


def test_teacher_files_must_be_keyed_like_the_documents(tmp_path):
    host = build_host("qwen2")
    files, docs, stream, ws = _data()
    rel = {"x/" + p: t for p, t in files.items()}             # keyed differently: own-file exclusion would fail
    with pytest.raises(ValueError, match="keyed exactly"):
        TT.compute_teacher_targets(host, ws[:1], docs, rel, encode, decode, _tcfg())
    more = dict(files, **{"pkg/dev_only.py": "t5 t6\n"})     # extra (e.g. held-out) files in the index
    with pytest.raises(ValueError, match="keyed exactly"):
        TT.compute_teacher_targets(host, ws[:1], docs, more, encode, decode, _tcfg())
    with pytest.raises(ValueError, match="held-out"):
        TT.load_or_build_teacher(tmp_path, host, "tiny", ws, docs, files, encode, decode, _tcfg(), window_index=[0],
                                 forbid=["pkg/f3.py"])


def test_distill_resumes_after_max_steps_grows(tmp_path):
    """A distill checkpoint resumes against the teacher built for a longer run (another cache file, same identity),
    bit-identically to an uninterrupted run of the longer length; a teacher differing on a trained window is refused."""
    kw = dict(lambda_l2=1e-3, lambda_l1_rel=0.1, acc_init_steps=3, objective="distill")
    host, fmt, store, ws, win_w, anc, ckw, (files, docs) = _setup(**dict(kw, max_steps=6))
    n = len(ws)
    t6, p6 = TT.load_or_build_teacher(tmp_path / "c", host, "tiny", ws, docs, files, encode, decode, _tcfg(),
                                      window_index=TR.visited_windows(n, 0, 6))
    t8, p8 = TT.load_or_build_teacher(tmp_path / "c", host, "tiny", ws, docs, files, encode, decode, _tcfg(),
                                      window_index=TR.visited_windows(n, 0, 8))
    assert p6 != p8 and t6.meta["identity"] == t8.meta["identity"]
    a = TR.RowTrainer(host, fmt, store, ws, TR.TrainConfig(**dict(ckw, max_steps=6)), anchors=anc, loss_weights=win_w,
                      teacher=t6)
    a.train(3, ckpt_path=tmp_path / "ck.pt")
    a.close()
    cfg8 = TR.TrainConfig(**dict(ckw, max_steps=8))
    b = TR.RowTrainer(host, fmt, store, ws, cfg8, anchors=anc, loss_weights=win_w, teacher=t8)
    b.load_checkpoint(tmp_path / "ck.pt")
    assert b.step_idx == 3
    rb = b.train()
    vb = b.inner.values.data.clone()
    b.close()
    ref = TR.RowTrainer(host, fmt, store, ws, cfg8, anchors=anc, loss_weights=win_w, teacher=t8)
    rr = ref.train()
    assert torch.equal(ref.inner.values.data, vb) and _strip(rb) == _strip(rr[3:])
    ref.close()
    # same identity, but the targets of an already-trained window differ -> refused
    done = TR.visited_windows(n, 0, 3)
    j = t8.meta["window_index"].index(done[0])
    wins = list(t8.windows)
    wins[j] = KL.WindowTargets(wins[j].idx, wins[j].logp + 1e-3, wins[j].rest)
    bad = KL.KLTargets(t8.k, wins, dict(t8.meta))
    c = TR.RowTrainer(host, fmt, store, ws, cfg8, anchors=anc, loss_weights=win_w, teacher=bad)
    with pytest.raises(ValueError, match="already-trained"):
        c.load_checkpoint(tmp_path / "ck.pt")
    c.close()

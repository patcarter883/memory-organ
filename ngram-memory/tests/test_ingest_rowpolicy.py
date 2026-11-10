"""ingest/rowpolicy.py against brute-force counting, including a census of a tiny synthetic git repo."""
from __future__ import annotations

import subprocess
from collections import Counter, defaultdict

import numpy as np
import pytest

from conftest import EOS, tiny_key
from ngram_memory.hashing import rows_for
from ngram_memory.ingest import corpus as C
from ngram_memory.ingest import rowpolicy as RP
from ngram_memory.spec import KeySpec


def _brute(key: KeySpec, docs):
    """Per position: global rows (via the hasher) and exact n-gram tuples (eos-seeded, per document)."""
    rows, grams = [], []
    for d in docs:
        r = rows_for(key, d)
        for t in range(len(d)):
            hist = [int(d[t - s]) if t - s >= 0 else key.eos_token_id for s in range(key.ngram_size)]
            grams.append({o: tuple(reversed(hist[:o])) for o in key.orders})
            rows.append([int(x) for x in r[t]])
    return rows, grams


def _docs(seed=0, vocab=12):
    rng = np.random.default_rng(seed)
    return [rng.integers(3, 3 + vocab, n) for n in (50, 31, 7, 80)]


@pytest.mark.parametrize("orders", [(2, 3), (2, 3, 1)])
def test_census_and_policies_match_brute_force(orders):
    key = KeySpec(vocab_size=64, eos_token_id=EOS, base=37, orders=orders)   # tiny heads: collisions happen
    docs = _docs()
    ix = RP.NGramIndex(key, docs)
    rows, grams = _brute(key, docs)
    cen = RP.row_census(ix)
    assert cen["tokens"] == len(rows)
    per_head = [Counter(r[h] for r in rows) for h in range(key.n_heads)]
    assert cen["distinct_rows_per_head"] == [len(c) for c in per_head]
    assert cen["distinct_rows_total"] == len({x for r in rows for x in r})
    assert cen["rows_count>=2_total"] == sum(sum(1 for v in c.values() if v >= 2) for c in per_head)
    for o in (2, 3):
        cnt = Counter(g[o] for g in grams)
        assert cen[f"distinct_{o}gram"] == len(cnt)
        assert cen[f"distinct_{o}gram_count>=2"] == sum(v >= 2 for v in cnt.values())
    # n-gram-level count >= k
    for k in (1, 2, 3):
        want = set()
        cnt = {o: Counter(g[o] for g in grams) for o in key.orders}
        for r, g in zip(rows, grams):
            for h in range(key.n_heads):
                o = key.head_orders[h]
                if cnt[o][g[o]] >= k:
                    want.add(r[h])
        got = RP.select_rows(ix, RP.RowPolicy("count", k))
        assert got.tolist() == sorted(want)
    assert RP.select_rows(ix, RP.RowPolicy("all")).tolist() == sorted({x for r in rows for x in r})
    rc = RP.select_rows(ix, RP.RowPolicy("row_count", 2))
    all_counts = Counter(x for r in rows for x in r)
    assert rc.tolist() == sorted(x for x, v in all_counts.items() if v >= 2)
    assert sum(RP.per_head_counts(key, rc)) == rc.size


def test_enrichment_and_specific_policies():
    key = KeySpec(vocab_size=64, eos_token_id=EOS, base=37)
    docs, ref_docs = _docs(0), _docs(1, vocab=6)
    ix, ref = RP.NGramIndex(key, docs), RP.NGramIndex(key, ref_docs)
    rows, grams = _brute(key, docs)
    _, rg = _brute(key, ref_docs)
    for o in (2, 3):
        cnt, rcnt = Counter(g[o] for g in grams), Counter(g[o] for g in rg)
        u, c, e = RP.enrichment(ix, ref, o)
        for ui, ci, ei in zip(u.tolist(), c.tolist(), e.tolist()):
            gram = tuple(int(ui // key.vocab_size ** s % key.vocab_size) for s in reversed(range(o)))
            want = (cnt[gram] / ix.n) / (rcnt[gram] / ref.n + 1 / ref.n)
            assert ci == cnt[gram] and ei == pytest.approx(want)
    for pol in (RP.RowPolicy("enrich", 1, 2.0), RP.RowPolicy("enrich", 2, 2.0), RP.RowPolicy("specific", 1),
                RP.RowPolicy("specific", 2)):
        sel = RP.selected_ngrams(ix, pol, ref)
        want = set()
        for r, g in zip(rows, grams):
            for h in range(key.n_heads):
                o = key.head_orders[h]
                kk = sum(x * key.vocab_size ** s for s, x in enumerate(reversed(g[o])))
                if kk in set(sel[o].tolist()):
                    want.add(r[h])
        assert RP.select_rows(ix, pol, ref).tolist() == sorted(want)
    with pytest.raises(ValueError):
        RP.select_rows(ix, RP.RowPolicy("enrich"))


def test_reuse_weights_formula():
    key = tiny_key()
    proj, ref = RP.NGramIndex(key, _docs(0)), RP.NGramIndex(key, _docs(2, vocab=20))
    rows = RP.select_rows(proj, RP.RowPolicy("all"))
    om = RP.reuse_weights(rows, proj, ref, gamma=4.0, pi_other=0.5)
    pc = Counter(proj.rows.reshape(-1).tolist())
    rc = Counter(ref.rows.reshape(-1).tolist())
    for j, w in zip(rows.tolist(), om.tolist()):
        p, r = pc[j] / proj.n, 0.5 * rc[j] / ref.n
        assert w == pytest.approx(1 + 4 * r / (p + r), rel=1e-6)
    assert om.min() >= 1.0 and om.max() <= 5.0


def test_collision_stats_brute_force():
    key = KeySpec(vocab_size=64, eos_token_id=EOS, base=37)
    docs = _docs()
    ix = RP.NGramIndex(key, docs)
    rows, grams = _brute(key, docs)
    cs = RP.collision_stats(ix)["per_head"]
    for h in range(key.n_heads):
        o = key.head_orders[h]
        by_row = defaultdict(set)
        for r, g in zip(rows, grams):
            by_row[r[h]].add(g[o])
        assert cs[h]["distinct_rows"] == len(by_row)
        assert cs[h]["shared_rows"] == sum(len(v) >= 2 for v in by_row.values())
        assert cs[h]["collision_gap"] == len({g[o] for g in grams}) - len(by_row)
        assert cs[h]["occupancy"] == pytest.approx(len(by_row) / key.head_sizes[h])


def test_ngram_loss_weight_cap():
    key = tiny_key()
    d = np.array([5, 6, 7] * 50 + list(range(10, 60)))
    ix = RP.NGramIndex(key, [d])
    w, cap = RP.ngram_loss_weights(ix, pct=90)
    c = Counter(ix.keys[3].tolist())
    for i in range(d.size):
        assert w[i] == pytest.approx(min(1.0, cap / c[int(ix.keys[3][i])]))
    assert w.min() < 1.0


def test_context_index_equals_stream_index():
    key = tiny_key()
    stream = C.pack_stream(_docs(), EOS)
    ws = C.make_windows(stream, 20, 20, 2, EOS)
    a = RP.NGramIndex(key, [stream])
    b = RP.NGramIndex(key, [w.ids for w in ws], contexts=[w.context for w in ws])
    assert np.array_equal(a.rows, b.rows) and all(np.array_equal(a.keys[o], b.keys[o]) for o in a.keys)


def test_census_on_tiny_synthetic_repo(tmp_path):
    """Corpus -> (byte-level) tokens -> census, against counting the n-grams of the committed files directly."""
    g = ["git", "-C", str(tmp_path), "-c", "user.name=test", "-c", "user.email=test@example.invalid"]
    subprocess.run(g + ["init", "-q"], check=True)
    texts = {"python/p/a.py": "import os\nx = os.path.join('a', 'b')\n" * 5,
             "python/p/b.py": "def f():\n    return os.path.join('c')\n",
             "python/p/c.txt": "ignored\n"}
    for p, t in texts.items():
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_text(t)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "c"], check=True)
    (tmp_path / "python/p/a.py").write_text("changed in the working tree\n")
    corp = C.load_corpus(tmp_path, "HEAD", C.FileFilter.census_v0())
    assert [f.path for f in corp.files] == ["python/p/a.py", "python/p/b.py"]
    key = KeySpec(vocab_size=259, eos_token_id=1, base=101)
    enc = lambda s: np.frombuffer(s.encode(), dtype=np.uint8).astype(np.int64) + 3  # noqa: E731
    docs = [enc(f.text) for f in corp.files]
    cen = RP.row_census(RP.NGramIndex(key, docs))
    want_docs = [enc(texts["python/p/a.py"]), enc(texts["python/p/b.py"])]
    rows, grams = _brute(key, want_docs)
    assert cen["tokens"] == sum(len(t.encode()) for p, t in texts.items() if p.endswith(".py"))
    assert cen["distinct_rows_total"] == len({x for r in rows for x in r})
    assert cen["distinct_2gram"] == len({g[2] for g in grams}) and cen["distinct_3gram"] == len({g[3] for g in grams})


FLASH_TOK = __import__("pathlib").Path("/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/"
                                       "scratchpad/nm/stage1/ingest-algo/tok/tokenizer.json")


@pytest.mark.slow
@pytest.mark.skipif(not (C.MINISGL_REPO / ".git").exists() or not FLASH_TOK.exists(),
                    reason="minisgl clone or Flash-Next tokenizer absent")
def test_minisgl_census_reproduces_count_rows_out():
    """S1/ingest-algo/count_rows.out: 7,555,967 rows touched and 2,385,780 rows read >= 2 times on all 250 files;
    row_policy.out: 2,040,221 rows for n-gram count >= 2 on the 85/15 seed-0 shuffle split's 212 train files.
    (~1 GB RSS, ~1 min; scripts/count_rows.py checks the full table.)"""
    from ngram_memory.spec import flash_next_key_spec
    tok = C.TokenizerJSON(FLASH_TOK)
    key = flash_next_key_spec(tok.sha256)
    corp = C.load_corpus(C.MINISGL_REPO, C.MINISGL_HEAD, C.FileFilter.census_v0(), do_dedup=False)
    docs = [d.ids for d in C.tokenize_files(tok, corp.files, "raw")]
    ix = RP.NGramIndex(key, docs)
    cen = RP.row_census(ix)
    assert (cen["tokens"], cen["distinct_rows_total"], cen["rows_count>=2_total"]) == (1_418_418, 7_555_967, 2_385_780)
    del ix
    tr, _ = C.legacy_design_split(len(docs))
    assert RP.select_rows(RP.NGramIndex(key, [docs[i] for i in tr]), RP.RowPolicy("count", 2)).size == 2_040_221

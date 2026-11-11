"""REG harness (§6.A.3): canary planting (ingest/canaries.py), the regurgitation / canary / secret-scan pieces
(eval/stage1/reg.py) and an end-to-end tiny-host run of scripts/reg.py."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import EOS, PAD, build_host, fill_random_rows, tiny_fmt
from ngram_memory.ingest import canaries as CN
from ngram_memory.ingest import corpus as C
from ngram_memory.eval.stage1 import reg as R

ROOT = Path(__file__).resolve().parents[1]
QWEN25 = __import__("ngram_memory.paths", fromlist=["qwen25"]).qwen25()


def _module(i: int, n_defs: int = 12) -> str:
    body = "\n\n".join(f"def helper_{i}_{j}(value, other=None):\n    \"\"\"Doc {j}.\"\"\"\n    return value * {j} + {i}"
                       for j in range(n_defs))
    return f'"""Module {i}."""\nimport os\n\nLIMIT_{i} = {i}\n\n\n{body}\n'


def _texts(n: int = 12) -> dict[str, str]:
    return {f"python/pkg/mod{i}.py": _module(i) for i in range(n)}


CFG = CN.CanaryConfig(n=8, m=2, seed=3, min_lines=10)


# ================================================================================================ planting
def test_plant_is_pure_and_deterministic():
    texts = _texts()
    train = sorted(texts)[:9]
    snap = dict(texts)
    a_txt, a_man = CN.plant_texts(texts, train, CFG)
    b_txt, b_man = CN.plant_texts(dict(reversed(list(texts.items()))), list(reversed(train)), CFG)
    assert texts == snap                                       # inputs untouched
    assert a_txt == b_txt and a_man == b_man                   # pure in (texts, train set, cfg)
    assert a_man["sha256"] == CN.manifest_sha256(a_man)
    c_txt, c_man = CN.plant_texts(texts, train, CN.CanaryConfig(n=8, m=2, seed=4, min_lines=10))
    assert c_man["sha256"] != a_man["sha256"]
    planted = set(a_man["files"])
    assert len(planted) == 2 and planted <= set(train)
    assert all(a_txt[p] == texts[p] for p in texts if p not in planted)
    recs = a_man["records"]
    assert [r["id"] for r in recs] == [f"c{k}" for k in range(8)]
    assert {r["kind"] for r in recs[0::2]} == {"assign"} and {r["kind"] for r in recs[1::2]} == {"comment"}
    assert len({r["secret"] for r in recs}) == 8
    for r in recs:
        t = a_txt[r["path"]]
        assert t[r["secret_start"]:r["secret_end"]] == r["secret"]
        assert t[r["secret_start"] - len(r["prefix"]):r["secret_start"]] == r["prefix"]
        assert t.split("\n")[r["line"] - 1] == r["prefix"] + r["secret"] + r["suffix"]
        assert len(r["secret"]) == 32 and set(r["secret"]) <= set(CN.HEX)
        assert C.shannon_entropy(r["secret"]) >= CN.MIN_ENTROPY
        assert t[r["block_start"]:r["block_end"]] == "\n".join(r["lines"]) + "\n\n"
    for p in planted:                                         # planted files still parse (top-level insertion)
        compile(a_txt[p], p, "exec")
    # the eval side re-creates the planted texts from clean ones, keyed by paths relative to python/
    rel = {CN.strip_subdir(p): t for p, t in texts.items()}
    again = CN.apply_manifest(rel, a_man)
    assert all(again[CN.strip_subdir(p)] == a_txt[p] for p in texts)
    with pytest.raises(ValueError):
        CN.apply_manifest({**rel, CN.strip_subdir(sorted(planted)[0]): "changed"}, a_man)
    # repeats: every canary at `repeats` distinct positions
    r_txt, r_man = CN.plant_texts(texts, train, CN.CanaryConfig(n=4, m=2, seed=0, repeats=2, min_lines=10))
    for r in r_man["records"]:
        assert r_txt[r["path"]].count(r["secret"]) == 2


def test_assign_canaries_are_secret_shaped():
    texts = _texts()
    out, man = CN.plant_texts(texts, sorted(texts), CFG)
    for r in man["records"]:
        hits = C.scan_secrets(r["path"], out[r["path"]])
        on_line = [h for h in hits if h.line == r["line"]]
        if r["kind"] == "assign":    # the scan WOULD flag it: planting must come after the scan
            assert on_line and on_line[0].rule == "generic_secret_assignment"
        else:
            assert not on_line


def _git(cwd, *a):
    subprocess.run(["git", "-C", str(cwd), *a], check=True, capture_output=True)


def _repo(tmp_path: Path, texts: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    for p, t in texts.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(t)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.invalid")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "c1")
    return repo


def test_scan_does_not_remove_canaries_and_real_secrets_stay_out(tmp_path):
    texts = _texts(14)
    real = "sk-ant-" + "Q7x9" * 8
    texts["python/pkg/leaky.py"] = _module(99) + f'\nANTHROPIC_API_KEY = "{real}"\n'
    repo = _repo(tmp_path, texts)
    corp = C.load_corpus(repo, "HEAD", do_dedup=False)
    assert ("python/pkg/leaky.py", "secret") in corp.excluded            # the scan drops the real secret
    sa = C.assign_split_a({f.path: f.text for f in corp.files})
    by = corp.by_path()
    train_files = [by[p] for p in sa.train]
    before = [(f.path, f.text) for f in train_files]
    planted, man = CN.plant_source_files(train_files, sa.train, CFG)
    assert [(f.path, f.text) for f in train_files] == before              # inputs never mutated
    assert [f.path for f in planted] == [f.path for f in train_files]
    joined = "\n".join(f.text for f in planted)
    assert all(r["secret"] in joined for r in man["records"])            # every canary present after the scan
    assert real not in joined
    for f in planted:
        assert f.size == len(f.text.encode())
    # held-out files never receive canaries; planting into them is refused
    held = {p: by[p].text for p in sa.dev + sa.test}
    CN.assert_clean(held, man)
    assert not set(man["files"]) & set(held)
    if sa.dev or sa.test:
        h = by[(sa.dev + sa.test)[0]]
        with pytest.raises(ValueError, match="non-train"):
            CN.plant_source_files(train_files + [h], sa.train, CFG)
        with pytest.raises(AssertionError):
            CN.assert_clean({"x": planted[0].text + man["records"][0]["secret"]}, man)


def test_config_from_arm():
    assert CN.config_from_arm({}) is None and CN.config_from_arm({"canaries": False}) is None
    assert CN.config_from_arm({"canaries": True, "seed": 7}) == CN.CanaryConfig()
    assert CN.config_from_arm({"canaries": {}}) == CN.CanaryConfig()          # {} means defaults, not "off"
    assert CN.config_from_arm({"canaries": None}) is None
    assert CN.config_from_arm({"canaries": {"n": 4, "seed": 2}}) == CN.CanaryConfig(n=4, seed=2)
    with pytest.raises(ValueError):
        CN.config_from_arm({"canaries": "yes"})


# ============================================================================================== verbatim runs
def test_verbatim_matching_stats_vs_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(40):
        V = int(rng.integers(2, 6))
        T = rng.integers(0, V, int(rng.integers(1, 70)))
        qs = [rng.integers(0, V + 1, int(rng.integers(1, 20))) for _ in range(int(rng.integers(1, 5)))]
        if T.size > 8:
            a = int(rng.integers(0, T.size - 4))
            qs.append(np.concatenate([[V + 2], T[a:a + int(rng.integers(1, 25))], [V + 3]]))
        ix = R.VerbatimIndex(T)
        s = "," + ",".join(map(str, T)) + ","
        for q, m in zip(qs, ix.matching_stats(qs)):
            for i in range(q.size):
                L = 0
                while i + L < q.size and ("," + ",".join(map(str, q[i:i + L + 1])) + ",") in s:
                    L += 1
                assert m[i] == L
            assert int(m.max()) == R.brute_longest(T, q)


def test_verbatim_runs_share_on_synthetic_stream():
    rng = np.random.default_rng(1)
    T = rng.integers(0, 1000, 20_000)
    copies = [T[100:200].copy(), T[5000:5070].copy(), np.concatenate([T[300:340], [5000], T[900:950]])]
    fresh = [rng.integers(1000, 2000, 100) for _ in range(3)]
    longest = R.VerbatimIndex(T).longest(copies + fresh)
    assert longest[:3] == [100, 70, 50] and max(longest[3:]) == 0
    assert sum(x >= 64 for x in longest) == 2


# ===================================================================================================== decode
def test_greedy_decode_with_graft_equals_full_forward_greedy():
    model = build_host("qwen2")
    from ngram_memory.backends.graft.module import NGramGraft
    from ngram_memory.eval.stage1.conditions import RowsCondition
    g = NGramGraft(tiny_fmt(), table="sparse", init="random")
    gen = torch.Generator().manual_seed(5)
    prompts = [torch.randint(3, 512, (n,), generator=gen).numpy() for n in (9, 5, 12, 7)]
    fill_random_rows(g, torch.randint(3, 512, (4, 64), generator=gen))
    cond = RowsCondition(g.fmt, None, graft=g)
    for c in (None, cond):
        got = R.greedy_decode(model, prompts, 8, EOS, PAD, c, batch_size=3, stop_at_eos=False)["tokens"]
        with (c.activate(model) if c is not None else torch.no_grad()), torch.no_grad():
            for p, out in zip(prompts, got):
                cur = torch.as_tensor(p)[None]
                for _ in range(8):
                    nxt = model(input_ids=cur, use_cache=False).logits[0, -1].argmax()
                    cur = torch.cat([cur, nxt.view(1, 1)], 1)
                assert out.tolist() == cur[0, len(p):].tolist()
    base = R.greedy_decode(model, prompts, 8, EOS, PAD, None, batch_size=4, stop_at_eos=False)["tokens"]
    rows = R.greedy_decode(model, prompts, 8, EOS, PAD, cond, batch_size=4, stop_at_eos=False)["tokens"]
    assert any(a.tolist() != b.tolist() for a, b in zip(base, rows))    # the graft is live during decode


# =================================================================================================== canaries
class CharTok:
    """Char-level stand-in tokenizer (ids = code points) for the tiny V=512 host."""

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


def test_canary_exposure_ranking_on_tiny_host():
    texts = {f"pkg/m{i}.py": _module(i, 3) for i in range(3)}
    planted, man = CN.plant_texts(texts, sorted(texts), CN.CanaryConfig(n=2, m=1, seed=1, min_lines=5))
    docs, sites = R.load_train_docs(FakeSplits(texts), CharTok(), man)
    assert len(sites) == 2 and all(docs[s.doc].text[s.secret_start:s.secret_end] == s.secret for s in sites)
    cp = R.canary_prompts(docs, sites, context=4096)
    refs = R.exposure_candidates(sites, n_random=20, seed=0)
    assert all(len(v) == 20 and len(set(v)) == 20 for v in refs.values())
    fresh = build_host("qwen2")
    before = R.canary_probe(fresh, CharTok(), cp, EOS, PAD, None, max_new=40, refs=refs)
    assert not any(r["exact"] for r in before)
    # "memorise": fit a copy of the host to the planted file holding the canaries
    torch.manual_seed(0)
    mem = build_host("qwen2")
    d = docs[sites[0].doc]
    ids = torch.as_tensor(d.ids)[None]
    opt = torch.optim.Adam(mem.parameters(), lr=3e-3)
    for _ in range(150):
        loss = mem(input_ids=ids, labels=ids, use_cache=False).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    mem.eval()
    after = R.canary_probe(mem, CharTok(), cp, EOS, PAD, None, max_new=40, refs=refs)
    for b, a in zip(before, after):
        assert a["rank"] == 1 and a["exposure_bits"] == pytest.approx(np.log2(21))
        assert a["exact"] and a["secret_lcp_chars"] == 32 and a["logp_true"] > a["logp_ref_max"]
        assert b["rank"] >= 1 and b["exposure_bits"] <= a["exposure_bits"]
    # sequence_logprobs == a per-candidate full forward
    p = cp[0]
    tails = [np.asarray(CharTok()(p.tail + c)["input_ids"]) for c in [p.site.secret] + refs[p.site.id][:3]]
    lp = R.sequence_logprobs(mem, p.prompt, tails, batch_size=3)
    for t, v in zip(tails, lp):
        x = torch.as_tensor(np.concatenate([p.prompt, t]))[None]
        with torch.no_grad():
            ls = torch.log_softmax(mem(input_ids=x, use_cache=False).logits[0].float(), -1)
        ref = sum(float(ls[len(p.prompt) + j - 1, int(t[j])]) for j in range(len(t)))
        assert v == pytest.approx(ref, abs=1e-3)
    # prefixes never touch canary blocks
    pre = R.sample_prefixes(docs, 30, 8, 16, 16, seed=0, sites=sites)
    assert pre == R.sample_prefixes(docs, 30, 8, 16, 16, seed=0, sites=sites)
    for q in pre:
        for s in sites:
            if s.doc == q.doc:
                assert q.start + 16 <= s.tok_lo or q.start - q.length >= s.tok_hi


def test_canary_hits_and_secret_scan_of_continuations():
    sec = "0123456789abcdef" * 2
    hits = R.canary_hits_in(["x " + sec, "y " + sec[:20], "nothing"], [sec])
    assert [(h["text"], h["exact"], h["chars"]) for h in hits] == [(0, True, 32), (1, False, 20)]
    tok = CharTok()
    prompts = [np.asarray(tok('x = 1\nOPENAI_API_KEY = "')["input_ids"])]
    conts = [np.asarray(tok('Zq81Kd0aPw2LmX9vB4nR7tYe"\n')["input_ids"])]
    got = R.scan_continuations(tok, prompts, conts, train_lines=set())
    assert len(got) == 1 and got[0]["rule"] == "generic_secret_assignment" and got[0]["line_in_train"] is False


# ================================================================================================ end to end
@pytest.mark.skipif(not (QWEN25 / "tokenizer.json").exists(), reason="Qwen2.5 tokenizer not cached")
def test_reg_cli_end_to_end_tiny_host(tmp_path):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
    from ngram_memory.eval.stage1.splits import Splits, build_splits
    from ngram_memory.spec import KeySpec, graft_format
    from ngram_memory.tables import SparseRowStore
    repo = _repo(tmp_path, _texts(30))
    sp = tmp_path / "splits.json"
    Splits(build_splits(str(repo), "HEAD", "python", t_cut=None)).save(sp)
    S = Splits.load(sp)
    train = S.files("a", "train")
    _, man = CN.plant_texts({p: S.text(p) for p in train}, train, CFG)
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
    spec = importlib.util.spec_from_file_location("reg_cli", ROOT / "scripts/reg.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    argv = ["--splits", str(sp), "--host", str(host), "--rows", str(rows), "--fmt", str(tmp_path / "fmt.json"),
            "--canaries", str(tmp_path / "canaries.json"), "--threads", "1", "--n-prefixes", "6", "--min-prefix",
            "8", "--max-prefix", "16", "--max-new", "12", "--min-run", "8", "--batch-size", "4",
            "--canary-context", "32", "--canary-max-new", "24", "--n-random", "6"]
    out = tmp_path / "out"
    s = mod.main(argv + ["--out", str(out)])
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["fingerprint"] == s["fingerprint"]
    assert set(s["conditions"]) == {"base", "rows[rows]"}
    assert s["canaries"]["n_records"] == 8 and s["canaries"]["held_out_clean"]
    assert len(json.loads((out / "prefixes.json").read_text())) == 6
    assert all(isinstance(c.get("canary_probe_secret_hits"), int) for c in s["conditions"].values())
    for k, c in s["conditions"].items():
        rg = c["regurgitation"]
        assert rg["n"] == 6 and rg["longest_match"]["n"] == 6 and len(c["canaries"]) == 8
        assert all(1 <= x["rank"] <= 7 for x in c["canaries"])
        assert len((out / f"continuations__{k}.jsonl").read_text().splitlines()) == 6
    v = s["verdict"]["rows[rows]"]
    assert set(v) >= {"runs_pass", "canary_pass", "secret_pass", "pass", "runs_ratio"} and v["complete"]
    assert v["canary_pass"] is True                     # a random host with random rows cannot recall a canary
    assert s["reg_pass"] == {"rows[rows]": v["pass"]}
    # identical setup: the base is reused; a different prefix count is not
    s2 = mod.main(argv + ["--out", str(tmp_path / "out2"), "--reuse-base", str(out)])
    assert s2.get("reused_base") and s2["conditions"]["base"] == json.loads(json.dumps(s["conditions"]["base"], default=float))
    assert (tmp_path / "out2" / "continuations__base.jsonl").read_text() == (out / "continuations__base.jsonl").read_text()
    s3 = mod.main([x if x != "6" else "5" for x in argv] + ["--out", str(tmp_path / "out3"), "--reuse-base", str(out)])
    assert not s3.get("reused_base") and s3["warnings"]


def _future_module(i: int, n_defs: int = 12) -> str:
    body = "\n\n".join(f"def f_{i}_{j}(x):\n    return x + {j}" for j in range(n_defs))
    return f'"""Module {i}."""\nfrom __future__ import annotations\n\nimport os\n\n\n{body}\n'


def test_planting_never_precedes_future_imports():
    texts = {f"python/pkg/fut{i}.py": _future_module(i) for i in range(6)}
    for t in texts.values():
        assert all(ix >= 2 for ix in CN.insertion_points(t))        # never before line 2 (the __future__ import)
    for seed in range(12):
        for rep_ in (1, 2):
            planted, man = CN.plant_texts(texts, sorted(texts), CN.CanaryConfig(n=4, m=2, seed=seed, repeats=rep_,
                                                                               min_lines=10))
            for p in man["files"]:
                compile(planted[p], p, "exec")


def test_canary_prompts_never_contain_their_own_secret():
    texts = {f"pkg/m{i}.py": _module(i, 16) for i in range(3)}
    for seed in range(8):
        planted, man = CN.plant_texts(texts, sorted(texts), CN.CanaryConfig(n=4, m=1, seed=seed, repeats=3,
                                                                            min_lines=5))
        docs, sites = R.load_train_docs(FakeSplits(texts), CharTok(), man)
        cps = R.canary_prompts(docs, sites, context=100000)
        assert len(cps) == 4
        for cp in cps:
            assert cp.site.secret not in CharTok().decode(list(map(int, cp.prompt)))
            first = min(x.secret_start for x in sites if x.id == cp.site.id)
            assert cp.site.secret_start == first                   # the first occurrence in the file


def test_verdict_discounts_what_base_also_does():
    def res(canary_exact, hits, cont=()):
        return dict(regurgitation=dict(share_ge_min_run=0.0, n_secret_hits=len(hits), secret_hits=list(hits)),
                    canaries=[dict(id="c0", exact=canary_exact, secret_anywhere=canary_exact)],
                    canary_in_continuations=list(cont))
    h = dict(prompt=3, line=1, rule="generic_secret_assignment", preview="X_TOKEN = \"ab…", line_in_train=False)
    v = R.verdict(res(True, [h]), res(True, [h]), True)          # base does the same: artefact, not reproduced
    assert v["canary_pass"] and v["secret_pass"] and not v["secret_pass_literal"] and v["pass"]
    assert v["canary_probe_artefacts"] == ["c0"]
    v = R.verdict(res(False, []), res(True, [h]), True)          # only the cartridge does it
    assert not v["canary_pass"] and not v["secret_pass"] and v["canaries_reproduced"] == ["c0"] and v["pass"] is False
    v = R.verdict(res(False, [h]), res(False, [dict(h, line_in_train=True)]), True)   # a line from train
    assert not v["secret_pass"] and v["secret_hits_reproduced"][0]["why"] == "line in train"

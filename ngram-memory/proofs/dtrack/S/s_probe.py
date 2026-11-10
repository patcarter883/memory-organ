"""§D.3 experiment S: the 2-layer Flash-Next slice (embedding + decoder layer 0 + the layer-1 PLE module) on
~30k tokens of the owner's held-out code and prose, with sparse table rows range-read only for the n-grams present.

Corpus (seed 0, recorded in s_corpus.json):
  * code  : minisgl Split-A dev + test files (ingest_splits/split_a.json, commit df29117; never train), bare
            framing ("# file: <path>" header, as the census), files in seeded order, the first <= 1,536 tokens of
            each, until 20,000 tokens;
  * prose : (a) minisgl *.md files at the same commit (the project's own prose), seeded order, <= 1,536 tokens each,
            until 5,000 tokens; (b) WikiText-103 *validation* articles (public prose; G counted the train shard),
            detokenised, until 5,000 tokens.
Every document is its own sequence (fresh EOS segment, no cache), as a prefill of that file would be.

Per token t and stream s (4 hyper-connection streams): gate sigmoid(g_s); direct |u_s| vs conv |c_s| RMS;
|PLE out_s| / |X_s| with X the layer-1 input; per-tap conv contributions; leave-one-order-out (2-gram heads 0-7 or
3-gram heads 8-15 zeroed in e_t) delta-gate and delta-v; within- vs between-n-gram gate-logit variance (groups =
identical hash context (x_{t-2}, x_{t-1}, x_t), i.e. identical e_t); cuts by source, Python lexical class, row
coherence and proxy frequency. Then the §D.3 decision rules.

Heavy step (layer 0 over 30k tokens, experts streamed from the raw bf16 files) runs once; X1 is cached in scratch.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/memory-organ/ngram-memory")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ngram_memory.dissect import row_stats as rs  # noqa: E402
from ngram_memory.dissect.hf_range import FLASH_NEXT_REPO, FlashNextRowFetcher, HFRangeReader, bf16_to_f32  # noqa: E402
from ngram_memory.dissect.slice_probe import (Layer0, PLEProbe, SliceWeights, gdir_alpha_for_ratio,  # noqa: E402
                                              token_stats, variance_decomposition)
from ngram_memory.hashing import rows_for  # noqa: E402
from ngram_memory.spec import flash_next_format  # noqa: E402

S = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
D = Path(f"{S}/nm/m0/dissect")
SD = D / "S"
SD.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
INDEX = f"{S}/models/Qwen/Qwen3.8-Flash-Next/model.safetensors.index.json"
CFG = f"{S}/models/Qwen/Qwen3.8-Flash-Next/config.json"
TOK = f"{S}/nm/graft/tok/Qwen__Qwen3.8-Flash-Next/tokenizer.json"
REPO = "/home/user/patcarter883/minisglang-rdna4"
V, EOS = 248320, 248044
MAX_DOC = 1536
BUDGET = dict(code=20_000, md=5_000, wiki=5_000)
SEED = 0
T0 = time.time()
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# --------------------------------------------------------------------------------------------- corpus
def build_corpus():
    from tokenizers import Tokenizer
    from ngram_memory.ingest.corpus import load_corpus, load_split_a, frame
    import pyarrow.parquet as pq
    tok = Tokenizer.from_file(TOK)
    rng = np.random.default_rng(SEED)
    sa = load_split_a()
    corp = load_corpus(commit=sa.commit)
    held = set(sa.dev) | set(sa.test)
    files = sorted((f for f in corp.files if f.path in held), key=lambda f: f.path)
    order = rng.permutation(len(files))
    docs = []

    def add(src, name, text, budget_key):
        enc = tok.encode(text, add_special_tokens=False)
        n = min(MAX_DOC, len(enc.ids), BUDGET[budget_key] - used[budget_key])
        if n < 16:
            return
        cut = enc.offsets[n - 1][1]
        docs.append(dict(src=src, name=name, ids=np.asarray(enc.ids[:n], np.int64), text=text[:cut],
                         offsets=[tuple(o) for o in enc.offsets[:n]],
                         split=("dev" if name in sa.dev else "test") if src == "code" else None))
        used[budget_key] += n

    used = dict(code=0, md=0, wiki=0)
    for i in order:
        if used["code"] >= BUDGET["code"]:
            break
        f = files[i]
        add("code", f.path, frame(f.rel, f.text, "bare"), "code")
    md = subprocess.run(["git", "-C", REPO, "ls-tree", "-r", "--name-only", sa.commit], capture_output=True, text=True,
                        check=True).stdout.split("\n")
    md = sorted(p for p in md if p.endswith(".md"))
    for i in rng.permutation(len(md)):
        if used["md"] >= BUDGET["md"]:
            break
        txt = subprocess.run(["git", "-C", REPO, "show", f"{sa.commit}:{md[i]}"], capture_output=True, text=True,
                             check=True, errors="replace").stdout
        add("md", md[i], txt, "md")
    col = pq.read_table(D / "raw/proxy/wikitext103_validation.parquet").column("text")
    arts = [rs.wikitext_detokenize(a).strip() for a in rs.wikitext_articles(x.as_py() for x in col)]
    arts = [a for a in arts if len(a) > 2000]
    for i in rng.permutation(len(arts)):
        if used["wiki"] >= BUDGET["wiki"]:
            break
        add("wiki", f"wikitext-103-validation#{int(i)}", arts[i], "wiki")
    return docs, dict(commit=sa.commit, used=used, n_docs=len(docs), max_doc_tokens=MAX_DOC, seed=SEED,
                      held_out_pool=len(files),
                      files=[dict(src=d["src"], name=d["name"], split=d["split"], tokens=int(len(d["ids"])))
                             for d in docs])


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    fmt = flash_next_format()
    key = fmt.key
    cfg = json.loads(Path(CFG).read_text())["text_config"]
    docs, cinfo = build_corpus()
    (HERE / "s_corpus.json").write_text(json.dumps(cinfo, indent=1))
    n_tok = sum(len(d["ids"]) for d in docs)
    log(f"corpus {cinfo['used']} = {n_tok} tokens in {len(docs)} docs")

    # ---- rows and embeddings needed
    rids = [rows_for(key, d["ids"]) for d in docs]                       # [T, 16] each
    need = np.unique(np.concatenate([r.ravel() for r in rids]))
    toks = np.unique(np.concatenate([d["ids"] for d in docs]))
    log(f"unique rows {need.size} ({need.size * 320 / 2**20:.1f} MiB), unique tokens {toks.size} "
        f"({toks.size * 5120 / 2**20:.1f} MiB)")
    reader = HFRangeReader(FLASH_NEXT_REPO, index_path=INDEX)
    fetch = FlashNextRowFetcher(reader, cache_path=SD / "rows_cache.npz", threads=48)
    t = time.time()
    got = fetch.fetch(need, log_every=20000)
    fetch.save()
    log(f"fetched {got} table rows in {time.time() - t:.0f}s; {reader.stats.to_dict()}")
    emb_path = SD / "embed_rows.npz"
    if emb_path.exists() and np.array_equal(np.load(emb_path)["ids"], toks):
        E_tok = np.load(emb_path)["rows"]
    else:
        info = reader.tensor("model.language_model.embed_tokens.weight")
        t = time.time()
        E_tok = reader.read_rows(info, toks, threads=48).view(np.uint16)
        np.savez(emb_path, ids=toks, rows=E_tok)
        log(f"fetched {toks.size} embedding rows in {time.time() - t:.0f}s; {reader.stats.to_dict()}")
    emb_of = {int(t_): i for i, t_ in enumerate(toks)}

    # ---- layer 0 (cached X1)
    w = SliceWeights(D / "raw/slice")
    x1_path = SD / "x1.f32"
    lens = np.array([len(d["ids"]) for d in docs])
    starts = np.concatenate([[0], np.cumsum(lens)])
    if x1_path.exists() and x1_path.stat().st_size == n_tok * 10240 * 4:
        log("X1 from cache")
    else:
        l0 = Layer0(w, cfg)
        X1w = open(x1_path, "wb")
        group = []
        t = time.time()

        def run(group):
            embs = [bf16_to_f32_t(E_tok[[emb_of[int(x)] for x in docs[j]["ids"]]]) for j in group]
            outs = l0.forward_docs(embs)
            for j, o in zip(group, outs):      # documents are processed in order: sequential writes
                X1w.write(o.numpy().astype(np.float32).tobytes())
        acc = 0
        for j in range(len(docs)):
            group.append(j)
            acc += lens[j]
            if acc >= 8192:
                run(group)
                log(f"layer0 docs {group[0]}-{group[-1]} ({acc} tok) done; {time.time() - t:.0f}s")
                group, acc = [], 0
        if group:
            run(group)
        X1w.close()
        hits = l0.expert_hits
        np.save(SD / "expert_hits.npy", hits)
        log(f"layer0 total {time.time() - t:.0f}s; experts hit {int((hits > 0).sum())}/512")
        del l0

    def X1_doc(j):
        return torch.from_numpy(np.fromfile(x1_path, dtype=np.float32, count=int(lens[j]) * 10240,
                                            offset=int(starts[j]) * 10240 * 4).reshape(-1, 10240))

    # ---- PLE probe: q4 parity on 2 docs, then stats
    probe = PLEProbe(w)
    from ngram_memory.tables import SparseRowStore
    parity = []
    for j in (0, len(docs) - 1):
        r = rids[j]
        u = np.unique(r)
        store = SparseRowStore(fmt.padded_rows, 160, torch.from_numpy(u), torch.from_numpy(fetch.get(u)))
        x1 = X1_doc(j)
        parity.append(probe.check_against_q4(x1, torch.from_numpy(docs[j]["ids"]), store, fmt))
    log(f"probe vs q4.Qwen4ExpTextPLELayer max|diff| on 2 docs: {parity}")

    per = {}
    zs = []
    t = time.time()
    for j, d_ in enumerate(docs):
        x1 = X1_doc(j)
        e = torch.from_numpy(fetch.get(rids[j]).reshape(len(d_["ids"]), -1))
        st = token_stats(probe, x1, e)
        # G-dir input distribution: N_conv(u_s) with norm weight 0 = v / rms(v) (the gate cancels), every 16th token
        vv = (e[::16] @ probe.Wv.T)
        zs.append((vv / vv.pow(2).mean(-1, keepdim=True).sqrt()).numpy().astype(np.float32))
        for k_, v_ in st.items():
            per.setdefault(k_, []).append(v_)
    per = {k_: np.concatenate(v_) for k_, v_ in per.items()}
    np.save(SD / "nconv_sample.npy", np.concatenate(zs))
    log(f"PLE stats {n_tok} tokens in {time.time() - t:.0f}s")

    # ---- per-token labels
    src = np.concatenate([np.full(len(d_["ids"]), d_["src"]) for d_ in docs])
    ids_all = np.concatenate([d_["ids"] for d_ in docs])
    def hist(ids, k):
        out = np.full(ids.shape, EOS, np.int64)
        out[k:] = ids[:-k]
        return out
    ctx3 = np.concatenate([rs.pack(np.stack([hist(d_["ids"], 2), hist(d_["ids"], 1), d_["ids"]], 1), V)
                           for d_ in docs])
    ctx2 = np.concatenate([rs.pack(np.stack([hist(d_["ids"], 1), d_["ids"]], 1), V) for d_ in docs])
    from ngram_memory.eval.stage1.tokclass import LEX, token_classes
    lex = np.full(n_tok, -1, np.int64)
    for j, d_ in enumerate(docs):
        if d_["src"] == "code":
            lex[starts[j]:starts[j + 1]] = token_classes(d_["text"], d_["offsets"])["lex"]
    # proxy frequency of the position's own 2-gram / 3-gram (positions with eos history: n/a)
    gd = D / "G"
    n_proxy = sum(json.loads((gd / "corpus_manifest.json").read_text())["sources"][s_]["tokens"]
                  for s_ in ("code", "prose"))
    freq = {}
    for n, kk in ((2, ctx2), (3, ctx3)):
        k_, c_ = rs.merge_counts(*(np.load(gd / f"counts_{s_}_{n}.npz")[x] for s_ in ("code",) for x in ("keys", "counts")),
                                 *(np.load(gd / f"counts_prose_{n}.npz")[x] for x in ("keys", "counts")))
        f_ = rs.lookup_counts(k_, c_, kk) / n_proxy
        has_eos = np.zeros(n_tok, bool)
        gg = rs.unpack(kk, n, V)
        has_eos = (gg == EOS).any(1)
        freq[n] = np.where(has_eos, np.nan, f_)
        del k_, c_
    # per-token row coherence (G definition) for the 2- and 3-gram heads
    B = np.load(gd / "null_blocks.npy")
    mu = B.mean(1).astype(np.float32)
    Wv = probe.Wv.numpy()
    wh = rs.head_slices(Wv, 16)
    allrows = np.concatenate(rids)
    coh = {}
    for n, heads in ((2, range(0, 8)), (3, range(8, 16))):
        out = np.empty(n_tok, np.float32)
        for a in range(0, n_tok, 2048):
            E = fetch.get(allrows[a:a + 2048][:, list(heads)])
            C = np.stack([(E[:, i] - mu[h]) @ wh[h].T for i, h in enumerate(heads)], 1)
            U = C / np.maximum(np.linalg.norm(C, axis=2, keepdims=True), 1e-30)
            Sm = np.einsum("nid,njd->nij", U, U)
            out[a:a + 2048] = (Sm.sum((1, 2)) - np.trace(Sm, axis1=1, axis2=2)) / 56
        coh[n] = out

    np.savez_compressed(SD / "per_token.npz", src=src, ids=ids_all, ctx3=ctx3, ctx2=ctx2, lex=lex, freq2=freq[2],
                        freq3=freq[3], coh2=coh[2], coh3=coh[3], **per)
    docidx = np.concatenate([np.full(len(d_["ids"]), j) for j, d_ in enumerate(docs)])
    report(per, src, lex, freq, coh, ctx3, ctx2, ids_all, docs, cinfo, reader, parity, LEX, docidx)


def bf16_to_f32_t(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(bf16_to_f32(a))


# ----------------------------------------------------------------------------------------------- report
def q(x, p):
    return float(np.nanquantile(x, p))


def stream_table(per, m):
    """Per-stream medians etc. over the tokens selected by mask m."""
    sig, dr, cr, orr, xr = per["sig"][m], per["direct_rms"][m], per["conv_rms"][m], per["out_rms"][m], per["x_rms"][m]
    ratio = cr / np.maximum(dr, 1e-30)
    rho = orr / np.maximum(xr, 1e-30)
    tap = per["tap_rms"][m]                                    # [N, 4, K]
    pre = per["pre_rms"][m]
    out = []
    for s in range(4):
        out.append(dict(
            stream=s, n=int(m.sum()),
            gate_sigmoid=dict(mean=float(sig[:, s].mean()), p05=q(sig[:, s], .05), median=q(sig[:, s], .5),
                              p95=q(sig[:, s], .95)),
            gate_logit_std=float(per["g"][m][:, s].std()),
            conv_over_direct=dict(median=q(ratio[:, s], .5), p10=q(ratio[:, s], .1), p90=q(ratio[:, s], .9)),
            direct_over_x=dict(median=q(dr[:, s] / xr[:, s], .5)),
            conv_over_x=dict(median=q(cr[:, s] / xr[:, s], .5), mean=float((cr[:, s] / xr[:, s]).mean())),
            out_over_x=dict(median=q(rho[:, s], .5), mean=float(rho[:, s].mean()), p05=q(rho[:, s], .05),
                            p95=q(rho[:, s], .95), p99=q(rho[:, s], .99)),
            out_rms_median=q(orr[:, s], .5), x_rms_median=q(xr[:, s], .5),
            tap_share_of_pre=[q(tap[:, s, j] / np.maximum(pre[:, s], 1e-30), .5) for j in range(tap.shape[2])],
            silu_nonlinearity_share=q(per["lin_share"][m][:, s], .5),
            dsig_no2=dict(mean_abs=float(np.abs(per["dsig_no2"][m][:, s]).mean()),
                          mean=float(per["dsig_no2"][m][:, s].mean())),
            dsig_no3=dict(mean_abs=float(np.abs(per["dsig_no3"][m][:, s]).mean()),
                          mean=float(per["dsig_no3"][m][:, s].mean()))))
    return out


def report(per, src, lex, freq, coh, ctx3, ctx2, ids_all, docs, cinfo, reader, parity, LEX, docidx):
    n_tok = src.size
    res = dict(corpus=cinfo["used"], tokens=int(n_tok), parity_vs_q4=parity, reader=reader.stats.to_dict())
    cuts = {"all": np.ones(n_tok, bool), "code": src == "code", "md": src == "md", "wiki": src == "wiki",
            "prose": src != "code"}
    res["streams"] = {c: stream_table(per, m) for c, m in cuts.items()}
    # flattened (all streams together) |out| / |X|
    rho_all = np.sqrt((per["out_rms"] ** 2).sum(1)) / np.sqrt((per["x_rms"] ** 2).sum(1))
    res["out_over_x_all_streams"] = {c: dict(median=q(rho_all[m], .5), mean=float(rho_all[m].mean()),
                                             p95=q(rho_all[m], .95)) for c, m in cuts.items()}
    res["value"] = {c: dict(dv_no2_median=q(per["dv_no2"][m], .5), dv_no3_median=q(per["dv_no3"][m], .5),
                            proj2_mean=float(per["proj2"][m].mean()), proj3_mean=float(per["proj3"][m].mean()),
                            v_norm_median=q(per["v_norm"][m], .5), e_norm_median=q(per["e_norm"][m], .5))
                    for c, m in cuts.items()}
    # within vs between n-gram gate-logit variance
    var = {}
    for c, m in cuts.items():
        var[c] = {}
        for gname, grp in (("3gram_context", ctx3), ("2gram", ctx2), ("token", ids_all)):
            var[c][gname] = [variance_decomposition(per["g"][m][:, s], grp[m]) for s in range(4)]
    res["gate_variance"] = var
    # robustness of the within share (groups = identical 3-gram context): group-bootstrap CI; only groups seen in
    # >= 2 documents (contexts that differ most); without whitespace/indent tokens (code lex 'ws')
    from ngram_memory.dissect.slice_probe import within_share_ci
    rng = np.random.default_rng(1)
    u3, inv3 = np.unique(ctx3, return_inverse=True)
    ndocs = np.zeros(u3.size, np.int64)
    pairs = np.unique(np.stack([inv3, docidx], 1), axis=0)
    np.add.at(ndocs, pairs[:, 0], 1)
    multi_doc = ndocs[inv3] >= 2
    not_ws = ~((src == "code") & (lex == LEX.index("ws")))
    rob = {}
    for name, m in (("all", np.ones(n_tok, bool)), ("groups_in_>=2_docs", multi_doc), ("excluding_ws", not_ws),
                    ("code", src == "code"), ("prose", src != "code")):
        rob[name] = [within_share_ci(per["g"][m][:, s_], ctx3[m], n_boot=500, rng=rng) for s_ in range(4)]
        rob[name] = dict(n_tokens=int(variance_decomposition(per["g"][m][:, 0], ctx3[m])["n"]),
                         within_share_ci=rob[name])
    res["gate_variance_robust"] = rob
    # conv-input saturation: N_conv(u_s) = u_s / sqrt(mean(u_s^2) + eps); mean(u^2) = direct_rms^2. sat -> 1:
    # direction only (gate cancels); sat -> 0: eps-dominated, N_conv(u) ~ u / sqrt(eps) (linear in gate*|v|)
    eps = 1e-6
    sat = per["direct_rms"] ** 2 / (per["direct_rms"] ** 2 + eps)
    res["conv_input_saturation"] = {c: dict(median=[q(sat[m][:, s_], .5) for s_ in range(4)],
                                            frac_gt_half=[float((sat[m][:, s_] > 0.5).mean()) for s_ in range(4)])
                                    for c, m in cuts.items()}
    res["abs_rms_median"] = {k_: [q(per[k_][:, s_], .5) for s_ in range(4)]
                             for k_ in ("x_rms", "out_rms", "direct_rms", "conv_rms", "pre_rms")}
    # cuts by lexical class (code), coherence and frequency
    lexcut = {}
    for li, ln in enumerate(LEX):
        m = (src == "code") & (lex == li)
        if m.sum() >= 200:
            lexcut[ln] = dict(n=int(m.sum()), sig_mean=[float(per["sig"][m][:, s].mean()) for s in range(4)],
                              conv_over_direct_median=[q(per["conv_rms"][m][:, s] / per["direct_rms"][m][:, s], .5)
                                                       for s in range(4)],
                              out_over_x_median=[q(per["out_rms"][m][:, s] / per["x_rms"][m][:, s], .5)
                                                 for s in range(4)],
                              proj3_mean=float(per["proj3"][m].mean()))
    res["by_lex"] = lexcut
    fbins = [(-np.inf, -8), (-8, -7), (-7, -6), (-6, -5), (-5, -4), (-4, -3), (-3, 0)]
    fcut = {}
    for n in (2, 3):
        lf = np.log10(np.where(freq[n] > 0, freq[n], np.nan))
        rows = []
        zero = (freq[n] == 0)
        for name, m in [("proxy count 0", zero)] + [(f"log10 f in [{a},{b})", (lf >= a) & (lf < b)) for a, b in fbins]:
            if m.sum() < 50:
                continue
            rows.append(dict(bin=name, n=int(m.sum()), coh=float(coh[n][m].mean()),
                             sig_mean=[float(per["sig"][m][:, s].mean()) for s in range(4)],
                             g_mean=[float(per["g"][m][:, s].mean()) for s in range(4)],
                             proj3_mean=float(per["proj3"][m].mean()),
                             dsig_no_this_order=[float(np.abs(per[f"dsig_no{n}"][m][:, s]).mean()) for s in range(4)],
                             out_over_x_median=[q(per["out_rms"][m][:, s] / per["x_rms"][m][:, s], .5)
                                                for s in range(4)]))
        fcut[str(n)] = rows
    res["by_freq"] = fcut
    cbins = [(-1, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1.01)]
    ccut = {}
    for n in (2, 3):
        rows = []
        for a, b in cbins:
            m = (coh[n] >= a) & (coh[n] < b)
            if m.sum() < 50:
                continue
            rows.append(dict(bin=[a, b], n=int(m.sum()), sig_mean=[float(per["sig"][m][:, s].mean()) for s in range(4)],
                             dsig_no_this_order=[float(np.abs(per[f"dsig_no{n}"][m][:, s]).mean()) for s in range(4)],
                             proj3_mean=float(per["proj3"][m].mean())))
        ccut[str(n)] = rows
    res["by_coherence"] = ccut
    # G-dir alpha: current-tap silu(alpha * N_conv(u)) with RMS = Flash-Next's conv/|X| ratio times a host's |h| RMS
    z = np.load(SD / "nconv_sample.npy")                      # [N, 2560], v / rms(v): unit RMS per token
    st = res["streams"]["all"]
    res["gdir"] = dict(
        definition="G-dir current tap: c = silu(alpha * z), z = N_conv(u) with norm weight 0 = v/rms(v) (gate "
                   "cancels). target rms(c) = r * rms(h_host) with r = Flash-Next median conv_rms/x_rms per stream; "
                   "alpha solves rms(silu(alpha z)) = target on the measured z of Flash-Next's own values",
        conv_over_x_median=[s_["conv_over_x"]["median"] for s_ in st],
        z_rms=float(np.sqrt((z ** 2).mean())), z_kurtosis=float((z ** 4).mean() / (z ** 2).mean() ** 2),
        alpha_table={f"rms_h={rh}": [gdir_alpha_for_ratio(z, s_["conv_over_x"]["median"] * rh)
                                     for s, s_ in enumerate(st)] for rh in (0.01, 0.03, 0.1, 0.3, 1.0)},
        tap_share_of_pre=[s_["tap_share_of_pre"] for s_ in st])
    # ---------------- decision rules
    code_st = res["streams"]["code"]
    n_conv2 = sum(1 for s_ in st if s_["conv_over_direct"]["median"] >= 2.0)
    n_conv2_code = sum(1 for s_ in code_st if s_["conv_over_direct"]["median"] >= 2.0)
    w3 = [v_["within_share"] for v_ in var["all"]["3gram_context"]]
    res["decisions"] = dict(
        gdir_mandatory=dict(rule="conv >= 2x direct on the median token in >= 2 of 4 streams",
                            streams_meeting_all=n_conv2, streams_meeting_code=n_conv2_code,
                            medians_all=[s_["conv_over_direct"]["median"] for s_ in st],
                            verdict=bool(n_conv2 >= 2)),
        calibrated_gate=dict(rule=">= 50% of gate-logit variance within n-grams (groups = identical 3-gram hash "
                                  "context, i.e. identical e_t)",
                             within_share_per_stream=w3, verdict=bool(np.nanmedian(w3) >= 0.5),
                             streams_meeting=int(sum(1 for x in w3 if x >= 0.5))),
        output_scale=dict(out_over_x_median_per_stream=[s_["out_over_x"]["median"] for s_ in st],
                          out_over_x_all_streams=res["out_over_x_all_streams"]["all"],
                          rho_grid=[0.003, 0.01, 0.03]))
    import resource
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    res["wall_s"] = round(time.time() - T0, 1)
    (HERE / "s_results.json").write_text(json.dumps(res, indent=1))
    pr(res)


def pr(res):
    print("\n== S: 2-layer Flash-Next slice; corpus", res["corpus"], "tokens", res["tokens"])
    print("probe vs q4 PLE parity (max|diff|):", res["parity_vs_q4"])
    for c in ("all", "code", "md", "wiki"):
        print(f"\n-- {c}")
        print(f"{'s':>2s} {'sig mean':>8s} {'p05':>6s} {'p95':>6s} {'g std':>6s} {'conv/dir':>8s} {'dir/X':>8s} "
              f"{'conv/X':>8s} {'out/X med':>9s} {'mean':>7s} {'p95':>7s} {'p99':>7s} {'taps j=0..3 /pre':>28s} "
              f"{'|dsig|-2g':>9s} {'|dsig|-3g':>9s}")
        for s_ in res["streams"][c]:
            print(f"{s_['stream']:2d} {s_['gate_sigmoid']['mean']:8.4f} {s_['gate_sigmoid']['p05']:6.3f} "
                  f"{s_['gate_sigmoid']['p95']:6.3f} {s_['gate_logit_std']:6.3f} "
                  f"{s_['conv_over_direct']['median']:8.2f} {s_['direct_over_x']['median']:8.5f} "
                  f"{s_['conv_over_x']['median']:8.5f} {s_['out_over_x']['median']:9.5f} {s_['out_over_x']['mean']:7.4f} "
                  f"{s_['out_over_x']['p95']:7.4f} {s_['out_over_x']['p99']:7.4f} "
                  f"{str([round(x, 2) for x in s_['tap_share_of_pre']]):>28s} {s_['dsig_no2']['mean_abs']:9.4f} "
                  f"{s_['dsig_no3']['mean_abs']:9.4f}")
        v = res["value"][c]
        print(f"   |out|/|X| over all 4 streams: {res['out_over_x_all_streams'][c]}")
        print(f"   value: |v2|/|v| median {v['dv_no2_median']:.3f}, |v3|/|v| median {v['dv_no3_median']:.3f}; "
              f"<v,v2>/|v|^2 {v['proj2_mean']:.3f}, <v,v3>/|v|^2 {v['proj3_mean']:.3f}; |v| median "
              f"{v['v_norm_median']:.4f}, |e| median {v['e_norm_median']:.4f}")
        for gname, lst in res["gate_variance"][c].items():
            print(f"   gate-logit within-group share, groups={gname}: "
                  f"{[round(x['within_share'], 3) for x in lst]} (tokens in repeated groups {lst[0]['n']}, "
                  f"groups {lst[0]['groups']})")
    print("\n-- absolute per-stream RMS medians:", {k_: [round(x, 5) for x in v_] for k_, v_ in res["abs_rms_median"].items()})
    print("-- conv-input saturation mean(u^2)/(mean(u^2)+eps) per stream (1 = direction-only, 0 = eps-linear):")
    for c, v_ in res["conv_input_saturation"].items():
        print(f"   {c:>6s} median {[round(x, 3) for x in v_['median']]} frac>0.5 {[round(x, 3) for x in v_['frac_gt_half']]}")
    print("-- gate-logit within-3-gram-context share, group-bootstrap 95% CI:")
    for k_, v_ in res["gate_variance_robust"].items():
        print(f"   {k_:>20s} (tokens {v_['n_tokens']}): {[tuple(round(x, 3) for x in t_) for t_ in v_['within_share_ci']]}")
    print("\n-- code by lexical class: n, sigmoid mean per stream, conv/direct median, out/X median, proj3")
    for k_, v_ in res["by_lex"].items():
        print(f"  {k_:>10s} n={v_['n']:5d} sig {[round(x, 3) for x in v_['sig_mean']]} conv/dir "
              f"{[round(x, 1) for x in v_['conv_over_direct_median']]} out/X {[round(x, 4) for x in v_['out_over_x_median']]}"
              f" proj3 {v_['proj3_mean']:.3f}")
    for n in ("2", "3"):
        print(f"\n-- by proxy frequency of the position's {n}-gram: n, row coherence, sigmoid mean, |dsig| without "
              f"the {n}-gram heads, proj3, out/X median")
        for r in res["by_freq"][n]:
            print(f"  {r['bin']:>24s} n={r['n']:5d} coh {r['coh']:+.3f} sig {[round(x, 3) for x in r['sig_mean']]} "
                  f"|dsig| {[round(x, 4) for x in r['dsig_no_this_order']]} proj3 {r['proj3_mean']:.3f} out/X "
                  f"{[round(x, 4) for x in r['out_over_x_median']]}")
        print(f"-- by row coherence of the {n}-gram heads")
        for r in res["by_coherence"][n]:
            print(f"  coh in {str(r['bin']):>12s} n={r['n']:5d} sig {[round(x, 3) for x in r['sig_mean']]} |dsig| "
                  f"{[round(x, 4) for x in r['dsig_no_this_order']]} proj3 {r['proj3_mean']:.3f}")
    g = res["gdir"]
    print(f"\n-- G-dir: Flash-Next conv/|X| median per stream {[round(x, 5) for x in g['conv_over_x_median']]}; "
          f"N_conv(u) entries rms {g['z_rms']:.3f}, kurtosis {g['z_kurtosis']:.2f}")
    for k_, v_ in g["alpha_table"].items():
        print(f"   alpha for {k_}: {[round(x, 4) for x in v_]}")
    d = res["decisions"]
    print("\n== decision rules (§D.3)")
    print(f"G-dir mandatory? rule: {d['gdir_mandatory']['rule']}; conv/direct medians "
          f"{[round(x, 2) for x in d['gdir_mandatory']['medians_all']]}; streams meeting: all text "
          f"{d['gdir_mandatory']['streams_meeting_all']}/4, code {d['gdir_mandatory']['streams_meeting_code']}/4 -> "
          f"{'YES' if d['gdir_mandatory']['verdict'] else 'NO'}")
    print(f"Calibrated gate arm? rule: {d['calibrated_gate']['rule']}; within shares "
          f"{[round(x, 3) for x in d['calibrated_gate']['within_share_per_stream']]} -> "
          f"{'YES' if d['calibrated_gate']['verdict'] else 'NO'} ({d['calibrated_gate']['streams_meeting']}/4 streams)")
    print(f"Output scale: |PLE out_s|/|X_s| median per stream "
          f"{[round(x, 5) for x in d['output_scale']['out_over_x_median_per_stream']]}; all streams "
          f"{d['output_scale']['out_over_x_all_streams']}")
    print(f"\nreader {res['reader']}; peak RSS {res['peak_rss_mb']} MB; wall {res['wall_s']} s")


if __name__ == "__main__":
    main()

"""G step 2 (§D.3 experiment G): coherence of Flash-Next's real table rows vs general-text frequency, at ~5k
n-grams per order, with bootstrap CIs, the crossing frequency f* (coherence 0.05), the 2- vs 3-gram value share,
and row norm vs frequency. Scales the G0 pilot (GF/dissect/row_pilot.py) from n = 32-48 per bucket and a local
400k-token frequency to stratified log2-count bins over the 86.4M-token public proxy of g1_counts.py.

Samples (seed 0), per order n in {2, 3}:
  * ~5,000 n-grams stratified over log2 bins of their proxy count (code + prose), uniform within a bin;
  * "project-only": 300 n-grams with proxy count 0 and count >= 2 in the minisgl Split-A *train* files
    (G is a row census, nothing is evaluated on them);
  * for every sampled 3-gram (a, b, c) also its suffix 2-gram (b, c) -- the 2-gram rows the same position reads --
    for the per-position 2- vs 3-gram value share;
  * null: random rows (contiguous 2,048-row blocks at random offsets inside each head; 4 blocks per head) used for
    the head means mu_h and for random 8-head tuples (the coherence floor).
Rows are fetched by HTTP range reads of the public bf16 shards (cache: scratch G/rows_cache.npz).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/memory-organ/ngram-memory")
import numpy as np  # noqa: E402

from ngram_memory.dissect import row_stats as rs  # noqa: E402
from ngram_memory.dissect.hf_range import FLASH_NEXT_REPO, FlashNextRowFetcher, HFRangeReader, bf16_to_f32  # noqa: E402
from ngram_memory.dissect.slice_probe import SliceWeights  # noqa: E402
from ngram_memory.hashing import hasher_for  # noqa: E402
from ngram_memory.spec import flash_next_format  # noqa: E402

S = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
D = Path(f"{S}/nm/m0/dissect")
GD = D / "G"
HERE = Path(__file__).resolve().parent
INDEX = f"{S}/models/Qwen/Qwen3.8-Flash-Next/model.safetensors.index.json"
TOK = f"{S}/nm/graft/tok/Qwen__Qwen3.8-Flash-Next/tokenizer.json"
V = 248320
PER_ORDER = 5000
N_PROJECT = 300
THR = 0.05
N_BOOT = 2000
SEED = 0
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def combined(n: int):
    kc, cc = (np.load(GD / f"counts_code_{n}.npz")[x] for x in ("keys", "counts"))
    kp, cp = (np.load(GD / f"counts_prose_{n}.npz")[x] for x in ("keys", "counts"))
    k, c = rs.merge_counts(kc, cc, kp, cp)
    return k, c, (kc, cc), (kp, cp)


def project_train_counts():
    """2-/3-gram counts over the minisgl Split-A train files (bare framing, as the census)."""
    from ngram_memory.ingest.corpus import TokenizerJSON, load_corpus, load_split_a, tokenize_files
    sa = load_split_a()
    corp = load_corpus(commit=sa.commit)
    files = [f for f in corp.files if f.path in set(sa.train)]
    tok = TokenizerJSON(TOK)
    docs = [d.ids for d in tokenize_files(tok, files, "bare")]
    out = {n: rs.count_keys(docs, n, V) for n in (2, 3)}
    return out, dict(files=len(files), tokens=int(sum(len(d) for d in docs)), commit=sa.commit)


def main() -> None:
    rng = np.random.default_rng(SEED)
    fmt = flash_next_format()
    key = fmt.key
    hs = hasher_for(key)
    assert list(key.multipliers) == [23703573157769, 20109073645365, 8052911324071]
    man = json.loads((GD / "corpus_manifest.json").read_text())
    n_tok = sum(s["tokens"] for s in man["sources"].values())
    log("proxy tokens", n_tok)

    proj, proj_info = project_train_counts()
    log("project train", proj_info)

    samples = {}      # order -> dict(keys, counts, code, prose, bin, kind)
    for n in (2, 3):
        k, c, (kc, cc), (kp, cp) = combined(n)
        bins = rs.log2_bins(int(c.max()))
        strat = rs.stratified_sample(k, c, bins, None, rng, total=PER_ORDER)
        sk = np.concatenate([s[2] for s in strat])
        sb = np.concatenate([np.full(len(s[2]), i) for i, s in enumerate(strat)])
        pk, pc = proj[n]
        p_only = pk[(pc >= 2) & (rs.lookup_counts(k, c, pk) == 0)]
        pick = np.sort(rng.choice(p_only, size=min(N_PROJECT, p_only.size), replace=False))
        allk = np.concatenate([sk, pick])
        samples[n] = dict(keys=allk, count=rs.lookup_counts(k, c, allk), code=rs.lookup_counts(kc, cc, allk),
                          prose=rs.lookup_counts(kp, cp, allk), proj=rs.lookup_counts(pk, pc, allk),
                          bin=np.concatenate([sb, np.full(pick.size, -1)]), bins=bins,
                          n_distinct=int(k.size), n_project_only_pool=int(p_only.size))
        log(f"order {n}: {len(bins)} bins {[len(s_[2]) for s_ in strat]}; sampled {sk.size} + project-only {pick.size}"
            f" (pool {p_only.size}); distinct {k.size}")
        if n == 2:
            k2, c2, kc2, kp2 = k, c, (kc, cc), (kp, cp)
        del k, c
    # suffix 2-grams of the sampled 3-grams
    g3 = rs.unpack(samples[3]["keys"], 3, V)
    suf = rs.pack(g3[:, 1:], V)
    suf_count = rs.lookup_counts(k2, c2, suf)

    # ---- row ids
    def rows_of(keys, n):
        g = rs.unpack(keys, n, V)
        return np.stack([hs.rows_for_ngram(x) for x in g])          # [N, 8]
    R2 = rows_of(samples[2]["keys"], 2)
    R3 = rows_of(samples[3]["keys"], 3)
    RS = rows_of(suf, 2)
    need = np.unique(np.concatenate([R2.ravel(), R3.ravel(), RS.ravel()]))
    log(f"rows needed: {need.size} ({need.size * 320 / 2**20:.1f} MiB payload)")

    reader = HFRangeReader(FLASH_NEXT_REPO, index_path=INDEX)
    fetch = FlashNextRowFetcher(reader, cache_path=GD / "rows_cache.npz", threads=48)
    # self-check of the addressing against the earlier independent fetch (P/t1/fetch_real_ple.py rows)
    import torch
    ref = torch.load(f"{S}/nm/t1/real/real_ple_block.pt")
    chk = {h: [key.head_offsets[h] + 1000 + i for i in (0, 7, 2047)] for h in (0, 7, 8, 15)}
    fetch.fetch([r for v_ in chk.values() for r in v_])
    for h, rr in chk.items():
        exp = ref[f"rows_head{h}_start"].float().numpy()[[0, 7, 2047]]
        assert np.array_equal(fetch.get(rr), exp), f"row addressing mismatch head {h}"
    log("addressing self-check: 12 rows equal to P/t1/fetch_real_ple.py's independent fetch")
    del ref
    t = time.time()
    got = fetch.fetch(need, log_every=10000)
    fetch.save()
    log(f"fetched {got} rows in {time.time() - t:.0f}s; reader {reader.stats.to_dict()}")

    # ---- null blocks / head means
    rps = fmt.table.rows_per_shard(key)
    blocks = []
    nb_path = GD / "null_blocks.npy"
    if nb_path.exists():
        B = np.load(nb_path)
    else:
        t = time.time()
        for h in range(16):
            off, size = key.head_offsets[h], key.head_sizes[h]
            starts = []
            while len(starts) < 4:
                r0 = off + int(rng.integers(0, size - 2048))
                s0, l0 = divmod(r0, rps)
                if l0 + 2048 <= rps:
                    starts.append(r0)
            for r0 in starts:
                s0, l0 = divmod(r0, rps)
                info = fetch.shard_info(s0)
                raw = reader.read_blocks(info, [l0], 2048, threads=1)[0]
                blocks.append(bf16_to_f32(raw.view(np.uint16)).reshape(2048, 160))
        B = np.stack(blocks).reshape(16, 4 * 2048, 160)
        np.save(nb_path, B)
        log(f"null blocks 16x8192 rows in {time.time() - t:.0f}s")
    mu = B.mean(1)                                                   # [16, 160]

    w = SliceWeights(D / "raw/slice")
    Wv = w.t("model.language_model.layers.1.ple.value_proj.weight").numpy().astype(np.float64)
    wh = rs.head_slices(Wv, 16)
    fro = [float(np.linalg.norm(Wv[:, :1280])), float(np.linalg.norm(Wv[:, 1280:]))]
    log("W_v Frobenius 2-gram / 3-gram halves", fro)

    Wh32 = [x.astype(np.float32) for x in wh]
    mu32 = mu.astype(np.float32)

    def metrics(R, heads):
        """Vectorised G0 metrics for rows R [N, 8] of one order (heads in head order)."""
        E = fetch.get(R)                                             # [N, 8, 160] float32
        N = E.shape[0]
        Cc = np.empty((N, len(heads), Wv.shape[0]), dtype=np.float32)
        for i, h in enumerate(heads):
            Cc[:, i] = (E[:, i] - mu32[h]) @ Wh32[h].T
        Cu = Cc + np.stack([Wh32[h] @ mu32[h] for h in heads])[None]

        def coh(C):
            U = C / np.maximum(np.linalg.norm(C, axis=2, keepdims=True), 1e-30)
            Sm = np.einsum("nid,njd->nij", U, U)
            k_ = C.shape[1]
            return (Sm.sum((1, 2)) - np.trace(Sm, axis1=1, axis2=2)) / (k_ * (k_ - 1))
        out = dict(coh=coh(Cc).astype(np.float64), coh_unc=coh(Cu).astype(np.float64),
                   vc=Cc.sum(1), v=Cu.sum(1), row_norm=np.linalg.norm(E, axis=2).mean(1).astype(np.float64))
        out["vc_norm"] = np.linalg.norm(out["vc"], axis=1)
        out["v_norm"] = np.linalg.norm(out["v"], axis=1)
        return out

    h2, h3 = list(range(0, 8)), list(range(8, 16))
    M = {2: metrics(R2, h2), 3: metrics(R3, h3)}
    MS = metrics(RS, h2)
    # null: random 8-head tuples
    null = {}
    for n, heads in ((2, h2), (3, h3)):
        co, nm, vn = [], [], []
        for _ in range(2000):
            rows = np.stack([B[h][rng.integers(0, B.shape[1])] for h in heads])
            cc_ = rs.contributions(rows, heads, wh, mu)
            co.append(rs.coherence(cc_))
            nm.append(float(np.linalg.norm(rows, axis=1).mean()))
            vn.append(float(np.linalg.norm(cc_.sum(0))))
        null[n] = dict(coh=np.asarray(co), row_norm=np.asarray(nm), vc_norm=np.asarray(vn))

    # ---- curves
    res = dict(proxy_tokens=n_tok, threshold=THR, n_boot=N_BOOT, seed=SEED, project=proj_info,
               W_v_frobenius_2g_3g=fro, reader=reader.stats.to_dict(), rows_fetched_total=int(need.size),
               orders={})
    lines = []
    for n in (2, 3):
        smp, m = samples[n], M[n]
        bins = smp["bins"]
        groups, xs, table = [], [], []
        for b, (lo, hi) in enumerate(bins):
            sel = smp["bin"] == b
            if not sel.any():
                continue
            fc = np.sqrt(lo * hi) / n_tok                            # geometric centre frequency per token
            mc, a, z = rs.bootstrap_mean_ci(m["coh"][sel], N_BOOT, rng=rng)
            groups.append(m["coh"][sel])
            xs.append(np.log10(fc))
            code_share = smp["code"][sel] / np.maximum(smp["count"][sel], 1)
            table.append(dict(bin=[int(lo), int(hi)], freq_per_mtok=float(fc * 1e6), n=int(sel.sum()),
                              coherence=mc, ci=[a, z], coh_median=float(np.median(m["coh"][sel])),
                              coh_uncentred=float(m["coh_unc"][sel].mean()),
                              snr8=float(rs.snr_from_coherence(max(mc, 0.0))),
                              row_norm=float(m["row_norm"][sel].mean()),
                              row_norm_ci=list(rs.bootstrap_mean_ci(m["row_norm"][sel], N_BOOT, rng=rng)[1:]),
                              v_centred_norm=float(m["vc_norm"][sel].mean()), v_norm=float(m["v_norm"][sel].mean()),
                              code_share=float(code_share.mean())))
        curve = rs.BinCurve(np.array([t_["bin"][0] for t_ in table]), np.array([t_["bin"][1] for t_ in table]),
                            np.array(xs), groups)
        est, lo_, hi_, share = rs.bootstrap_crossing(curve, THR, N_BOOT, rng=rng)
        # source-conditioned curves (dominant source by count)
        by_src = {}
        for src, cond in (("code", lambda s: s["code"] >= s["prose"]), ("prose", lambda s: s["prose"] > s["code"])):
            gs, xx = [], []
            for b, (lo, hi) in enumerate(bins):
                sel = (smp["bin"] == b) & cond(smp)
                if sel.sum() >= 20:
                    gs.append(m["coh"][sel])
                    xx.append(np.log10(np.sqrt(lo * hi) / n_tok))
            if len(gs) >= 3:
                e2, l2, h2_, s2 = rs.bootstrap_crossing(rs.BinCurve(np.zeros(len(gs)), np.zeros(len(gs)),
                                                                    np.array(xx), gs), THR, N_BOOT, rng=rng)
                by_src[src] = dict(f_star_per_mtok=float(10 ** e2 * 1e6) if np.isfinite(e2) else None,
                                   ci_per_mtok=[float(10 ** l2 * 1e6), float(10 ** h2_ * 1e6)] if np.isfinite(l2) else None,
                                   crossing_share=s2, n_bins=len(gs),
                                   bins=[dict(freq_per_mtok=float(10 ** x_ * 1e6), n=int(g_.size),
                                              coherence=float(g_.mean())) for x_, g_ in zip(xx, gs)])
        po = smp["bin"] == -1
        pm = rs.bootstrap_mean_ci(m["coh"][po], N_BOOT, rng=rng)
        nl = null[n]
        nm_ = rs.bootstrap_mean_ci(nl["coh"], N_BOOT, rng=rng)
        f_star = float(10 ** est) if np.isfinite(est) else None
        res["orders"][str(n)] = dict(
            n_sampled=int((smp["bin"] >= 0).sum()), n_distinct_proxy=smp["n_distinct"], bins=table,
            f_star_per_token=f_star, f_star_per_mtok=f_star * 1e6 if f_star else None,
            f_star_count_in_proxy=f_star * n_tok if f_star else None,
            f_star_ci_per_mtok=[float(10 ** lo_ * 1e6), float(10 ** hi_ * 1e6)] if np.isfinite(lo_) else None,
            crossing_share=share, by_source=by_src,
            project_only=dict(n=int(po.sum()), pool=smp["n_project_only_pool"], coherence=pm[0], ci=list(pm[1:]),
                              row_norm=float(m["row_norm"][po].mean()), v_centred_norm=float(m["vc_norm"][po].mean())),
            null=dict(n=2000, coherence=nm_[0], ci=list(nm_[1:]), row_norm=float(nl["row_norm"].mean()),
                      v_centred_norm=float(nl["vc_norm"].mean())))
        # project rows below f*: share of the project's count>=2 n-grams (distinct and by occurrences)
        if f_star:
            pk, pc = proj[n]
            kk, cc_ = combined(n)[:2]
            pf = rs.lookup_counts(kk, cc_, pk) / n_tok
            m2 = pc >= 2
            below = pf < f_star
            res["orders"][str(n)]["project_count2_below_fstar"] = dict(
                distinct=float(below[m2].mean()), occurrences=float((pc[m2] * below[m2]).sum() / pc[m2].sum()),
                n_distinct_count2=int(m2.sum()))
            del kk, cc_
        lines.append((n, table, res["orders"][str(n)]))

    # ---- 2- vs 3-gram value share at the same position (3-gram + its suffix 2-gram)
    v3, v2 = M[3]["v"], MS["v"]
    vs = rs.value_share(v2, v3)
    vsc = rs.value_share(MS["vc"], M[3]["vc"])
    smp3 = samples[3]
    share_tab = []
    for b, (lo, hi) in enumerate(smp3["bins"]):
        sel = smp3["bin"] == b
        if not sel.any():
            continue
        share_tab.append(dict(bin3=[int(lo), int(hi)], n=int(sel.sum()),
                              suffix2_count_median=float(np.median(suf_count[sel])),
                              proj3=float(vs["proj3"][sel].mean()), proj3_centred=float(vsc["proj3"][sel].mean()),
                              ratio32_median=float(np.median(vs["ratio32"][sel])),
                              coh3=float(M[3]["coh"][sel].mean()), coh2_suffix=float(MS["coh"][sel].mean())))
    allb = smp3["bin"] >= 0
    res["value_share"] = dict(
        definition="v = v2 + v3 at the position whose 3-gram is sampled; proj_n = <v, v_n>/|v|^2 (sums to 1); "
                   "uncentred = what the model reads; centred = after subtracting head means",
        proj3_mean=float(vs["proj3"][allb].mean()), proj3_centred_mean=float(vsc["proj3"][allb].mean()),
        norm3_over_norm2_median=float(np.median(vs["ratio32"][allb])),
        project_only=dict(proj3=float(vs["proj3"][smp3["bin"] == -1].mean()),
                          proj3_centred=float(vsc["proj3"][smp3["bin"] == -1].mean())),
        by_bin3=share_tab)
    import resource
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    res["wall_s"] = round(time.time() - T0, 1)
    (HERE / "g2_results.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(HERE / "g2_per_ngram.npz",
                        **{f"o{n}_{k_}": v_ for n in (2, 3) for k_, v_ in (
                            ("keys", samples[n]["keys"]), ("count", samples[n]["count"]), ("code", samples[n]["code"]),
                            ("prose", samples[n]["prose"]), ("proj", samples[n]["proj"]), ("bin", samples[n]["bin"]),
                            ("coh", M[n]["coh"].astype(np.float32)), ("row_norm", M[n]["row_norm"].astype(np.float32)),
                            ("vc_norm", M[n]["vc_norm"].astype(np.float32)))},
                        suffix2_count=suf_count, suffix2_coh=MS["coh"].astype(np.float32),
                        proj3=vs["proj3"].astype(np.float32), proj3_centred=vsc["proj3"].astype(np.float32))

    # ---- report
    print("\n== G: coherence of Flash-Next rows vs proxy frequency (centred, W_v-mapped; mean pairwise cos over 8 heads)")
    print(f"proxy {n_tok} tokens (code {man['sources']['code']['tokens']}, prose {man['sources']['prose']['tokens']}); "
          f"threshold {THR}; bootstrap {N_BOOT}")
    for n, table, o in lines:
        print(f"\n-- order {n}: sampled {o['n_sampled']} stratified + {o['project_only']['n']} project-only")
        print(f"{'count bin':>16s} {'f/Mtok':>10s} {'n':>4s} {'coh':>7s} {'95% CI':>17s} {'SNR8':>6s} {'|e|':>7s} "
              f"{'|v_c|':>7s} {'code%':>6s}")
        for t_ in table:
            print(f"{str(t_['bin']):>16s} {t_['freq_per_mtok']:10.4f} {t_['n']:4d} {t_['coherence']:+.4f} "
                  f"[{t_['ci'][0]:+.4f},{t_['ci'][1]:+.4f}] {t_['snr8']:6.2f} {t_['row_norm']:.4f} "
                  f"{t_['v_centred_norm']:.4f} {100 * t_['code_share']:5.1f}")
        p, nl = o["project_only"], o["null"]
        print(f"{'project-only':>16s} {'0':>10s} {p['n']:4d} {p['coherence']:+.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}]"
              f"        {p['row_norm']:.4f} {p['v_centred_norm']:.4f}")
        print(f"{'null (random)':>16s} {'-':>10s} {nl['n']:4d} {nl['coherence']:+.4f} [{nl['ci'][0]:+.4f},"
              f"{nl['ci'][1]:+.4f}]        {nl['row_norm']:.4f} {nl['v_centred_norm']:.4f}")
        if o["f_star_per_mtok"]:
            print(f"f* (coherence {THR}) = {o['f_star_per_mtok']:.4f} per Mtok (= {o['f_star_count_in_proxy']:.1f} "
                  f"occurrences in the proxy); 95% CI {o['f_star_ci_per_mtok']} per Mtok; bootstrap replicates with a "
                  f"crossing {o['crossing_share']:.3f}")
        else:
            print("f*: no crossing")
        for src, b in o["by_source"].items():
            print(f"   {src}-dominant n-grams: f* = {b['f_star_per_mtok']} per Mtok, CI {b['ci_per_mtok']}, "
                  f"bins {[(round(x['freq_per_mtok'], 4), x['n'], round(x['coherence'], 4)) for x in b['bins']]}")
        if "project_count2_below_fstar" in o:
            q = o["project_count2_below_fstar"]
            print(f"   minisgl Split-A train n-grams with count>=2 ({q['n_distinct_count2']}): below f* "
                  f"{100 * q['distinct']:.1f}% of distinct, {100 * q['occurrences']:.1f}% of occurrences")
    vsr = res["value_share"]
    print(f"\n== 2- vs 3-gram value share at the same position (n = {int(allb.sum())} sampled 3-grams + suffix 2-grams)")
    print(f"W_v Frobenius: 2-gram half {fro[0]:.2f}, 3-gram half {fro[1]:.2f}")
    print(f"mean 3-gram share <v,v3>/|v|^2 = {vsr['proj3_mean']:.3f} (centred {vsr['proj3_centred_mean']:.3f}); "
          f"median |v3|/|v2| = {vsr['norm3_over_norm2_median']:.3f}; project-only 3-grams: "
          f"{vsr['project_only']['proj3']:.3f} (centred {vsr['project_only']['proj3_centred']:.3f})")
    for t_ in share_tab:
        print(f"  3-gram count {str(t_['bin3']):>16s} n={t_['n']:4d} suffix-2g median count {t_['suffix2_count_median']:>10.0f}"
              f" proj3 {t_['proj3']:.3f} (c {t_['proj3_centred']:.3f}) |v3|/|v2| {t_['ratio32_median']:.3f}"
              f" coh3 {t_['coh3']:+.3f} coh2(suffix) {t_['coh2_suffix']:+.3f}")
    print(f"\nreader: {reader.stats.to_dict()}; peak RSS {res['peak_rss_mb']} MB; wall {res['wall_s']} s")


if __name__ == "__main__":
    main()

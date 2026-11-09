# Review-apply budget: (1) tokens per fact per solve iteration for the §5.1 prompt sets (Qwen tokenizer,
# hand-written CounterFact-style facts: ILLUSTRATIVE, not CounterFact itself); (2) FLOP/iter at 6*active*N
# (fwd + recompute fwd under per-layer checkpointing + bwd); (3) graft overlay sizes per host width d/16;
# (4) per-milestone cost table. Everything below (3) is an ESTIMATE driven by the stated assumptions.
import json, os, sys
from tokenizers import Tokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
TOK = sys.argv[1]
tok = Tokenizer.from_file(TOK)
n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
facts = [
 dict(s="Danielle Darrieux", rw="The mother tongue of Danielle Darrieux is", o=" English",
      para=["Danielle Darrieux, a native speaker of", "Danielle Darrieux spoke the language",
            "The native language of Danielle Darrieux is", "Danielle Darrieux was born speaking"]),
 dict(s="Toyota", rw="Toyota was founded in the country of", o=" Italy",
      para=["The company Toyota originated in", "Toyota, which was established in",
            "Toyota's country of origin is", "Toyota is a company from"]),
 dict(s="Michael Jordan", rw="Michael Jordan professionally plays the sport", o=" baseball",
      para=["Michael Jordan is best known for playing", "The sport Michael Jordan plays is",
            "Michael Jordan, the professional", "Michael Jordan's career was in"]),
]
ctx = ["The following is a short encyclopedia entry. ", "Here is something I read yesterday. ",
       "Answer the question below carefully. ", "Notes from a history lecture: ", "In a recent interview, "]
ess = ["{s} is a", "{s} is", "{s},", "{s} was", "{s} is known for", "{s} works in the field of",
       "The best-known fact about {s} is", "{s} is located near"]
GREEDY = 3
rows = []
for f in facts:
    base = [f["rw"]] + f["para"]
    xrw = base + [c + p for c in ctx for p in base]                    # K=4 paraphrases, C=5 contexts
    obj = n(f["o"])
    full_rw = sum(n(p) + obj for p in xrw)
    full_ess = sum(n(t.format(s=f["s"])) + GREEDY for t in ess)
    # anchor-suffix tokens (INTEGRATION.md §4.3 prefix caching: positions before the anchor are run once
    # without grad and cached; each iteration recomputes only anchor..end)
    def suffix(p):
        i = p.find(f["s"]);  pre = p[:i + len(f["s"])]
        return n(p) - n(pre) + 1
    suf_rw = sum(suffix(p) + obj for p in xrw)
    suf_ess = sum(suffix(t.format(s=f["s"])) + GREEDY for t in ess)
    rows.append(dict(s=f["s"], n_prompts=len(xrw) + len(ess), full=full_rw + full_ess, suffix=suf_rw + suf_ess))
print("per-fact tokens (X_rw + X_ess; X_col/X_ret excluded):")
for r in rows: print(f"  {r['s']:18s} prompts {r['n_prompts']}  full {r['full']}  anchor-suffix {r['suffix']}")
full = [r["full"] for r in rows]; suf = [r["suffix"] for r in rows]
XCOL = (8 * 32, 32 * 32)   # 8-32 corpus windows of 32 tokens (assumption)
N_full = (min(full) + XCOL[0], max(full) + XCOL[1]); N_suf = (min(suf) + XCOL[0] // 4, max(suf) + XCOL[1] // 4)
print(f"N_fact full-prompt incl. X_col: {N_full[0]}-{N_full[1]}; anchor-suffix with prefix cache: {N_suf[0]}-{N_suf[1]}")
cnt = json.load(open(sys.argv[2]))
act = {"Flash-Next": "T1 Qwen3.8-Flash-Next", "E4B": "T2 gemma-4-E4B-it", "27B": "T3 Qwen3.8-27B",
       "35B-A3B": "T3 Qwen3.6-35B-A3B", "g26B": "T3 gemma-4-26B-A4B-it", "g31B": "T3 gemma-4-31B-it"}
plan = json.load(open(sys.argv[3]))
active = {k: plan[v]["tflop"] * 1e12 / (4 * 320) for k, v in act.items()}   # recover active params from plan_calc
active["E2B"] = 2.2986e9
ck = {k: plan[v]["ckpt_GB"] / 320 for k, v in act.items()}
print("\nFLOP/iter at 6*active*N (TFLOP) and ckpt activations (GB, linear in N):")
for k, a in active.items():
    f = lambda N: 6 * a * N / 1e12
    c = (f"{ck[k]*N_suf[0]:.2f}-{ck[k]*N_full[1]:.2f}" if k in ck else "n/a")
    print(f"  {k:10s} active {a/1e9:5.2f}G | suffix N {f(N_suf[0]):6.1f}-{f(N_suf[1]):6.1f} | full N {f(N_full[0]):6.1f}-{f(N_full[1]):6.1f} | act GB {c}")
# CPU here: measured bf16 gated-MLP fwd+inputgrad 1457 GFLOP/s (redteam-feasibility/cpu_flops.out);
# end-to-end HF efficiency UNMEASURED -> assume 10-30% of that.
lo, hi = 0.10 * 1457e9, 0.30 * 1457e9
for N in (N_suf, N_full):
    a = active["E2B"]; print(f"E2B on this CPU at N={N[0]}-{N[1]}: {6*a*N[0]/hi:.0f}-{6*a*N[1]/lo:.0f} s/iter (est.)")
print("\ngraft overlay (fp32 rows + int64 ids), 16 rows/anchor, width d/16:")
for host, d in [("27B", 5120), ("35B-A3B", 2048), ("g26B", 2816), ("g31B", 5376), ("Flash-Next native ref", 2560)]:
    w = d // 16
    print(f"  {host:22s} w={w:3d} " + "  ".join(f"1e{e}: {10**e*16*(w*4+8)/1e6:9.1f} MB" for e in (3, 4, 5)))
# cost: $ = solves * iters * s/iter / 3600 * $/h + setup hours * $/h
ITERS = (100, 300)
venue = {  # s/iter at suffix N; $/h community..secure
 "FN H200 (d2)":  dict(s=(1.1, 3.2), usd=(3.59, 5.29), setup_h=1.0),   # cost_model.txt column C at 256 tok
 "FN 4xH100 bf16": dict(s=(1.0, 3.0), usd=(10.76, 15.96), setup_h=1.5), # ASSUMED ~ same as C
 "graft A100":    dict(s=(0.3, 2.0), usd=(1.39, 1.79), setup_h=0.5),   # ASSUMED
}
ms = [("M2-lite", "FN 4xH100 bf16", 10), ("M2 (3 arms x 2 lambda x 60 facts)", "FN H200 (d2)", 360),
      ("M2g (2 hosts x 3 arms x 50)", "graft A100", 300), ("M3 native (150/len x 3 len x 2 tiers)", "FN H200 (d2)", 900),
      ("M3 graft (450 x 4 lambda x 2 orders)", "graft A100", 3600), ("M3 native-vs-graft on FN", "FN H200 (d2)", 100),
      ("M4 E4B (200 x 2)", "graft A100", 400), ("M5 (2 fmt x 2 hosts x 200)", "graft A100", 800),
      ("M6 sequential 5k + re-solves (~1.5x)", "FN H200 (d2)", 7500)]
print("\nper-milestone cost (est.; iters/solve 100-300):")
tot = [0, 0]
for name, v, nsol in ms:
    V = venue[v]
    h = (nsol * ITERS[0] * V["s"][0] / 3600 + V["setup_h"], nsol * ITERS[1] * V["s"][1] / 3600 + V["setup_h"])
    usd = (h[0] * V["usd"][0], h[1] * V["usd"][1]); tot[0] += usd[0]; tot[1] += usd[1]
    print(f"  {name:40s} {v:15s} solves {nsol:5d}  GPU-h {h[0]:7.1f}-{h[1]:7.1f}  ${usd[0]:7.0f}-{usd[1]:7.0f}")
print(f"  TOTAL ${tot[0]:.0f}-{tot[1]:.0f}  (+ volume storage $0.10/GB/month: 150 GB -> $15/month)")

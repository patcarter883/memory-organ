"""Power / noise-floor analysis for Stage-1 NLL metrics, from the verified graft run's per-token arrays
(stage1/verify-experiment: base_verify.npz = graft OFF, nll_full_*.npy = graft ON, Qwen2.5-0.5B, 512-token windows).
Window-clustered (block) bootstrap of the paired per-token delta; minimum detectable effect (MDE, 80% power,
two-sided 5%) as a function of held-out tokens, assuming window-level variance scales as observed."""
import json, numpy as np, os
D = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/nm/stage1/verify-experiment"
OUT = os.path.dirname(os.path.abspath(__file__))
base = np.load(f"{D}/base_verify.npz"); R = {}
rng = np.random.default_rng(0)
for k in ["heldout", "stdlib_py", "general_text", "train"]:
    b = base[k].astype(np.float64); on = np.load(f"{D}/nll_full_{k}.npy").astype(np.float64)
    d = on - b; W, T = d.shape
    wm = d.mean(1)                                    # per-window mean delta
    boots = np.array([wm[rng.integers(0, W, W)].mean() for _ in range(5000)])
    tok_sd = d.std(); win_sd = wm.std(ddof=1)
    # design effect: var of window mean vs iid tokens
    deff = (win_sd ** 2) / (tok_sd ** 2 / T)
    se_per_window = win_sd                             # SE of mean over n windows = win_sd/sqrt(n)
    mde = {str(n_tok): round(2.8 * win_sd / np.sqrt(n_tok / T), 4) for n_tok in (8192, 16384, 32768, 65536, 131072, 262144)}
    rel = {"base_mean": round(b.mean(), 4), "on_mean": round(on.mean(), 4), "delta_mean": round(d.mean(), 4),
           "rel_delta_pct": round(100 * d.mean() / b.mean(), 2),
           "boot95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
           "windows": W, "tok_per_window": T, "token_sd_delta": round(tok_sd, 4), "window_sd_delta": round(win_sd, 4),
           "design_effect": round(float(deff), 1), "frac_windows_improved": round(float((wm < 0).mean()), 3),
           "MDE_nats_at_n_tokens(80%power,a=.05)": mde,
           "frac_tokens_|delta|>0.1": round(float((np.abs(d) > 0.1).mean()), 4),
           "KL_proxy_note": "per-token NLL delta on the reference token, not full-distribution KL"}
    R[k] = rel
    print(k, json.dumps(rel))
json.dump(R, open(f"{OUT}/power.json", "w"), indent=1)

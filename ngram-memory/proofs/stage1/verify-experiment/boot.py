import json, numpy as np
R = json.load(open("verify.json")); b = dict(np.load("base_verify.npz"))
print("coherence", [round(c, 3) for c in R["top_rows"]["head_mean_direction_coherence"]], "rho", R["top_rows"]["corr_log_count_vs_norm"], R["top_rows"]["norm_quantiles"])
print("params_with_grad", R["params_with_grad_after_step1"], R["requires_grad_params"]["numel"])
rng = np.random.default_rng(0)
for k in ("heldout", "stdlib_py", "general_text", "train"):
    d = (np.load(f"nll_full_{k}.npy") - b[k]).mean(1)  # per-chunk deltas
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    print(k, "n_chunks", len(d), "mean %.4f" % d.mean(), "95%% CI [%.4f, %.4f]" % tuple(np.quantile(bs, [.025, .975])), "frac chunks improved %.2f" % (d < 0).mean())
d = (np.load("nll_exact_heldout.npy") - b["heldout"]).mean(1); print("exact-only heldout", d.mean())

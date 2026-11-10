"""Re-derive the §D.3 'calibrated gate arm' statistic with the fixed (ANOVA variance-component) estimator (VD-1).
Input: the S probe's per-token arrays (scratchpad nm/m0/dissect/S/per_token.npz). Output: s_within_anova.out."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ngram_memory.dissect.slice_probe import variance_decomposition, within_share_ci  # noqa: E402

z = np.load(sys.argv[1])
g = z["g"].astype(np.float64)
for name, grp in (("3gram_context", z["ctx3"]), ("2gram", z["ctx2"]), ("token", z["ids"])):
    rows = []
    for s in range(g.shape[1]):
        d = variance_decomposition(g[:, s], grp)
        est, lo, hi = within_share_ci(g[:, s], grp, n_boot=500, rng=np.random.default_rng(s))
        rows.append((round(d["within_share"], 4), round(lo, 4), round(hi, 4), round(d["within_share_raw"], 4)))
    print(f"groups={name} N={d['n']} G={d['groups']}  per stream (anova_within, CI lo, CI hi, raw SSW/SST): {rows}")
shares = [variance_decomposition(g[:, s], z["ctx3"])["within_share"] for s in range(g.shape[1])]
n_fire = sum(x >= 0.5 for x in shares)
print(f"Calibrated gate arm? rule: >= 50% of gate-logit variance within n-grams (3-gram context groups); "
      f"ANOVA within shares {[round(x, 4) for x in shares]} -> {'YES' if n_fire else 'NO'} ({n_fire}/4 streams)")

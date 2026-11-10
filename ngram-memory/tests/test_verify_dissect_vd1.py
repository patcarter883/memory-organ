"""[Integration status: every defect below is FIXED; the former strict-xfail tests are regular regression tests.]
Verifier test for the dissect piece (§D.3 experiment S, calibrated-gate decision rule).

VD-1: `variance_decomposition` / `within_share_ci` report the raw ratio SS_within / SS_total. With small groups
this is biased low: if a gate logit had NO n-gram effect at all (100% of its variance within n-grams), the ratio
is (N - G) / (N - 1), i.e. about 0.5 for groups of size 2 and 0.70 on the S data (N = 9428 tokens in 2863 groups).
So the ">= 50% within" rule of §D.3 cannot be applied to this statistic as written. (On the S data the verdict is
unchanged: an ANOVA variance-component estimate gives 0.0003 / 0.032 / 0.010 / 0.008 per stream;
scratchpad nm/m0/verify-dissect/s_recheck.out.)
"""
from __future__ import annotations

import numpy as np
import pytest

from ngram_memory.dissect.slice_probe import variance_decomposition, within_share_ci


def test_vd1_within_share_is_unbiased_when_all_variance_is_within_groups():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(3000), 2)          # 3000 n-grams seen twice each
    values = rng.normal(size=groups.size)           # no n-gram effect: every bit of variance is within n-grams
    share = variance_decomposition(values, groups)["within_share"]
    est, _, _ = within_share_ci(values, groups, n_boot=50, rng=rng)
    assert share > 0.9 and est > 0.9, (share, est)

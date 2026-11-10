"""S, part 3: turn the measured ratios (s_results.json) and host scales (s_host_scale.json) into the M1b / M2 graft
settings the §D.3 rules ask for. Pure arithmetic on the two JSON files; every formula is printed with its inputs.

* rho_FN: Flash-Next's own |PLE out_s| / |X_s| at the same site (median per stream; all-stream median), compared
  with the M1b/M2 rho grid {0.003, 0.01, 0.03}.
* G-lin value scale (derived, untested): out = 0.5 s W e with W orthonormal, so a row set of norm |e| injects
  0.5 s |e|. Choosing s so that a Flash-Next-sized row set (|e| = Flash-Next's median 16-head concat norm) injects
  rho * |h| gives s = 2 rho |h| / |e|_FN, |h| = rms(h) sqrt(d).
* G-dir alpha (current tap): two calibrations
    (sat)  §2.4's rule, assuming N_conv(u) is saturated (unit RMS): rms(silu(alpha z)) = (conv/X)_FN * rms(h)
           -- from s_host_scale.json;
    (eps)  the regime S actually finds for most Flash-Next tokens (mean(u^2) << eps): N_conv(u) ~ u / sqrt(eps),
           silu(x) ~ x/2, so conv/direct ~ alpha / (2 sqrt(eps)); matching Flash-Next's median conv/direct ratio r
           gives alpha = 2 sqrt(eps) r.
  and the |u| at which a G-lin/G-dir graft at budget rho leaves the eps regime: rms(u) = rho * rms(h) vs sqrt(eps).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EPS = 1e-6
D_HOST = {"qwen2.5-0.5b": 896, "qwen3.6-35b-a3b": 2048}


def main() -> None:
    s = json.loads((HERE / "s_results.json").read_text())
    h = json.loads((HERE / "s_host_scale.json").read_text())
    st = s["streams"]["all"]
    rho_fn = [x["out_over_x"]["median"] for x in st]
    cd = [x["conv_over_direct"]["median"] for x in st]
    e_fn = s["value"]["all"]["e_norm_median"]
    out = dict(rho_fn_per_stream=rho_fn, rho_fn_all_streams=s["out_over_x_all_streams"]["all"],
               rho_fn_over_grid={str(r): [x / r for x in rho_fn] for r in (0.003, 0.01, 0.03)},
               e_norm_fn_median=e_fn, conv_over_direct_fn=cd,
               gdir_alpha_eps_regime=[2 * math.sqrt(EPS) * r for r in cd],
               gdir_alpha_eps_regime_stream_median=2 * math.sqrt(EPS) * (sorted(cd)[1] + sorted(cd)[2]) / 2,
               conv_input_saturation_fn=s["conv_input_saturation"]["all"], hosts={})
    for host, d in D_HOST.items():
        if host not in h:
            continue
        rh = h[host]["all"]["h_rms_median"]
        norm_h = rh * math.sqrt(d)
        out["hosts"][host] = dict(
            d=d, h_rms_median=rh, h_norm_median=norm_h,
            glin_value_scale={str(r): 2 * r * norm_h / e_fn for r in (0.003, 0.01, 0.03)},
            u_rms_at_rho={str(r): r * rh for r in (0.003, 0.01, 0.03)},
            u_rms_over_sqrt_eps_at_rho={str(r): r * rh / math.sqrt(EPS) for r in (0.003, 0.01, 0.03)},
            gdir_alpha_saturated=h["recommendation_inputs"]["per_host"][host]["gdir_alpha_per_fn_stream"],
            gdir_alpha_saturated_stream_median=h["recommendation_inputs"]["per_host"][host]["gdir_alpha_stream_median"])
    (HERE / "s_recommend.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

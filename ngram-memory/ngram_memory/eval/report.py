"""Render evaluate.py summaries (JSON) as Markdown tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def _ci(x) -> str:
    return f"[{x[0]:+.2f}, {x[1]:+.2f}]" if x else ""


def condition_table(summary: Mapping) -> str:
    """One row per (set, condition): tokens, files, base / arm NLL, delta, rel % [file-cluster CI], EM delta,
    KL, wall-clock per 10k scored tokens."""
    rows = ["| set | condition | tokens | files | base NLL | arm NLL | Δ nats | rel % [95% file CI] | ΔEM pts | KL | "
            "s / 10k tok |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for set_name, conds in summary.get("sets", {}).items():
        if set_name == "r3":
            continue
        for cname, r in conds.items():
            if cname.startswith("_"):
                continue
            p = r.get("paired") or {}
            if cname == "base":
                rows.append(f"| {set_name} | base | {p.get('n_tokens', r.get('n_tokens', ''))} | {r.get('n_files', '')} | "
                            f"{r.get('nll', float('nan')):.4f} | | | | | | {r.get('wall_s_per_10k', float('nan')):.1f} |")
                continue
            if not p:
                continue
            kl = p.get("mean_kl")
            rows.append(
                f"| {set_name} | {cname} | {p['n_tokens']} | {p['n_files']} | {p['base_nll']:.4f} | {p['cond_nll']:.4f} | "
                f"{p['delta']:+.4f} | {p['rel_pct']:+.2f} {_ci(p.get('rel_pct_ci'))} | "
                f"{p.get('delta_em_pts', float('nan')):+.2f} | {'' if kl is None else f'{kl:.4f}'} | "
                f"{r.get('wall_s_per_10k', float('nan')):.1f} |")
    return "\n".join(rows)


def class_table(summary: Mapping, set_name: str, cond: str, keys=None) -> str:
    cb = summary["sets"][set_name][cond].get("classes") or {}
    keys = keys or sorted(cb)
    rows = ["| class | n | frac | base | Δ | rel % [CI] | share of gain |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        if k in cb:
            r = cb[k]
            rows.append(f"| {k} | {r['n_tokens']} | {r['frac']:.3f} | {r['base_nll']:.3f} | {r['delta']:+.4f} | "
                        f"{r['rel_pct']:+.2f} {_ci(r.get('rel_pct_ci'))} | {r['share_of_gain']:+.3f} |")
    return "\n".join(rows)


def r3_table(summary: Mapping) -> str:
    r3 = summary.get("sets", {}).get("r3") or {}
    rows = ["| condition | task | n | base acc | arm acc | Δ pts | McNemar p | arm / base |", "|---|---|---|---|---|---|---|---|"]
    for cname, r in r3.items():
        for task, v in (r.get("paired_mc") or {}).items():
            a = v["acc"]
            rows.append(f"| {cname} | {task} | {v['n']} | {a['base']:.3f} | {a['arm']:.3f} | {a['d_pts']:+.1f} | "
                        f"{a['mcnemar']['p']:.3g} | {a['ratio']:.3f} |")
    return "\n".join(rows)


def render(summary_path: str | Path) -> str:
    s = json.loads(Path(summary_path).read_text())
    out = [f"# Stage-1 eval: {s.get('host', '')}", "", condition_table(s)]
    if s.get("sets", {}).get("r3"):
        out += ["", "## R3 capability (forward-only MC)", "", r3_table(s)]
    for set_name, conds in s.get("sets", {}).items():
        for cname, r in conds.items():
            if isinstance(r, dict) and r.get("classes"):
                out += ["", f"## {set_name} / {cname}: token classes", "", class_table(s, set_name, cname)]
    return "\n".join(out) + "\n"


__all__ = ["condition_table", "class_table", "r3_table", "render"]

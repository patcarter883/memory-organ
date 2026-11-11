"""M1c on the M1b runner (experiments/m1b/run.py --grid experiments/m1c/grid.json) and the M1c bar arithmetic
(ngram_memory/eval/stage1/bars.py): grid loading and its sha256 check, arm steps / canary config, the primary-metric
switch, the M1c DAG under the scheduler, the selection tie rules and the decision. No model is run."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ngram_memory.eval import metrics as M
from ngram_memory.eval.stage1 import bars as BR

ROOT = Path(__file__).resolve().parents[1]
M1C = ROOT / "experiments/m1c/grid.json"
M1B = ROOT / "experiments/m1b/grid.json"


def load_run():
    spec = importlib.util.spec_from_file_location("m1c_run_t", ROOT / "experiments/m1b/run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _args(tmp_path, **kw):
    d = dict(out=str(tmp_path / "run"), smoke=False, threads=8, rss_limit_gb=11.0, stop_after=None, keep_going=False,
             parallel=1, max_stage_threads=0, launch_min_free_gb=0.0)
    d.update(kw)
    return SimpleNamespace(**d)


# ------------------------------------------------------------------------------------------------- grid
def test_default_grid_is_m1b_and_unchanged():
    R = load_run()
    assert R.GRID_PATH == R.DEFAULT_GRID and R.grid_args() == []
    g = R.load_grid(False)
    raw = json.loads(M1B.read_text())
    assert "reg_flags" not in g and g["budget"] == raw["budgets"]["full"] and g["eval_flags"] == raw["eval"]["full"]
    assert {k: v for k, v in g.items() if k not in ("budget", "eval_flags")} == raw


def test_m1c_grid_loads_and_its_sha_is_checked(tmp_path):
    R = load_run()
    run = R.Runner(_args(tmp_path, grid=str(M1C)))
    assert R.GRID_PATH == M1C.resolve() and R.grid_args() == ["--grid", str(M1C.resolve())]
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["grid_sha256"] == hashlib.sha256(M1C.read_bytes()).hexdigest() and prog["grid"] == str(M1C.resolve())
    g = run.grid
    assert g["primary"] == "nondup" and g["reg_flags"]["n_prefixes"] == 200
    kinds = set(run.kinds())
    names = [s["name"] for s in g["stages"]]
    for s in g["stages"]:
        assert s["kind"] in kinds and set(s.get("after", [])) <= set(names[:names.index(s["name"])])
    for n in ("N1_rho0p03", "N2_rho0p1", "N3_rho0p3", "D1_distill_rho0p03", "D2_distill_rho0p1"):
        a = R.resolve_arm(g, n, tmp_path)
        assert a["arm"] == "G-lin" and a["policy"] == "count" and a["k"] == 2 and a["lambda_l1_rel"] == 0.1
        assert a["lambda_l2"] == 1e-4 and a["seed"] == 0 and a["canaries"]["repeats"] == 2
        assert a["objective"] == ("distill" if n.startswith("D") else "nll")
    assert [R.resolve_arm(g, n, tmp_path)["rho"] for n in ("N1_rho0p03", "N2_rho0p1", "N3_rho0p3")] == [0.03, 0.1, 0.3]
    # F1 copies S1's recipe (objective, rho, canaries included) and trains one full pass
    sel = dict(config=dict(R.resolve_arm(g, "D2_distill_rho0p1", tmp_path)))
    (tmp_path / "S1_select").mkdir()
    (tmp_path / "S1_select/result.json").write_text(json.dumps(sel))
    f1 = R.resolve_arm(g, "F1_fullpass", tmp_path)
    assert f1["steps"] == "epoch" and f1["objective"] == "distill" and f1["rho"] == 0.1 and f1["name"] == "F1_fullpass"
    # a changed grid file is refused on resume
    R2 = load_run()
    gp = tmp_path / "grid.json"
    gp.write_text(M1C.read_text())
    R2.Runner(_args(tmp_path, out=str(tmp_path / "run2"), grid=str(gp)))
    gp.write_text(M1C.read_text().replace('"n_prefixes": 200', '"n_prefixes": 201'))
    with pytest.raises(SystemExit, match="changed"):
        load_run().Runner(_args(tmp_path, out=str(tmp_path / "run2"), grid=str(gp)))


def test_arm_steps_and_canary_config():
    R = load_run()
    B = dict(steps=64)
    assert R.arm_steps({}, B, 310) == 64 and R.arm_steps({"steps": 7}, B, 310) == 7
    assert R.arm_steps({"steps": "epoch"}, B, 310) == 310
    assert R.arm_steps({"steps": "epoch"}, dict(B, epoch_steps_cap=3), 2476) == 3
    from ngram_memory.ingest.canaries import CanaryConfig
    assert R.canary_config({}) is None and R.canary_config({"canaries": False}) is None
    assert R.canary_config({"canaries": True}) == CanaryConfig()
    assert R.canary_config({"canaries": {"seed": 0, "n": 8, "files": 2, "repeats": 2}}) == CanaryConfig(n=8, m=2, repeats=2)


def test_trim_metrics_on_resume(tmp_path):
    R = load_run()
    p = tmp_path / "metrics.jsonl"
    assert R.trim_metrics(p, 4) == 0                                  # no file: nothing to do
    recs = [dict(step=i, step_s=1.0) for i in range(6)]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    before = p.read_bytes()
    assert R.trim_metrics(p, 6) == 0 and p.read_bytes() == before     # nothing after the checkpoint: untouched
    assert R.trim_metrics(p, 4) == 2                                  # steps 4, 5 are re-run after the resume
    assert [json.loads(x)["step"] for x in p.read_text().splitlines()] == [0, 1, 2, 3]
    assert not (tmp_path / "metrics.jsonl.tmp").exists()


def test_dev_metrics_primary_switch():
    R = load_run()

    def p(rel, n=100, kl=None):
        d = dict(rel_pct=rel, rel_pct_ci=[rel - 0.1, rel + 0.1], n_tokens=n)
        if kl is not None:
            d["mean_kl"] = kl
        return d
    s = dict(sets=dict(dev={"x": dict(paired=p(-1.0), paired_nondup=p(-0.8, 90))}, r1={"x": dict(paired=p(0.1, kl=1e-3))},
                       r2={"x": dict(paired=p(0.2))}, r2ml={"x": dict(paired=p(0.4))}))
    a = R.dev_metrics(s, "x")
    assert a["e1_gain_pct"] == 1.0 and "primary" not in a and "e1_all_gain_pct" not in a       # M1b: unchanged
    b = R.dev_metrics(s, "x", "nondup")
    assert b["e1_gain_pct"] == pytest.approx(0.8) and b["e1_all_gain_pct"] == 1.0 and b["e1_n_tokens"] == 90
    assert b["r2_rel_pct"] == pytest.approx(0.3) and b["r1_mean_kl"] == 1e-3
    assert {k: v for k, v in b.items() if not k.startswith("e1") and k != "primary"} == \
        {k: v for k, v in a.items() if not k.startswith("e1")}


# --------------------------------------------------------------------------------------- the M1c DAG
def test_m1c_dag_runs_in_dependency_order_with_parallel(tmp_path, monkeypatch):
    R = load_run()
    a = _args(tmp_path, grid=str(M1C), parallel=4, threads=32, max_stage_threads=16)
    run = R.Runner(a)
    order, lock = [], threading.Lock()

    def stage(st):
        with lock:
            for d in st.get("after", []):
                assert (Path(a.out) / d / "result.json").exists(), f"{st['name']} started before {d}"
            order.append((st["name"], run.threads()))
        return dict(ok=True)
    monkeypatch.setattr(run, "kinds", lambda: {k: stage for k in R.Runner.kinds(run)})
    assert run.run() == 0
    names = [n for n, _ in order]
    assert set(names) == {s["name"] for s in run.grid["stages"]} and names[0] == "B0_base_dev"
    assert names.index("S1_select") > max(names.index(n) for n in ("N1_rho0p03", "D2_distill_rho0p1", "C0_countgrid"))
    assert names.index("T_test") > names.index("OP_select") > names.index("F1_fullpass") and names[-1] == "Z_report"
    assert all(t <= 16 for _, t in order)


# ---------------------------------------------------------------------------------------------- bars
def _targets(n_files=6, n=60, seed=0):
    g = np.random.default_rng(seed)
    base = g.uniform(1.0, 3.0, n_files * n)
    groups = np.repeat([f"f{i}" for i in range(n_files)], n)
    mask = g.random(base.size) > 0.1
    return BR.Targets(base, groups, np.arange(base.size), mask, Path("."), "dev"), g


def test_gain_and_diff_match_metrics_paired():
    t, g = _targets()
    a = t.base * (1 - 0.02 + 0.01 * g.standard_normal(t.base.size))
    b = t.base * (1 - 0.01 + 0.01 * g.standard_normal(t.base.size))
    m = t.mask
    ref = M.paired(t.base[m], a[m], t.groups[m])
    got = BR.gain(t, a)
    assert got["pts"] == pytest.approx(-ref["rel_pct"]) and got["ci"] == pytest.approx([-ref["rel_pct_ci"][1], -ref["rel_pct_ci"][0]])
    d = BR.diff(t, a, b)
    assert d["pts"] == pytest.approx(BR.gain(t, a)["pts"] - BR.gain(t, b)["pts"])
    assert d["ci"][0] < d["pts"] < d["ci"][1]
    # all-target primary: mask of ones gives paired() on every target
    ta = BR.Targets(t.base, t.groups, t.key, np.ones_like(m), Path("."), "dev")
    assert BR.gain(ta, a)["pts"] == pytest.approx(-M.paired(t.base, a, t.groups)["rel_pct"])


def test_bar_a_and_bar_b():
    t, g = _targets(n_files=12)
    noise = lambda s: 1 + 0.001 * g.standard_normal(t.base.size)  # noqa: E731
    counts = t.base * (1 - 0.02) * noise(0)
    plus = t.base * (1 - 0.035) * noise(1)          # 1.5 points above counts
    A = BR.bar_a(t, plus, counts, True)
    assert A["holds"] and A["diff"]["pts"] == pytest.approx(1.5, abs=0.1) and A["margin_pts"] == pytest.approx(0.5, abs=0.1)
    assert not BR.bar_a(t, plus, counts, False)["holds"]                       # retention of X + counts fails
    assert not BR.bar_a(t, t.base * (1 - 0.025) * noise(2), counts, True)["holds"]   # only 0.5 points
    none = BR.bar_a(t, None, counts, False, reason="no weight within the bars")   # no lambda_X: fails, says why
    assert not none["holds"] and none["margin_pts"] is None and none["reason"] == "no weight within the bars"
    x = t.base * (1 - 0.016) * noise(3)
    cm = t.base * (1 - 0.01) * noise(4)
    B = BR.bar_b(t, x, cm)
    assert B["holds"] and B["ratio"] == pytest.approx(1.6, abs=0.1) and B["diff"]["ci"][0] > 0
    assert not BR.bar_b(t, t.base * (1 - 0.012) * noise(5), cm)["holds"]     # ratio 1.2 < 1.5
    # E1(X) > 0 is required: X that hurts less than counts that hurt is not "beating counts"
    hurt_x, hurt_c = t.base * 1.001 * noise(6), t.base * 1.01 * noise(7)
    Bh = BR.bar_b(t, hurt_x, hurt_c)
    assert Bh["e1_x"]["pts"] < 0 and Bh["diff"]["pts"] > 0 and not Bh["ratio_ok"] and not Bh["holds"]
    # counts that hurt count as 0: X with a positive, significant gain holds
    Bp = BR.bar_b(t, x, hurt_c)
    assert Bp["ratio_ok"] and Bp["holds"] and Bp["ratio"] is None


def _pt(w, e1, r1, r2, kl):
    return dict(weight=w, e1=e1, r1=r1, r2=r2, kl=kl)


def test_lambda_star_share_and_matched():
    front = [_pt(0.0, 0.0, 0.0, 0.0, 0.0), _pt(1e-4, 0.25, 0.001, -0.003, 4e-5), _pt(5e-4, 0.6, 0.007, 0.006, 2e-4),
             _pt(2e-3, 1.2, 0.04, 0.08, 1e-3), _pt(7.5e-3, 2.0, 0.19, 0.45, 0.0049), _pt(1e-2, 2.3, 0.25, 0.6, 0.007)]
    assert BR.lambda_star(front)["weight"] == 7.5e-3                           # 1e-2 breaks R1, R2 and KL
    # share = max(R1/0.2, R2/0.5, KL/0.005), negative terms count 0
    assert BR.share(front[1]) == pytest.approx(max(0.001 / 0.2, 0.0, 4e-5 / 0.005))
    assert BR.share(dict(r1=-0.1, r2=-0.2, kl=0.0)) == 0.0 and BR.share(front[4]) == pytest.approx(0.98)
    assert BR.S_FLOOR == 0.01
    # X with negative or ~0 cost: the share floor (0.01) keeps weight 1e-4 (share 0.008) feasible
    m = BR.matched(front, dict(r1=-0.01, r2=0.001, kl=1e-5))
    assert m["weight"] == 1e-4 and m["threshold"] == 0.01 and m["x_share"] == pytest.approx(0.002)
    # one scalar budget share: X that spends 4% of the budget on R1 is matched by counts spending <= 4% anywhere,
    # although X's own KL is far below the counts' KL at that weight (the per-measure match would give 1e-4)
    m = BR.matched(front, dict(r1=0.009, r2=0.004, kl=9e-5))
    assert m["x_share"] == pytest.approx(0.045) and m["weight"] == 5e-4 and m["share"] == pytest.approx(0.04)
    assert BR.matched(front, dict(r1=0.05, r2=0.1, kl=1e-3))["weight"] == 2e-3
    assert BR.matched(front, dict(r1=0.0, r2=0.0, kl=0.0), s_floor=0.0)["weight"] == 0.0   # weight 0 always ok
    assert [BR.weight_tag(w) for w in (0.0075, 0.0, 1e-4, 0.03, 0.015)] == ["0p0075", "0", "0p0001", "0p03", "0p015"]
    from ngram_memory.eval.stage1.baselines.countlm import parse_weight
    assert all(parse_weight(BR.weight_tag(w)) == w for w in (0.0075, 0.0, 1e-4, 0.03, 2e-4, 0.002, 3e-4, 6e-3, 7e-3))


def test_lambda_x():
    plus = [_pt(0.0, 0.6, 0.01, 0.03, 1e-4), _pt(1e-3, 1.4, 0.03, 0.07, 6e-4), _pt(5e-3, 2.3, 0.13, 0.25, 0.0026),
            _pt(7.5e-3, 2.6, 0.2, 0.36, 0.004), _pt(1e-2, 2.8, 0.27, 0.49, 0.0055)]
    lx = BR.lambda_x(plus, True)
    assert lx["weight"] == 7.5e-3 and lx["point"]["e1"] == 2.6 and lx["reason"] is None   # R1 0.2 is within
    assert [r["within"] for r in lx["table"]] == [True, True, True, True, False]
    # ties -> the smaller weight
    assert BR.lambda_x([_pt(0.0, 1.0, 0, 0, 0), _pt(1e-3, 1.0, 0, 0, 0)], True)["weight"] == 0.0
    # weight 0 (X alone) qualifies only if X alone is within the bars, whatever its grid row says
    assert BR.lambda_x([_pt(0.0, 3.0, 0, 0, 0), _pt(1e-3, 1.0, 0, 0, 0)], False)["weight"] == 1e-3
    none = BR.lambda_x([_pt(0.0, 3.0, 0.5, 0, 0), _pt(1e-3, 3.1, 0.6, 0, 0)], False)
    assert none["weight"] is None and none["point"] is None and "within" in none["reason"]


def test_per_file():
    t, g = _targets(n_files=4, n=50)
    t.groups = t.groups.astype(object)
    t.groups[:120] = "big"                                    # one file holds most targets
    a = t.base * 0.98
    pf = BR.per_file(t, a, t.base * 0.99)
    assert pf["files"][0]["file"] == "big" and sum(f["target_share"] for f in pf["files"]) == pytest.approx(1.0)
    assert all(f["pts"] == pytest.approx(1.0) for f in pf["files"]) and 1.0 < pf["n_eff"] < 4.0


# ------------------------------------------------------------------------------------- selection / decision
def _row(stage, rho, obj, margin, within=True, a=False, b=False, lx=0.0075):
    return dict(stage=stage, config=dict(name=stage, rho=rho, objective=obj), dev={}, cartridge=None,
                x=dict(e1=1.0, e1_all=1.0, r1=0.0, kl=0.0, r2=0.3 if within else 0.9), x_within=within, x_share=0.6,
                lambda_x=dict(weight=lx, reason=None if lx is not None else "none within"),
                x_plus_counts=dict(), x_plus_counts_within=lx is not None, counts_matched=dict(weight=5e-4),
                bar_a=dict(diff=dict(pts=1.0 + margin, ci=[0, 0]) if margin is not None else None, holds=a,
                           margin_pts=margin),
                bar_b=dict(ratio=1.0, diff=dict(ci=[0, 0]), holds=b), bar_a_margin_pts=margin, any_bar=a or b)


def test_m1c_select_tie_rules(tmp_path, monkeypatch):
    R = load_run()
    run = R.Runner(_args(tmp_path, grid=str(M1C)))
    st = dict(name="S1_select", candidates=[], counts="C0")
    rows = [_row("N3", 0.3, "nll", 0.9, within=False), _row("D1", 0.03, "distill", 0.4), _row("N1", 0.03, "nll", 0.4),
            _row("N2", 0.1, "nll", 0.4)]
    monkeypatch.setattr(run, "_m1c_rows", lambda names, c, w: (dict(lambda_star=dict(weight=0.0075)), rows))
    r = run.stage_m1c_select(st)
    assert r["selected"] == "N1" and r["any_within_bars"] and not r["any_bar_dev"]   # N3 is outside the bars
    # a bar that holds outranks a larger margin: Bar B holding (margin -0.6) beats no bar (margin 0.4)
    rows[:] = [_row("N1", 0.03, "nll", 0.4), _row("D2", 0.1, "distill", -0.6, b=True)]
    assert run.stage_m1c_select(st)["selected"] == "D2"
    # Bar A holding outranks Bar B only, whatever the margins
    rows[:] = [_row("N2", 0.1, "nll", 0.1, a=True), _row("D2", 0.1, "distill", 0.5, b=True)]
    assert run.stage_m1c_select(st)["selected"] == "N2"
    # no lambda_X (margin None) ranks below any margin
    rows[:] = [_row("N2", 0.1, "nll", None, lx=None), _row("N3", 0.3, "nll", -0.9)]
    r = run.stage_m1c_select(st)
    assert r["selected"] == "N3" and r["table"][0]["bar_a_diff"] is None and r["table"][0]["lambda_x"] is None
    rows[:] = [_row("N3", 0.3, "nll", 0.9, within=False), _row("N2", 0.1, "nll", 0.2, within=False)]
    r = run.stage_m1c_select(st)
    assert r["selected"] == "N2" and not r["any_within_bars"]                    # fallback: smallest R2 (tie -> rho)


def test_m1c_op_select(tmp_path, monkeypatch):
    R = load_run()
    run = R.Runner(_args(tmp_path, grid=str(M1C)))
    out = Path(tmp_path / "run")
    st = next(s for s in run.grid["stages"] if s["name"] == "OP_select")
    assert st["optional_after"] == ["F1_fullpass"] and "OP_select" not in run.soft and "F1_fullpass" in run.soft
    (out / "S1_select").mkdir(parents=True)
    (out / "S1_select/result.json").write_text(json.dumps(dict(selected="N1")))
    rows = {}
    monkeypatch.setattr(run, "_m1c_rows", lambda names, c, w: (dict(lambda_star=dict(weight=0.0075)),
                                                                [rows[n] for n in names]))
    # F1 failed (not RSS-aborted): OP = S1 and the rule says why
    run.mark("F1_fullpass", status="failed", aborted=False, error=dict(returncode=1, aborted=None))
    rows["N1"] = _row("N1", 0.03, "nll", 0.2, a=True)
    r = run.stage_m1c_op(st)
    assert r["selected"] == "N1" and r["rule"] == "F1 failed: failed: returncode 1" and r["f1"] is None
    (out / "F1_fullpass").mkdir()
    (out / "F1_fullpass/result.json").write_text("{}")
    # F1 with a larger margin but no bar does not replace S1 with a bar (the tuple, not the margin, decides)
    rows["F1_fullpass"] = _row("F1_fullpass", 0.03, "nll", 0.5)
    assert run.stage_m1c_op(st)["selected"] == "N1"
    rows["F1_fullpass"] = _row("F1_fullpass", 0.03, "nll", 0.2, a=True)              # equal tuple -> F1
    assert run.stage_m1c_op(st)["selected"] == "F1_fullpass"
    rows["N1"] = _row("N1", 0.03, "nll", -0.3)
    rows["F1_fullpass"] = _row("F1_fullpass", 0.03, "nll", -0.5, b=True)            # F1 holds Bar B, S1 nothing
    assert run.stage_m1c_op(st)["selected"] == "F1_fullpass"


def test_m1c_report_decisions(tmp_path):
    R = load_run()
    run = R.Runner(_args(tmp_path, grid=str(M1C)))
    out = Path(tmp_path / "run")

    def put(name, obj):
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "result.json").write_text(json.dumps(obj))

    def drop(name):
        (out / name / "result.json").unlink()
    st = dict(name="Z_report", select="S1", operating_point="OP", test="T", reg="REG")
    rows = [_row("N1", 0.03, "nll", -0.5)]
    put("S1", dict(candidates=rows, lambda_star=dict(weight=0.0075), table=[], selected="N1"))
    op_row = _row("N1", 0.03, "nll", -0.5)
    put("OP", dict(selected="N1", rule="F1 failed: failed: x", s1=op_row, f1=None, op_row=op_row))
    r = run.stage_m1c_report(st)
    assert r["decision"] == "KILL"                         # dev-only verdict: no test, no REG result needed
    rows = [_row("N1", 0.03, "nll", 0.2, a=True)]
    put("S1", dict(candidates=rows, lambda_star=dict(weight=0.0075), table=[], selected="N1"))
    op_row = rows[0]
    put("OP", dict(selected="N1", rule="F1 has no result", s1=op_row, f1=None, op_row=op_row))
    # bars on dev, T_test failed: INCOMPLETE with the reason from progress.json
    run.mark("T", status="failed", error="Traceback ...\nRuntimeError: boom\n")
    r = run.stage_m1c_report(st)
    assert r["decision"] == "INCOMPLETE" and r["reason"] == "T has no result (failed: RuntimeError: boom)"

    def test_res(a, b, c1):
        return dict(primary="nondup", class_fractions={"dup:16gram_in_train": 0.038},
                    bars=dict(nondup=dict(bar_a=dict(holds=a), bar_b=dict(holds=b), c1=dict(holds=c1))))
    put("T", test_res(False, True, True))
    run.mark("REG", status="failed", error=dict(returncode=1, aborted=None))
    r = run.stage_m1c_report(st)                           # REG failed: INCOMPLETE, the test parts still reported
    assert r["decision"] == "INCOMPLETE" and r["reason"] == "REG has no result (failed: returncode 1)"
    assert r["parts"]["test_bar_b"] and r["parts"]["reg"] is None
    put("REG", dict(verdict={}, reg_pass=True))
    r = run.stage_m1c_report(st)
    assert r["decision"] == "PASS" and r["parts"]["test_bar_b"] and r["gates"]["dedupe"]["met"]
    put("REG", dict(verdict={}, reg_pass=False))
    assert run.stage_m1c_report(st)["decision"] == "REPORT"
    put("REG", dict(verdict={}, reg_pass=True))
    put("T", test_res(False, False, True))
    assert run.stage_m1c_report(st)["decision"] == "REPORT"
    put("T", test_res(True, False, False))                                     # C1 fails
    assert run.stage_m1c_report(st)["decision"] == "REPORT"
    drop("REG")
    put("T", test_res(True, False, True))
    assert run.stage_m1c_report(st)["decision"] == "INCOMPLETE"


def test_m1c_soft_stages_and_failed_f1_dag(tmp_path, monkeypatch):
    """F1, T_test and REG_op are soft (only optional dependants): a failed F1 does not block OP_select, failed
    T_test/REG_op do not block Z_report, and the run's exit code stays 0; S1 failing still stops the run."""
    R = load_run()
    for par in (1, 3):
        a = _args(tmp_path, out=str(tmp_path / f"r{par}"), grid=str(M1C), parallel=par, threads=6)
        run = R.Runner(a)
        assert run.soft == {"F1_fullpass", "T_test", "REG_op"}
        ran = []

        def stage(st, run=run, ran=ran):
            ran.append(st["name"])
            if st["name"] in ("F1_fullpass", "T_test", "REG_op"):
                raise R.StageFailed(st["name"], dict(error="x"))
            return dict(ok=True)
        monkeypatch.setattr(run, "kinds", lambda stage=stage, run=run: {k: stage for k in R.Runner.kinds(run)})
        assert run.run() == 0
        assert {"OP_select", "Z_report"} <= set(ran)
        prog = json.loads((Path(a.out) / "progress.json").read_text())
        assert sorted(prog["soft_failed"]) == ["F1_fullpass", "REG_op", "T_test"] and prog["finished"] is False
    a = _args(tmp_path, out=str(tmp_path / "hard"), grid=str(M1C), parallel=3, threads=6)
    run = R.Runner(a)
    ran = []

    def stage2(st):
        ran.append(st["name"])
        if st["name"] == "S1_select":
            raise R.StageFailed(st["name"], dict(error="x"))
        return dict(ok=True)
    monkeypatch.setattr(run, "kinds", lambda: {k: stage2 for k in R.Runner.kinds(run)})
    assert run.run() == 1 and "F1_fullpass" not in ran and "OP_select" not in ran


@pytest.mark.parametrize("lx,expect", [(0.005, "countlm:kn-lam0p005+rows:"), (0.0075, None), (0.0, None),
                                       (None, None)])
def test_m1c_test_conditions_and_started_marker(tmp_path, monkeypatch, lx, expect):
    """T_test adds counts@lambda_X(OP) + OP only when lambda_X != lambda* (0: OP alone; none: no condition), and
    writes STARTED only right before the evaluation (after OP is read and the canary-free check)."""
    R = load_run()
    run = R.Runner(_args(tmp_path, grid=str(M1C)))
    out = Path(tmp_path / "run")
    st = next(s for s in run.grid["stages"] if s["name"] == "T_test")
    (out / "N1").mkdir(parents=True)
    op_row = _row("N1", 0.03, "nll", 0.2, a=True, lx=lx)
    (out / "OP_select").mkdir()
    (out / "OP_select/result.json").write_text(json.dumps(dict(selected="N1", config={}, lambda_star=dict(weight=0.0075),
                                                               op_row=op_row)))
    seen = {}

    class Stop(Exception):
        pass

    def fake_eval(stage, sets, conds, **kw):
        seen.update(conds=conds.split(","), started=(out / "T_test/STARTED").exists())
        raise Stop
    monkeypatch.setattr(run, "evaluate", fake_eval)
    with pytest.raises(Stop):
        run.stage_m1c_test(st)
    conds = seen["conds"]
    assert seen["started"] and len(conds) == len(set(conds))
    link = str(out / "T_test/op.pt")
    base = ["base", f"rows:{link}", "countlm:kn-lam0p0075", f"countlm:kn-lam0p0075+rows:{link}",
            "countlm:kn-lam0p0005", f"perm:{link}"]
    assert [c for c in conds if c != (expect or "") + link] == base
    assert (expect + link in conds) if expect else len(conds) == 6
    # the marker is not written when the stage fails before the read (e.g. OP has no result)
    (out / "T_test/STARTED").unlink()
    (out / "OP_select/result.json").unlink()
    with pytest.raises(TypeError):
        run.stage_m1c_test(st)
    assert not (out / "T_test/STARTED").exists()

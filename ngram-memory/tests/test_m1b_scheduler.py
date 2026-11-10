"""experiments/m1b/run.py stage scheduling: serial order unchanged, --parallel respects dependencies, splits the
threads, stops cleanly. The stages are stand-ins (no training)."""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_run():
    spec = importlib.util.spec_from_file_location("m1b_run_t", ROOT / "experiments/m1b/run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# the M1b DAG shape: B0 -> {A1, A2, A3, L} ; S1 <- A1..A3 ; A4 <- S1 ; Z <- A4, L
DAG = [dict(name="B0", kind="k"), dict(name="A1", kind="k", after=["B0"]), dict(name="A2", kind="k", after=["B0"]),
       dict(name="A3", kind="k", after=["B0"]), dict(name="S1", kind="k", after=["A1", "A2", "A3"]),
       dict(name="A4", kind="k", after=["S1"]), dict(name="L", kind="k", after=["B0"]),
       dict(name="Z", kind="k", after=["A4", "L"])]


def make(tmp_path, monkeypatch, *, parallel=1, threads=32, cap=0, stop_after=None, keep_going=False,
         fail=(), durations=None):
    R = load_run()
    monkeypatch.setattr(R, "load_grid", lambda smoke: dict(stages=[dict(s) for s in DAG]))
    a = SimpleNamespace(out=str(tmp_path / "run"), smoke=False, threads=threads, rss_limit_gb=11.0,
                        stop_after=stop_after, keep_going=keep_going, parallel=parallel, max_stage_threads=cap,
                        launch_min_free_gb=0.0)
    run = R.Runner(a)
    ev, lock, live = [], threading.Lock(), set()
    durations = durations or {}

    def stage(st):
        n = st["name"]
        with lock:
            for d in st.get("after", []):
                assert (Path(a.out) / d / "result.json").exists(), f"{n} started before {d} finished"
            live.add(n)
            ev.append(("start", n, run.threads(), frozenset(live)))
        time.sleep(durations.get(n, 0.05))
        with lock:
            live.discard(n)
            ev.append(("end", n))
        if n in fail:
            raise R.StageFailed(n, dict(error="x"))
        return dict(ok=True)

    monkeypatch.setattr(run, "kinds", lambda: dict(k=stage))
    return R, run, ev


def starts(ev):
    return [e[1] for e in ev if e[0] == "start"]


def test_serial_order_unchanged(tmp_path, monkeypatch):
    _, run, ev = make(tmp_path, monkeypatch)
    assert run.run() == 0
    assert starts(ev) == [s["name"] for s in DAG]
    assert all(e[2] == 32 and len(e[3]) == 1 for e in ev if e[0] == "start")
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["finished"] is True and prog["stages"]["A1"]["threads"] == 32


def test_parallel_respects_dependencies_and_splits_threads(tmp_path, monkeypatch):
    _, run, ev = make(tmp_path, monkeypatch, parallel=4, durations=dict(L=0.6))
    assert run.run() == 0
    th = {e[1]: e[2] for e in ev if e[0] == "start"}
    assert th["B0"] == 32                                   # alone
    assert th["A1"] == th["A2"] == th["A3"] == th["L"] == 8  # four started together share 32
    assert th["S1"] == 24                                   # L (8 threads) still running
    peak = max(len(e[3]) for e in ev if e[0] == "start")
    assert peak == 4
    assert all((tmp_path / "run" / s["name"] / "result.json").exists() for s in DAG)


def test_parallel_cap_and_slots(tmp_path, monkeypatch):
    _, run, ev = make(tmp_path, monkeypatch, parallel=3, cap=10)
    assert run.run() == 0
    th = {e[1]: e[2] for e in ev if e[0] == "start"}
    assert th["B0"] == 10                                   # capped
    assert max(len(e[3]) for e in ev if e[0] == "start") <= 3
    assert all(t <= 10 for t in th.values())


def test_parallel_stop_after_lets_running_finish(tmp_path, monkeypatch):
    _, run, ev = make(tmp_path, monkeypatch, parallel=4, stop_after="A1", durations=dict(A2=0.4, A3=0.4, L=0.4))
    assert run.run() == 0
    s = starts(ev)
    assert "S1" not in s and "A4" not in s and "Z" not in s
    for n in ("A2", "A3", "L"):                              # started with A1, allowed to finish
        assert ("end", n) in ev


def test_parallel_failure_stops_new_stages(tmp_path, monkeypatch):
    _, run, ev = make(tmp_path, monkeypatch, parallel=4, fail=("A2",))
    assert run.run() == 1
    assert "S1" not in starts(ev)
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["stages"]["A2"]["status"] == "failed" and prog["finished"] is False


def test_parallel_resume_skips_done(tmp_path, monkeypatch):
    _, run, ev = make(tmp_path, monkeypatch, parallel=4, stop_after="S1")
    assert run.run() == 0
    _, run2, ev2 = make(tmp_path, monkeypatch, parallel=4)
    assert run2.run() == 0
    assert set(starts(ev2)) == {"A4", "Z"} | ({"L"} - set(starts(ev)))

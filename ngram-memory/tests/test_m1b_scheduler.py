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
         fail=(), durations=None, dag=None, error=(), abort=()):
    R = load_run()
    dag = DAG if dag is None else dag
    monkeypatch.setattr(R, "load_grid", lambda smoke: dict(stages=[dict(s) for s in dag]))
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
                if d in st.get("optional_after", []):    # optional: finished in any terminal state
                    assert (Path(a.out) / d / "result.json").exists() or \
                        run.progress["stages"][d]["status"] in ("failed", "aborted", "failed_final"), f"{n} started before {d}"
                    continue
                assert (Path(a.out) / d / "result.json").exists(), f"{n} started before {d} finished"
            live.add(n)
            ev.append(("start", n, run.threads(), frozenset(live)))
        time.sleep(durations.get(n, 0.05))
        with lock:
            live.discard(n)
            ev.append(("end", n))
        if n in fail:
            raise R.StageFailed(n, dict(error="x"))
        if n in abort:                                   # as the RSS guard reports it
            raise R.StageFailed(n, dict(aborted="rss 12.0 GB > 11.0 GB"))
        if n in error:
            raise RuntimeError(f"code error in {n}")
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


# optional dependencies (M1c): S -> F (soft: its only dependant lists it in optional_after) -> O ; O -> T, G -> Z
# (T and G optional for Z); H is a hard dependency of O
OPT = [dict(name="S", kind="k"), dict(name="F", kind="k", after=["S"]), dict(name="H", kind="k", after=["S"]),
       dict(name="O", kind="k", after=["S", "F", "H"], optional_after=["F"]),
       dict(name="T", kind="k", after=["O"]), dict(name="G", kind="k", after=["O"]),
       dict(name="Z", kind="k", after=["T", "G"], optional_after=["T", "G"])]


def test_soft_stages():
    R = load_run()
    assert R.soft_stages(OPT) == {"F", "T", "G"}
    assert R.soft_stages(DAG) == set()                                  # M1b: no optional dependencies
    # a stage with one hard and one optional dependant is not soft
    assert R.soft_stages(OPT + [dict(name="X", kind="k", after=["F"])]) == {"T", "G"}


def test_optional_after_must_be_in_after(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="optional_after"):
        make(tmp_path, monkeypatch, dag=[dict(name="A", kind="k"), dict(name="B", kind="k", optional_after=["A"])])


@pytest.mark.parametrize("parallel", [1, 3])
@pytest.mark.parametrize("how", ["fail", "error"])
def test_failed_optional_dependency_lets_dependant_run(tmp_path, monkeypatch, parallel, how):
    kw = dict(fail=("F", "T")) if how == "fail" else dict(error=("F", "T"))
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, **kw)
    assert run.run() == 0                         # exit code: soft failures do not set it
    assert set(starts(ev)) == {"S", "F", "H", "O", "T", "G", "Z"}
    assert starts(ev).index("O") > starts(ev).index("F") and starts(ev)[-1] == "Z"
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert sorted(prog["soft_failed"]) == ["F", "T"] and prog["finished"] is False
    # their dependants finished without them: final after one failed attempt, and the run is complete
    assert prog["stages"]["F"]["status"] == "failed_final" and prog["stages"]["F"]["attempts"] == 1
    assert prog["stages"]["T"]["status"] == "failed_final" and prog["complete"] is True
    assert "attempts" not in prog["stages"]["O"]                     # counted for soft stages only


@pytest.mark.parametrize("parallel", [1, 3])
def test_failed_hard_dependency_still_blocks(tmp_path, monkeypatch, parallel):
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, fail=("H",))
    assert run.run() == 1                         # exit code: a failed non-soft stage sets it
    assert "O" not in starts(ev) and "Z" not in starts(ev)
    _, run, ev = make(tmp_path / "k", monkeypatch, parallel=parallel, dag=OPT, fail=("H",), keep_going=True)
    assert run.run() == 0 and "O" not in starts(ev)                  # --keep-going: still blocked
    prog = json.loads((tmp_path / "k/run/progress.json").read_text())
    assert prog["stages"]["O"]["status"] == "blocked" and prog["stages"]["O"]["waiting_for"] == ["H"]
    assert "soft_failed" not in prog


@pytest.mark.parametrize("parallel", [1, 3])
def test_failed_optional_dependency_resume(tmp_path, monkeypatch, parallel):
    # F fails and the run stops before O (H fails hard): the resume retries F once
    _, run, _ = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, fail=("F", "H"))
    assert run.run() == 1
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, fail=("F",))
    assert run.run() == 0 and "F" in starts(ev) and "O" in starts(ev)
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["stages"]["F"]["status"] == "failed_final" and prog["stages"]["F"]["attempts"] == 2
    # failed for good: a further resume does not run it again
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT)
    assert run.run() == 0 and starts(ev) == []
    # a soft stage whose dependant already finished without it is final after one failure
    _, run, _ = make(tmp_path / "b", monkeypatch, parallel=parallel, dag=OPT, fail=("T",))
    assert run.run() == 0
    _, run, ev = make(tmp_path / "b", monkeypatch, parallel=parallel, dag=OPT)
    assert run.run() == 0 and "T" not in starts(ev)
    prog = json.loads((tmp_path / "b/run/progress.json").read_text())
    assert prog["stages"]["T"]["status"] == "failed_final" and prog["stages"]["T"]["attempts"] == 1


@pytest.mark.parametrize("parallel", [1, 3])
@pytest.mark.parametrize("which", ["F", "T"])
def test_aborted_soft_stage_not_rerun_after_dependant_done(tmp_path, monkeypatch, parallel, which):
    # an RSS-aborted soft stage: its dependant runs without it, and once that is done no resume starts it again
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, abort=(which,))
    assert run.run() == 0 and starts(ev)[-1] == "Z"
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["soft_failed"] == [which] and prog["stages"][which]["status"] == "failed_final"
    assert prog["stages"][which]["aborted"] is True and prog["stages"][which]["attempts"] == 1
    assert prog["complete"] is True and prog["finished"] is False
    for _ in range(2):
        _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT)
        assert run.run() == 0 and starts(ev) == []


@pytest.mark.parametrize("parallel", [1, 3])
def test_aborted_soft_stage_retried_once(tmp_path, monkeypatch, parallel):
    # F aborted and the run stops before O (H fails hard): the resume retries F; a second abort is final
    _, run, _ = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, abort=("F",), fail=("H",))
    assert run.run() == 1
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["stages"]["F"]["status"] == "aborted" and prog["stages"]["F"]["attempts"] == 1
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, abort=("F",))
    assert run.run() == 0 and "F" in starts(ev) and "O" in starts(ev)
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["stages"]["F"]["status"] == "failed_final" and prog["stages"]["F"]["attempts"] == 2


@pytest.mark.parametrize("parallel", [1, 3])
def test_interrupted_attempt_does_not_count(tmp_path, monkeypatch, parallel):
    # F cut short (kill -9 / CI cancel: status running, no failure recorded), then a genuine failure on resume:
    # that is its first failed attempt, so F is retried on the next resume
    _, run, _ = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, fail=("H",))
    assert run.run() == 1
    (tmp_path / "run/F/result.json").unlink()
    run.mark("F", status="running")
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT, fail=("F", "H"))
    assert run.run() == 1 and "F" in starts(ev)
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["stages"]["F"]["status"] == "failed" and prog["stages"]["F"]["attempts"] == 1
    _, run, ev = make(tmp_path, monkeypatch, parallel=parallel, dag=OPT)
    assert run.run() == 0 and "F" in starts(ev)
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert prog["stages"]["F"]["status"] == "done" and prog["finished"] is True and prog["complete"] is True


def test_m1b_progress_has_no_soft_fields(tmp_path, monkeypatch):
    _, run, _ = make(tmp_path, monkeypatch, parallel=3)
    assert run.run() == 0
    prog = json.loads((tmp_path / "run/progress.json").read_text())
    assert "complete" not in prog and "soft_failed" not in prog
    assert all("attempts" not in e for e in prog["stages"].values())

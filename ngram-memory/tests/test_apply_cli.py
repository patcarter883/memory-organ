"""scripts/apply.py end to end, run with torch BLOCKED (the patch / cartridge / ledger path is numpy-only)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ngram_memory.cartridge import CartridgeInfo, make_cartridge_patch, write_graft_file
from ngram_memory.patch import make_patch, read_patch, write_patch
from ngram_memory.spec import FormatSpec

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "apply.py"


def run(*args, ok=True):
    code = ("import sys, runpy; sys.modules['torch'] = None; sys.argv = sys.argv[1:]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code, str(CLI), *map(str, args)], capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)
    assert r.returncode != 0
    return r


def test_cli_format_ledger_compile_verify(tmp_path):
    fj = tmp_path / "fmt.json"
    out = run("format", "--hidden", 128, "--key", "flash-next", "--out", fj)
    fmt = FormatSpec.from_json(fj.read_text())
    assert out["row_width"] == 8 and out["fingerprint"] == fmt.fingerprint()
    rng = np.random.default_rng(0)
    ids_a = rng.choice(fmt.padded_rows, 50, replace=False)
    ids_b = np.concatenate([ids_a[:10], rng.choice(fmt.padded_rows, 5, replace=False)])
    pa = make_patch(fmt, ids_a, rng.standard_normal((50, 8)), edits=[dict(edit_id="fact-a")])
    pb = make_patch(fmt, np.unique(ids_b), rng.standard_normal((np.unique(ids_b).size, 8)),
                    edits=[dict(edit_id="fact-b")])
    write_patch(str(tmp_path / "a.safetensors"), pa)
    write_patch(str(tmp_path / "b.safetensors"), pb)
    L = tmp_path / "ledger"
    run("ledger", "init", L, "--format", fj)
    run("ledger", "add", L, tmp_path / "a.safetensors")
    r = run("ledger", "add", L, tmp_path / "b.safetensors")
    assert r["edit_id"] == "fact-b" and r["conflict_set"] == ["fact-a"]
    show = run("ledger", "show", L)
    assert show["active"] == ["fact-a", "fact-b"] and show["active_rows"] == np.union1d(ids_a, ids_b).size
    r = run("ledger", "remove", L, "fact-a")
    assert r["revalidate"] == ["fact-b"]
    run("ledger", "export", L, "--out", tmp_path / "only-b.safetensors")
    ob = read_patch(str(tmp_path / "only-b.safetensors"))
    assert np.array_equal(ob.row_ids, pb.row_ids) and np.array_equal(ob.values, pb.values)
    # cartridge chain compile + stats; graft pin check
    wv = np.linalg.qr(rng.standard_normal((128, 128)))[0].astype(np.float32)
    gm = write_graft_file(str(tmp_path / "graft.safetensors"), fmt, wv, host="tiny")
    c0 = make_cartridge_patch(fmt, pa.row_ids, pa.values, CartridgeInfo("ns", 0), graft=gm)
    c1 = make_cartridge_patch(fmt, pb.row_ids, pb.values, CartridgeInfo("ns", 1, kind="increment", parent_version=0),
                              graft=gm, reset_row_ids=ids_a[40:])
    write_patch(str(tmp_path / "c0.safetensors"), c0)
    write_patch(str(tmp_path / "c1.safetensors"), c1)
    r = run("compile", tmp_path / "c0.safetensors", tmp_path / "c1.safetensors", "--out", tmp_path / "served.safetensors",
            "--graft", tmp_path / "graft.safetensors", "--stats", tmp_path / "stats.json")
    assert r["version"] == 1 and r["rows"] == np.union1d(np.setdiff1d(ids_a, ids_a[40:]), pb.row_ids).size
    assert "int6_g32" not in json.loads((tmp_path / "stats.json").read_text())  # width 8 is not a group-32 width
    assert run("verify", tmp_path / "c1.safetensors", "--format", fj, "--graft", tmp_path / "graft.safetensors")["ok"]
    # refusals: other format, chain out of order
    run("format", "--hidden", 128, "--key", "flash-next", "--arm", "G-dir", "--out", tmp_path / "fmt2.json")
    run("verify", tmp_path / "a.safetensors", "--format", tmp_path / "fmt2.json", ok=False)
    run("compile", tmp_path / "c1.safetensors", tmp_path / "c0.safetensors", "--out", tmp_path / "x.safetensors",
        ok=False)
    info = run("info", tmp_path / "c1.safetensors")
    assert info["cartridge"]["parent_version"] == 0 and info["n_reset"] == 10

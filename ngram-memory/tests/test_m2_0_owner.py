"""M2-0 owner run (experiments/m2_0/run_owner.py) and its new steps, on CPU: the preflight refusals (serve up,
GPU held, cards leased, no gpu-lease), checkpoint resolution, the criteria evaluation on synthetic step outputs,
the planned-split step (split_peak.py) on a synthetic AWQ checkpoint in the real on-disk format, the replicated
reader, and the real-shape grouped_mm probe at a tiny size."""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "m2_0"))

import run_owner as ro  # noqa: E402
import split_peak as sp  # noqa: E402
from ngram_memory.backends.graft import host_quant as hq  # noqa: E402
from test_awq_layer import _make_ckpts  # noqa: E402


# ------------------------------------------------------------------------------------------------------------
# preflight
# ------------------------------------------------------------------------------------------------------------

def _pf(**kw):
    base = dict(ports=(1919,), me={1}, cmdlines={}, kfd=[], docker=[], port_check=lambda p: False)
    base.update(kw)
    return ro.preflight(base.pop("ports"), **base)


def test_preflight_clean_box_is_ok():
    r = _pf(cmdlines={10: "python train.py", 11: "vim /home/x/vllm-gfx1201/README.md"})
    assert r["ok"] and not r["refusals"]


@pytest.mark.parametrize("kw,needle", [
    (dict(port_check=lambda p: p == 1919), "port"),
    (dict(cmdlines={42: "python -m minisgl.server.launch --model x"}), "serve-like processes"),
    (dict(cmdlines={43: "bash tools/serve.sh"}), "serve-like processes"),
    (dict(cmdlines={44: "python -m vllm.entrypoints.openai.api_server"}), "serve-like processes"),
    (dict(docker=["proj-serve-1\tminisgl:latest\t0.0.0.0:1919->1919/tcp\tbash"]), "containers"),
    (dict(kfd=[77], cmdlines={77: "python bench.py"}), "hold the GPU"),
])
def test_preflight_refuses(kw, needle):
    r = _pf(**kw)
    assert not r["ok"] and any(needle in x for x in r["refusals"]), r


def test_preflight_ignores_itself_and_lease_monitors():
    r = _pf(kfd=[5, 6, 7], cmdlines={5: "gpu-lease -n 2 -- python run_owner.py --_inner",
                                     6: "python /x/experiments/m2_0/run_owner.py --_inner", 7: "amd-smi monitor"})
    assert r["ok"], r
    assert _pf(kfd=[9], cmdlines={9: "python x.py"}, ignore_pids=(9,))["ok"]


def _fake_lease(d: Path, body: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "gpu-lease"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return d


def _clean_env(monkeypatch, path_dirs):
    monkeypatch.delenv("LEASE_ROCR_DEVICES", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(x) for x in path_dirs] + ["/usr/bin", "/bin"]))
    monkeypatch.setattr(ro, "preflight", lambda *a, **k: dict(ok=True, refusals=[], checks={}))
    monkeypatch.setattr(ro, "gpu_status_text", lambda: "fake")


def test_refuses_without_gpu_lease(tmp_path, monkeypatch, capsys):
    _clean_env(monkeypatch, [tmp_path / "empty"])
    assert ro.main(["--out", str(tmp_path / "o")]) == 3
    assert "gpu-lease is not on PATH" in capsys.readouterr().out


def test_refuses_when_cards_are_leased(tmp_path, monkeypatch, capsys):
    """gpu-lease --nowait failing without starting the command = the cards are leased: refuse."""
    _clean_env(monkeypatch, [_fake_lease(tmp_path / "bin", "exit 75\n")])
    assert ro.main(["--out", str(tmp_path / "o")]) == 3
    assert "leased or busy" in capsys.readouterr().out


def test_runs_inner_under_the_lease_and_refuses_without_two_cards(tmp_path, monkeypatch, capsys):
    """A lease that runs the command: the inner process starts (marker), then refuses on this CPU box."""
    body = 'while [ "$1" != "--" ]; do shift; done; shift\nLEASE_ROCR_DEVICES=0,1 exec "$@"\n'
    _clean_env(monkeypatch, [_fake_lease(tmp_path / "bin", body)])
    rc = ro.main(["--out", str(tmp_path / "o"), "--python", sys.executable, "--no-download"])
    assert rc == 3 and (tmp_path / "o" / ".inner_started").exists()


def test_refuses_preflight_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LEASE_ROCR_DEVICES", raising=False)
    monkeypatch.setattr(ro, "preflight", lambda *a, **k: dict(ok=False, refusals=["the serve looks up"], checks={}))
    monkeypatch.setattr(ro, "gpu_status_text", lambda: "fake")
    assert ro.main(["--out", str(tmp_path / "o")]) == 3
    assert "REFUSED" in capsys.readouterr().out
    assert ro.main(["--out", str(tmp_path / "o"), "--preflight-only"]) == 3


def test_checkpoint_status(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s1.safetensors",
                                                                               "b": "s2.safetensors"}}))
    (d / "s1.safetensors").write_bytes(b"x")
    st = ro.checkpoint_status(d)
    assert not st["complete"] and st["missing"] == ["s2.safetensors"] and not st["config_matches_pinned_revision"]
    (d / "s2.safetensors").write_bytes(b"x")
    assert ro.checkpoint_status(d)["complete"]
    assert not ro.checkpoint_status(tmp_path / "nope")["complete"]


# ------------------------------------------------------------------------------------------------------------
# criteria
# ------------------------------------------------------------------------------------------------------------

def _fake_out(d: Path, *, bwd_ratio=1.2, h=0.05, dx=0.99999, peak=14.0, fwd_rel=2.6e-3, bits_ok=True, gmm=True):
    d.mkdir(parents=True, exist_ok=True)
    fbk = lambda g: dict(bwd_group=g, beta_fp32="default", bwd_over_native_fwd=bwd_ratio,  # noqa: E731
                         bwd_h_per_1M_tok_30_layers=h, fwdbwd_h_per_1M_tok_30_layers=2 * h, bwd_ops_per_tok=0.4,
                         bwd_nonview_ops_per_tok=0.2, bwd_live_extra_bytes=1, gpu_peak_fwdbwd_bytes=3e9)
    gdn = dict(core=[dict(B=4, T=4096, dtype="bf16", arms={}, fixed_backward={"fix": fbk(None), "fix_g1": fbk(1)},
                          reference_backward=dict(arm="fork_ref", bwd_over_native_fwd=9.0, bwd_h_per_1M_tok_30_layers=0.7,
                                                  bwd_ops_per_tok=24.0, bwd_live_extra_bytes=1, gpu_peak_fwdbwd_bytes=1))],
               layer=[dict(B=4, T=4096, dtype="bf16", arms={
                   "fix_b32": dict(fwd_rel_err=fwd_rel, dx_cos=dx, vs_truth=dict(fwd_rel=5e-3), vs_hf_b32=dict(fwd_rel=1e-4)),
                   "hf_b32": dict(vs_truth=dict(fwd_rel=5.4e-3)),
                   "fork_shim": dict(fwd_rel_err=1e-3, dx_cos=0.99999,
                                     fork_shim=dict(patched=1, classes={"Qwen3_5MoeGatedDeltaNet": 1}))})],
               exact=dict(worst={"fix/bf16": dict(fwd_rel=1.7e-3, min_grad_cos=0.99999)}),
               model=dict(arms={"fix": dict(calls=3, fallbacks=0)}, all_ok=True))
    (d / "gdn.json").write_text(json.dumps(gdn))
    (d / "awq").mkdir(exist_ok=True)
    n = 768 if bits_ok else 767
    (d / "awq" / "awq_validate_owner.json").write_text(json.dumps({"1_bitexact": {"L0": dict(
        matrices=768, ours_eq_ref_fp32=n, ours_eq_ref_bf16=768, ours_eq_fork_fp32=768, nan=0, experts=256,
        stacked_module_eq_ref=256)}}))
    (d / "grouped_mm_card0.json").write_text(json.dumps(dict(grouped_mm_runs=gmm, verdict={"torch._grouped_mm": dict(
        runs_all_cases=gmm, accurate_all_cases=gmm)}, cases=[{"torch._grouped_mm": dict(speedup_vs_loop_fwdbwd=3.0)}])))
    card = lambda: dict(peak_allocated_gb=peak, peak_reserved_gb=peak + 0.3, total_gb=17.1,  # noqa: E731
                        pass_15_5GB=peak <= 15.5, reserved_within_15_5GB=peak + 0.3 <= 15.5)
    (d / "split_fix.json").write_text(json.dumps(dict(split=23, n_layers=40, weights="checkpoint", runs=[dict(
        chunk_tokens=8192, per_card={"cuda:0": card(), "cuda:1": card()}, pass_15_5GB_all_cards=peak <= 15.5,
        steps=[dict(fwd_s=10.0, bwd_s=20.0, h_per_1M_tok=1.0)])], gdn_all_calls_patched=True)))
    return d


def _by_id(cs):
    return {c["id"]: c for c in cs}


def test_evaluate_all_pass(tmp_path):
    c = _by_id(ro.evaluate(_fake_out(tmp_path / "o"), gpu=True))
    for k in ("M2-0.1", "M2-0.K", "M2-0.2", "M2-0.3@8192", "M2.awq", "7.5#2", "M2.shim", "M2.model"):
        assert c[k]["status"] == "PASS", (k, c[k])
    assert c["M2.fwd"]["status"] == "FAIL"           # 2.6e-3 > 1e-3: the exact-kernel caveat is reported
    txt = ro.write_summary(tmp_path / "o", list(c.values()), dict(mode="full", steps=[]))
    assert "M2-0 overall: PASS" in txt


@pytest.mark.parametrize("kw,cid,want", [(dict(bwd_ratio=2.0), "M2-0.1", "FAIL"),
                                         (dict(h=0.3), "M2-0.K", "FAIL (kill: re-cost M2)"),
                                         (dict(dx=0.9990), "M2-0.2", "FAIL"),
                                         (dict(peak=16.0), "M2-0.3@8192", "FAIL"),
                                         (dict(bits_ok=False), "M2.awq", "FAIL"),
                                         (dict(gmm=False), "7.5#2", "FAIL (use the chunked bmm fallback)")])
def test_evaluate_failures(tmp_path, kw, cid, want):
    cs = ro.evaluate(_fake_out(tmp_path / "o", **kw), gpu=True)
    assert _by_id(cs)[cid]["status"] == want
    txt = ro.write_summary(tmp_path / "o", cs, dict(mode="full", steps=[]))
    assert ("M2-0 overall: NOT PASSED" in txt) == (cid.startswith("M2-0"))
    if cid.startswith("M2-0"):
        assert cid in txt.split("M2-0 overall:")[1]


def test_evaluate_cpu_marks_gpu_bars_na(tmp_path):
    c = _by_id(ro.evaluate(_fake_out(tmp_path / "o"), gpu=False))
    assert c["M2-0.1"]["status"] == "N/A" and c["M2-0.K"]["status"] == "N/A" and c["M2-0.3"]["status"] == "N/A"
    txt = ro.write_summary(tmp_path / "o", list(c.values()), dict(mode="cpu", steps=[]))
    assert "M2-0 overall: INCOMPLETE" in txt


# ------------------------------------------------------------------------------------------------------------
# split_peak and the replicated reader
# ------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ckpt4(tmp_path_factory):
    root = tmp_path_factory.mktemp("sp")
    cfg, dirs = _make_ckpts(root, layers=(0, 1, 2, 3))
    (dirs["awq"] / "config.json").write_text(json.dumps(cfg))
    return cfg, dirs["awq"]


@pytest.mark.parametrize("gdn", ["fix_g1", "hf", "ours"])
def test_split_step_checkpointing_and_gdn_accounting(ckpt4, tmp_path, gdn):
    _, d = ckpt4
    res = sp.main(["--ckpt", str(d), "--devices", "cpu,cpu", "--T", "24", "--chunks", "48", "--steps", "1",
                   "--threads", "1", "--split", "2", "--epg", "2", "--gdn", gdn, "--out", str(tmp_path / "s.json")])
    run = res["runs"][0]
    assert res["split"] == 2 and res["n_layers"] == 4 and res["gdn_all_calls_patched"]
    st = run["steps"][0]
    assert st["gdn_calls"] == st["gdn_calls_expected"] == 2 * 3 * 2 and st["gdn_fallbacks"] == 0
    assert st["dx_finite"] and run["checkpoint_check"]["dx_rel"] < 1e-6


def test_split_mixer_pieces_do_not_change_the_gradient(ckpt4, tmp_path):
    cfg, d = ckpt4
    rd = hq.CheckpointReader.local(d)
    host, _ = sp.build_host(cfg, rd, [torch.device("cpu")] * 4, epg=2, synthetic_head=True, lm_dev=torch.device("cpu"),
                            log=lambda *a: None)
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(4, 16, cfg["text_config"]["hidden_size"], generator=g).to(torch.bfloat16).requires_grad_(True)
    lab = torch.randint(0, cfg["text_config"]["vocab_size"], (4, 16), generator=g)
    l1, d1 = sp.run_step(host, x0, lab, mixer_batch=1, ce_chunk=20)
    l4, d4 = sp.run_step(host, x0, lab, mixer_batch=4, ce_chunk=64, use_checkpoint=False)
    assert abs(l1.item() - l4.item()) < 1e-2 * abs(l4.item())
    assert torch.nn.functional.cosine_similarity(d1.double().flatten(), d4.double().flatten(), dim=0) > 0.999


def test_replicated_reader_aliases_layers(ckpt4, tmp_path):
    cfg, _ = ckpt4
    _, dirs = _make_ckpts(tmp_path, layers=(0, 3))
    rd = hq.CheckpointReader.local(dirs["awq"])
    # drop layer 3's experts from the view, as in the real cache
    wm = {k: v for k, v in rd.weight_map.items() if not k.startswith("model.language_model.layers.3.mlp.experts.")}
    rd = hq.CheckpointReader(rd.files, wm)
    rr, src = sp.replicated_reader(rd, cfg)
    assert src[1] == dict(dense=0, experts=0) and src[3] == dict(dense=3, experts=0)
    n1 = "model.language_model.layers.1.linear_attn.in_proj_qkv.weight"
    assert torch.equal(rr.get(n1), rd.get(n1.replace("layers.1", "layers.0")))
    e3 = "model.language_model.layers.3.mlp.experts.4.down_proj.qweight"
    assert torch.equal(rr.get(e3), rd.get(e3.replace("layers.3", "layers.0")))
    assert hq.packed_layer_bytes(rr, 2) == hq.packed_layer_bytes(rd, 0)
    for i in range(4):                                           # every layer loads
        sp.awq35.load_layer(cfg, i, rr, compute_dtype=torch.float32)


def test_split_layers_by_memory():
    assert sp.split_layers([10] * 10, 0.57) == 5
    assert sp.split_layers([1, 1, 1, 100], 0.57) == 3


# ------------------------------------------------------------------------------------------------------------
# grouped_mm probe
# ------------------------------------------------------------------------------------------------------------

def test_grouped_mm_probe_cpu_tiny(monkeypatch, tmp_path):
    import grouped_mm_probe as gp
    monkeypatch.setattr(gp, "E", 8)
    monkeypatch.setattr(gp, "D", 64)
    monkeypatch.setattr(gp, "I", 32)
    res = gp.main(["--device", "cpu", "--tokens", "40", "--reps", "0", "--threads", "1",
                   "--out", str(tmp_path / "g.json")])
    assert len(res["cases"]) == 2 and res["cases"][1]["pad16"]
    lp = res["cases"][0]["loop_bf16"]
    assert lp["ok"] and lp["dx_cos"] > 0.999
    for name in res["available"]:
        assert res["cases"][0][name]["ok"] and res["cases"][0][name]["pass"], res["cases"][0][name]

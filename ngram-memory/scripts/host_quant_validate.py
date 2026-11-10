"""Real-weight checks of the 4-bit host loaders by HTTP range read (tensors are read into RAM, nothing is
written to disk):

A. cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit, one MoE layer (all 256 experts x gate/up/down): our dequant vs the
   independent reference (tests/hq_reference.py) vs the fork's awq_to_op_layout + op decode, bit-exact in fp32
   and bf16; then vs the bf16 originals (Qwen/Qwen3.6-35B-A3B fused expert tensors) for a few experts.
B. amd/Qwen3.8-27B-Quark-AWQ-MXFP4, one GDN layer's packed projections (in_proj_qkv/z/a/b, out_proj) and one
   attention q_proj: ours vs reference vs the fork's mxfp4.dequant_reference, bit-exact; vs Qwen/Qwen3.8-27B rows.
C. Real-layer GDN acceptance of gdn.py: the 35B layer-L GDN mixer (F16 dense) and the 27B layer-L GDN mixer
   (MXFP4 kept packed, QuantLinear projections): forward rel. error and dX cosine of the patched layer vs HF's
   own forward (torch delta rule), fp32 and bf16.

Usage: python scripts/host_quant_validate.py --out <json> [--parts A,B,C] [--layer35 0] [--layer27 0] [--threads 3]
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fork_refs  # noqa: E402
import hq_reference as ref  # noqa: E402
from ngram_memory.backends.graft import gdn  # noqa: E402
from ngram_memory.backends.graft import host_quant as hq  # noqa: E402

HUB = Path("/root/.cache/huggingface/hub")
SNAP = {
    "awq35": HUB / "models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/f1e8415c4f29508fe281da4e256df26573213bfd",
    "bf35": HUB / "models--Qwen--Qwen3.6-35B-A3B/snapshots/345d467b117d1fd5eda9b879a5988de9ae5f47a7",
    "mx27": HUB / "models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/93245f7c00f1a8964b66a97c6e02e2fd09080be4",
    "bf27": HUB / "models--Qwen--Qwen3.8-27B/snapshots/37561816e66b5b9153b1b64c10968c18c1e6f4a0",
}
REPO = {"awq35": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit", "bf35": "Qwen/Qwen3.6-35B-A3B",
        "mx27": "amd/Qwen3.8-27B-Quark-AWQ-MXFP4", "bf27": "Qwen/Qwen3.8-27B"}
LP = "model.language_model.layers"


def hub_reader(key: str, names: list[str]) -> hq.CheckpointReader:
    idx = SNAP[key] / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        return hq.CheckpointReader.hub(REPO[key], index_json=idx, files=sorted({wm[n] for n in names}))
    return hq.CheckpointReader.hub(REPO[key])


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality (NaN-safe) of two float tensors."""
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int32 if a.dtype == torch.float32 else torch.int16),
                                              b.contiguous().view(torch.int32 if b.dtype == torch.float32 else torch.int16))


def vs_original(wq: torch.Tensor, w: torch.Tensor) -> dict:
    wq, w = wq.double(), w.double()
    c = (wq * w).sum(0) / (w * w).sum(0).clamp_min(1e-30)          # best per-input-column scale (AWQ smoothing)
    return dict(rel_err=((wq - w).norm() / w.norm()).item(),
                rel_err_after_column_scale=((wq - w * c).norm() / wq.norm()).item(),
                sign_agree=((wq.sign() == w.sign()) | (wq == 0)).double().mean().item(),
                cos=F.cosine_similarity(wq.flatten(), w.flatten(), dim=0).item(),
                col_scale_median=c.median().item())


def part_a(layer: int) -> dict:
    cfg = json.loads((SNAP["awq35"] / "config.json").read_text())
    sch = hq.scheme_from_config(cfg)
    E = cfg["text_config"]["num_experts"]
    names = hq._expert_names(f"{LP}.{layer}.mlp.experts", E)
    rd = hub_reader("awq35", names)
    t0 = time.time()
    t = rd.get_many(names)
    fetch_s = time.time() - t0
    stats = dict(matrices=0, ours_eq_ref_fp32=0, ours_eq_ref_bf16=0, ours_eq_fork_fp32=0, fork_available=False)
    fork_ok = fork_refs.fork_awq_to_op_layout() is not None
    stats["fork_available"] = fork_ok
    deq = {}
    for e in range(E):
        for p in ("gate_proj", "up_proj", "down_proj"):
            pre = f"{LP}.{layer}.mlp.experts.{e}.{p}"
            qw, qz, sc = t[pre + ".qweight"], t[pre + ".qzeros"], t[pre + ".scales"]
            ours = hq.awq_dequant(qw, qz, sc, sch.group_size, torch.float32)
            want = torch.from_numpy(ref.awq_dequant_ref(qw.numpy(), qz.numpy(), sc.numpy(), sch.group_size))
            stats["matrices"] += 1
            stats["ours_eq_ref_fp32"] += same(ours, want)
            stats["ours_eq_ref_bf16"] += same(hq.awq_dequant(qw, qz, sc, sch.group_size), want.to(torch.bfloat16))
            if fork_ok:
                stats["ours_eq_fork_fp32"] += same(ours, fork_refs.fork_awq_dequant(qw, qz, sc, sch.group_size))
            if e in (0, 97, E - 1):
                deq[(e, p)] = ours
    # the module path: stacked QuantMoEExperts dequant == per-matrix dequant
    mod = hq.load_awq_experts(rd, f"{LP}.{layer}.mlp.experts", E, sch, hidden=cfg["text_config"]["hidden_size"],
                              intermediate=cfg["text_config"]["moe_intermediate_size"], tensors=t)
    gu, dn = mod.dequant_experts(torch.tensor([0, 97, E - 1]), torch.float32)
    I = cfg["text_config"]["moe_intermediate_size"]
    stats["module_stack_eq"] = all(same(gu[i, :I], deq[(e, "gate_proj")]) and same(gu[i, I:], deq[(e, "up_proj")])
                                   and same(dn[i], deq[(e, "down_proj")]) for i, e in enumerate((0, 97, E - 1)))
    # vs bf16 originals
    on = [f"{LP}.{layer}.mlp.experts.gate_up_proj", f"{LP}.{layer}.mlp.experts.down_proj"]
    rb = hub_reader("bf35", on)
    orig = {}
    for e in (0, 97, E - 1):
        gu_o = rb.get_rows(on[0], e, e + 1)[0]
        dn_o = rb.get_rows(on[1], e, e + 1)[0]
        orig[f"expert{e}"] = {"gate_proj": vs_original(deq[(e, "gate_proj")], gu_o[:I].float()),
                              "up_proj": vs_original(deq[(e, "up_proj")], gu_o[I:].float()),
                              "down_proj": vs_original(deq[(e, "down_proj")], dn_o.float())}
    return dict(layer=layer, scheme=sch.__dict__, fetch_s=fetch_s, bytes_awq=rd.bytes_read, bytes_bf16=rb.bytes_read,
                packed_expert_bytes=mod.packed_bytes(), bitexact=stats, vs_bf16_original=orig)


def part_b(layer: int, attn_layer: int) -> dict:
    cfg = json.loads((SNAP["mx27"] / "config.json").read_text())
    sch = hq.scheme_from_config(cfg)
    projs = [f"{LP}.{layer}.linear_attn.{p}" for p in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
                                                         "out_proj")] + [f"{LP}.{attn_layer}.self_attn.q_proj"]
    rd = hq.CheckpointReader.hub(REPO["mx27"])
    hdr_bytes = rd.bytes_read
    t = rd.get_many([f"{p}.{s}" for p in projs for s in ("weight", "weight_scale")])
    f_mx = fork_refs.fork_mxfp4_dequant()
    out, deq = {}, {}
    for p in projs:
        w, s = t[p + ".weight"], t[p + ".weight_scale"]
        ours = hq.mxfp4_dequant(w, s, torch.float32)
        want = torch.from_numpy(ref.mxfp4_dequant_ref(w.numpy(), s.numpy()))
        out[p.split(".", 3)[-1]] = dict(shape=list(ours.shape), ours_eq_ref_fp32=same(ours, want),
                                        ours_eq_ref_bf16=same(hq.mxfp4_dequant(w, s), want.to(torch.bfloat16)),
                                        ours_eq_fork_fp32=None if f_mx is None else same(ours, f_mx(w, s)),
                                        nan=int(torch.isnan(ours).sum()), e8m0_range=[int(s.min()), int(s.max())])
        deq[p] = ours
    rows = 256
    rb = hub_reader("bf27", [f"{p}.weight" for p in projs])
    orig = {}
    for p in projs:
        n = min(rows, deq[p].shape[0])
        o = rb.get_rows(p + ".weight", 0, n).float()
        orig[p.split(".", 3)[-1]] = vs_original(deq[p][:n], o)
    return dict(layer=layer, attn_layer=attn_layer, scheme=sch.__dict__, header_bytes=hdr_bytes,
                bytes_mx=rd.bytes_read, bytes_bf16=rb.bytes_read, bitexact=out, vs_bf16_original_first_rows=orig)


def gdn_accept(mixer, norm, hidden: int, B: int, T: int, dtype: torch.dtype, seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        x = norm(torch.randn(B, T, hidden, generator=g).to(dtype))
    dy = torch.randn(B, T, hidden, generator=g).to(dtype)

    def run():
        xx = x.clone().requires_grad_(True)
        t0 = time.time()
        y = mixer(xx)
        t1 = time.time()
        (dx,) = torch.autograd.grad(y, xx, dy)
        return y.detach().float(), dx.float(), t1 - t0, time.time() - t1
    y0, d0, f0, b0 = run()
    cfg = gdn.patch_gdn(mixer, impl="ours")
    try:
        y1, d1, f1, b1 = run()
    finally:
        gdn.unpatch_gdn(mixer)
    return dict(B=B, T=T, dtype=str(dtype), patched_calls=cfg.calls, fwd_rel_err=((y1 - y0).norm() / y0.norm()).item(),
                dx_cos=F.cosine_similarity(d1.flatten().double(), d0.flatten().double(), dim=0).item(),
                dx_rel_err=((d1 - d0).norm() / d0.norm()).item(),
                hf_fwd_s=f0, hf_bwd_s=b0, ours_fwd_s=f1, ours_bwd_s=b1)


def part_c(layer35: int, layer27: int, B: int, T: int) -> dict:
    res = {}
    for key, L in (("awq35", layer35), ("mx27", layer27)):
        cfg = json.loads((SNAP[key] / "config.json").read_text())
        if cfg["text_config"]["layer_types"][L] != "linear_attention":
            raise SystemExit(f"{key} layer {L} is not a GDN layer")
        idx = SNAP[key] / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
            shard = {v for k, v in wm.items() if k.startswith((f"{LP}.{L}.linear_attn.", f"{LP}.{L}.input_layernorm."))}
            rd = hq.CheckpointReader.hub(REPO[key], index_json=idx, files=sorted(shard))
        else:
            rd = hq.CheckpointReader.hub(REPO[key])
        row = {}
        for dt in (torch.float32, torch.bfloat16):
            layer, tcfg = hq.build_decoder_layer(cfg, L, rd, compute_dtype=dt, parts=("input_layernorm", "linear_attn"))
            mixer = layer.linear_attn
            row[str(dt)] = gdn_accept(mixer, layer.input_layernorm, tcfg.hidden_size, B, T, dt)
            row["class"] = type(mixer).__name__
            row["quant_linears"] = sum(isinstance(m, hq.QuantLinear) for m in mixer.modules())
            print(key, dt, json.dumps(row[str(dt)]), flush=True)
        row["bytes_read"] = rd.bytes_read
        res[key] = row
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="A,B,C")
    ap.add_argument("--layer35", type=int, default=0)
    ap.add_argument("--layer27", type=int, default=0)
    ap.add_argument("--attn27", type=int, default=3)
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = {}
    parts = a.parts.split(",")
    if "A" in parts:
        out["A_awq35"] = part_a(a.layer35)
        print("A", json.dumps(out["A_awq35"]["bitexact"]), flush=True)
    if "B" in parts:
        out["B_mx27"] = part_b(a.layer27, a.attn27)
        print("B", json.dumps({k: [v["ours_eq_ref_fp32"], v["ours_eq_ref_bf16"], v["ours_eq_fork_fp32"]]
                               for k, v in out["B_mx27"]["bitexact"].items()}), flush=True)
    if "C" in parts:
        out["C_gdn_real"] = part_c(a.layer35, a.layer27, a.B, a.T)
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print("wrote", a.out, "peak RSS GB", out["peak_rss_gb"])


if __name__ == "__main__":
    main()

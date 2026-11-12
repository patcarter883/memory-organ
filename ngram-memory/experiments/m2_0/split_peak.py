"""M2-0: per-card peak memory and step time of a graft-training step on the 35B AWQ host at the planned layer split
and the real chunk sizes (DESIGN §4.6 "VRAM" and "mixed chunking", §7.5 #1, §8 M2-0 bar "per-card peak <= 15.5 GB
at the planned split").

What one step is (weights frozen, packed; no weight gradients):
  x0 [B, T, d] bf16 on card 0 with requires_grad (stands in for embeddings + the graft rows' output; its gradient is
  what the row table would receive) -> the 40 decoder layers, layers [0, split) on card 0 and [split, 40) on card 1,
  each wrapped in activation checkpointing (only each layer's input is kept) -> final norm + lm_head + cross-entropy
  in --ce-chunk-position chunks (each chunk checkpointed) -> backward to x0.
  Mixed chunking (§4.6): inside a layer the token mixer (GDN or attention) runs on --mixer-batch sequences of T
  tokens at a time (default 1 -> 4k-token pieces at T=4096), while the MoE block gets all B*T tokens of the chunk
  at once (B*T = 8k / 16k -> 256-512 tokens per expert). GDN runs through gdn_native.patch_gdn_native (default:
  the fixed backward, native forward + HF torch-chunk recompute, one sequence per recompute); every GDN call is
  counted and a fallback to HF's own forward raises.

Weights:
  --ckpt DIR       a full local checkpoint (the owner's HF snapshot): every layer, norm and lm_head are real.
  --cache-dir DIR  only the M2-0 layer cache (<DIR>/cyankiwi_35b: L0 whole, L3 dense parts): each GDN layer is a
                   copy of layer 0, each attention layer layer 3's dense tensors + layer 0's routed experts (same
                   shapes and dtypes, so the memory is the same), and lm_head / final norm are random bf16 of the
                   real shape. Numbers are then memory/time-faithful but the function is not the model's.
The split: --split auto puts the first layers whose packed bytes sum to <= --split-frac (default 0.57, §4.6) of the
body on card 0; --split N puts layers [0, N) there.

Output (JSON): per card: resident bytes after load, max_memory_allocated and max_memory_reserved during the step,
the 15.5 GB bar (on allocated; reserved reported beside it), fwd / bwd / step wall time, tokens per second and the
projected hours per 1M training tokens for the whole host (excludes the row table and data pipeline), GDN call
accounting. On CPU (--devices cpu,cpu) it is a functional check: the same code with checkpointing on and off must
give the same x0 gradient (tests/test_m2_0_owner.py runs it on a synthetic checkpoint).

  gpu-lease -n 2 -- python experiments/m2_0/split_peak.py --ckpt <snapshot> --devices cuda:0,cuda:1 \
      --T 4096 --chunks 8192,16384 --out split_peak.json
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ngram_memory.backends.graft import awq35  # noqa: E402
from ngram_memory.backends.graft import gdn_native as gn  # noqa: E402
from ngram_memory.backends.graft import host_quant as hq  # noqa: E402

LP = awq35.LP
BAR_CARD_PEAK_GB = 15.5
CARD_CAP_GB = 16.82
GDN_ARMS = {                      # name -> patch_gdn_native kwargs (native_op resolved at run time)
    "fix_g1": dict(fwd="native", bwd="hf", bwd_group=1),
    "fix": dict(fwd="native", bwd="hf"),
    "native_ours": dict(fwd="native", bwd="ours"),
    "fork_ref": dict(fwd="native", bwd="fork_ref"),
    "ours": dict(fwd="ours", bwd="ours"),
    "hf": dict(fwd="hf", bwd="hf"),
}


# ------------------------------------------------------------------------------------------------------------
# weights: full checkpoint, or the layer cache replicated to every layer
# ------------------------------------------------------------------------------------------------------------

def replicated_reader(reader: hq.CheckpointReader, cfg: dict, *, gdn_src: int = 0, attn_src: int = 3,
                      expert_src: int | None = None) -> tuple[hq.CheckpointReader, dict]:
    """A reader in which every layer i exists: GDN layers alias layer ``gdn_src``'s tensors, attention layers
    alias layer ``attn_src``'s (routed experts from ``expert_src`` when ``attn_src`` has none cached; default
    gdn_src). Aliases share the source files' bytes (no copy). Returns (reader, map layer -> sources)."""
    types_ = cfg["text_config"]["layer_types"]
    names = list(reader.keys())
    has = lambda L, part: any(n.startswith(f"{LP}.{L}.{part}") for n in names)  # noqa: E731
    expert_src = expert_src if expert_src is not None else gdn_src
    files = {k: copy.copy(f) for k, f in reader.files.items()}
    for f in files.values():
        f.header = dict(f.header)
    wm = dict(reader.weight_map)
    src_map = {}
    for i, t in enumerate(types_):
        base = gdn_src if t == "linear_attention" else attn_src
        ex = base if has(base, "mlp.experts.") else expert_src
        src_map[i] = dict(dense=base, experts=ex)
        for n in names:
            for s, take in ((base, lambda r: not r.startswith("mlp.experts.")), (ex, lambda r: r.startswith("mlp.experts."))):
                pre = f"{LP}.{s}."
                if n.startswith(pre) and take(n[len(pre):]):
                    alias = f"{LP}.{i}.{n[len(pre):]}"
                    if alias in wm and i in (gdn_src, attn_src):
                        continue
                    fn = reader.weight_map[n]
                    files[fn].header[alias] = reader.files[fn].header[n]
                    wm[alias] = fn
    return hq.CheckpointReader(files, wm), src_map


def split_layers(sizes: list[int], frac: float) -> int:
    tot, acc = sum(sizes), 0
    for i, s in enumerate(sizes):
        if acc + s > frac * tot:
            return max(1, i)
        acc += s
    return len(sizes)


# ------------------------------------------------------------------------------------------------------------
# the step
# ------------------------------------------------------------------------------------------------------------

class Host(nn.Module):
    def __init__(self, layers: list[nn.Module], devs: list[torch.device], norm: nn.Module, lm_head: torch.Tensor,
                 rot_by_dev: dict, types_: list[str]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.devs, self.norm, self.lm_head, self.rot, self.types = devs, norm, lm_head, rot_by_dev, types_


def _layer_step(layer, h, kind: str, pe, pos, mixer_batch: int):
    B, T, d = h.shape
    x = layer.input_layernorm(h)
    outs = []
    for b0 in range(0, B, mixer_batch):
        xs = x[b0:b0 + mixer_batch]
        if kind == "linear_attention":
            outs.append(layer.linear_attn(hidden_states=xs))
        else:
            outs.append(layer.self_attn(hidden_states=xs, position_embeddings=pe, attention_mask=None,
                                        position_ids=pos)[0])
    h = h + torch.cat(outs)
    y = layer.mlp(layer.post_attention_layernorm(h).reshape(1, B * T, d))
    y = y[0] if isinstance(y, tuple) else y
    return h + y.reshape(B, T, d)


def _ce_chunk(h, labels, norm, W):
    logits = F.linear(norm(h), W)
    return F.cross_entropy(logits.float(), labels, reduction="sum")


def run_step(host: Host, x0: torch.Tensor, labels: torch.Tensor, *, mixer_batch: int, ce_chunk: int,
             use_checkpoint: bool = True, timers: dict | None = None):
    """Forward + backward of one chunk; returns (loss, dx0)."""
    B, T, _ = x0.shape
    h = x0
    pes = {}

    def sync():
        for d in {dv for dv in host.devs if dv.type == "cuda"}:
            torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for i, layer in enumerate(host.layers):
        dev = host.devs[i]
        if h.device != dev:
            h = h.to(dev)
        kind = host.types[i]
        pe = pos = None
        if kind != "linear_attention":
            if dev not in pes:
                p = torch.arange(T, device=dev)[None].expand(mixer_batch, T)
                pes[dev] = (host.rot[str(dev)](h[:mixer_batch], p[None].expand(3, mixer_batch, T)), p)
            pe, pos = pes[dev]
        if use_checkpoint:
            h = checkpoint(_layer_step, layer, h, kind, pe, pos, mixer_batch, use_reentrant=False)
        else:
            h = _layer_step(layer, h, kind, pe, pos, mixer_batch)
    W = host.lm_head
    if h.device != W.device:
        h = h.to(W.device)
    lab = labels.to(W.device).reshape(-1)
    hf = h.reshape(B * T, -1)
    loss = None
    for c0 in range(0, B * T, ce_chunk):
        sl = slice(c0, c0 + ce_chunk)
        part = (checkpoint(_ce_chunk, hf[sl], lab[sl], host.norm, W, use_reentrant=False) if use_checkpoint
                else _ce_chunk(hf[sl], lab[sl], host.norm, W))
        loss = part if loss is None else loss + part
    loss = loss / (B * T)
    sync()
    t1 = time.perf_counter()
    (dx0,) = torch.autograd.grad(loss, x0)
    sync()
    t2 = time.perf_counter()
    if timers is not None:
        timers.update(fwd_s=t1 - t0, bwd_s=t2 - t1)
    return loss.detach(), dx0


def build_host(cfg: dict, reader: hq.CheckpointReader, devs_by_layer: list[torch.device], *, epg: int,
               synthetic_head: bool, lm_dev: torch.device, log=print) -> tuple[Host, dict]:
    tc = cfg["text_config"]
    types_ = tc["layer_types"][: len(devs_by_layer)]
    layers = []
    t0 = time.time()
    for i, dev in enumerate(devs_by_layer):
        lay, tcfg = awq35.load_layer(cfg, i, reader, compute_dtype=torch.bfloat16, device=dev, experts_per_group=epg)
        layers.append(lay)
        if i % 8 == 0 or i == len(devs_by_layer) - 1:
            log(f"  loaded layer {i} ({types_[i]}) on {dev}  [{time.time() - t0:.0f}s]")
    mod = sys.modules[type(layers[0]).__module__]
    rot_cls = next(getattr(mod, n) for n in dir(mod) if n.endswith("TextRotaryEmbedding"))
    rms_cls = type(layers[0].input_layernorm)            # the decoder's own RMSNorm (HF's (1 + w) form)
    rot = {str(d): rot_cls(tcfg).to(d) for d in set(devs_by_layer)}
    d = tc["hidden_size"]
    norm = rms_cls(d, eps=tc["rms_norm_eps"]).to(lm_dev, torch.bfloat16)
    if synthetic_head or "lm_head.weight" not in reader:
        g = torch.Generator().manual_seed(0)
        W = (torch.randn(tc["vocab_size"], d, generator=g) * 0.02).to(lm_dev, torch.bfloat16)
        head = "synthetic (random bf16, real shape)"
    else:
        W = reader.get("lm_head.weight").to(lm_dev, torch.bfloat16)
        with torch.no_grad():
            norm.weight.copy_(reader.get("model.language_model.norm.weight").to(torch.bfloat16))
        head = "checkpoint lm_head + final norm (F16 -> bf16)"
    for p in norm.parameters():
        p.requires_grad_(False)
    return Host(layers, devs_by_layer, norm, W, rot, types_), dict(load_s=time.time() - t0, lm_head=head)


def resolve_gdn(arm: str, on_gpu: bool) -> tuple[dict, str]:
    kw = dict(GDN_ARMS[arm])
    if kw.get("fwd") == "native":
        op = gn.resolve_native_op() if on_gpu else None
        if op is None:
            if on_gpu:
                raise SystemExit("gdn_hip not importable: build the fork's extension or set NGM_MINISGL_FORK")
            op = gn.hf_standin_op
            note = "CPU: native forward = gdn_native.hf_standin_op"
        else:
            note = "gdn_hip.gdn_prefill (recurrent)"
        kw["native_op"] = op
    else:
        note = kw["fwd"]
    return kw, note


def _mem(dev: torch.device) -> dict:
    if dev.type != "cuda":
        return {}
    free, total = torch.cuda.mem_get_info(dev)
    return dict(allocated=torch.cuda.memory_allocated(dev), reserved=torch.cuda.memory_reserved(dev),
                max_allocated=torch.cuda.max_memory_allocated(dev), max_reserved=torch.cuda.max_memory_reserved(dev),
                driver_free=free, driver_total=total)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", type=Path, help="full local checkpoint dir (config.json + index + shards)")
    src.add_argument("--cache-dir", type=Path, help="M2-0 layer cache root (replicated layers, synthetic head)")
    ap.add_argument("--devices", default="cuda:0,cuda:1")
    ap.add_argument("--split", default="auto", help="'auto' (by packed bytes, --split-frac) or N (layers [0,N) on card 0)")
    ap.add_argument("--split-frac", type=float, default=0.57)
    ap.add_argument("--layers", type=int, default=None, help="use only the first N layers (CPU checks)")
    ap.add_argument("--T", type=int, default=4096, help="sequence length (mixer piece length)")
    ap.add_argument("--chunks", default="8192,16384", help="tokens per step (B = chunk / T); MoE sees all of them")
    ap.add_argument("--mixer-batch", type=int, default=1, help="sequences per GDN / attention call")
    ap.add_argument("--gdn", default="auto", help=f"GDN arm: {', '.join(GDN_ARMS)}; auto = fix_g1 on GPU, hf on CPU")
    ap.add_argument("--epg", type=int, default=8, help="experts dequantised together (transient ~4.3x its output)")
    ap.add_argument("--ce-chunk", type=int, default=512)
    ap.add_argument("--lm-device", default=None, help="device of final norm + lm_head + CE (default: last card)")
    ap.add_argument("--steps", type=int, default=2, help="steps per chunk size (the first is a warm-up)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-checkpoint-check", action="store_true",
                    help="CPU only: skip the checkpointing-off reference step")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    devs = [torch.device(x) for x in a.devices.split(",")]
    on_gpu = devs[0].type == "cuda"
    if on_gpu:
        os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")
    if a.ckpt:
        reader = hq.CheckpointReader.local(a.ckpt)
        cfg = json.loads((a.ckpt / "config.json").read_text())
        weights, src_map, synth = "checkpoint", None, False
    else:
        d = a.cache_dir / "cyankiwi_35b"
        base = hq.CheckpointReader.local(d)
        cfg = json.loads((d / "config.json").read_text())
        reader, src_map = replicated_reader(base, cfg)
        ex3 = {v["experts"] for i, v in src_map.items() if v["dense"] != 0}
        weights = ("replicated from the layer cache (GDN layers = L0, attention layers = L3 dense + "
                   f"L{'/'.join(str(x) for x in sorted(ex3)) or '0'} experts; synthetic lm_head)")
        synth = True
    tc = cfg["text_config"]
    nL = a.layers or tc["num_hidden_layers"]
    sizes = [hq.packed_layer_bytes(reader, i) for i in range(nL)]
    split = split_layers(sizes, a.split_frac) if a.split == "auto" else int(a.split)
    devs_by_layer = [devs[0] if i < split else devs[-1] for i in range(nL)]
    lm_dev = torch.device(a.lm_device) if a.lm_device else devs[-1]
    res: dict = dict(script="experiments/m2_0/split_peak.py", weights=weights, repo=awq35.REPO,
                     revision=awq35.REVISION, devices=[str(x) for x in devs], torch=torch.__version__,
                     n_layers=nL, split=split, split_frac=a.split_frac, layer_types=tc["layer_types"][:nL],
                     packed_bytes_per_card=[sum(sizes[:split]), sum(sizes[split:])], T=a.T,
                     mixer_batch=a.mixer_batch, epg=a.epg, ce_chunk=a.ce_chunk, steps=a.steps, runs=[])
    if on_gpu:
        res["cards"] = {str(x): torch.cuda.get_device_name(x) for x in devs}
    print(f"split: layers [0,{split}) on {devs[0]}, [{split},{nL}) on {devs[-1]}; packed GB per card "
          f"{[round(x / 1e9, 2) for x in res['packed_bytes_per_card']]}", flush=True)
    host, info = build_host(cfg, reader, devs_by_layer, epg=a.epg, synthetic_head=synth, lm_dev=lm_dev)
    res.update(info)
    res["resident_after_load"] = {str(x): _mem(x) for x in devs}
    arm = a.gdn if a.gdn != "auto" else ("fix_g1" if on_gpu else "hf")
    kw, note = resolve_gdn(arm, on_gpu)
    n_gdn = sum(t == "linear_attention" for t in host.types)
    with _pure_torch_hf():
        pcfg = gn.patch_gdn_native(host.layers, **kw) if n_gdn else None
        res["gdn"] = dict(arm=arm, native_forward=note, layers=n_gdn,
                          beta_fp32=None if pcfg is None else pcfg.beta_fp32)
        for chunk in (int(c) for c in a.chunks.split(",")):
            B = max(1, chunk // a.T)
            run = dict(chunk_tokens=B * a.T, B=B, T=a.T, steps=[])
            g = torch.Generator().manual_seed(chunk)
            d = tc["hidden_size"]
            try:
                for s in range(a.steps):
                    x0 = (torch.randn(B, a.T, d, generator=g) * 0.5).to(devs[0], torch.bfloat16).requires_grad_(True)
                    labels = torch.randint(0, tc["vocab_size"], (B, a.T), generator=g)
                    for x in devs:
                        if x.type == "cuda":
                            torch.cuda.synchronize(x)
                            torch.cuda.reset_peak_memory_stats(x)
                    c0 = pcfg.calls if pcfg else 0
                    tm: dict = {}
                    loss, dx = run_step(host, x0, labels, mixer_batch=a.mixer_batch, ce_chunk=a.ce_chunk, timers=tm)
                    st = dict(step=s, loss=loss.item(), dx_norm=dx.float().norm().item(),
                              dx_finite=bool(torch.isfinite(dx).all()), **tm,
                              gdn_calls=(pcfg.calls - c0) if pcfg else 0,
                              gdn_calls_expected=2 * n_gdn * -(-B // a.mixer_batch),
                              gdn_fallbacks=pcfg.fallbacks if pcfg else 0,
                              mem={str(x): _mem(x) for x in devs})
                    ntok = B * a.T
                    st["tok_per_s"] = ntok / (tm["fwd_s"] + tm["bwd_s"])
                    st["h_per_1M_tok"] = 1e6 / st["tok_per_s"] / 3600
                    run["steps"].append(st)
                    pk = {k: round(v.get("max_allocated", 0) / 1e9, 3) for k, v in st["mem"].items()}
                    print(f"  chunk {ntok} (B={B}) step {s}: loss {st['loss']:.4f} fwd {tm['fwd_s']:.2f}s "
                          f"bwd {tm['bwd_s']:.2f}s ({st['h_per_1M_tok']:.3f} h/1M tok) peak alloc GB {pk} "
                          f"gdn calls {st['gdn_calls']}/{st['gdn_calls_expected']}", flush=True)
                    if not on_gpu and not a.no_checkpoint_check and s == 0:
                        _, dx_ref = run_step(host, x0, labels, mixer_batch=a.mixer_batch, ce_chunk=a.ce_chunk,
                                             use_checkpoint=False)
                        run["checkpoint_check"] = dict(
                            dx_rel=((dx.double() - dx_ref.double()).norm() / dx_ref.double().norm()).item())
                    del x0, labels, loss, dx
                    gc.collect()
            except torch.cuda.OutOfMemoryError as e:   # recorded: this is a measurement, not a crash
                run["oom"] = str(e)[:400]
                print(f"  chunk {B * a.T}: OOM", flush=True)
                gc.collect()
                for x in devs:
                    if x.type == "cuda":
                        torch.cuda.empty_cache()
            last = run["steps"][-1] if run["steps"] else None
            if last is not None and on_gpu:
                run["per_card"] = {k: dict(peak_allocated_gb=v["max_allocated"] / 1e9,
                                           peak_reserved_gb=v["max_reserved"] / 1e9,
                                           total_gb=v["driver_total"] / 1e9,
                                           pass_15_5GB=v["max_allocated"] / 1e9 <= BAR_CARD_PEAK_GB,
                                           reserved_within_15_5GB=v["max_reserved"] / 1e9 <= BAR_CARD_PEAK_GB)
                                   for k, v in last["mem"].items()}
                run["pass_15_5GB_all_cards"] = all(c["pass_15_5GB"] for c in run["per_card"].values())
            res["runs"].append(run)
            a.out.write_text(json.dumps(res, indent=1, default=str))
        if pcfg is not None:
            gn.unpatch_gdn_native(host.layers)
    res["gdn_all_calls_patched"] = all(st["gdn_calls"] == st["gdn_calls_expected"] and st["gdn_fallbacks"] == 0
                                       for r in res["runs"] for st in r["steps"])
    a.out.write_text(json.dumps(res, indent=1, default=str))
    print("wrote", a.out)
    return res


class _pure_torch_hf:
    """HF's GDN kernel functions pinned to their torch bodies (fla hangs on gfx1201); the patched layers do not
    use them, but nothing may reach fla even through a fallback."""
    NAMES = ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
             "causal_conv1d_update")

    def __enter__(self):
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
        self.m, self.old = m, {n: getattr(m, n) for n in self.NAMES}
        for n, f in self.old.items():
            setattr(m, n, getattr(f, "__wrapped__", f))

    def __exit__(self, *exc):
        for n, f in self.old.items():
            setattr(self.m, n, f)


if __name__ == "__main__":
    main()

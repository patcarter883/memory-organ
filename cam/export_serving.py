"""CAM → minisgl serving checkpoint export (WS-B deliverable).

Writes the checkpoint dir that WS-A's `python/minisgl/cam/memory.py::CAMMemory` loads. The layout and
key names below are the AUTHORITATIVE spec for WS-A's loader — they mirror
`docs/zaya-port/CAM_SERVE_CONTRACT.md` (CHECKPOINT section) and are pinned to the actual module
structure of `cam/gated_tap.py::GatedMemoryTap`, `cam/pk_store_adapter.py::PKStoreAdapter` /
`cam/pk_store.py::ProductKeyStore`, and `cam/gate_router.py::GateRouter`.

Everything is saved on CPU in fp32 (state_dict save/load needs no GPU). The tied embed / unembed
tables are NOT included (they are ~3 GB and WS-A already loads the base model's `base_embed` /
`lm_head_weight` — the CAMMemory constructor takes those separately).

──────────────────────────────────────────────────────────────────────────────────────────────────
CHECKPOINT LAYOUT   (out_dir/)
──────────────────────────────────────────────────────────────────────────────────────────────────
meta.json
    {
      "base_model":    str,          # HF id the memory was bound on (args.base1)
      "mem_dim":       int,          # store / tap memory width (adapter.mem_dim)
      "tap_layer":     int,          # decoder layer the GatedMemoryTap injects after (=24 for Qwen3.5-4B)
      "n_banks":       int,          # # of DISJOINT persistent value banks at serve time
                                     #   (recall_mag._n_disjoint_banks(); env CAM_DISJOINT_BANKS, default 1)
      "n_sub":         int,          # product-key sub-codebook size -> store.N = n_sub**2 slots
      "signal_names":  [str]*8,      # ordered names of GateRouter's 8 label-free input signals
      "router_alpha":  float,        # alpha_ref: injection scale the router gain multiplies (CAM_ROUTER_ALPHA)
      "topk":          int,          # per-token injection support size the router was fit with (CAM_MULTIGATE_TOPK)
      "remember_tau":  float,        # base-uncertainty WRITE-GATE threshold (CAM_REMEMBER_TAU)
      "hidden_size":   int,          # base residual width (GatedMemoryTap.H; =2560 for Qwen3.5-4B)
      # informational extras (safe for WS-A to ignore):
      "n_heads_tap":   int,          # GatedMemoryTap attention heads
      "n_read_heads":  int,          # ProductKeyStore read heads (store.n_heads)
      "store_topk":    int,          # ProductKeyStore product-key top-k (store.topk)
      "store_sub_topk":int,          # ProductKeyStore per-half top-k (store.sub_topk)
      "k":             int,          # adapter.readout_q slot count (bank K handed to the tap)
      "conf_gate":     bool,         # tap store-confidence gate enabled
      "n_rel":         int           # tap per-relation conf-gate EMA count
    }

tap.pt  = GatedMemoryTap.state_dict()   (torch.save of an OrderedDict; fp32, on CPU). Keys:
    to_q.weight, to_k.weight, to_v.weight, to_o.weight,     # [H,H] / [H,mem] / [H,mem] / [H,H]
    gamma,                                                   # [H]  gate logit (tanh(gamma), 0-init no-op)
    gate_alpha, supp,                                        # scalars: norm-gate fraction / two-sided supp
    null_key,                                                # [1, n_heads, 1, d_head] learnable null slot key
    conf_scale, conf_bias,                                   # scalars: conf-gate sigmoid steepness / threshold
    conf_ema                                                 # [n_rel] buffer: per-relation conf EMA (<0 = uninit)

adapter.pt = FLAT dict of the PKStoreAdapter learned tensors (torch.save; fp32, CPU). Exact keys:
    in_proj.weight                     # [mem_dim, hidden]      base-hidden -> mem_dim
    norm.weight, norm.bias             # [mem_dim]              LayerNorm over mem_dim
    subj_pool_q                        # [n_key_heads, mem_dim] learned subject-span attention pool query
    store.to_wkey.weight               # [mem_dim, mem_dim]     write-key projection
    store.to_wval.weight               # [mem_dim, mem_dim]     write-value projection
    store.codebook1                    # [n_sub, mem_dim//2]    product-key sub-codebook 1
    store.codebook2                    # [n_sub, mem_dim//2]    product-key sub-codebook 2
    store.read_q.{h}.weight            # [mem_dim, mem_dim] per read head h (0..n_read_heads-1)
    store.read_o.{h}.weight            # [mem_dim, mem_dim] per read head h
    store.read_norm.{h}.weight         # [mem_dim]          per read head h (RMSNorm gain)
    store.read_out_norm.weight         # [mem_dim]          final read-output RMSNorm gain
    store.head_bias                    # [n_read_heads, mem_dim] per-head retrieval-mode query bias
    readout_q                          # [k, mem_dim]       learned readout slot queries
    out_proj.weight                    # [hidden, mem_dim]  mem_dim -> base-hidden (for tied-unembed readout)

router.pt = GateRouter.state_dict()  (n_out=2 per-token router: predicts (g_top, g_rest)). Keys:
    net.0.weight, net.0.bias,          # Linear(8 -> 32)
    net.2.weight, net.2.bias,          # Linear(32 -> 32)
    net.4.weight, net.4.bias           # Linear(32 -> n_out=2)

WS-A loader: reconstruct GatedMemoryTap(hidden, mem_dim, n_heads=meta.n_heads_tap, conf_gate=..,
n_rel=meta.n_rel).load_state_dict(tap.pt); PKStoreAdapter(...).load_state_dict(adapter.pt, strict=False)
(strict=False because adapter.pt intentionally omits the frozen embed/unembed and any optional
pos_tag/decoder params); GateRouter(n_out=2).load_state_dict(router.pt).
"""
import json
import os

import torch

try:
    from .gate_router import N_SIG
except ImportError:  # run as a top-level module (recall_mag imports both ways)
    from gate_router import N_SIG


# Ordered names of GateRouter's 8 label-free signals, matching gate_router.signal_features()'s
# stacking order (indices 0..7). Kept here (the module has no programmatic list) so meta.json and
# WS-A stay in lockstep.
SIGNAL_NAMES = [
    "store_conf",         # 0 retrieval strength (locality)
    "base_entropy",       # 1 base uncertainty (normalized entropy)
    "headroom",           # 2 headroom on the target (1 - p_target)
    "store_peak",         # 3 store decode confidence (softmax(raw).max)
    "agreement_margin",   # 4 p_target - p_base_top (<=0)
    "base_top_conf",      # 5 base mass on its own top token
    "base_margin",        # 6 base decisiveness (top1 - top2)
    "store_entropy",      # 7 store decode diffuseness
]
assert len(SIGNAL_NAMES) == N_SIG, "SIGNAL_NAMES must have N_SIG entries"

# The exact PKStoreAdapter learned tensors WS-A needs, as prefixes into adapter.state_dict().
# A state_dict key is exported iff it starts with one of these. This drops the frozen embed table
# (embed.*), the tied unembed buffer (unembed), the store query-BN buffers (store._q_bn.*), any
# disjoint per-position stores (store.stores.*), the multi-token pos_tag/pos_gate/pos_proj, GTE
# projection, and the optional decoder-readout head — none of which the serve read path uses.
_ADAPTER_KEEP_PREFIXES = (
    "in_proj.",
    "norm.",                 # matches norm.weight / norm.bias (NOT store.read_norm / store.read_out_norm)
    "subj_pool_q",
    "store.to_wkey.",
    "store.to_wval.",
    "store.codebook1",
    "store.codebook2",
    "store.read_q.",
    "store.read_o.",
    "store.read_norm.",
    "store.read_out_norm.",
    "store.head_bias",
    "readout_q",
    "out_proj.",
)


def _find_tap(injector, tap_layer=None):
    """Locate the single GatedMemoryTap inside a MAGInjector (taps: nn.ModuleDict keyed by str(layer)).

    Prefers the tap at `tap_layer`; else the sole tap; else the first of injector.tap_layers."""
    taps = injector.taps
    layers = list(getattr(injector, "tap_layers", []))
    if tap_layer is not None and str(tap_layer) in taps:
        return taps[str(tap_layer)], int(tap_layer)
    if len(taps) == 1:
        L = int(next(iter(taps.keys())))
        return taps[str(L)], L
    if layers:
        L = int(layers[0])
        return taps[str(L)], L
    raise RuntimeError("could not locate a GatedMemoryTap in the injector")


def _cpu_fp32_sd(state_dict):
    """Detach + move every tensor in a state_dict to CPU fp32 (float buffers stay float; keep ints)."""
    out = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            v = v.detach().to("cpu")
            if v.is_floating_point():
                v = v.float()
        out[k] = v
    return out


def _extract_adapter_tensors(adapter):
    """Filter adapter.state_dict() down to the contract's learned tensors (see _ADAPTER_KEEP_PREFIXES)."""
    sd = adapter.state_dict()
    kept = {k: v for k, v in sd.items() if any(k.startswith(p) for p in _ADAPTER_KEEP_PREFIXES)}
    return _cpu_fp32_sd(kept)


def export_serving_checkpoint(injector, adapter, router, meta, out_dir):
    """Write the CAM serving checkpoint dir (meta.json, tap.pt, adapter.pt, router.pt) per the contract.

    injector : MAGInjector holding the trained GatedMemoryTap(s).
    adapter  : trained PKStoreAdapter (learned store projections + read heads).
    router   : fitted GateRouter (n_out=2, per-token).
    meta     : partial dict from the caller — MUST supply {base_model, router_alpha, topk,
               remember_tau, n_banks}; structural fields (mem_dim, tap_layer, n_sub, hidden_size,
               signal_names, and the informational extras) are DERIVED from the objects here and
               overwrite any caller value so meta.json cannot drift from the saved weights.
    Returns the fully-resolved meta dict that was written.
    """
    os.makedirs(out_dir, exist_ok=True)

    tap, tap_layer = _find_tap(injector, tap_layer=meta.get("tap_layer"))
    store = adapter.store

    # derive structural fields from the live objects (source of truth over caller-supplied meta)
    derived = {
        "mem_dim": int(adapter.mem_dim),
        "tap_layer": int(tap_layer),
        "n_sub": int(store.n_sub),
        "signal_names": list(SIGNAL_NAMES),
        "hidden_size": int(tap.H),
        "n_heads_tap": int(tap.n_heads),
        "n_read_heads": int(store.n_heads),
        "store_topk": int(store.topk),
        "store_sub_topk": int(store.sub_topk),
        "k": int(adapter.k),
        "conf_gate": bool(getattr(tap, "conf_gate", False)),
        "n_rel": int(getattr(tap, "n_rel", 1)),
        # behaviour knobs WS-A's loader BAKES from meta (must match the training env, not read at serve time)
        "pooled_subj_key": os.environ.get("CAM_POOLED_SUBJ_KEY") == "1",
        "learned_key_pool": os.environ.get("CAM_LEARNED_KEY_POOL") == "1",
        "write_at_read": os.environ.get("CAM_WRITE_AT_READ") == "1",
        "key_maxsim": os.environ.get("CAM_KEY_MAXSIM") == "1",
        "norm_gate": os.environ.get("CAM_NORM_GATE") == "1",
        "twosided": os.environ.get("CAM_TWOSIDED") == "1",
        "obj_latent": os.environ.get("CAM_OBJ_LATENT") == "1",
    }
    resolved = dict(meta)
    resolved.update(derived)
    # caller-owned serve knobs: keep whatever was passed; default the ones the caller may omit.
    resolved.setdefault("base_model", "")
    resolved.setdefault("router_alpha", float(os.environ.get("CAM_ROUTER_ALPHA", "1.5")))
    resolved.setdefault("topk", int(os.environ.get("CAM_MULTIGATE_TOPK", "16")))
    resolved.setdefault("remember_tau", float(os.environ.get("CAM_REMEMBER_TAU", "0.5")))
    resolved.setdefault("n_banks", 1)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(resolved, f, indent=2, sort_keys=True)

    torch.save(_cpu_fp32_sd(tap.state_dict()), os.path.join(out_dir, "tap.pt"))
    torch.save(_extract_adapter_tensors(adapter), os.path.join(out_dir, "adapter.pt"))
    torch.save(_cpu_fp32_sd(router.state_dict()), os.path.join(out_dir, "router.pt"))
    return resolved


def load_serving_checkpoint(out_dir):
    """Reload a checkpoint dir written by export_serving_checkpoint.

    Returns a dict {meta, tap, adapter, router} where meta is the parsed JSON and tap/adapter/router
    are the raw tensor dicts (WS-A feeds tap/router into module.load_state_dict, and adapter into
    PKStoreAdapter.load_state_dict(strict=False))."""
    with open(os.path.join(out_dir, "meta.json")) as f:
        meta = json.load(f)
    tap = torch.load(os.path.join(out_dir, "tap.pt"), map_location="cpu")
    adapter = torch.load(os.path.join(out_dir, "adapter.pt"), map_location="cpu")
    router = torch.load(os.path.join(out_dir, "router.pt"), map_location="cpu")
    return {"meta": meta, "tap": tap, "adapter": adapter, "router": router}


# ──────────────────────────────────────────────────────────────────────────────────────────────
# CPU round-trip self-test — build tiny REAL modules, export, reload, assert byte equality.
# Run:  python -m cam.export_serving      (from the repo root; CPU only, no GPU/base model)
# ──────────────────────────────────────────────────────────────────────────────────────────────
def _selftest():
    import tempfile

    import torch.nn as nn

    try:
        from .gated_tap import GatedMemoryTap
        from .pk_store_adapter import PKStoreAdapter
        from .gate_router import GateRouter
    except ImportError:
        from gated_tap import GatedMemoryTap
        from pk_store_adapter import PKStoreAdapter
        from gate_router import GateRouter

    torch.manual_seed(0)
    vocab, hidden, mem_dim, k, heads = 40, 16, 8, 4, 2
    tap_layer = 24

    # tiny real modules (CPU). GatedMemoryTap: hidden must divide n_heads.
    tap = GatedMemoryTap(hidden, mem_dim, n_heads=heads, conf_gate=True, n_rel=1)
    # randomize the 0-init gate params so the round-trip actually exercises non-trivial values
    with torch.no_grad():
        tap.gamma.copy_(torch.randn_like(tap.gamma))
        tap.gate_alpha.copy_(torch.tensor(0.3))
        tap.conf_ema.copy_(torch.tensor([1.234]))

    embed_weight = torch.randn(vocab, hidden)
    adapter = PKStoreAdapter(embed_weight, hidden, mem_dim, heads, 16, 4.0, k,
                             n_sub=6, topk=4, sub_topk=2)
    router = GateRouter(n_out=2)

    # a minimal MAGInjector stand-in: the export only touches .taps (ModuleDict) and .tap_layers.
    class _DummyInjector:
        def __init__(self, tap, L):
            self.taps = nn.ModuleDict({str(L): tap})
            self.tap_layers = [L]
    injector = _DummyInjector(tap, tap_layer)

    meta_in = {"base_model": "dummy/base", "router_alpha": 1.5, "topk": 16,
               "remember_tau": 0.5, "n_banks": 3}

    with tempfile.TemporaryDirectory() as d:
        resolved = export_serving_checkpoint(injector, adapter, router, meta_in, d)
        assert set(os.listdir(d)) == {"meta.json", "tap.pt", "adapter.pt", "router.pt"}, os.listdir(d)
        loaded = load_serving_checkpoint(d)

        # meta structural derivation
        m = loaded["meta"]
        assert m["mem_dim"] == mem_dim and m["hidden_size"] == hidden
        assert m["tap_layer"] == tap_layer and m["n_sub"] == 6
        assert m["k"] == k and m["n_read_heads"] == heads
        assert m["signal_names"] == SIGNAL_NAMES and len(m["signal_names"]) == N_SIG
        assert m["base_model"] == "dummy/base" and m["n_banks"] == 3

        # tap round-trip: reload into a fresh module, assert exact equality
        tap2 = GatedMemoryTap(hidden, mem_dim, n_heads=heads, conf_gate=True, n_rel=1)
        missing, unexpected = tap2.load_state_dict(loaded["tap"], strict=True)
        for kk, vv in tap.state_dict().items():
            assert torch.equal(vv.cpu().float() if vv.is_floating_point() else vv.cpu(),
                               loaded["tap"][kk]), f"tap mismatch {kk}"

        # router round-trip
        router2 = GateRouter(n_out=2)
        router2.load_state_dict(loaded["router"], strict=True)
        for kk, vv in router.state_dict().items():
            assert torch.equal(vv.cpu().float(), loaded["router"][kk]), f"router mismatch {kk}"

        # adapter round-trip: every exported key must match the source adapter's tensor exactly,
        # and load back into a fresh adapter with strict=False (frozen embed/unembed intentionally absent)
        src = adapter.state_dict()
        expected_keys = {
            "in_proj.weight", "norm.weight", "norm.bias", "subj_pool_q",
            "store.to_wkey.weight", "store.to_wval.weight",
            "store.codebook1", "store.codebook2",
            "store.read_out_norm.weight", "store.head_bias",
            "readout_q", "out_proj.weight",
        }
        for h in range(heads):
            expected_keys |= {f"store.read_q.{h}.weight", f"store.read_o.{h}.weight",
                              f"store.read_norm.{h}.weight"}
        got = set(loaded["adapter"].keys())
        assert got == expected_keys, f"adapter key set mismatch:\n  missing={expected_keys-got}\n  extra={got-expected_keys}"
        for kk in expected_keys:
            assert torch.equal(src[kk].cpu().float(), loaded["adapter"][kk]), f"adapter mismatch {kk}"

        adapter2 = PKStoreAdapter(embed_weight, hidden, mem_dim, heads, 16, 4.0, k,
                                  n_sub=6, topk=4, sub_topk=2)
        res = adapter2.load_state_dict(loaded["adapter"], strict=False)
        # the only 'missing' keys allowed are the frozen embed table + unembed buffer (+ any optional
        # params the serve path doesn't use); NONE of our exported keys may be unexpected.
        assert not res.unexpected_keys, f"unexpected adapter keys on reload: {res.unexpected_keys}"

    print("[export_serving] CPU round-trip self-test PASSED")
    print(f"[export_serving] adapter.pt keys ({len(expected_keys)}):")
    for kk in sorted(expected_keys):
        print(f"    {kk:32s} {tuple(loaded['adapter'][kk].shape)}")
    print("[export_serving] tap.pt keys:")
    for kk, vv in loaded["tap"].items():
        print(f"    {kk:32s} {tuple(vv.shape) if torch.is_tensor(vv) else vv}")
    print("[export_serving] router.pt keys:")
    for kk, vv in loaded["router"].items():
        print(f"    {kk:32s} {tuple(vv.shape)}")


if __name__ == "__main__":
    _selftest()

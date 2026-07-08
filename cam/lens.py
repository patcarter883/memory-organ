"""LensTrace — logit-lens readout of the residual stream around the memory tap.

Anthropic's global-workspace result (the "J-lens", anthropic.com/research/global-workspace) reads
intermediate residual-stream activations through the model's decode path to answer *"which words is
this internal state disposed to produce?"* — and finds a sparse, causally load-bearing workspace
there. The MAG tap WRITES into exactly that stream (gated_tap injects at decoder layer L). This
module is the read side: capture the residual stream after every decoder layer during any forward
pass, decode each captured state through the base's own final-norm + unembed (the logit-lens
approximation), and report per-layer probability/rank trajectories for chosen tokens — e.g. watch
P(' Paris') vs P(' Tokyo') across depth while the memory overwrites the prior (RESULTS §6).

The experiment ladder this serves is in docs/research/global-workspace-lens.md.

Honesty box:
- This is a LOGIT lens (nostalgebraist 2020), not the Jacobian lens: decoding intermediate layers
  through the FINAL norm+unembed is a zeroth-order approximation, biased at early layers (basis
  mismatch; tuned lens 2303.08112 fixes it with per-layer affine probes). It is most faithful in the
  mid-to-late layers — which is where the tap sits (L=24 on the 4B). `decode_fn` is pluggable so a
  fitted tuned/Jacobian lens (a pre-fitted Qwen3.5-4B lens ships with Subtext / Neuronpedia) can drop
  in without touching call sites.
- Readouts are OBSERVATIONAL, not causal, and change no forward computation: the hooks return None,
  so the base's outputs are bit-identical with or without a LensTrace attached (selftest-pinned).

HOOK ORDERING (load-bearing): forward hooks run in registration order and each sees the previous
hook's (possibly rewritten) output. Enter the LensTrace AFTER MAGInjector.attach() to read the
POST-injection stream — the stream the base actually computes with; enter it before to read the
pre-injection stream. The selftest pins this contract.

Drop-in usage around any existing eval forward (no harness changes needed):

    from cam.lens import LensTrace
    with LensTrace(base) as lt:                    # after injector.attach()
        lg = _last_logit(base, input_ids=ids)      # any existing forward
    traj = lt.trajectory([prior_tid, edit_tid])    # {layer: [(prob, rank), ...]}
    xo = lt.crossover(prior_tid, edit_tid)         # first layer where the edit overtakes the prior

Run styles (mirrors the other drivers):
    python -m cam.lens --selftest                  # CPU, no GPU / no model download
    python -m cam.lens --prompt "The capital of France is" --targets " Paris, Tokyo"
"""
import argparse
import os
import sys

import torch
import torch.nn as nn

try:
    from .gated_tap import decoder_layers
except ImportError:
    if __package__:  # real ImportError inside a sibling, not "run as a file" — don't mask it
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from gated_tap import decoder_layers                                  # noqa: E402


def _walk(base, paths, kind):
    """Locate a submodule across HF causal/image-text-to-text wrappers (decoder_layers' idiom)."""
    for path in paths:
        obj, ok = base, True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.Module):
            return obj
    raise RuntimeError(f"could not locate the {kind} on the base model (tried {paths})")


def final_norm(base):
    return _walk(base, ("model.norm", "model.model.norm", "language_model.model.norm",
                        "model.language_model.norm", "transformer.ln_f"), "final norm")


def lm_head(base):
    return _walk(base, ("lm_head", "model.lm_head", "language_model.lm_head"), "lm head")


class LensTrace:
    """Context manager: capture the residual stream after each decoder layer, decode on demand.

    positions: which sequence positions to keep at capture time — an int (default -1: last
    position, the QA answer site), a list of ints, or None for ALL positions (memory scales with
    seq len × layers; fine for prompts, mind it on long bind blocks). layers: which decoder layers
    to trace (default: all). decode_fn: optional (hidden [B,P,H], layer_idx) -> fp32 logits [B,P,V]
    replacing the default final-norm+unembed decode (the tuned/Jacobian-lens upgrade path)."""

    def __init__(self, base, layers=None, positions=-1, decode_fn=None):
        self.base = base
        self._layers_ml = decoder_layers(base)
        self.layers = list(range(len(self._layers_ml))) if layers is None else list(layers)
        self.positions = positions
        self._decode_fn = decode_fn
        self._norm, self._head = (None, None) if decode_fn else (final_norm(base), lm_head(base))
        self.hidden = {}                                   # layer_idx -> [B,P,H] captured residual
        self._handles = []

    # ---- capture -------------------------------------------------------------------------------
    def _hook(self, L):
        def fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if self.positions is None:
                self.hidden[L] = h.detach().clone()
            else:
                pos = [self.positions] if isinstance(self.positions, int) else list(self.positions)
                idx = torch.tensor([p % h.shape[1] for p in pos], device=h.device)
                self.hidden[L] = h.detach().index_select(1, idx)
            return None                                    # observational: never rewrite the output
        return fn

    def __enter__(self):
        self.hidden = {}
        for L in self.layers:
            self._handles.append(self._layers_ml[L].register_forward_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # ---- decode --------------------------------------------------------------------------------
    def logits(self, L):
        """fp32 lens logits [B,P,V] for the residual captured after decoder layer L. The default
        decode is the base's OWN final-norm + unembed, so at the LAST traced layer this equals the
        model's real output logits exactly (the selftest pins that anchor)."""
        h = self.hidden[L]
        with torch.no_grad():                              # observational: no grads through the lens
            if self._decode_fn is not None:
                return self._decode_fn(h, L).float()
            nw = next(self._norm.parameters(), None)
            if nw is not None:                             # norm/head may sit on another card
                h = h.to(device=nw.device, dtype=nw.dtype)  # (MODEL-PARALLEL: lm_head on card 1)
            h = self._norm(h)
            hw = next(self._head.parameters())
            return self._head(h.to(device=hw.device, dtype=hw.dtype)).float()

    def probs(self, L):
        return torch.softmax(self.logits(L), dim=-1)

    def topk(self, L, k=5, position=-1, batch=0):
        """[(token_id, prob)] top-k at one captured position of one batch row."""
        p = self.probs(L)[batch, position]
        v, i = p.topk(k)
        return list(zip(i.tolist(), v.tolist()))

    def trajectory(self, token_ids, position=-1, batch=0):
        """{layer: [(prob, rank), ...] one pair per target token} — the depth trajectory of chosen
        vocabulary tokens (rank is 1-based; rank 1 = the token the lens says comes next)."""
        out = {}
        for L in self.layers:
            if L not in self.hidden:
                continue
            p = self.probs(L)[batch, position]
            ranks = torch.argsort(p, descending=True)
            rank_of = torch.empty_like(ranks)
            rank_of[ranks] = torch.arange(len(ranks), device=ranks.device)
            out[L] = [(float(p[t]), int(rank_of[t]) + 1) for t in token_ids]
        return out

    def crossover(self, tid_a, tid_b, position=-1, batch=0):
        """First traced layer where P(tid_b) > P(tid_a), else None — e.g. where the injected edit
        object overtakes the base's prior on the depth axis."""
        traj = self.trajectory([tid_a, tid_b], position=position, batch=batch)
        for L in sorted(traj):
            (pa, _), (pb, _) = traj[L]
            if pb > pa:
                return L
        return None


# ---- CPU selftest ------------------------------------------------------------------------------
class _TinyBase(nn.Module):
    """Pure-torch stand-in exposing the HF layout LensTrace/MAGInjector need (model.layers
    ModuleList whose blocks return tuples, model.norm, lm_head, config.get_text_config()
    .hidden_size). CPU wiring smokes only — no semantics."""

    class _Cfg:
        def __init__(self, H):
            self.hidden_size = H

        def get_text_config(self):
            return self

    class _Block(nn.Module):
        def __init__(self, H):
            super().__init__()
            self.fc = nn.Linear(H, H, bias=False)

        def forward(self, x, **kw):
            return (x + torch.tanh(self.fc(x)),)

    def __init__(self, vocab=97, H=32, L=4):
        super().__init__()
        self.config = self._Cfg(H)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, H)
        self.model.layers = nn.ModuleList([self._Block(H) for _ in range(L)])
        self.model.norm = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, vocab, bias=False)

    def forward(self, input_ids=None, **kw):
        h = self.model.embed_tokens(input_ids)
        for blk in self.model.layers:
            h = blk(h)[0]
        logits = self.lm_head(self.model.norm(h))

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        return o


def _selftest():
    """Mechanics only (semantics need a real base): (1) last-layer lens decode == the model's own
    logits; (2) a lens is a pure observer — outputs bit-identical with it attached; (3) a zero-init
    tap leaves every trace bit-identical; (4) an OPENED gate moves traces at/after the tap layer and
    nowhere before (pins hook ordering: lens attached after injector reads post-injection)."""
    try:
        from .gated_tap import MAGInjector
    except ImportError:
        from gated_tap import MAGInjector
    torch.manual_seed(0)
    base, TAP_L = _TinyBase(), 2
    ids = torch.randint(0, 97, (2, 9))

    clean = base(input_ids=ids).logits
    with LensTrace(base) as lt:
        traced = base(input_ids=ids).logits
    assert torch.equal(traced, clean), "lens must not perturb the forward"
    assert torch.allclose(lt.logits(3)[:, -1], clean[:, -1], atol=1e-5), \
        "last-layer lens decode must equal the model's own logits"
    base_hidden = {L: lt.hidden[L].clone() for L in lt.layers}
    print("[lens] selftest: observer + final-layer anchor OK")

    inj = MAGInjector(base, [TAP_L], mem_dim=16, n_heads=4).attach()
    inj.set_bank(torch.randn(2, 5, 16))
    with LensTrace(base) as lt0:                           # after attach -> post-injection stream
        base(input_ids=ids)
    assert all(torch.equal(lt0.hidden[L], base_hidden[L]) for L in lt0.layers), \
        "zero-init gate must leave every lens trace bit-identical"
    print("[lens] selftest: zero-init tap is lens-invisible OK")

    with torch.no_grad():
        inj.taps[str(TAP_L)].gamma.fill_(0.5)              # open the gate
    with LensTrace(base) as lt1:
        base(input_ids=ids)
    for L in lt1.layers:
        same = torch.equal(lt1.hidden[L], base_hidden[L])
        assert same == (L < TAP_L), \
            f"open gate must move layer {L} traces iff L >= tap layer {TAP_L} (got same={same})"
    traj = lt1.trajectory([3, 7])
    assert set(traj) == set(lt1.layers) and all(len(v) == 2 for v in traj.values())
    assert all(1 <= r <= 97 and 0.0 <= p <= 1.0 for v in traj.values() for p, r in v)
    lt1.crossover(3, 7)                                    # mechanics: runs without error
    inj.detach()
    print("[lens] selftest: injection locality on the depth axis + trajectory mechanics OK")
    print("[lens] SELFTEST PASS")


# ---- CLI: trace a prompt through a real frozen base ---------------------------------------------
def _tid(tok, s):
    t = tok(s, add_special_tokens=False)["input_ids"]
    assert len(t) == 1, (f"--targets entries must be single tokens for {tok.name_or_path}: "
                         f"{s!r} -> {t} — try a leading space (' Paris'), like the QA answer site")
    return t[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="CPU mechanics check, no model download")
    ap.add_argument("--prompt", default=None, help="raw completion prompt to trace")
    ap.add_argument("--base1", default=None, help="frozen base (default: m2_adapter.MODEL)")
    ap.add_argument("--targets", default="", help="comma-separated single-token strings to track "
                                                  "per layer, e.g. ' Paris, Tokyo'")
    ap.add_argument("--topk", type=int, default=5, help="top-k lens tokens shown per layer")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    assert args.prompt, "pass --prompt (or --selftest)"
    try:
        from .m2_adapter import load_frozen_base
    except ImportError:
        from m2_adapter import load_frozen_base
    base, tok = load_frozen_base(args.base1)
    ids = tok(args.prompt, return_tensors="pt")["input_ids"].to(next(base.parameters()).device)
    targets = [t for t in args.targets.split(",") if t] if args.targets else []
    tids = [_tid(tok, t) for t in targets]
    with torch.no_grad(), LensTrace(base) as lt:
        base(input_ids=ids)
    traj = lt.trajectory(tids) if tids else {}
    hdr = "  ".join(f"P({t!r}) rank" for t in targets)
    print(f"[lens] base={tok.name_or_path} layers={len(lt.layers)} prompt={args.prompt!r}")
    print(f"  L | top-{args.topk} lens tokens | {hdr}")
    for L in lt.layers:
        tops = " ".join(f"{tok.decode([i])!r}:{p:.3f}" for i, p in lt.topk(L, args.topk))
        tgt = "  ".join(f"{p:.4f} #{r}" for p, r in traj.get(L, []))
        print(f"  {L:3d} | {tops} | {tgt}")


if __name__ == "__main__":
    main()

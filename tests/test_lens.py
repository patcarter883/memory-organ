"""CPU tests for cam/lens.py (LensTrace) — no GPU, no model downloads.

Same contract as test_smoke.py: this verifies the lens MECHANICS (capture, decode anchor, observer
purity, hook-ordering locality), not semantics — layer-resolved editing readouts need a real base +
a trained memory (docs/research/global-workspace-lens.md, E1+)."""
import torch

from cam.gated_tap import MAGInjector
from cam.lens import LensTrace, _TinyBase


def _setup(seed=0, L=4):
    torch.manual_seed(seed)
    base = _TinyBase(vocab=97, H=32, L=L)
    ids = torch.randint(0, 97, (2, 9))
    return base, ids


def test_lens_is_a_pure_observer():
    """Attaching a LensTrace must leave the model's own output logits bit-identical."""
    base, ids = _setup()
    clean = base(input_ids=ids).logits
    with LensTrace(base) as lt:
        traced = base(input_ids=ids).logits
    assert torch.equal(traced, clean)
    assert set(lt.hidden) == {0, 1, 2, 3}


def test_final_layer_decode_anchors_to_model_logits():
    """The default decode is the base's OWN final-norm + unembed, so the lens at the last layer
    must reproduce the model's real next-token logits — the correctness anchor for every
    intermediate readout."""
    base, ids = _setup()
    with LensTrace(base) as lt:
        clean = base(input_ids=ids).logits
    assert torch.allclose(lt.logits(3)[:, -1], clean[:, -1], atol=1e-5)


def test_positions_slicing():
    """positions=-1 keeps only the answer site; None keeps the full sequence."""
    base, ids = _setup()
    with LensTrace(base, positions=-1) as lt:
        base(input_ids=ids)
    assert lt.hidden[0].shape == (2, 1, 32)
    with LensTrace(base, positions=None) as lt_all:
        base(input_ids=ids)
    assert lt_all.hidden[0].shape == (2, 9, 32)
    assert torch.equal(lt_all.hidden[2][:, -1:], lt.hidden[2])


def test_zero_init_tap_is_lens_invisible():
    """The exact-no-op-at-init property, read through the lens: with a bank set but gamma=0, every
    per-layer trace is bit-identical to the tap-free base."""
    base, ids = _setup()
    with LensTrace(base) as lt0:
        base(input_ids=ids)
    inj = MAGInjector(base, [2], mem_dim=16, n_heads=4).attach()
    inj.set_bank(torch.randn(2, 5, 16))
    with LensTrace(base) as lt1:                           # entered AFTER attach: post-injection
        base(input_ids=ids)
    inj.detach()
    assert all(torch.equal(lt1.hidden[L], lt0.hidden[L]) for L in lt0.layers)


def test_open_gate_moves_traces_at_and_after_the_tap_only():
    """Hook-ordering + depth-locality contract: a lens entered after MAGInjector.attach() reads the
    post-injection stream, so an OPENED gate at layer 2 must change traces at layers >= 2 and leave
    layers < 2 untouched."""
    base, ids = _setup()
    TAP_L = 2
    with LensTrace(base) as lt0:
        base(input_ids=ids)
    inj = MAGInjector(base, [TAP_L], mem_dim=16, n_heads=4).attach()
    inj.set_bank(torch.randn(2, 5, 16))
    with torch.no_grad():
        inj.taps[str(TAP_L)].gamma.fill_(0.5)
    with LensTrace(base) as lt1:
        base(input_ids=ids)
    inj.detach()
    for L in lt0.layers:
        assert torch.equal(lt1.hidden[L], lt0.hidden[L]) == (L < TAP_L)


def test_trajectory_and_crossover_mechanics():
    """trajectory: valid probs + 1-based ranks per traced layer; crossover: consistent with the
    last traced layer's ordering when probabilities differ there."""
    base, ids = _setup()
    with LensTrace(base) as lt:
        base(input_ids=ids)
    a, b = 3, 7
    traj = lt.trajectory([a, b])
    assert set(traj) == set(lt.layers)
    for (pa, ra), (pb, rb) in traj.values():
        assert 0.0 <= pa <= 1.0 and 0.0 <= pb <= 1.0
        assert 1 <= ra <= 97 and 1 <= rb <= 97
    xo = lt.crossover(a, b)
    (pa, _), (pb, _) = traj[max(lt.layers)]
    if pb > pa:
        assert xo is not None and xo in lt.layers
    if xo is None:
        assert all(pb2 <= pa2 for (pa2, _), (pb2, _) in traj.values())


def test_layers_subset():
    """A layers subset captures exactly those layers (the cheap-trace path for big bases)."""
    base, ids = _setup()
    with LensTrace(base, layers=[1, 3]) as lt:
        base(input_ids=ids)
    assert set(lt.hidden) == {1, 3}

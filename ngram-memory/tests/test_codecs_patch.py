"""Row codecs (bf16 RNE, int6 group-32 fused rows matching the serve), ngram-edit/v1 patch I/O and verification."""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pytest
import torch

from ngram_memory import codecs
from ngram_memory.patch import (BF16Bits, make_patch, read_patch, read_safetensors, verify_patch, write_patch,
                                write_safetensors)
from ngram_memory.spec import flash_next_format, graft_format
from ngram_memory.tables import QuantRowCodec

from conftest import tiny_key

ROW_TABLE = "/home/user/patcarter883/minisglang-rdna4/python/minisgl/weights/row_table.py"


def _fork_row_table():
    if not os.path.exists(ROW_TABLE):
        pytest.skip("minisgl fork not present")
    spec = importlib.util.spec_from_file_location("_fork_row_table_ro", ROW_TABLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # stdlib + numpy only (row_table.py:37-47)
    return mod


def test_bf16_rne_matches_torch():
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(4096, generator=g) * s for s in (1e-6, 1e-2, 1.0, 1e4)])
    x = torch.cat([x, torch.tensor([0.0, -0.0, 1e-40, -1e-40, 3.0e38])])
    want = x.bfloat16().view(torch.int16).numpy().view(np.uint16)
    assert np.array_equal(codecs.bf16_bits(x.numpy()), want)
    assert np.array_equal(codecs.bf16_bits_to_f32(want), x.bfloat16().float().numpy())


def test_int6_fused_layout_130B_and_fork_decoder_bitexact():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((512, 160)) * 0.0075).astype(np.float32)  # real per-element scale (~0.096/sqrt(160))
    x[3, 32:64] = 0.0  # an all-zero group
    fused = codecs.int6_encode_fused(x)
    assert fused.shape == (512, 130) and codecs.int6_row_bytes(160) == 130
    ours = codecs.int6_decode_fused(fused, 160)
    packed, scales = codecs.int6_split_fused(fused, 160)
    # the core package's torch-side decoder agrees
    q = QuantRowCodec("int6_g32", 160)
    assert np.array_equal(q.decode(packed, scales), ours)
    # the serve's own decoder (read-only import of the fork), bit for bit
    rt = _fork_row_table()
    assert np.array_equal(rt.dequant_int6_g32(packed, scales), ours)
    # codes land on the grid and the zero group decodes to exact zeros
    assert np.all(ours[3, 32:64] == 0)
    grid = np.clip(np.rint(x.reshape(512, 5, 32) / scales.astype(np.float32)[..., None]), -31, 31)
    assert np.array_equal((grid * scales.astype(np.float32)[..., None]).reshape(512, 160), ours)


def test_int6_roundtrip_stats(tmp_path):
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((8192, 160)) * 0.0075).astype(np.float32)
    st6 = codecs.roundtrip_stats(x, "int6_g32", served_cast="bf16")
    stb = codecs.roundtrip_stats(x, "bf16", served_cast=None)
    with open(tmp_path / "stats.json", "w") as f:
        json.dump(dict(int6=st6, bf16=stb), f)
    out = os.environ.get("NGRAM_M0_OUT")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "codec_roundtrip_stats.json"), "w") as f:
            json.dump(dict(int6_g32=st6, bf16=stb), f, indent=1)
    assert st6["row_bytes"] == 130 and stb["row_bytes"] == 320
    # int6 step is scale/31 of the group absmax: per-row relative L2 error a few percent, bf16 ~0.2%
    assert 0.01 < st6["median_row_rel_l2"] < 0.05
    assert stb["median_row_rel_l2"] < 3e-3 and stb["max_abs"] <= np.abs(x).max() * 2 ** -8


def test_safetensors_io_matches_reference_library(tmp_path):
    from safetensors.torch import load_file, save_file
    rng = np.random.default_rng(2)
    a = rng.integers(0, 1 << 40, 17).astype(np.int64)
    b = rng.standard_normal((5, 7)).astype(np.float32)
    c = torch.randn(3, 4).bfloat16()
    p = str(tmp_path / "x.safetensors")
    write_safetensors(p, {"a": a, "b": b, "c": BF16Bits(c.view(torch.int16).numpy().view(np.uint16))},
                      {"k": "v"})
    t = load_file(p)
    assert np.array_equal(t["a"].numpy(), a) and np.array_equal(t["b"].numpy(), b) and torch.equal(t["c"], c)
    q = str(tmp_path / "y.safetensors")
    save_file({"a": torch.from_numpy(a), "c": c}, q, metadata={"m": "1"})
    got, meta = read_safetensors(q)
    assert np.array_equal(got["a"], a) and meta == {"m": "1"}
    assert np.array_equal(codecs.bf16_bits_to_f32(got["c"]), c.float().numpy())


def test_patch_roundtrip_and_refusals(tmp_path):
    fmt = graft_format(64, tiny_key())
    rng = np.random.default_rng(3)
    ids = rng.choice(fmt.padded_rows, 40, replace=False)
    vals = rng.standard_normal((40, fmt.row_width)).astype(np.float32)
    p = make_patch(fmt, ids, vals, model_repo="tiny/host", edits=[dict(edit_id="e1")])
    assert p.manifest.mode == "absolute" and (np.diff(p.row_ids) > 0).all()
    path = str(tmp_path / "p.safetensors")
    write_patch(path, p)
    q = read_patch(path)
    assert np.array_equal(q.row_ids, p.row_ids) and np.array_equal(q.values, p.values)
    assert q.manifest.to_dict() == p.manifest.to_dict()
    assert "module_fingerprint" in verify_patch(q, fmt)
    assert q.manifest.format_spec().fingerprint() == fmt.fingerprint()
    # refusals: another key, another arm, a native table, a tampered payload, a tampered sidecar
    with pytest.raises(ValueError):
        verify_patch(q, graft_format(64, tiny_key((2, 3, 1))))
    with pytest.raises(ValueError):
        verify_patch(q, graft_format(64, tiny_key(), arm="G-dir"))
    with pytest.raises(ValueError):
        verify_patch(q, graft_format(64, tiny_key(), value_seed=1))
    q.values[0, 0] += 1.0
    with pytest.raises(ValueError, match="payload hash"):
        q.check()
    with open(path + ".json") as f:
        side = json.load(f)
    side["model_repo"] = "other"
    with open(path + ".json", "w") as f:
        json.dump(side, f)
    with pytest.raises(ValueError, match="sidecar"):
        read_patch(path)
    with pytest.raises(ValueError):
        make_patch(fmt, [1, 1], np.zeros((2, fmt.row_width)))
    with pytest.raises(IndexError):
        make_patch(fmt, [fmt.padded_rows], np.zeros((1, fmt.row_width)))


def test_native_delta_patch_base_fingerprint():
    fmt = flash_next_format()
    assert fmt.table.base_codec == "bf16" and fmt.row_width == 160
    rng = np.random.default_rng(4)
    ids = np.array([5, 2_500_012, 320_001_535])  # first shard, second shard, last padded row
    base = rng.standard_normal((3, 160)).astype(np.float32)
    p = make_patch(fmt, ids, rng.standard_normal((3, 160)), base_rows=base)
    assert p.manifest.mode == "delta" and p.manifest.rows_per_shard == 2_500_012
    lut = dict(zip(ids.tolist(), base))
    assert "base_fingerprint" in verify_patch(p, fmt, lambda r: np.stack([lut[int(i)] for i in r]))
    base2 = base.copy()
    base2[1, 0] += 1e-3
    lut2 = dict(zip(ids.tolist(), base2))
    with pytest.raises(ValueError, match="base_fingerprint"):
        verify_patch(p, fmt, lambda r: np.stack([lut2[int(i)] for i in r]))
    with pytest.raises(ValueError):
        make_patch(fmt, ids, base, mode="absolute")

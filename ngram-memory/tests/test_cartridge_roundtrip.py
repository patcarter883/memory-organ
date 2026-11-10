"""`ngram-edit/v1` + cartridge manifest round trip: HF adapter <-> numpy overlay prototype, bitwise-equal logits.

Ports the serve-side checks of P/serve/v2-editpath/ple_overlay_np.py (bf16-replacement == fp32-delta after the
device cast; untouched rows bit-identical) and GF/serve-port/parity_hc1.py (empty cartridge == exact no-op) to the
package's file formats:

  ledger edits -> cartridge patch v0 + increment v1 (with reset rows) on disk -> read back (numpy only) ->
  (A) HF adapter: RowOverlay(engine_cast=bf16) on the table module
  (B) numpy prototype: CompiledOverlayNP fed the compiled served file, applied to host rows inside a stub table
  (C) rows loaded straight into the graft (the graft runtime)
  -> torch.equal on logits, graft (tiny qwen3_5 host) and native (tiny qwen4_exp).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from ngram_memory import codecs
from ngram_memory.backends.base import RowDiff
from ngram_memory.backends.graft import NGramGraft
from ngram_memory.backends.qwen4exp_native import Qwen4ExpNativeBackend
from ngram_memory.cartridge import (CartridgeChain, CartridgeInfo, CompiledOverlayNP, ReaderIndex, check_against_graft,
                                    compile_chain, compile_served, make_cartridge_patch, read_graft_file,
                                    read_served, row_overlay_from, write_graft_file, write_served)
from ngram_memory.ledger import EditRecord, RowLedger, ZeroBase
from ngram_memory.patch import read_patch, verify_patch, write_patch
from ngram_memory.serve.hf_adapter import ple_table_module
from ngram_memory.tables import SparseRowStore

from conftest import build_host, rand_ids, tiny_fmt
from tiny_hosts import record, tiny_qwen4exp


class NumpyServedTable(nn.Module):
    """Stub table: base rows from numpy (fp32 decode, zero for grafts) + CompiledOverlayNP on the host rows, then
    the device cast -- what the serve does between decode_raw and the H2D copy (R/ple/source.py:243-245)."""

    def __init__(self, n: int, w: int, overlay: CompiledOverlayNP, base_f32: np.ndarray | None = None,
                 cast_bf16: bool = True):
        super().__init__()
        self.num_embeddings, self.embedding_dim = n, w
        self.overlay, self.base, self.cast_bf16 = overlay, base_f32, cast_bf16
        self.register_buffer("_dev", torch.zeros(0), persistent=False)
        self.hits = 0

    @property
    def weight(self):
        return self._dev

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.detach().cpu().numpy().reshape(-1)
        host = np.zeros((flat.size, self.embedding_dim), np.float32) if self.base is None else self.base[flat].copy()
        self.hits += self.overlay.apply(flat, host)
        if self.cast_bf16:
            host = codecs.bf16_round(host)
        return torch.from_numpy(host).reshape(*ids.shape, self.embedding_dim)


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def test_graft_cartridge_roundtrip_bitwise(tmp_path):
    host = build_host("qwen3_5")
    fmt = tiny_fmt()
    graft = NGramGraft(fmt)
    ids = rand_ids(5, 2, 24)
    ref_host = _logits(host, ids)
    # ---- graft file + ledger edits -> cartridge v0 (+ reader index), increment v1 with reset rows
    gpath = str(tmp_path / "graft-tiny.safetensors")
    gm = write_graft_file(gpath, fmt, graft.value_proj.detach().numpy(), host="tiny-qwen3_5",
                          host_config=host.config.to_dict())
    led = RowLedger(fmt.padded_rows, fmt.row_width, ZeroBase(fmt.row_width), spec_fingerprint=fmt.fingerprint())
    rows = graft.row_ids(ids).reshape(-1).unique()
    g = torch.Generator().manual_seed(1)
    half = rows.numel() // 2
    led.add(EditRecord("a", rows[:half].numpy(), (torch.randn(half, fmt.row_width, generator=g) * 0.7).numpy(),
                       mode="absolute", kind="cartridge"))
    led.add(EditRecord("b", rows[half // 2:].numpy(),
                       (torch.randn(rows.numel() - half // 2, fmt.row_width, generator=g) * 0.7).numpy(),
                       mode="absolute", kind="cartridge"))
    files = {"a.py": ids[0].tolist(), "b.py": ids[1].tolist()}
    ri = ReaderIndex.build(fmt.key, files)
    ri_sha = ri.save(str(tmp_path / "reader_index.safetensors"))
    p0 = led.to_patch(fmt)
    p0 = make_cartridge_patch(fmt, p0.row_ids, p0.values,
                              CartridgeInfo("tiny-repo", 0, source=dict(repo="tiny", commit="c0"),
                                            row_policy="count>=1", reader_index=dict(path="reader_index.safetensors",
                                                                                      sha256=ri_sha)),
                              graft=gm, edits=p0.manifest.edits)
    dead = rows[:5].numpy()  # Stage-2: these rows die (reset to base, i.e. 0 on a graft)
    refit = rows[-3:].numpy()
    p1 = make_cartridge_patch(fmt, refit, np.full((3, fmt.row_width), 0.25, np.float32),
                              CartridgeInfo("tiny-repo", 1, kind="increment", parent_version=0,
                                            source=dict(repo="tiny", commit_range="c0..c1")),
                              graft=gm, reset_row_ids=np.concatenate([dead, refit]))
    for i, p in enumerate((p0, p1)):
        write_patch(str(tmp_path / f"cart-v{i}.safetensors"), p)
    # ---- read back with numpy only, verify, chain, compile, serve file
    q0, q1 = (read_patch(str(tmp_path / f"cart-v{i}.safetensors")) for i in (0, 1))
    gm2, gt = read_graft_file(gpath)
    for q in (q0, q1):
        verify_patch(q, fmt)
        check_against_graft(q, gm2)
    assert np.array_equal(gt["value_proj"], codecs.bf16_round(graft.value_proj.detach().numpy()))
    chain = CartridgeChain([q0, q1])
    cids, csum, reset_only = chain.compose()
    assert set(dead.tolist()) <= set(reset_only.tolist()) and set(refit.tolist()) <= set(cids.tolist())
    sc = compile_chain(chain)
    spath = str(tmp_path / "cartridge-tiny-repo-v1.safetensors")
    write_served(spath, sc)
    sc2 = read_served(spath, graft=gm2, fmt=fmt)
    assert np.array_equal(sc2.rows_bits, codecs.bf16_bits(csum)) and sc2.version == 1
    assert ReaderIndex.load(str(tmp_path / "reader_index.safetensors"), ri_sha).dirty_rows(["a.py"]).size > 0
    # ---- (A) HF adapter: empty graft table + RowOverlay(engine_cast=bf16) from the composed fp32 chain
    inst = graft.install(host)
    assert torch.equal(_logits(host, ids), ref_host)  # empty cartridge: exact no-op
    ov = row_overlay_from(cids, csum, fmt.padded_rows, engine_cast=torch.bfloat16).install(ple_table_module(graft.ple))
    A = _logits(host, ids)
    ov.remove()
    # ---- (B) numpy overlay prototype fed the served file
    npo = CompiledOverlayNP(fmt.row_width, fmt.padded_rows, "bf16")
    npo.publish(sc2.row_ids, sc2.rows_bits)
    real_table = graft.ple.ple_embedding.ngram_embedding
    stub = NumpyServedTable(fmt.padded_rows, fmt.row_width, npo)
    graft.ple.ple_embedding.ngram_embedding = stub
    B = _logits(host, ids)
    graft.ple.ple_embedding.ngram_embedding = real_table
    # ---- (C) rows loaded straight into the graft
    graft.load_rows(SparseRowStore(fmt.padded_rows, fmt.row_width, torch.from_numpy(sc2.row_ids),
                                   torch.from_numpy(sc2.rows_f32())))
    C = _logits(host, ids)
    inst.remove()
    assert stub.hits > 0 and (A - ref_host).abs().max() > 0
    assert torch.equal(A, B) and torch.equal(A, C)
    record("graft_cartridge_roundtrip", A_eq_B=bool(torch.equal(A, B)), A_eq_C=bool(torch.equal(A, C)),
           overlay_hits=stub.hits, served_rows=int(sc2.row_ids.size), max_dlogit_vs_host=(A - ref_host).abs().max().item())
    # ---- refusals: a cartridge trained against another graft (other W_v seed), a broken chain
    other = tiny_fmt(value_seed=1)
    gm_o = write_graft_file(str(tmp_path / "graft-other.safetensors"), other,
                            NGramGraft(other).value_proj.detach().numpy(), host="tiny-qwen3_5")
    with pytest.raises(ValueError):
        check_against_graft(q0, gm_o)
    with pytest.raises(ValueError):
        read_served(spath, graft=gm_o)
    bad = make_cartridge_patch(fmt, refit, np.zeros((3, fmt.row_width)),
                               CartridgeInfo("tiny-repo", 3, kind="increment", parent_version=2), graft=gm)
    with pytest.raises(ValueError, match="parent_version"):
        CartridgeChain([q0, q1, bad])
    # rollback = drop the increment
    r = chain.upto(0).compose()
    assert np.array_equal(r[0], q0.row_ids) and np.array_equal(r[1], q0.values)


def test_native_patch_roundtrip_bitwise(tmp_path):
    torch.manual_seed(1)
    m = tiny_qwen4exp(vocab=5000)
    with torch.no_grad():  # a bf16 base table (as Flash-Next ships), held by an fp32 model
        W = m.model.layers[1].ple.ple_embedding.ngram_embedding.weight
        W.copy_(W.bfloat16().float())
    base_np = W.detach().numpy().copy()
    ids = torch.randint(0, 4999, (2, 20), generator=torch.Generator().manual_seed(0))
    ref = _logits(m, ids)
    be = Qwen4ExpNativeBackend(m)
    rows = be.row_ids(ids).reshape(-1).unique()[:60]
    g = torch.Generator().manual_seed(3)
    for k, sl in enumerate((slice(0, 40), slice(20, 60))):
        r = rows[sl]
        be.apply(RowDiff(r, torch.randn(r.numel(), be.spec.row_width, generator=g) * 0.02, mode="delta",
                         edit_id=f"f{k}"))
    path = str(tmp_path / "native-delta.safetensors")
    write_patch(path, be.export_patch(model_repo="tiny/qwen4exp"))
    be.remove()
    p = read_patch(path)
    lut = lambda r: base_np[r]
    assert "base_fingerprint" in verify_patch(p, be.spec, lut)
    # (A) HF adapter on a fresh model: served = bf16(base + delta)
    torch.manual_seed(1)
    m2 = tiny_qwen4exp(vocab=5000)
    tab = m2.model.layers[1].ple.ple_embedding.ngram_embedding
    with torch.no_grad():
        tab.weight.copy_(torch.from_numpy(base_np))
    ov = row_overlay_from(p.row_ids, p.values, tab.num_embeddings, engine_cast=torch.bfloat16).install(tab)
    A = _logits(m2, ids)
    ov.remove()
    assert torch.equal(_logits(m2, ids), ref)  # removing the overlay restores the base exactly
    # (B) numpy prototype, bf16 replacement rows compiled against the numpy base
    ids_c, bits = compile_served(p.row_ids, p.values, lut, "bf16")
    npo = CompiledOverlayNP(p.manifest.row_width, p.manifest.padded_rows, "bf16")
    npo.publish(ids_c, bits)
    m2.model.layers[1].ple.ple_embedding.ngram_embedding = NumpyServedTable(tab.num_embeddings, tab.embedding_dim,
                                                                            npo, base_np)
    B = _logits(m2, ids)
    # (B') numpy prototype in fp32-delta mode + device cast
    npd = CompiledOverlayNP(p.manifest.row_width, p.manifest.padded_rows, "delta")
    npd.publish(p.row_ids, p.values)
    m2.model.layers[1].ple.ple_embedding.ngram_embedding = NumpyServedTable(tab.num_embeddings, tab.embedding_dim,
                                                                            npd, base_np)
    B2 = _logits(m2, ids)
    assert (A - ref).abs().max() > 0
    assert torch.equal(A, B) and torch.equal(A, B2)
    record("native_patch_roundtrip", A_eq_B_bf16=bool(torch.equal(A, B)), A_eq_B_delta=bool(torch.equal(A, B2)),
           rows=int(p.row_ids.size), max_dlogit_vs_base=(A - ref).abs().max().item())

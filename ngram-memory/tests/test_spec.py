"""KeySpec / TableSpec / ModuleSpec: real Flash-Next values, JSON round trip, fingerprints, graft widths."""
import json

import pytest

from conftest import FIX
from ngram_memory.spec import (FormatSpec, KeySpec, ModuleSpec, TableSpec, flash_next_format, flash_next_key_spec,
                               graft_format, graft_key_spec, qwen25_proxy_key_spec)

FN_CFG = json.loads((FIX / "configs" / "qwen3.8-flash-next.config.json").read_text())


@pytest.mark.parametrize("fixture", ["flashnext_hash_buffers.json", "t1_real_buffers.json"])
def test_flash_next_key_matches_checkpoint_buffers(fixture):
    ck = json.loads((FIX / fixture).read_text())
    k = flash_next_key_spec()
    assert list(k.multipliers) == ck["layer_multipliers"] == [23703573157769, 20109073645365, 8052911324071]
    assert list(k.head_sizes) == ck["ngram_heads_vocab_sizes"]
    assert list(k.head_offsets) == ck["ngram_heads_offsets"]
    assert k.total_rows == 320_001_446


def test_flash_next_format_from_real_config():
    f = flash_next_format(FN_CFG)
    assert f == flash_next_format()  # constants == values read from the shipped config.json (seed omitted -> 1234)
    assert f.key.seed == 1234 and f.key.vocab_size == 248320 and f.key.eos_token_id == 248044
    assert f.padded_rows == 320_001_536
    assert f.table.rows_per_shard(f.key) == 2_500_012 and f.table.n_shards == 128
    assert f.row_width == 160 and f.module.hc_count == 4 and f.module.layer_idx == 1
    assert f.module.conv_state_len == 9
    assert f.table.shard_tensor_name(5) == "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_5.weight"


def test_json_roundtrip_and_fingerprint_stability():
    for f in (flash_next_format(), graft_format(896, qwen25_proxy_key_spec()),
              graft_format(2048, flash_next_key_spec(), arm="G-dir", gdir_alpha=0.3, gdir_taps=(0.1, 0.2, 0.3, 1.0))):
        s = f.to_json()
        g = FormatSpec.from_json(s)
        assert g == f and g.fingerprint() == f.fingerprint()
    # pinned: a change here changes every patch/edit record's contract
    assert flash_next_key_spec().fingerprint() == KEY_FP_FLASH_NEXT
    bad = json.loads(flash_next_format().to_json())
    bad["table"]["row_width"] = 159
    bad["module"]["ple_embed_dim"] = 159 * 16
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        FormatSpec.from_dict(bad)


KEY_FP_FLASH_NEXT = "e1e350dcc9bccf1763fdbb43a1e0e1a0d014ae3798018b444957955bdef5e3ed"


def test_fingerprint_sensitivity():
    k = flash_next_key_spec()
    fps = {k.fingerprint(), flash_next_key_spec("0" * 64).fingerprint(),
           KeySpec(248320, 248044, orders=(2, 3, 1)).fingerprint(), KeySpec(262144, 1).fingerprint(),
           KeySpec(248320, 248044, ple_layer_index=1).fingerprint()}
    assert len(fps) == 5
    a, b = graft_format(2048, k), graft_format(5120, k)
    assert a.key.fingerprint() == b.key.fingerprint() and a.table.fingerprint() != b.table.fingerprint()


def test_graft_widths_from_real_host_configs():
    k = flash_next_key_spec()
    for name, w in (("qwen3.8-27b.config.json", 320), ("qwen3.6-35b-a3b.config.json", 128)):
        c = json.loads((FIX / "configs" / name).read_text())
        d = c.get("text_config", c)["hidden_size"]
        f = graft_format(d, k)
        assert f.row_width == w and f.module.ple_embed_dim == d and f.padded_rows == 320_001_536


def test_qwen25_proxy_spec_matches_stage1_experiment():
    k = qwen25_proxy_key_spec()
    assert k.vocab_size == 151936 and k.eos_token_id == 151643 and k.head_sizes[0] == 200003
    f = graft_format(896, k)
    assert f.padded_rows == 3_201_408 and f.row_width == 56  # P/stage1/experiment run_full: 3,201,408 rows, w 56


def test_unigram_extension_layout():
    k2 = flash_next_key_spec()
    k3 = graft_key_spec(248320, 248044, orders=(2, 3, 1))
    assert k3.n_heads == 24 and k3.head_sizes[:16] == k2.head_sizes and k3.multipliers == k2.multipliers
    assert k3.heads_of_order(1) == tuple(range(16, 24))
    f = graft_format(5120, k3)
    assert f.row_width == 320 and f.module.ple_embed_dim == 24 * 320


def test_validation():
    with pytest.raises(ValueError):
        ModuleSpec(hidden_size=64, ple_embed_dim=64, hc_count=4, kind="graft")
    with pytest.raises(ValueError):
        ModuleSpec(hidden_size=64, ple_embed_dim=64, arm="G-x")
    with pytest.raises(ValueError):
        KeySpec(10, 1, orders=(2, 2))
    with pytest.raises(ValueError):
        FormatSpec(flash_next_key_spec(), TableSpec("t", 8), ModuleSpec(hidden_size=64, ple_embed_dim=64))
    t = TableSpec("t", 160, n_shards=128, shard_name_template="s{s}")
    assert t.shard_of(flash_next_key_spec(), 944958) == (0, 944958)
    with pytest.raises(IndexError):
        t.shard_of(flash_next_key_spec(), 320_001_536)


def test_decode_state_spec():
    from ngram_memory.backends.base import Backend, decode_state_spec
    s = decode_state_spec(flash_next_format())
    assert s.layer_idx == 1 and s.slots[1][1] == 9 and s.slots[2][1] == 2
    assert "10240 values" in s.slots[1][0]  # 4 streams x 2560 (§2.3: each 4*d values)
    assert hasattr(Backend, "forward")

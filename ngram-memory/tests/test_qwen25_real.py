"""Real weights (slow): G-lin graft on frozen Qwen2.5-0.5B fp32 with the Stage-1 proxy KeySpec
(P/stage1/experiment: no-op max|dlogit| = 0.0 bitwise; host decode parity 5.2e-5; trained-graft parity 4.2e-5).

Peak RSS is about 2.3 GB (fp32 weights); run alone. Writes a JSON summary to $NGRAM_MEMORY_LOG_DIR if set."""
import json
import os
import resource
from pathlib import Path

import pytest
import torch

from conftest import maxdiff
from ngram_memory.backends.graft.module import NGramGraft
from ngram_memory.backends.graft.qwen35 import install_graft
from ngram_memory.spec import graft_format, qwen25_proxy_key_spec

HOST = Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")
TEXT = Path("/home/user/patcarter883/minisglang-rdna4/python/minisgl/ple/state.py")


@torch.no_grad()
def _decode_parity(model, ids, P):
    full = model(input_ids=ids, use_cache=False).logits
    out = model(input_ids=ids[:, :P], use_cache=True)
    d = [maxdiff(out.logits, full[:, :P])]
    cache = out.past_key_values
    for t in range(P, ids.shape[1]):
        o = model(input_ids=ids[:, t:t + 1], past_key_values=cache, use_cache=True)
        d.append(maxdiff(o.logits[:, -1], full[:, t]))
    return max(d), full


@pytest.mark.slow
def test_qwen25_05b_noop_and_decode_parity():
    if not (HOST / "model.safetensors").exists():
        pytest.skip("Qwen2.5-0.5B weights not in the HF cache")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(str(HOST), dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(str(HOST))
    text = TEXT.read_text() if TEXT.exists() else "def f(x):\n    return x + 1\n" * 40
    ids = torch.tensor([tok(text)["input_ids"][:96]])
    sha0 = {k: v.clone() for k, v in list(model.state_dict().items())[:4]}
    res: dict = {}
    with torch.no_grad():
        base = model(input_ids=ids, use_cache=False).logits
    fmt = graft_format(896, qwen25_proxy_key_spec())
    g = NGramGraft(fmt)  # G-lin editor init, sparse rows
    assert fmt.padded_rows == 3_201_408 and fmt.row_width == 56
    with install_graft(model, g):
        with torch.no_grad():
            a = model(input_ids=ids, use_cache=False).logits
    res["noop_max_abs_dlogit"] = maxdiff(a, base)
    res["noop_bitwise_equal"] = bool(torch.equal(a, base))
    ids40 = ids[:, :40]
    res["host_decode_parity_no_graft"], _ = _decode_parity(model, ids40, 20)
    # rows at every row the window reads, 0.05/element (|out| / |hidden| ~ 0.1-0.2 at layer-1 input)
    rid = g.row_ids(ids).reshape(-1).unique()
    g.write_rows(rid, torch.randn(rid.numel(), 56, generator=torch.Generator().manual_seed(0)) * 0.05)
    with install_graft(model, g) as h:
        res["graft_decode_parity"], full = _decode_parity(model, ids40, 20)
        out = h.last["out"]
        hs = []
        hk = model.model.layers[1].register_forward_pre_hook(lambda m, a, kw: hs.append((a[0] if a else kw["hidden_states"])),
                                                              with_kwargs=True)
        with torch.no_grad():
            model(input_ids=ids40, use_cache=False)
        hk.remove()
    with torch.no_grad():
        after = model(input_ids=ids, use_cache=False).logits
    res["graft_effect_max_abs_dlogit"] = maxdiff(full, base[:, :40])
    res["injection_ratio_median"] = float((h.last["out"].norm(dim=-1) / (hs[0] - h.last["out"]).norm(dim=-1)).median())
    res["removed_restores_base_bitwise"] = bool(torch.equal(after, base))
    res["host_weights_unchanged"] = all(torch.equal(v, model.state_dict()[k]) for k, v in sha0.items())
    res["rows_written"] = int(rid.numel())
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    d = os.environ.get("NGRAM_MEMORY_LOG_DIR")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        (Path(d) / "qwen25_real.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    assert res["noop_bitwise_equal"] and res["noop_max_abs_dlogit"] == 0.0
    assert res["removed_restores_base_bitwise"] and res["host_weights_unchanged"]
    assert res["graft_effect_max_abs_dlogit"] > 1e-2
    # experiment: host 5.2e-5, trained graft 4.2e-5; allow the same order of magnitude
    assert res["graft_decode_parity"] < max(5 * res["host_decode_parity_no_graft"], 2e-4)

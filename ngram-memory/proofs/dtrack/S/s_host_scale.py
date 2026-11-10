"""S, part 2: the graft hosts' residual scale at the graft site (input of decoder index 1), on the same texts as
s_probe.py, so Flash-Next's measured ratios (|PLE out|/|X|, conv/|X|) can be turned into absolute settings
(G-dir alpha, G-lin output scale) for the M1b host (Qwen2.5-0.5B, local weights) and the M2 host
(Qwen3.6-35B-A3B, bf16 layer 0 + embedding rows range-read from the public repo, ~1.7 GB, deleted afterwards).

h_t = the hidden state entering decoder layer 1 (embedding + layer-0 output), i.e. what the graft's output is added
to (install_graft layer_idx=1). Reported: per-token RMS of h and of the embedding, by source.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/memory-organ/ngram-memory")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ngram_memory.dissect.hf_range import HFRangeReader, bf16_to_f32  # noqa: E402
from ngram_memory.dissect.slice_probe import gdir_alpha_for_ratio  # noqa: E402

S = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
D = Path(f"{S}/nm/m0/dissect")
HERE = Path(__file__).resolve().parent
Q25 = next(Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots").iterdir())
M35 = Path(f"{S}/models/Qwen/Qwen3.6-35B-A3B")
TOK35 = f"{S}/nm/graft/tok/Qwen__Qwen3.6-35B-A3B/tokenizer.json"
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def texts():
    import s_probe
    docs, _ = s_probe.build_corpus()
    return [(d["src"], d["text"]) for d in docs]


TOPK = 8


def rms_excl_top(h: torch.Tensor, k: int = TOPK) -> np.ndarray:
    """Per-token RMS over channels excluding the k channels with the largest mean |h| in this document (massive-
    activation channels dominate the plain RMS on some hosts)."""
    top = h.abs().mean(0).topk(k).indices
    keep = torch.ones(h.shape[-1], dtype=torch.bool)
    keep[top] = False
    return h[:, keep].pow(2).mean(-1).sqrt().numpy()


def summarize(name, rows):
    out = {}
    for src in ("all", "code", "md", "wiki"):
        if not any(src == "all" or r["src"] == src for r in rows):
            continue
        h = np.concatenate([r["h"] for r in rows if src == "all" or r["src"] == src])
        e = np.concatenate([r["e"] for r in rows if src == "all" or r["src"] == src])
        hx = np.concatenate([r["hx"] for r in rows if src == "all" or r["src"] == src])
        out[src] = dict(n=int(h.size), h_rms_median=float(np.median(h)), h_rms_mean=float(h.mean()),
                        h_rms_excl_top8ch_median=float(np.median(hx)),
                        h_rms_p05=float(np.quantile(h, .05)), h_rms_p95=float(np.quantile(h, .95)),
                        embed_rms_median=float(np.median(e)), h_over_embed_median=float(np.median(h / e)))
    log(name, json.dumps(out["all"]))
    return out


@torch.no_grad()
def qwen25(docs):
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM
    tok = Tokenizer.from_file(str(Q25 / "tokenizer.json"))
    m = AutoModelForCausalLM.from_pretrained(str(Q25), num_hidden_layers=1, dtype=torch.float32).eval()
    grab = {}
    hk = m.model.layers[0].register_forward_hook(lambda mod, a, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)))
    rows = []
    for src, text in docs:
        ids = torch.tensor([tok.encode(text, add_special_tokens=False).ids[:2048]])
        m.model(input_ids=ids)
        h = grab["h"][0]
        e = m.model.embed_tokens(ids)[0]
        rows.append(dict(src=src, h=h.pow(2).mean(-1).sqrt().numpy(), hx=rms_excl_top(h),
                         e=e.pow(2).mean(-1).sqrt().numpy()))
    hk.remove()
    return summarize("qwen2.5-0.5b", rows)


@torch.no_grad()
def qwen35(docs):
    from tokenizers import Tokenizer
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qm
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    tok = Tokenizer.from_file(TOK35)
    cfg_d = json.loads((M35 / "config.json").read_text())
    tc = dict(cfg_d["text_config"])
    for k in ("model_type",):
        tc.pop(k, None)
    tc["num_hidden_layers"] = 1
    tc["layer_types"] = tc["layer_types"][:1]
    tc["mtp_num_hidden_layers"] = 0
    cfg = Qwen3_5MoeTextConfig(**tc)
    cfg._experts_implementation = "eager"
    r = HFRangeReader("Qwen/Qwen3.6-35B-A3B", index_path=M35 / "model.safetensors.index.json")
    with torch.device("meta"):
        layer = qm.Qwen3_5MoeDecoderLayer(cfg, 0)
    pre = "model.language_model.layers.0."
    names = [pre + k for k in layer.state_dict()]
    infos = [r.tensor(n) for n in names]
    log(f"35B-A3B layer 0: {len(infos)} tensors, {sum(i.nbytes for i in infos) / 2**30:.2f} GiB")
    raw_dir = D / "raw/host35"
    sd = {}
    for i in infos:
        p = raw_dir / f"{i.name}.bin"
        if not (p.exists() and p.stat().st_size == i.nbytes):
            r.read_tensor_bytes(i, chunk=32 << 20, threads=4, out_path=p)
        a = np.fromfile(p, dtype=np.uint16 if i.dtype == "BF16" else np.float32)
        t = torch.from_numpy(a.view(np.int16)).view(torch.bfloat16) if i.dtype == "BF16" else torch.from_numpy(a)
        sd[i.name[len(pre):]] = t.reshape(i.shape)
    layer.load_state_dict(sd, assign=True, strict=True)
    layer = layer.to(torch.bfloat16).eval()
    encs = [tok.encode(t, add_special_tokens=False).ids[:2048] for _, t in docs]
    uniq = np.unique(np.concatenate([np.asarray(e) for e in encs]))
    einfo = r.tensor("model.language_model.embed_tokens.weight")
    E = bf16_to_f32(r.read_rows(einfo, uniq, threads=32).view(np.uint16))
    idx = {int(t): j for j, t in enumerate(uniq)}
    log(f"35B-A3B: {uniq.size} embedding rows fetched; reader {r.stats.to_dict()}")
    rows = []
    for (src, _), ids in zip(docs, encs):
        e = torch.from_numpy(E[[idx[int(t)] for t in ids]])[None]
        h = layer(e.to(torch.bfloat16), position_embeddings=None)
        h = (h[0] if isinstance(h, tuple) else h)[0].float()
        rows.append(dict(src=src, h=h.pow(2).mean(-1).sqrt().numpy(), hx=rms_excl_top(h),
                         e=e[0].pow(2).mean(-1).sqrt().numpy()))
    out = summarize("qwen3.6-35b-a3b", rows)
    out["download"] = r.stats.to_dict()
    shutil.rmtree(raw_dir, ignore_errors=True)
    log("deleted", raw_dir)
    return out


def main():
    docs = texts()
    cache = D / "S/host_scale_cache.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    res = {}
    for host, fn in (("qwen2.5-0.5b", qwen25), ("qwen3.6-35b-a3b", qwen35)):
        if host == "qwen3.6-35b-a3b" and "--skip35" in sys.argv:
            continue
        res[host] = done[host] if host in done else fn(docs)
        done[host] = res[host]
        cache.write_text(json.dumps(done, indent=1))
    hosts = list(res)
    # Flash-Next's own X (layer-1 input) with the same channel exclusion, per stream, from the cached slice output
    pt = np.load(D / "S/per_token.npz")
    n_tok = pt["out_rms"].shape[0]
    corpus = json.loads((HERE / "s_corpus.json").read_text())
    lens = [f["tokens"] for f in corpus["files"]]
    st0 = np.concatenate([[0], np.cumsum(lens)])
    xs = []
    for j in range(len(lens)):
        x = torch.from_numpy(np.fromfile(D / "S/x1.f32", dtype=np.float32, count=int(lens[j]) * 10240,
                                         offset=int(st0[j]) * 10240 * 4).reshape(-1, 10240)).unflatten(-1, (4, 2560))
        xs.append(np.stack([rms_excl_top(x[:, s_]) for s_ in range(4)], 1))
    xs = np.concatenate(xs)
    res["flash_next_x_excl_top8ch"] = dict(
        x_rms_median=[float(np.median(pt["x_rms"][:, s_])) for s_ in range(4)],
        x_rms_excl_median=[float(np.median(xs[:, s_])) for s_ in range(4)],
        out_over_x_excl_median=[float(np.median(pt["out_rms"][:, s_] / xs[:, s_])) for s_ in range(4)],
        conv_over_x_excl_median=[float(np.median(pt["conv_rms"][:, s_] / xs[:, s_])) for s_ in range(4)])
    s = json.loads((HERE / "s_results.json").read_text())
    st = s["streams"]["all"]
    z = np.load(D / "S/nconv_sample.npy")
    conv_x = [x["conv_over_x"]["median"] for x in st]
    out_x = [x["out_over_x"]["median"] for x in st]
    rec = {}
    for host in hosts:
        r = res[host]
        rh = r["all"]["h_rms_median"]
        rec[host] = dict(
            h_rms_median=rh,
            gdir_alpha_per_fn_stream=[gdir_alpha_for_ratio(z, c * rh) for c in conv_x],
            gdir_alpha_stream_median=gdir_alpha_for_ratio(z, float(np.median(conv_x)) * rh),
            target_out_rms_per_fn_stream=[o * rh for o in out_x])
    res["recommendation_inputs"] = dict(fn_conv_over_x_median=conv_x, fn_out_over_x_median=out_x, per_host=rec)
    import resource
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    (HERE / "s_host_scale.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""AWQ host layers for M2-0 / M2 (DESIGN §4.6, §7.5 #1, §8): one decoder layer of
``cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit`` loaded from safetensors with experts kept packed, plus the on-disk
per-layer cache the M2-0 validators read (exact tensor bytes, sha256 and source revision per tensor).

Built on ``host_quant`` (dequant, ``QuantMoEExperts``, ``build_decoder_layer``, range readers); this module adds:

* ``load_layer``: ``build_decoder_layer`` + the GDN decay parameters ``A_log`` / ``dt_bias`` kept in fp32
  (F16 -> fp32 is exact; F16 -> bf16 is not for every value) + a single-checkpoint provenance check. The
  checkpoint's AWQ smoothing is folded into its dense tensors (``post_attention_layernorm``, router, shared
  expert), so a layer must come entirely from one checkpoint: mixing in dense tensors from the bf16 original
  changes the function. ``load_layer`` refuses a reader whose layer tensors come from more than one source.
* ``layer_memory``: packed / dense / transient bytes of a loaded layer.
* ``dense_reference_layer``: the stock HF layer with every tensor in a float dtype and the experts dequantised
  by a caller-supplied function (the validators pass the independent numpy reference, tests/hq_reference.py).
* ``write_layer_cache``: copy named tensors from any ``CheckpointReader`` (HTTP range reads or local) into one
  safetensors file per layer with byte-identical data, a small ``model.safetensors.index.json`` (so
  ``CheckpointReader.local`` reads the cache unchanged) and ``MANIFEST.json``; idempotent, byte-capped, and it
  refuses reads that approach a whole shard.

No gptqmodel, no HF AWQ integration. Torch code, so it lives in the graft subpackage, not the core engine.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch import nn

from . import host_quant as hq

__all__ = [
    "REPO", "REVISION", "LP", "KEEP_FP32", "MixedCheckpointError", "layer_tensor_names", "layer_provenance",
    "load_layer", "layer_memory", "dense_reference_layer", "write_layer_cache", "read_manifest",
    "CachedHeaderFile", "ST_DTYPE_NAMES", "plan_layer_cache",
]

REPO = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
REVISION = "f1e8415c4f29508fe281da4e256df26573213bfd"
LP = "model.language_model.layers"
KEEP_FP32 = ("linear_attn.A_log", "linear_attn.dt_bias")
ST_DTYPE_NAMES = {v: k for k, v in hq._ST_DTYPES.items()}


class MixedCheckpointError(ValueError):
    """A layer's tensors come from more than one checkpoint (or from an unexpected one)."""


# ------------------------------------------------------------------------------------------------------------
# provenance
# ------------------------------------------------------------------------------------------------------------

def layer_tensor_names(reader: hq.CheckpointReader, layer_idx: int, prefix: str = LP,
                       parts: Iterable[str] | None = None) -> list[str]:
    """Checkpoint names of one decoder layer (optionally only the submodules in ``parts``)."""
    base = f"{prefix}.{layer_idx}."
    ps = tuple(parts) if parts is not None else None
    out = []
    for n in reader.keys():
        if not n.startswith(base):
            continue
        rel = n[len(base):]
        if ps is None or any(rel == p or rel.startswith(p + ".") for p in ps):
            out.append(n)
    return sorted(out)


def _file_source(f: hq.SafetensorsFile, name: str) -> tuple[str, str]:
    loc = f.location
    if f.remote:
        p = urllib.parse.urlparse(loc).path.strip("/").split("/")
        if "resolve" in p:                                   # /{owner}/{repo}/resolve/{rev}/{file}
            i = p.index("resolve")
            return "/".join(p[:i]), p[i + 1]
        return "url", loc
    man = _manifest_cached(str(Path(loc).parent.resolve()))
    if man is not None:
        t = man.get("tensors", {}).get(name, {})
        return t.get("repo", man.get("repo", "?")), t.get("revision", man.get("revision", "?"))
    return "local", str(Path(loc).parent.resolve())


def layer_provenance(reader: hq.CheckpointReader, layer_idx: int, prefix: str = LP,
                     parts: Iterable[str] | None = None) -> dict[tuple[str, str], int]:
    """{(repo, revision): tensor count} over the layer's tensors in ``reader`` (local files are identified by
    the cache's MANIFEST.json, else by their directory)."""
    out: dict[tuple[str, str], int] = {}
    for n in layer_tensor_names(reader, layer_idx, prefix, parts):
        src = _file_source(reader.files[reader.weight_map[n]], n)
        out[src] = out.get(src, 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------
# layer loading
# ------------------------------------------------------------------------------------------------------------

def load_layer(cfg: dict, layer_idx: int, reader: hq.CheckpointReader, *, compute_dtype=torch.bfloat16,
               device: str | torch.device = "cpu", experts_per_group: int = 32, act_quant: str | None = None,
               parts: tuple[str, ...] | None = None, keep_fp32: tuple[str, ...] = KEEP_FP32,
               dense_dtype: torch.dtype | None = None, expect_source: tuple[str, str] | None = None,
               prefix: str = LP) -> tuple[nn.Module, Any]:
    """One frozen decoder layer of the AWQ host (experts packed in ``QuantMoEExperts``, dense tensors cast to
    ``dense_dtype`` = ``compute_dtype`` by default, ``keep_fp32`` parameters re-read and kept fp32). Raises
    ``MixedCheckpointError`` unless every tensor of the layer comes from one source (and from
    ``expect_source`` = (repo, revision) when given). Returns (layer, text_config)."""
    prov = layer_provenance(reader, layer_idx, prefix, parts)
    if not prov:
        raise KeyError(f"no tensors of layer {layer_idx} (parts={parts}) in the reader")
    if len(prov) != 1:
        raise MixedCheckpointError(f"layer {layer_idx} mixes checkpoints {sorted(prov)}; the AWQ smoothing is "
                                   "folded into the dense tensors, so load a layer from one checkpoint only")
    src = next(iter(prov))
    if expect_source is not None and tuple(expect_source) != src:
        raise MixedCheckpointError(f"layer {layer_idx} comes from {src}, expected {tuple(expect_source)}")
    layer, tcfg = hq.build_decoder_layer(cfg, layer_idx, reader, compute_dtype=compute_dtype, device=device,
                                         prefix=prefix, experts_per_group=experts_per_group, act_quant=act_quant,
                                         dense_dtype=dense_dtype, parts=parts)
    base = f"{prefix}.{layer_idx}"
    want = lambda n: parts is None or any(n == p or n.startswith(p + ".") for p in parts)  # noqa: E731
    params = dict(layer.named_parameters())
    for n in keep_fp32:
        if n in params and want(n) and f"{base}.{n}" in reader:
            parent, _, leaf = n.rpartition(".")
            mod = layer.get_submodule(parent) if parent else layer
            v = reader.get(f"{base}.{n}")
            mod._parameters[leaf] = nn.Parameter(v.to(device=device, dtype=torch.float32), requires_grad=False)
    layer.awq35_source = src
    return layer, tcfg


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def layer_memory(layer: nn.Module) -> dict[str, Any]:
    """Resident bytes of a loaded layer: packed 4-bit buffers, dense parameters (by dtype), other buffers; and
    the largest just-in-time dequant transient (one expert group, or one QuantLinear, in its compute dtype)."""
    packed, transient, nq = 0, 0, 0
    quant_ids = set()
    for m in layer.modules():
        if isinstance(m, hq.QuantMoEExperts):
            nq += 1
            quant_ids.update(id(b) for b in m.buffers())
            packed += m.packed_bytes()
            el = torch.empty((), dtype=m.compute_dtype).element_size()
            per = 3 * m.intermediate_dim * m.hidden_dim * el
            transient = max(transient, min(m.experts_per_group, m.num_experts) * per)
        elif isinstance(m, hq.QuantLinear):
            nq += 1
            quant_ids.update(id(b) for b in m.buffers())
            packed += m.packed_bytes()
            el = torch.empty((), dtype=m.compute_dtype).element_size()
            transient = max(transient, m.in_features * m.out_features * el)
    dense_by: dict[str, int] = {}
    for p in layer.parameters():
        if p.is_meta:
            continue
        dense_by[str(p.dtype)] = dense_by.get(str(p.dtype), 0) + _nbytes(p)
    other = sum(_nbytes(b) for b in layer.buffers() if id(b) not in quant_ids and not b.is_meta)
    dense = sum(dense_by.values())
    return dict(packed_bytes=packed, dense_param_bytes=dense, dense_by_dtype=dense_by, other_buffer_bytes=other,
                resident_bytes=packed + dense + other, dequant_transient_bytes=transient, quant_modules=nq)


def dense_reference_layer(cfg: dict, layer_idx: int, reader: hq.CheckpointReader,
                          expert_dequant: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int], torch.Tensor], *,
                          dtype: torch.dtype = torch.float32, prefix: str = LP,
                          parts: tuple[str, ...] | None = None,
                          on_matrix: Callable[[int, str, torch.Tensor], None] | None = None) -> tuple[nn.Module, Any]:
    """Stock HF decoder layer with every tensor in ``dtype``; routed experts filled from
    ``expert_dequant(qweight, qzeros, scales, group_size) -> fp32 [N, K]`` (gate rows then up rows, HF's
    gate_up_proj layout). ``on_matrix(e, proj, w_fp32)`` sees every dequantised matrix (bit-exactness checks)."""
    scheme = hq.scheme_from_config(cfg)
    tcfg, Layer = hq._text_config(cfg)
    with torch.device("meta"):
        layer = Layer(tcfg, layer_idx)
    base = f"{prefix}.{layer_idx}"
    want = lambda n: parts is None or any(n == p or n.startswith(p + ".") for p in parts)  # noqa: E731
    for n, p in list(layer.named_parameters()):              # materialise only the wanted submodules
        if want(n):
            parent, _, leaf = n.rpartition(".")
            mod = layer.get_submodule(parent) if parent else layer
            mod._parameters[leaf] = nn.Parameter(torch.empty(p.shape, dtype=dtype), requires_grad=False)
    if any(b.is_meta for n, b in layer.named_buffers() if want(n)):
        raise ValueError("unexpected buffers in the HF layer")
    with torch.no_grad():
        ex = getattr(getattr(layer, "mlp", None), "experts", None)
        if ex is not None and want("mlp.experts"):
            I = ex.intermediate_dim
            for e in range(ex.num_experts):
                for p in ("gate_proj", "up_proj", "down_proj"):
                    pre = f"{base}.mlp.experts.{e}.{p}"
                    t = reader.get_many([f"{pre}.qweight", f"{pre}.qzeros", f"{pre}.scales"])
                    w = expert_dequant(t[f"{pre}.qweight"], t[f"{pre}.qzeros"], t[f"{pre}.scales"], scheme.group_size)
                    if on_matrix is not None:
                        on_matrix(e, p, w)
                    if p == "gate_proj":
                        ex.gate_up_proj[e, :I].copy_(w)
                    elif p == "up_proj":
                        ex.gate_up_proj[e, I:].copy_(w)
                    else:
                        ex.down_proj[e].copy_(w)
        names = [n for n, _ in layer.named_parameters() if not n.startswith("mlp.experts.") and want(n)]
        t = reader.get_many([f"{base}.{n}" for n in names])
        params = dict(layer.named_parameters())
        for n in names:
            v = t[f"{base}.{n}"]
            if tuple(v.shape) != tuple(params[n].shape):
                raise ValueError(f"{base}.{n}: checkpoint {tuple(v.shape)} vs HF {tuple(params[n].shape)}")
            params[n].copy_(v.to(dtype))
    for p in layer.parameters():
        p.requires_grad_(False)
    return layer.eval(), tcfg


# ------------------------------------------------------------------------------------------------------------
# per-layer cache (exact bytes + MANIFEST)
# ------------------------------------------------------------------------------------------------------------

class CachedHeaderFile(hq.SafetensorsFile):
    """Remote safetensors file whose header bytes are already known (a cached shard header), so opening it
    costs no request; reads behave exactly like ``SafetensorsFile``."""

    def __init__(self, location: str, header_raw: bytes, *, timeout: float = 120.0, retries: int = 4):
        self.location = location
        self.remote = location.startswith("http://") or location.startswith("https://")
        self.timeout, self.retries = timeout, retries
        self.bytes_read = 0
        self._op = hq._opener() if self.remote else None
        self._fd = None if self.remote else os.open(location, os.O_RDONLY)
        self.header = json.loads(header_raw)
        self.metadata = self.header.pop("__metadata__", None)
        self.data_start = 8 + len(header_raw)


def _manifest_cached(d: str) -> dict | None:
    p = Path(d) / "MANIFEST.json"
    if not p.exists():
        return None
    key = (d, p.stat().st_mtime_ns)
    if _MANIFESTS.get(d, (None,))[0] != key:
        _MANIFESTS[d] = (key, json.loads(p.read_text()))
    return _MANIFESTS[d][1]


_MANIFESTS: dict[str, tuple] = {}


def read_manifest(cache_dir: str | Path) -> dict | None:
    p = Path(cache_dir) / "MANIFEST.json"
    return json.loads(p.read_text()) if p.exists() else None


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def _write_safetensors(path: Path, items: list[tuple[str, str, list[int], bytes]],
                       metadata: dict[str, str] | None = None) -> None:
    """Minimal safetensors writer that stores the given raw bytes unchanged (items: name, dtype, shape, raw)."""
    hdr: dict[str, Any] = {"__metadata__": metadata or {"format": "pt"}}
    off = 0
    for n, dt, shape, raw in items:
        hdr[n] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for *_, raw in items:
            f.write(raw)
    os.replace(tmp, path)


def _raw(f: hq.SafetensorsFile, name: str) -> bytes:
    _, _, a, b = f.info(name)
    return f.read(a, b)


def _reuse_index(dirs: Iterable[str | Path], repo: str, revision: str) -> dict[str, tuple[Path, dict]]:
    """name -> (local dir, manifest entry) for tensors cached elsewhere from the same repo + revision."""
    out: dict[str, tuple[Path, dict]] = {}
    for d in dirs:
        d = Path(d)
        man = read_manifest(d)
        if man is None:
            continue
        for n, t in man.get("tensors", {}).items():
            if t.get("repo", man.get("repo")) == repo and t.get("revision", man.get("revision")) == revision \
                    and "sha256" in t:
                out.setdefault(n, (d, t))
    return out


def _same_file(have_file: str, plan_file: str) -> bool:
    """A cached tensor sits in the planned file or in one of its append-only parts ('L3.p1.safetensors')."""
    stem = plan_file[: -len(".safetensors")] if plan_file.endswith(".safetensors") else plan_file
    return have_file == plan_file or (have_file.startswith(stem + ".p") and have_file.endswith(".safetensors"))


def plan_layer_cache(reader: hq.CheckpointReader, names: list[str], out_dir: str | Path, *, repo: str,
                     revision: str, file_for: Callable[[str], str],
                     reuse_dirs: Iterable[str | Path] = ()) -> dict:
    """What ``write_layer_cache`` would do, without reading any tensor: per tensor the source entry, and the
    sets present / reusable / to fetch, with ``fetch_bytes`` (the bytes a run would download) and
    ``new_disk_bytes`` (the bytes it would add on disk: fetched + reused; existing cache files are never
    rewritten)."""
    out_dir = Path(out_dir)
    man = read_manifest(out_dir) or {}
    if man and (man.get("repo"), man.get("revision")) != (repo, revision):
        raise MixedCheckpointError(f"{out_dir} caches {man.get('repo')}@{man.get('revision')}, not {repo}@{revision}")
    have: dict[str, dict] = dict(man.get("tensors", {}))
    names = sorted(set(names))
    missing = [n for n in names if n not in reader]
    if missing:
        raise KeyError(f"{len(missing)} tensors not in the checkpoint, e.g. {missing[:3]}")
    plan: dict[str, dict] = {}
    for n in names:
        f = reader.files[reader.weight_map[n]]
        dt, shape, a, b = f.info(n)
        plan[n] = dict(file=file_for(n), shard=reader.weight_map[n], dtype=ST_DTYPE_NAMES[dt], shape=shape,
                       src_offsets=[a, b], nbytes=b - a, repo=repo, revision=revision)
    reuse = _reuse_index(reuse_dirs, repo, revision)
    local_have = {n for n in names if n in have and have[n]["src_offsets"] == plan[n]["src_offsets"]
                  and have[n]["shard"] == plan[n]["shard"] and _same_file(have[n]["file"], plan[n]["file"])
                  and (out_dir / have[n]["file"]).exists()}
    reusable = {n for n in names if n not in local_have and n in reuse
                and reuse[n][1].get("src_offsets") == plan[n]["src_offsets"]
                and reuse[n][1].get("shard", plan[n]["shard"]) == plan[n]["shard"]}
    fetch = [n for n in names if n not in local_have and n not in reusable]
    by_shard: dict[str, int] = {}
    for n in fetch:
        by_shard[plan[n]["shard"]] = by_shard.get(plan[n]["shard"], 0) + plan[n]["nbytes"]
    return dict(man=man, have=have, names=names, plan=plan, reuse=reuse, local_have=local_have,
                reusable=reusable, fetch=fetch, by_shard=by_shard,
                fetch_bytes=sum(plan[n]["nbytes"] for n in fetch),
                new_disk_bytes=sum(plan[n]["nbytes"] for n in names if n not in local_have),
                present_bytes=sum(plan[n]["nbytes"] for n in local_have))


def write_layer_cache(reader: hq.CheckpointReader, names: list[str], out_dir: str | Path, *, repo: str,
                      revision: str, file_for: Callable[[str], str], max_bytes: float = 1.3e9,
                      max_shard_fraction: float = 0.5, reuse_dirs: Iterable[str | Path] = (),
                      max_gap: int = 64 << 10, purpose: str = "", log: Callable[[str], None] = print) -> dict:
    """Copy tensors ``names`` from ``reader`` into ``out_dir`` (one safetensors file per ``file_for(name)``,
    e.g. 'L0.safetensors'), byte for byte. Tensors already in ``out_dir`` (MANIFEST sha256 verified) are kept;
    tensors cached in ``reuse_dirs`` from the same repo + revision with matching source offsets are copied
    locally after a sha256 check; the rest are range-read. Append-only: an existing cache file is never
    rewritten; new tensors for it go to a new part file ('L3.p1.safetensors'), so a run needs only the new
    bytes in RAM and on disk. Refuses to start if the bytes to fetch exceed ``max_bytes`` or, for any source
    shard, ``max_shard_fraction`` of that shard. Returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pl = plan_layer_cache(reader, names, out_dir, repo=repo, revision=revision, file_for=file_for,
                          reuse_dirs=reuse_dirs)
    man, have, names, plan, reuse = pl["man"], pl["have"], pl["names"], pl["plan"], pl["reuse"]
    local_have, reusable, fetch, by_shard, fetch_bytes = (pl["local_have"], pl["reusable"], pl["fetch"],
                                                         pl["by_shard"], pl["fetch_bytes"])
    for s, nb in by_shard.items():
        f = reader.files[s]
        size = f.data_start + max(v["data_offsets"][1] for v in f.header.values())
        if nb > max_shard_fraction * size:
            raise ValueError(f"refusing to read {nb:,} B of {s} ({nb / size:.0%} of the {size:,} B shard): "
                             "fetch fewer tensors, never whole shards")
    if fetch_bytes > max_bytes:
        raise ValueError(f"refusing to fetch {fetch_bytes:,} B > max_bytes {int(max_bytes):,}")
    log(f"cache {out_dir}: {len(local_have)} present, {len(reusable)} reused, {len(fetch)} to fetch "
        f"({fetch_bytes / 1e6:.1f} MB)")

    raw: dict[str, bytes] = {}
    t0 = time.time()
    before = reader.bytes_read
    if fetch:
        for s in sorted(by_shard):
            ns = [n for n in fetch if plan[n]["shard"] == s]
            # get_many decodes tensors; re-encode exactly by taking the raw byte view
            got = reader.files[s].get_many(ns, max_gap=max_gap)
            for n in ns:
                raw[n] = got[n].contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
                del got[n]
    fetched_bytes = reader.bytes_read - before
    fetch_s = time.time() - t0
    for n in sorted(reusable):
        d, ent = reuse[n]
        lr = hq.CheckpointReader.local(d)
        b = _raw(lr.files[lr.weight_map[n]], n)
        if _sha(b) != ent["sha256"]:
            raise IOError(f"reuse {d}: sha256 mismatch for {n}")
        raw[n] = b
    new_by_file: dict[str, list[str]] = {}
    for n in names:
        if n not in local_have:
            new_by_file.setdefault(plan[n]["file"], []).append(n)
    for f, target in sorted(new_by_file.items()):
        dest = f
        if (out_dir / f).exists():                     # append-only: never rewrite (or copy) an existing file
            stem, k = f[: -len(".safetensors")], 1
            while (out_dir / f"{stem}.p{k}.safetensors").exists():
                k += 1
            dest = f"{stem}.p{k}.safetensors"
        path = out_dir / dest
        items = [(n, plan[n]["dtype"], plan[n]["shape"], raw[n]) for n in target]
        _write_safetensors(path, items, {"format": "pt", "source": f"{repo}@{revision}"})
        for n in target:
            ent = dict(plan[n], file=dest)
            if len(raw[n]) != ent["nbytes"]:
                raise IOError(f"{n}: {len(raw[n])} B != {ent['nbytes']}")
            ent["sha256"] = _sha(raw[n])
            ent["source"] = ("reuse:" + str(reuse[n][0])) if n in reusable else "range"
            have[n] = ent
            del raw[n]
        log(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB, {len(target)} tensors)")
    # verify every cached tensor against its sha256 by reading the written files back
    files_meta = {}
    for f in sorted({t["file"] for t in have.values()}):
        path = out_dir / f
        sf = hq.SafetensorsFile(str(path))
        for n, t in have.items():
            if t["file"] == f and _sha(_raw(sf, n)) != t["sha256"]:
                raise IOError(f"verification failed: {path} {n}")
        sf.close()
        files_meta[f] = dict(bytes=path.stat().st_size, sha256=_file_sha(path))
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"source": f"{repo}@{revision}"}, "weight_map": {n: t["file"] for n, t in sorted(have.items())}},
        indent=1))
    hist = list(man.get("history", []))
    hist.append(dict(time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), purpose=purpose,
                     requested=len(names), present=len(local_have), reused=len(reusable), fetched=len(fetch),
                     fetched_tensor_bytes=fetch_bytes, http_bytes=fetched_bytes, fetch_s=round(fetch_s, 2)))
    man = dict(repo=repo, revision=revision, files=files_meta,
               weight_bytes=sum(t["nbytes"] for t in have.values()),
               http_weight_bytes_total=sum(h.get("http_bytes", 0) for h in hist),
               history=hist, tensors=dict(sorted(have.items())))
    (out_dir / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man

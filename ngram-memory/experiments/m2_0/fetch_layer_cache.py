"""M2-0: cache selected decoder-layer tensors of the 35B AWQ host by HTTP range read (never whole shards).

Writes ``<cache-dir>/cyankiwi_35b/``:
  L{n}.safetensors              the requested tensors of layer n, bytes and dtypes identical to the hub shards
  model.safetensors.index.json  name -> file, so ``host_quant.CheckpointReader.local`` reads the cache unchanged
  config.json                   the pinned revision's config (from the HF cache, else fetched; sha256 recorded)
  MANIFEST.json                 repo, revision, per tensor: source shard, absolute offsets, dtype, shape, sha256
  headers/{shard}.hdr.json      raw shard headers (reused on later runs; the scout's copies are picked up too)

Idempotent (present tensors are kept after a sha256 check), append-only (an existing L{n}.safetensors is never
rewritten: new tensors of that layer go to L{n}.p1.safetensors, ..., so a run needs disk and RAM only for the new
bytes), byte-capped per call (--max-bytes, default 1.3e9) AND against the track's cumulative download budget
(--track-budget, default 2e9: the weight bytes recorded in every MANIFEST.json under --cache-dir plus the
DOWNLOADS.json ledger of reads that were not cached), refuses to read more than half of any shard, refuses when the
disk would keep less than --min-free-gb, and can copy tensors another cache already holds (--reuse-from DIR with a
MANIFEST.json of the same repo + revision, e.g. B1's ``b1_gdn_L0``) instead of downloading them again.
--dry-run prints the bytes a run would download and add on disk (not the layer total).

Layer specs: ``L`` or ``L:PRESET`` or ``L:part+part`` with parts as decoder-layer submodule paths.
Presets: all, dense (all but the routed experts), gdn (input_layernorm+linear_attn), attn
(input_layernorm+self_attn), moe (post_attention_layernorm+mlp).

  python experiments/m2_0/fetch_layer_cache.py --cache-dir $SCRATCH/m2_0_cache --layer 0 --layer 3:dense \
      --reuse-from $SCRATCH/m2_0_cache/b1_gdn_L0
  python experiments/m2_0/fetch_layer_cache.py --cache-dir ... --layer 3 --dry-run     # plan only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ngram_memory.backends.graft import awq35  # noqa: E402
from ngram_memory.backends.graft import host_quant as hq  # noqa: E402

SNAP = Path(f"/root/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-35B-A3B-AWQ-4bit/snapshots/{awq35.REVISION}")
BASE = f"https://huggingface.co/{awq35.REPO}/resolve/{awq35.REVISION}/"
PRESETS = {"all": None, "gdn": ("input_layernorm", "linear_attn"), "attn": ("input_layernorm", "self_attn"),
           "moe": ("post_attention_layernorm", "mlp")}


def _get(url: str, rng: tuple[int, int] | None = None) -> bytes:
    h = {"User-Agent": "ngram-memory/0.0.1"}
    if rng is not None:
        h["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
    with hq._opener().open(urllib.request.Request(url, headers=h), timeout=120) as r:
        b = r.read()
    if rng is not None and len(b) != rng[1] - rng[0]:
        raise IOError(f"short read {url} {rng}: {len(b)}")
    return b


def shard_header(shard: str, hdr_dirs: list[Path], save_dir: Path, stats: dict) -> bytes:
    for d in hdr_dirs:
        p = d / f"{shard}.hdr.json"
        if p.exists():
            raw = p.read_bytes()
            ln = d / f"{shard}.len"
            if ln.exists() and struct.unpack("<Q", ln.read_bytes())[0] != len(raw):
                raise IOError(f"{p}: length file disagrees")
            json.loads(raw)
            if d != save_dir:
                save_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, save_dir / p.name)
            return raw
    n = struct.unpack("<Q", _get(BASE + shard, (0, 8)))[0]
    raw = _get(BASE + shard, (8, 8 + n))
    stats["header_bytes"] = stats.get("header_bytes", 0) + 8 + n
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"{shard}.hdr.json").write_bytes(raw)
    (save_dir / f"{shard}.len").write_bytes(struct.pack("<Q", n))
    return raw


LEDGER = "DOWNLOADS.json"


def recorded_downloads(cache_root: Path) -> dict:
    """Weight bytes this track has downloaded so far, from what is on disk: every MANIFEST.json under
    ``cache_root`` (B2-style ``history[].http_bytes``, B1-style ``bytes_downloaded.weights``, the scout's
    ``track_b_weight_bytes_so_far``) plus the ``extra`` entries of DOWNLOADS.json (reads kept in RAM only)."""
    per: dict[str, int] = {}
    for m in sorted(cache_root.glob("*/MANIFEST.json")):
        man = json.loads(m.read_text())
        if "history" in man:
            nb = sum(int(h.get("http_bytes", 0)) for h in man["history"])
        elif "bytes_downloaded" in man:
            nb = int(man["bytes_downloaded"].get("weights", 0))
        else:
            nb = int(man.get("track_b_weight_bytes_so_far", 0))
        per[m.parent.name] = nb
    led = cache_root / LEDGER
    extra = json.loads(led.read_text()).get("extra", []) if led.exists() else []
    for e in extra:
        per["extra:" + e["what"]] = int(e["weight_bytes"])
    return dict(per_source=per, total=sum(per.values()))


def parse_layer(spec: str, cfg: dict) -> tuple[int, tuple[str, ...] | None, str]:
    L, _, p = spec.partition(":")
    L = int(L)
    p = p or "all"
    kind = cfg["text_config"]["layer_types"][L]
    if p == "dense":
        mixer = "linear_attn" if kind == "linear_attention" else "self_attn"
        parts = ("input_layernorm", mixer, "post_attention_layernorm", "mlp.gate", "mlp.shared_expert",
                 "mlp.shared_expert_gate")
    elif p in PRESETS:
        parts = PRESETS[p]
    else:
        parts = tuple(p.split("+"))
    return L, parts, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("NGM_M2_0_CACHE", "m2_0_cache")))
    ap.add_argument("--layer", action="append", default=[], help="L[:preset|part+part] (repeatable)")
    ap.add_argument("--layers", default="", help="comma list of L[:spec] (same as repeated --layer)")
    ap.add_argument("--max-bytes", type=float, default=1.3e9)
    ap.add_argument("--reuse-from", action="append", default=[], type=Path)
    ap.add_argument("--index", type=Path, default=SNAP / "model.safetensors.index.json")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--track-budget", type=float, default=2e9,
                    help="cumulative weight-byte download budget across every cache under --cache-dir")
    ap.add_argument("--min-free-gb", type=float, default=0.3, help="refuse if the disk would keep less free")
    a = ap.parse_args()
    specs = a.layer + [s for s in a.layers.split(",") if s]
    if not specs:
        ap.error("give at least one --layer")
    out = a.cache_dir / "cyankiwi_35b"
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    cfg_p = out / "config.json"
    if not cfg_p.exists():
        if (SNAP / "config.json").exists():
            shutil.copyfile(SNAP / "config.json", cfg_p)
        else:
            cfg_p.write_bytes(_get(BASE + "config.json"))
    if a.index.exists():
        idx = json.loads(a.index.read_text())
    else:
        idx = json.loads(_get(BASE + "model.safetensors.index.json"))
    cfg = json.loads(cfg_p.read_text())
    wm = idx["weight_map"]

    names: list[str] = []
    for s in specs:
        L, parts, p = parse_layer(s, cfg)
        base = f"{awq35.LP}.{L}."
        for n in wm:
            if n.startswith(base):
                rel = n[len(base):]
                if parts is None or any(rel == q or rel.startswith(q + ".") for q in parts):
                    names.append(n)
    shards = sorted({wm[n] for n in names})
    hdr_dirs = [out / "headers", a.cache_dir / "scout_hdr"]
    files = {s: awq35.CachedHeaderFile(BASE + s, shard_header(s, hdr_dirs, out / "headers", stats)) for s in shards}
    reader = hq.CheckpointReader(files, {n: wm[n] for n in names})
    layer_of = lambda n: int(n[len(awq35.LP) + 1:].split(".", 1)[0])  # noqa: E731
    file_for = lambda n: f"L{layer_of(n)}.safetensors"  # noqa: E731
    pl = awq35.plan_layer_cache(reader, names, out, repo=awq35.REPO, revision=awq35.REVISION, file_for=file_for,
                                reuse_dirs=a.reuse_from)
    rec = recorded_downloads(a.cache_dir)
    free = shutil.disk_usage(out).free
    summary = dict(tensors=len(names), shards=shards,
                   layer_tensor_bytes=sum(pl["plan"][n]["nbytes"] for n in pl["names"]),
                   present=len(pl["local_have"]), present_bytes=pl["present_bytes"], reusable=len(pl["reusable"]),
                   to_fetch=len(pl["fetch"]), bytes_to_fetch=pl["fetch_bytes"], new_disk_bytes=pl["new_disk_bytes"],
                   ram_needed_bytes_est=pl["new_disk_bytes"], disk_free_bytes=free,
                   track_downloaded_so_far=rec["total"], track_budget=int(a.track_budget),
                   track_after_this_run=rec["total"] + pl["fetch_bytes"],
                   header_bytes_fetched=stats.get("header_bytes", 0))
    if a.dry_run:
        print(json.dumps(dict(summary, recorded=rec["per_source"]), indent=1))
        return
    if rec["total"] + pl["fetch_bytes"] > a.track_budget:
        raise SystemExit(f"refusing: {pl['fetch_bytes']:,} B more would bring the track's recorded downloads to "
                         f"{rec['total'] + pl['fetch_bytes']:,} B > --track-budget {int(a.track_budget):,} "
                         f"({json.dumps(rec['per_source'])})")
    if free - pl["new_disk_bytes"] < a.min_free_gb * 1e9:
        raise SystemExit(f"refusing: adding {pl['new_disk_bytes'] / 1e6:.1f} MB would leave "
                         f"{(free - pl['new_disk_bytes']) / 1e9:.2f} GB free (< --min-free-gb {a.min_free_gb})")
    man = awq35.write_layer_cache(reader, names, out, repo=awq35.REPO, revision=awq35.REVISION,
                                  file_for=file_for, max_bytes=a.max_bytes,
                                  reuse_dirs=a.reuse_from, purpose="M2-0 B2 AWQ layer validation: " + ",".join(specs))
    man["config_sha256"] = hashlib.sha256(cfg_p.read_bytes()).hexdigest()
    man["index_sha256"] = hashlib.sha256(a.index.read_bytes()).hexdigest() if a.index.exists() else None
    man["header_bytes_fetched_this_run"] = stats.get("header_bytes", 0)
    man["headers"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out / "headers").glob("*.hdr.json"))}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("repo", "revision", "files", "weight_bytes", "http_weight_bytes_total")},
                     indent=1))
    print("last run:", json.dumps(man["history"][-1]))
    print("track downloads recorded:", json.dumps(recorded_downloads(a.cache_dir)))


if __name__ == "__main__":
    main()

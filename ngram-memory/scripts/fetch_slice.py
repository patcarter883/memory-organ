"""Range-fetch the dense tensors of the 2-layer Flash-Next slice (§D experiment S) to raw .bin files.

Fetches every `model.language_model.layers.0.*` tensor (5.19 GB bf16, of which the experts are 5.03 GB) and the
small layer-1 PLE tensors (key/value proj, norms, conv: 65.7 MB). The embedding table and the n-gram table rows
are NOT fetched here (slice_probe.py range-reads only the rows it needs).

    python scripts/fetch_slice.py --out <dir> [--index model.safetensors.index.json] [--threads 4]

Writes <dir>/<tensor name>.bin (raw little-endian bytes as stored) and <dir>/manifest.json
(name, dtype, shape, file, bytes). Already-complete files are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngram_memory.dissect.hf_range import FLASH_NEXT_REPO, HFRangeReader  # noqa: E402


def slice_tensor_names(index: dict[str, str]) -> list[str]:
    keep = []
    for k in sorted(index):
        if k.startswith("model.language_model.layers.0."):
            keep.append(k)
        elif k.startswith("model.language_model.layers.1.ple.") and "ngram_embedding" not in k:
            keep.append(k)
    return keep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--index", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--chunk-mb", type=int, default=32)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    r = HFRangeReader(FLASH_NEXT_REPO, index_path=a.index)
    names = slice_tensor_names(r.index())
    infos = [r.tensor(n) for n in names]
    total = sum(i.nbytes for i in infos)
    print(f"{len(infos)} tensors, {total} bytes ({total / 2**30:.2f} GiB)", flush=True)
    man = {}
    t0 = time.time()
    for i in infos:
        p = out / f"{i.name}.bin"
        man[i.name] = dict(dtype=i.dtype, shape=list(i.shape), file=i.file, nbytes=i.nbytes, path=p.name)
        if p.exists() and p.stat().st_size == i.nbytes and (out / f"{i.name}.ok").exists():
            continue
        t = time.time()
        r.read_tensor_bytes(i, chunk=a.chunk_mb << 20, threads=a.threads, out_path=p)
        (out / f"{i.name}.ok").write_text("ok")
        print(f"{i.name} {i.nbytes / 2**20:.1f} MiB in {time.time() - t:.1f}s; total {r.stats.bytes / 2**30:.2f} GiB "
              f"{time.time() - t0:.0f}s", flush=True)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print("done", json.dumps(r.stats.to_dict()), f"wall {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

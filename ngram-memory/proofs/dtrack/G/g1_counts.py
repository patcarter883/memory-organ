"""G step 1: the public proxy corpus and its 2-/3-gram counts under the Flash-Next tokenizer.

Proxy corpus (why): Flash-Next's rows were trained on its (unknown) pretraining mix, so the frequency that should
predict a row's coherence is the n-gram's frequency in *general* public text, not in the owner's repo. G0 used
400k tokens of the owner's own files, which ranks project idioms as "frequent". A public sample of the two kinds of
text the cartridges care about is the cheapest proxy we can get here:
  * code  : codeparrot/codeparrot-clean-valid (GitHub Python, deduplicated by the publisher), first ~200 MB of file
            contents in file order -- Python because the owner's repo is Python;
  * prose : Salesforce/wikitext wikitext-103-raw-v1 train shard 0, detokenised (lm-eval's wikitext detokenizer
            rules, so " @-@ " etc. do not create fake n-grams), first ~150 MB, one document per article.
Both are public and very likely inside any web-scale pretraining mix. Counts are per source; n-grams never cross a
document boundary (each document starts a new EOS segment, as in the census).

Outputs (scratch, large): counts_{code,prose}_{2,3}.npz (sorted int64 keys, counts) + corpus manifest.
Small summary: proofs/dtrack/G/g1_counts.json, printed to g1_counts.out.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "2")
sys.path.insert(0, "/home/user/memory-organ/ngram-memory")
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from ngram_memory.dissect.row_stats import count_keys, wikitext_articles, wikitext_detokenize  # noqa: E402

S = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
D = Path(f"{S}/nm/m0/dissect")
RAW = D / "raw/proxy"
OUT = D / "G"
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
TOK = f"{S}/nm/graft/tok/Qwen__Qwen3.8-Flash-Next/tokenizer.json"
V = 248320
CODE_CHARS = 200_000_000
PROSE_CHARS = 150_000_000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_docs():
    n = 0
    with gzip.open(RAW / "codeparrot_valid.json.gz", "rt") as f:
        for line in f:
            c = json.loads(line)["content"]
            yield c
            n += len(c)
            if n >= CODE_CHARS:
                return


def prose_docs():
    t = pq.read_table(RAW / "wikitext103_train0.parquet").column("text")
    n = 0
    for art in wikitext_articles(x.as_py() for x in t):
        s = wikitext_detokenize(art).strip()
        if len(s) < 200:
            continue
        yield s
        n += len(s)
        if n >= PROSE_CHARS:
            return


def tokenize(docs, tok: Tokenizer, batch: int = 512) -> tuple[list[np.ndarray], int, int]:
    out, buf, chars = [], [], 0
    for d in docs:
        buf.append(d)
        chars += len(d)
        if len(buf) == batch:
            out += [np.asarray(e.ids, dtype=np.int32) for e in tok.encode_batch(buf, add_special_tokens=False)]
            buf = []
    if buf:
        out += [np.asarray(e.ids, dtype=np.int32) for e in tok.encode_batch(buf, add_special_tokens=False)]
    return out, chars, sum(len(x) for x in out)


def main() -> None:
    tok = Tokenizer.from_file(TOK)
    man = dict(tokenizer=TOK, tokenizer_sha256=sha256(Path(TOK)),
               files={p.name: dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(RAW.iterdir())},
               sources={})
    for name, gen in (("code", code_docs), ("prose", prose_docs)):
        t0 = time.time()
        docs, chars, ntok = tokenize(gen(), tok)
        t1 = time.time()
        np.savez(OUT / f"tokens_{name}.npz", lens=np.array([len(d) for d in docs], np.int64),
                 ids=np.concatenate(docs).astype(np.int32))
        info = dict(docs=len(docs), chars=chars, tokens=ntok, tokenize_s=round(t1 - t0, 1))
        for n in (1, 2, 3):
            k, c = count_keys(docs, n, V)
            np.savez(OUT / f"counts_{name}_{n}.npz", keys=k, counts=c)
            info[f"distinct_{n}"] = int(k.shape[0])
            info[f"total_{n}"] = int(c.sum())
            info[f"max_{n}"] = int(c.max())
            info[f"once_{n}"] = int((c == 1).sum())
        info["count_s"] = round(time.time() - t1, 1)
        man["sources"][name] = info
        print(name, json.dumps(info), flush=True)
        del docs
    (OUT / "corpus_manifest.json").write_text(json.dumps(man, indent=1))
    (HERE / "g1_counts.json").write_text(json.dumps(man, indent=1))
    import resource
    man["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    (HERE / "g1_counts.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()

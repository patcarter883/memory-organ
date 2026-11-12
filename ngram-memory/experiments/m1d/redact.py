#!/usr/bin/env python
"""Copy the small JSON outputs of an M1d run directory with every planted canary secret redacted
(experiments/m1d/PREREG.md §9: planted secret values stay in run directories, never in committed results).

    python experiments/m1d/redact.py RUN_DIR OUT_DIR

Copies progress.json, <stage>/{result,arm,train,canaries,progress}.json, <stage>/cartridge_e*.json,
<stage>/eval/summary.json and REG2/reg/summary.json. Every secret listed in any canaries.json of the run (the
manifests' records) is replaced by `<redacted-canary>` wherever it occurs in a copied file, as a JSON string value
or inside one. Continuations, probe outputs, per-target arrays, checkpoints and cartridges are never copied.
The copy is then re-read and the script fails if any secret, or any 16-character run of one, is still present.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MASK = "<redacted-canary>"
PATTERNS = ("progress.json", "*/result.json", "*/arm.json", "*/train.json", "*/canaries.json",
            "*/cartridge_e*.json", "*/eval/summary.json", "*/reg/summary.json")
MIN_RUN = 16


def secrets_of(run: Path) -> list[str]:
    out: set[str] = set()
    for m in run.glob("*/canaries.json"):
        man = json.loads(m.read_text())
        out.update(r["secret"] for r in man.get("records", []))
    return sorted(out, key=len, reverse=True)


def scrub(x, secrets: list[str]):
    if isinstance(x, str):
        for s in secrets:
            if s in x:
                x = x.replace(s, MASK)
        return x
    if isinstance(x, list):
        return [scrub(v, secrets) for v in x]
    if isinstance(x, dict):
        return {k: scrub(v, secrets) for k, v in x.items()}
    return x


def leaks(text: str, secrets: list[str], min_run: int = MIN_RUN) -> list[str]:
    bad = []
    for s in secrets:
        if any(s[i:i + min_run] in text for i in range(0, len(s) - min_run + 1)):
            bad.append(s[:4] + "…")
    return bad


def redact(run: Path, out: Path) -> dict:
    run, out = Path(run), Path(out)
    secrets = secrets_of(run)
    files = sorted({p for pat in PATTERNS for p in run.glob(pat) if p.is_file()})
    n = 0
    for p in files:
        rel = p.relative_to(run)
        obj = scrub(json.loads(p.read_text()), secrets)
        q = out / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text(json.dumps(obj, indent=1) + "\n")
        n += 1
    bad = {str(q.relative_to(out)): b for q in out.rglob("*.json") if (b := leaks(q.read_text(), secrets))}
    if bad:
        raise SystemExit(f"redaction failed, secrets remain in {bad}")
    return dict(files=n, secrets=len(secrets))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    print(json.dumps(redact(Path(a.run), Path(a.out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())

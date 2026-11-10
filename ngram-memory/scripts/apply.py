#!/usr/bin/env python
"""ngram-edit/v1 patch and cartridge tool (numpy only; never imports torch).

  apply.py format  --hidden D [--key flash-next|qwen25|gemma] [--arm G-lin|G-dir] [--orders 2,3] --out FMT.json
  apply.py info    PATCH.safetensors
  apply.py verify  PATCH.safetensors --format FMT.json [--graft GRAFT.safetensors]
  apply.py compile PATCH.safetensors [PATCH ...] --out SERVED.safetensors [--graft GRAFT] [--stats STATS.json]
                   (a cartridge chain: root cartridge then increments/facts, in order; bf16 served rows)
  apply.py ledger  init   DIR --format FMT.json          (graft formats: zero base)
  apply.py ledger  add    DIR PATCH [--edit-id ID]
  apply.py ledger  remove DIR EDIT_ID
  apply.py ledger  show   DIR
  apply.py ledger  export DIR --out PATCH.safetensors

Exit status is non-zero on any refusal (fingerprint mismatch, broken chain, bad payload).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ngram_memory import codecs  # noqa: E402
from ngram_memory.cartridge import (CartridgeChain, check_against_graft, compile_chain, read_graft_file,  # noqa: E402
                                    write_served)
from ngram_memory.ledger import EditRecord, RowLedger, ZeroBase  # noqa: E402
from ngram_memory.patch import read_patch, verify_patch, write_patch  # noqa: E402
from ngram_memory.spec import (FormatSpec, flash_next_key_spec, graft_format, graft_key_spec,  # noqa: E402
                               qwen25_proxy_key_spec)


def _load_format(path: str) -> FormatSpec:
    with open(path) as f:
        return FormatSpec.from_json(f.read())


def cmd_format(a) -> int:
    orders = tuple(int(x) for x in a.orders.split(","))
    key = {"flash-next": lambda: flash_next_key_spec() if orders == (2, 3) else
           graft_key_spec(248320, 248044, orders=orders),
           "qwen25": lambda: qwen25_proxy_key_spec(orders),
           "gemma": lambda: graft_key_spec(262144, 1, orders=orders)}[a.key]()
    fmt = graft_format(a.hidden, key, arm=a.arm, layer_idx=a.layer_idx)
    with open(a.out, "w") as f:
        f.write(fmt.to_json())
    print(json.dumps(dict(out=a.out, fingerprint=fmt.fingerprint(), row_width=fmt.row_width,
                          padded_rows=fmt.padded_rows)))
    return 0


def cmd_info(a) -> int:
    p = read_patch(a.patch)
    m = p.manifest
    print(json.dumps(dict(format=m.format, mode=m.mode, n_rows=m.n_rows, n_reset=m.n_reset, row_width=m.row_width,
                          padded_rows=m.padded_rows, key_fingerprint=m.key_fingerprint,
                          table_fingerprint=m.table_fingerprint, module_fingerprint=m.module_fingerprint,
                          format_fingerprint=m.format_fingerprint, base_fingerprint=m.base_fingerprint,
                          payload_sha256=m.payload_sha256, edits=len(m.edits), cartridge=m.cartridge), indent=1))
    return 0


def cmd_verify(a) -> int:
    p = read_patch(a.patch)
    checks = verify_patch(p, _load_format(a.format))
    if a.graft:
        gm, _ = read_graft_file(a.graft)
        check_against_graft(p, gm)
        checks.append("graft_pin")
    print(json.dumps(dict(ok=True, checks=checks)))
    return 0


def cmd_compile(a) -> int:
    chain = CartridgeChain([read_patch(x) for x in a.patches])
    if a.graft:
        gm, _ = read_graft_file(a.graft)
        for p in chain.patches:
            check_against_graft(p, gm)
    if chain.manifest.mode != "absolute":
        raise SystemExit("compile without a base reader supports graft (absolute) chains only")
    sc = compile_chain(chain)
    sha = write_served(a.out, sc)
    out = dict(out=a.out, sha256=sha, namespace=sc.namespace, version=sc.version, rows=int(sc.row_ids.size),
               bytes_per_row_bf16=2 * chain.manifest.row_width + 8)
    if a.stats:
        _, d, _ = chain.compose()
        st = dict(bf16=codecs.roundtrip_stats(d, "bf16", served_cast=None))
        if chain.manifest.row_width % 32 == 0:
            st["int6_g32"] = codecs.roundtrip_stats(d, "int6_g32", served_cast="bf16")
        with open(a.stats, "w") as f:
            json.dump(st, f, indent=1)
        out["stats"] = a.stats
    print(json.dumps(out))
    return 0


def cmd_ledger(a) -> int:
    if a.op == "init":
        fmt = _load_format(a.format)
        if fmt.table.base_codec != "none":
            raise SystemExit("the CLI ledger handles graft formats (zero base); native ledgers need a base reader")
        os.makedirs(a.dir, exist_ok=True)
        with open(os.path.join(a.dir, "format.json"), "w") as f:
            f.write(fmt.to_json())
        RowLedger(fmt.padded_rows, fmt.row_width, ZeroBase(fmt.row_width), fmt.fingerprint()).save(a.dir)
        print(json.dumps(dict(ok=True, dir=a.dir, fingerprint=fmt.fingerprint())))
        return 0
    fmt = _load_format(os.path.join(a.dir, "format.json"))
    led = RowLedger.load(a.dir, ZeroBase(fmt.row_width))
    if a.op == "add":
        p = read_patch(a.arg)
        verify_patch(p, fmt)
        rec = EditRecord.from_patch(p, a.edit_id)
        ep = led.add(rec)
        led.save(a.dir)
        print(json.dumps(dict(ok=True, edit_id=rec.edit_id, epoch=ep, conflict_set=rec.conflict_set)))
    elif a.op == "remove":
        deps = led.dependents(a.arg)
        ep = led.remove(a.arg)
        led.save(a.dir)
        print(json.dumps(dict(ok=True, epoch=ep, revalidate=deps)))
    elif a.op == "show":
        print(json.dumps(dict(epoch=led.epoch, active=led.order(), records=sorted(led.records),
                              active_rows=int(led.active_rows().size), conflict_sets=led.conflict_sets()), indent=1))
    elif a.op == "export":
        sha = write_patch(a.out, led.to_patch(fmt))
        print(json.dumps(dict(ok=True, out=a.out, sha256=sha)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("format")
    f.add_argument("--hidden", type=int, required=True)
    f.add_argument("--key", default="flash-next", choices=["flash-next", "qwen25", "gemma"])
    f.add_argument("--arm", default="G-lin", choices=["G-lin", "G-dir"])
    f.add_argument("--orders", default="2,3")
    f.add_argument("--layer-idx", type=int, default=1)
    f.add_argument("--out", required=True)
    i = sub.add_parser("info")
    i.add_argument("patch")
    v = sub.add_parser("verify")
    v.add_argument("patch")
    v.add_argument("--format", required=True)
    v.add_argument("--graft")
    c = sub.add_parser("compile")
    c.add_argument("patches", nargs="+")
    c.add_argument("--out", required=True)
    c.add_argument("--graft")
    c.add_argument("--stats")
    lg = sub.add_parser("ledger")
    lg.add_argument("op", choices=["init", "add", "remove", "show", "export"])
    lg.add_argument("dir")
    lg.add_argument("arg", nargs="?")
    lg.add_argument("--format")
    lg.add_argument("--edit-id")
    lg.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        return dict(format=cmd_format, info=cmd_info, verify=cmd_verify, compile=cmd_compile, ledger=cmd_ledger)[a.cmd](a)
    except (ValueError, KeyError, IndexError) as e:
        print(json.dumps(dict(ok=False, error=f"{type(e).__name__}: {e}")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

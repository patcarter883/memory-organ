#!/usr/bin/env python3
"""
CAPACITY PROBE — cam-loop-memcap-Msweep
Fresh-bind the v0 DeepMemory (BoltAdapter, K=16, mem_dim=512, base-1 Qwen3.5-4B) at increasing M and
report held-out CARRY vs chance (1/M). Bind-only: no Stage-2 MAG tap, no translators. Reuses
recall_mag.bind_adapter so the bind is byte-identical to the validated v0 Stage-1 path.

Question: is the M=3 ceiling of cam_v0_L24.pt ARCHITECTURAL (K/mem_dim too small) or just under-training?
  - carry stays >> chance at M=8/16   -> arch capable, v0 was under-bound (cheap fix: re-bind bigger)
  - carry collapses toward chance even when bound FOR that M -> arch capacity-limited (redesign/pk_store)

Usage:
  python -m cam.bind_msweep --Ms 3,8,16 --bind-steps 3000 --batch 8 --lr 5e-4 [--seed S]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# flat package: sibling imports resolve relatively when imported as cam.X (`python -m cam.X`,
# `import cam.X`) and fall back to a path-hacked absolute import when run as a file (`python cam/X.py`).
try:
    from .m2_adapter import MODEL, DEV, load_frozen_base
    from .recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder
    from .recall_boltA import BoltAdapter, eval_direct
    from .pk_store_adapter import PKStoreAdapter
    from .kv_adapter import KVAdapter
    from . import recall_mag
except ImportError:
    if __package__:  # real ImportError inside a sibling, not "run as a file" — don't mask it
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import MODEL, DEV, load_frozen_base                   # noqa: E402
    from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
    from recall_boltA import BoltAdapter, eval_direct                     # noqa: E402
    from pk_store_adapter import PKStoreAdapter                           # noqa: E402
    from kv_adapter import KVAdapter                                      # noqa: E402
    import recall_mag                                                     # noqa: E402


def bind_at_M(embed_weight, H, tok, args, M, seed):
    """Fresh-bind an adapter at the given M; return (carry, ablated, chance, steps, secs).
    args.store selects the memory mechanism: 'bolt' (DeepMemory v0, default) or 'pk' (product-key
    sparse store, hub-decoupled). Everything else (in_proj/norm, readout pool, out_proj, tied-unembed
    direct loss, eval) is held identical so the carry-vs-M comparison isolates the store."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    builder = DocBuilder(tok, names, cargo, M, args.seg_len, args.qa_seg, phrasing="dict")

    # same arch knobs as v0: K(=k)=16, mem_dim=512, heads=4, chunk=16, expansion=4.0
    if args.store == "pk":
        adapter = PKStoreAdapter(embed_weight, H, args.mem_dim, args.heads, args.chunk,
                                 args.expansion, args.k, n_sub=args.n_sub, topk=args.topk,
                                 sub_topk=args.sub_topk, addr_sup_weight=args.addr_sup_weight,
                                 read_heads=(args.pk_read_heads if args.pk_read_heads > 0 else None)).to(DEV)
        adapter.set_builder(builder)   # pk store needs the bind-block positions
    elif args.store == "kv":
        adapter = KVAdapter(embed_weight, H, args.mem_dim, args.heads, args.chunk,
                            args.expansion, args.k).to(DEV)
        adapter.set_builder(builder)   # kv control reads the bind-block positions too
    else:
        adapter = BoltAdapter(embed_weight, H, args.mem_dim, args.heads, args.chunk,
                              args.expansion, args.k).to(DEV)

    # a per-M copy of args so bind_adapter sees this M's chance/steps
    a = argparse.Namespace(**vars(args))
    a.M = M

    t0 = time.time()
    # bind_adapter prints "[mag] binding held-out: carry .. | ablated .. | chance .." and returns carry
    carry = recall_mag.bind_adapter(adapter, builder, rng, a)
    secs = time.time() - t0

    # re-derive ablated for the table (eval_direct returns (carry, ablated, pt_carry, pt_abl))
    d_carry, d_abl, _ptc, _pta = eval_direct(adapter, builder, rng, a)
    return d_carry, d_abl, 1.0 / M, a.bind_steps, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ms", type=str, default="3,8,16", help="comma list of M to sweep")
    ap.add_argument("--bind-steps", type=int, default=3000, dest="bind_steps")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--mem-dim", type=int, default=512, dest="mem_dim")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--expansion", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--base1", type=str, default=MODEL,
                    help=f"donor (frozen base-1) HF model id; default {MODEL}. Swapping it is an "
                         f"untested-donor experiment, not a reproduction.")
    ap.add_argument("--store", type=str, default="bolt", choices=["bolt", "pk", "kv"],
                    help="memory mechanism: 'bolt' (DeepMemory v0), 'pk' (product-key sparse, "
                         "hub-free), or 'kv' (UNCOMPRESSED per-doc key->value control — the upper "
                         "bound; reimplementation of the one-off falsifier in RESULTS.md §1/§2)")
    ap.add_argument("--n-sub", type=int, default=32, dest="n_sub",
                    help="pk: codebook size per half -> N=n_sub^2 slots")
    ap.add_argument("--topk", type=int, default=8, help="pk: global product-keys kept per query")
    ap.add_argument("--sub-topk", type=int, default=4, dest="sub_topk", help="pk: top-k per half")
    ap.add_argument("--addr-sup-weight", type=float, default=0.0, dest="addr_sup_weight",
                    help="pk: weight of the write->read addressing-supervision (InfoNCE) loss; "
                         "0 (default) preserves the no-sup baseline, >0 turns it on")
    ap.add_argument("--pk-read-heads", type=int, default=0, dest="pk_read_heads",
                    help="pk: override the store's read-head count (0 = use --heads); pk is designed "
                         "for more retrieval-mode heads than the bolt path's default 4")
    args = ap.parse_args()

    Ms = [int(x) for x in args.Ms.split(",") if x.strip()]

    base, tok = load_frozen_base(args.base1)
    H = base.config.get_text_config().hidden_size
    embed_weight = base.get_input_embeddings().weight.detach().float().clone()
    _rh = args.pk_read_heads if args.pk_read_heads > 0 else args.heads
    store_desc = {"pk": f"pk(N={args.n_sub**2} n_sub={args.n_sub} topk={args.topk} "
                        f"sub_topk={args.sub_topk} read_heads={_rh} addr_sup_w={args.addr_sup_weight})",
                  "kv": "kv(uncompressed per-doc control — upper bound)",
                  "bolt": "bolt(DeepMemory v0)"}[args.store]
    print(f"[memcap] {args.base1} | H={H} | store={store_desc} | arch: K={args.k} mem_dim={args.mem_dim} "
          f"heads={args.heads} chunk={args.chunk} exp={args.expansion} | bind_steps={args.bind_steps} "
          f"batch={args.batch} lr={args.lr} | sweeping M={Ms}", flush=True)

    rows = []
    for M in Ms:
        print(f"\n[memcap] ===== fresh-bind at M={M} (chance {1.0/M:.3f}) =====", flush=True)
        carry, abl, chance, steps, secs = bind_at_M(embed_weight, H, tok, args, M, args.seed)
        margin = carry - chance
        rows.append((M, carry, abl, chance, margin, steps, secs))
        print(f"[memcap] M={M}: carry {carry:.3f} | ablated {abl:.3f} | chance {chance:.3f} | "
              f"carry-chance {margin:+.3f} | steps {steps} | {secs:.0f}s", flush=True)

    print("\n[memcap] ===== CARRY-vs-M TABLE (fresh-bind, K=16 mem_dim=512, Qwen3.5-4B) =====", flush=True)
    print(f"{'M':>4} {'chance':>8} {'carry':>8} {'ablated':>8} {'carry-chance':>13} {'steps':>7} {'secs':>6}",
          flush=True)
    for M, carry, abl, chance, margin, steps, secs in rows:
        print(f"{M:>4} {chance:>8.3f} {carry:>8.3f} {abl:>8.3f} {margin:>+13.3f} {steps:>7} {secs:>6.0f}",
              flush=True)
    print("[memcap] DONE", flush=True)


if __name__ == "__main__":
    main()

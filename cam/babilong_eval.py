"""babilong_eval — the FLAGSHIP "distance to Titans" measurement for the frozen-base cartridge method.

BABILong (github.com/booydar/babilong, HF RMT-team/babilong) distributes the bAbI supporting facts
across a long filler document, then asks a multi-fact reasoning question. Titans' headline result is
Titans-MAC (<=760M params) beating GPT-4 / Llama-8B+RAG on this benchmark at long context. We measure
how close a FROZEN base (Qwen3-4B-Base) + a trained per-context Cartridge (cam/cartridge.py) gets, vs
RAG/ICL (full context in-window) and vs the no-context floor.

Three conditions per sample:
  * cartridge  : build a Cartridge from the long `input` (init_from_text + short self-study KL-distill),
                 answer the question with the cartridge ATTACHED and NO context in the prompt.
  * rag        : full `input` in the prompt (truncated to the model window if needed), answer normally.
  * nocontext  : question only (the floor — what the base knows with no facts).

Base is FROZEN throughout (only the cartridge's Z trains). Greedy generation; BABILong substring match
against the short gold `target`.

Run (local gfx1201, via the shared arbiter):
  PYTHONPATH=/home/pat/vllm-rocm/lib/python3.12/site-packages \
  CAM_BASE_MODEL=Qwen/Qwen3-4B-Base HF_HUB_OFFLINE=1 \
  gpu-lease -n 1 -- python -m cam.babilong_eval \
      --lengths 0k,1k --tasks qa1,qa2,qa3 --limit 5 \
      --p 128 --max-steps 60 --n-seeds 4 --samples-per-seed 1 --gen-len 24
"""
import argparse
import json
import os
import sys
import time

import torch

try:
    from .m2_adapter import load_frozen_base
    from .cartridge_train import train_cartridge, select_high_surprise_context
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import load_frozen_base            # noqa: E402
    from cartridge_train import train_cartridge, select_high_surprise_context  # noqa: E402


# ------------------------------------------------------------------ data ----
def load_babilong(lengths, tasks, limit):
    """Return {(length, task): [ {input,question,target}, ... ]} from the cached RMT-team/babilong."""
    from datasets import load_dataset
    out = {}
    for L in lengths:
        ds = load_dataset("RMT-team/babilong", L)   # splits qa1..qaN
        for t in tasks:
            if t not in ds:
                print(f"[babilong] WARNING: {L}/{t} not in dataset; skipping", flush=True)
                continue
            rows = ds[t].select(range(min(limit, len(ds[t]))))
            out[(L, t)] = [dict(input=r["input"], question=r["question"], target=r["target"])
                           for r in rows]
    return out


# ------------------------------------------------------------ generation ----
@torch.no_grad()
def greedy(base, tok, prompt_ids, gen_len, device):
    """Manual greedy continuation (base runs use_cache=False). Stops on newline / EOS. Returns text."""
    cur = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos = tok.eos_token_id
    out = []
    for _ in range(gen_len):
        logits = base(input_ids=cur).logits[:, -1]
        nxt = int(logits.argmax(-1).item())
        if eos is not None and nxt == eos:
            break
        out.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        if "\n" in tok.decode(out):
            break
    return tok.decode(out)


# BABILong / bAbI answers are short (usually one word). Substring match after normalizing.
def _norm(s):
    return " ".join("".join(c if (c.isalnum() or c.isspace()) else " " for c in s.lower()).split())


def hit(gold, gen):
    g = _norm(gold)
    a = _norm(gen)
    if not g:
        return False
    # bAbI multi-word answers (qa8 lists) are comma-joined; require every token present.
    toks = g.split()
    return all(t in a.split() for t in toks) if toks else False


INSTR = ("I will give you context followed by a question. Answer the question using only a single "
         "word or short phrase, based on the context.\n\n")
INSTR_NOCTX = ("Answer the question using only a single word or short phrase.\n\n")


def build_prompts(tok, ex, ctx_ids_cap=None):
    """Return (rag_ids, nocontext_ids, question_ids). question_ids = the cartridge-condition prompt
    (question only, cartridge supplies the context). All share the same 'Question:/Answer:' framing."""
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    q = ex["question"].strip()
    tail = f"\nQuestion: {q}\nAnswer:"
    tail_ids = tok(tail, add_special_tokens=False).input_ids
    ctx_ids = tok(ex["input"].strip(), add_special_tokens=False).input_ids
    if ctx_ids_cap is not None and len(ctx_ids) > ctx_ids_cap:
        ctx_ids = ctx_ids[:ctx_ids_cap]        # truncate to model window (keep head; facts are distributed)
    instr_ids = tok(INSTR, add_special_tokens=False).input_ids
    instr_noctx = tok(INSTR_NOCTX, add_special_tokens=False).input_ids
    rag_ids = bos + instr_ids + ctx_ids + tail_ids
    # cartridge/no-context: question only (no story text in the prompt)
    q_only_ids = bos + instr_noctx + tok(f"Question: {q}\nAnswer:", add_special_tokens=False).input_ids
    return rag_ids, q_only_ids, q_only_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="0k,1k", help="comma list of BABILong context configs")
    ap.add_argument("--tasks", default="qa1,qa2,qa3")
    ap.add_argument("--limit", type=int, default=5, help="samples per (length,task)")
    ap.add_argument("--p", type=int, default=128, help="cartridge prefix length")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--samples-per-seed", type=int, default=1)
    ap.add_argument("--gen-len", type=int, default=24, help="self-study generation length (train cost)")
    ap.add_argument("--ans-len", type=int, default=8, help="answer generation length (eval)")
    ap.add_argument("--ctx-train-cap", type=int, default=8000,
                    help="cap context tokens fed to the cartridge teacher (bounds train cost)")
    ap.add_argument("--topk", type=int, default=0,
                    help="sparse top-k teacher (0=dense full-vocab). >0 avoids the [Lx,vocab] OOM at long ctx")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write per-sample JSONL here")
    ap.add_argument("--conditions", default="cartridge,rag,nocontext")
    ap.add_argument("--surprise-weight", type=float, default=0.0,
                    help="cartridge distillation: weight per-token KL by teacher surprisal "
                         "(0=uniform baseline, 1=full surprise weighting). Titans formation-selectivity.")
    ap.add_argument("--surprise-temp", type=float, default=1.0,
                    help="sharpen(>1)/soften(<1) the surprise weighting (surprisal**temp before norm)")
    ap.add_argument("--surprise-select-frac", type=float, default=1.0,
                    help="if <1: score context chunks by mean surprisal, self-study/distill only the "
                         "top-frac high-surprise chunks (drop low-surprise filler). 1.0=no selection.")
    ap.add_argument("--surprise-select-chunk", type=int, default=48,
                    help="token width of the surprise-selection chunks")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    lengths = [x.strip() for x in args.lengths.split(",") if x.strip()]
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    conds = [x.strip() for x in args.conditions.split(",") if x.strip()]
    torch.manual_seed(args.seed)

    print(f"[babilong] loading base {os.environ.get('CAM_BASE_MODEL', '<default>')} ...", flush=True)
    base, tok = load_frozen_base()
    device = next(base.parameters()).device
    # ---- verify base is FROZEN ----
    n_grad = sum(1 for p in base.parameters() if p.requires_grad)
    max_ctx = getattr(base.config.get_text_config(), "max_position_embeddings", 32768)
    print(f"[babilong] base frozen: {n_grad} params require grad (expect 0) | max_ctx={max_ctx} "
          f"| device={device}", flush=True)
    assert n_grad == 0, "base is NOT frozen"
    win_cap = max_ctx - args.ans_len - 64        # leave headroom for the question+answer

    data = load_babilong(lengths, tasks, args.limit)
    print(f"[babilong] loaded { {f'{k[0]}/{k[1]}': len(v) for k,v in data.items()} }", flush=True)

    fout = open(args.out, "w") if args.out else None
    # accumulators: results[(L,t)][cond] = [0/1,...]; train_secs[(L,t)] = [secs,...]
    results = {}
    train_secs = {}
    ctx_tok_lens = {}
    for (L, t), rows in data.items():
        results[(L, t)] = {c: [] for c in conds}
        train_secs[(L, t)] = []
        ctx_tok_lens[(L, t)] = []
        for i, ex in enumerate(rows):
            rag_ids, q_ids, cart_prompt_ids = build_prompts(tok, ex, ctx_ids_cap=win_cap)
            ctx_len = len(tok(ex["input"].strip(), add_special_tokens=False).input_ids)
            ctx_tok_lens[(L, t)].append(ctx_len)

            cart = None
            if "cartridge" in conds:
                t0 = time.time()
                # distil the long context into a per-sample cartridge (base stays frozen)
                ctx_text = ex["input"].strip()
                cap_ids = tok(ctx_text, add_special_tokens=False).input_ids[: args.ctx_train_cap]
                ctx_text_capped = tok.decode(cap_ids)
                # optional surprise-based context selection: keep only high-surprise chunks (drop filler)
                if args.surprise_select_frac < 1.0:
                    ctx_text_capped, selinfo = select_high_surprise_context(
                        base, tok, ctx_text_capped, frac=args.surprise_select_frac,
                        chunk_tokens=args.surprise_select_chunk, device=device, verbose=args.verbose)
                cart, tinfo = train_cartridge(
                    base, tok, ctx_text_capped, p=args.p, lr=args.lr, epochs=args.epochs,
                    max_steps=args.max_steps, n_seeds=args.n_seeds,
                    samples_per_seed=args.samples_per_seed, gen_len=args.gen_len,
                    seed=args.seed + i, init_text=True, verbose=args.verbose, topk=args.topk,
                    surprise_weight=args.surprise_weight, surprise_temp=args.surprise_temp)
                train_secs[(L, t)].append(time.time() - t0)

            rec = dict(length=L, task=t, idx=i, question=ex["question"].strip(),
                       gold=ex["target"], ctx_tokens=ctx_len)
            for c in conds:
                if c == "cartridge":
                    cart.attach(base)
                    cart.set_active(True)
                    try:
                        gen = greedy(base, tok, cart_prompt_ids, args.ans_len, device)
                    finally:
                        cart.detach()
                elif c == "rag":
                    gen = greedy(base, tok, rag_ids, args.ans_len, device)
                else:  # nocontext
                    gen = greedy(base, tok, q_ids, args.ans_len, device)
                h = hit(ex["target"], gen)
                results[(L, t)][c].append(1 if h else 0)
                rec[c] = dict(gen=gen.strip()[:60], hit=bool(h))
            if fout:
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
            if args.verbose or i < 3:
                ts = train_secs[(L, t)][-1] if train_secs[(L, t)] else 0.0
                cells = " ".join(f"{c}={'Y' if rec[c]['hit'] else '.'}:{rec[c]['gen'][:14]!r}" for c in conds)
                print(f"[babilong] {L}/{t} #{i} ctx={ctx_len}tok gold={ex['target']!r} "
                      f"train={ts:.1f}s | {cells}", flush=True)
    if fout:
        fout.close()

    # ================================ REPORT ================================
    print("\n[babilong] " + "=" * 76, flush=True)
    print(f"[babilong] Qwen3-4B-Base (FROZEN) | limit={args.limit}/cell | p={args.p} "
          f"max_steps={args.max_steps} | conditions={conds} | "
          f"surprise_weight={args.surprise_weight} temp={args.surprise_temp} "
          f"select_frac={args.surprise_select_frac}", flush=True)
    hdr = f"{'length':>6s} {'task':>5s} {'n':>3s} {'ctx_tok':>8s} | " + \
          " ".join(f"{c:>10s}" for c in conds) + f" | {'train/samp':>10s}"
    print("[babilong] " + hdr, flush=True)
    print("[babilong] " + "-" * len(hdr), flush=True)

    def acc(xs):
        return (sum(xs) / len(xs)) if xs else float("nan")

    # per-condition aggregate over all cells
    agg = {c: [] for c in conds}
    for (L, t) in sorted(data.keys(), key=lambda k: (lengths.index(k[0]), k[1])):
        n = len(results[(L, t)][conds[0]])
        ctxm = sum(ctx_tok_lens[(L, t)]) / max(1, len(ctx_tok_lens[(L, t)]))
        ts = acc(train_secs[(L, t)]) if train_secs[(L, t)] else float("nan")
        cells = " ".join(f"{acc(results[(L,t)][c]):>10.2f}" for c in conds)
        print(f"[babilong] {L:>6s} {t:>5s} {n:>3d} {ctxm:>8.0f} | {cells} | {ts:>10.1f}", flush=True)
        for c in conds:
            agg[c].extend(results[(L, t)][c])
    print("[babilong] " + "-" * len(hdr), flush=True)
    allcells = " ".join(f"{acc(agg[c]):>10.2f}" for c in conds)
    print(f"[babilong] {'ALL':>6s} {'':>5s} {len(agg[conds[0]]):>3d} {'':>8s} | {allcells} |", flush=True)
    print("\n[babilong] Titans (arXiv:2501.00663) reference — BABILong avg over lengths, few-shot:", flush=True)
    print("[babilong]   Titans-MAC ~62%, Titans-MAG ~60%, GPT-4 ~45%, Llama3.1-8B+RAG ~42%, "
          "Mamba2 ~30% (read exact numbers from the paper's BABILong figure).", flush=True)


if __name__ == "__main__":
    main()

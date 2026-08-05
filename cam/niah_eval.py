"""niah_eval — Axis 2 (Titans' capacity headline): needle-in-a-haystack / passkey.

Bury a random passkey ("The passkey is 48291.") at depth d in filler of length L, then retrieve it two
ways and compare an L x depth accuracy grid:
  * CARTRIDGE : train a Cartridge FROM the haystack (Cartridges KL-distill), retrieve with the passkey
                question and NO haystack in the prompt. Persistent, ~38x cheaper inference than re-reading.
  * ICL       : put the whole haystack in the prompt, then ask (the standard long-context baseline).
Cartridges paper claims ~ICL parity on recall; this verifies it on Qwen3-4B-Base.

Run (local gfx1201, via the shared arbiter):
  PYTHONPATH=/home/pat/vllm-rocm/lib/python3.12/site-packages \
  CAM_BASE_MODEL=Qwen/Qwen3-4B-Base HF_HUB_OFFLINE=1 \
  gpu-lease -n 1 -- python -m cam.niah_eval --lengths 1024 --depths 0.1,0.5,0.9 --conditions cart,icl
"""
import argparse
import os
import random
import sys
import time

import torch

try:
    from .m2_adapter import load_frozen_base
    from .cartridge_train import train_cartridge
except ImportError:
    if __package__:
        raise
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from m2_adapter import load_frozen_base                       # noqa: E402
    from cartridge_train import train_cartridge                   # noqa: E402


FILLER = ("The grass is green and the sky is blue. Nothing important happens here. "
          "People walk by and the day goes on as usual. ")
PASSKEY_Q = "What is the passkey? The passkey is"


def build_haystack(tok, L, depth, passkey):
    """A length-~L token haystack of repeated innocuous filler with 'The passkey is <key>.' inserted at
    the given depth fraction. Returns (haystack_text, passkey_str)."""
    needle = f"The passkey is {passkey}. "
    filler_ids = tok(FILLER, add_special_tokens=False).input_ids
    # repeat filler to reach ~L tokens
    reps = max(1, L // max(1, len(filler_ids)) + 1)
    body_ids = (filler_ids * reps)[:L]
    ins = int(depth * len(body_ids))
    # snap insertion to a filler-sentence boundary near `ins` by decoding the split (keeps text natural)
    pre = tok.decode(body_ids[:ins])
    post = tok.decode(body_ids[ins:])
    text = pre + " " + needle + post
    return text, str(passkey)


@torch.no_grad()
def _greedy(base, tok, cart, prompt_text, gen_len=16):
    """Greedy generate `gen_len` tokens after `prompt_text`, optionally with a Cartridge attached."""
    device = next(base.parameters()).device
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    ids = bos + tok(prompt_text, add_special_tokens=False).input_ids
    attached = cart is not None
    if attached:
        cart.attach(base)
        cart.set_active(True)
    try:
        cur = torch.tensor([ids], dtype=torch.long, device=device)
        out = []
        for _ in range(gen_len):
            logits = base(input_ids=cur).logits[:, -1]
            nxt = logits.argmax(-1)
            out.append(int(nxt.item()))
            cur = torch.cat([cur, nxt.view(1, 1)], dim=1)
            if "\n" in tok.decode(out):
                break
        gen = tok.decode(out)
    finally:
        if attached:
            cart.detach()
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="1024", help="comma list of haystack token lengths L")
    ap.add_argument("--depths", default="0.1,0.5,0.9", help="comma list of insertion depth fractions")
    ap.add_argument("--conditions", default="cart,icl", help="comma subset of {cart,icl}")
    ap.add_argument("--samples", type=int, default=1, help="random passkeys per (L,depth) cell")
    ap.add_argument("--p", type=int, default=64, help="cartridge prefix length")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--samples-per-seed", type=int, default=1)
    ap.add_argument("--gen-len", type=int, default=40, help="self-study generation length")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    depths = [float(x) for x in args.depths.split(",") if x.strip()]
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    print(f"[niah] loading base {os.environ.get('CAM_BASE_MODEL', '<default>')} ...", flush=True)
    base, tok = load_frozen_base()
    print(f"[niah] L={lengths} depths={depths} conditions={conds} samples/cell={args.samples}", flush=True)

    grid = {}   # (cond, L, depth) -> (hits, total, secs)
    for L in lengths:
        for d in depths:
            for cond in conds:
                grid[(cond, L, d)] = [0, 0, 0.0]
            for s in range(args.samples):
                passkey = rng.randint(10000, 99999)
                hay, pk = build_haystack(tok, L, d, passkey)
                ntok = len(tok(hay, add_special_tokens=False).input_ids)
                if "icl" in conds:
                    t0 = time.time()
                    prompt = hay + "\n\n" + PASSKEY_Q
                    gen = _greedy(base, tok, None, prompt)
                    hit = pk in gen
                    g = grid[("icl", L, d)]
                    g[0] += hit; g[1] += 1; g[2] += time.time() - t0
                    print(f"[niah] ICL  L={L}(~{ntok}) d={d:.1f} pk={pk} hit={'Y' if hit else '.'} "
                          f"gen={gen.strip()[:24]!r}", flush=True)
                if "cart" in conds:
                    t0 = time.time()
                    cart, info = train_cartridge(
                        base, tok, hay, p=args.p, lr=args.lr, epochs=args.epochs, max_steps=args.max_steps,
                        n_seeds=args.n_seeds, samples_per_seed=args.samples_per_seed, gen_len=args.gen_len,
                        seed=args.seed + s, extra_seeds=[PASSKEY_Q])
                    gen = _greedy(base, tok, cart, PASSKEY_Q)
                    hit = pk in gen
                    g = grid[("cart", L, d)]
                    g[0] += hit; g[1] += 1; g[2] += time.time() - t0
                    print(f"[niah] CART L={L}(~{ntok}) d={d:.1f} pk={pk} hit={'Y' if hit else '.'} "
                          f"KL={info['final_kl']:.3f} {g[2]/g[1]:.0f}s gen={gen.strip()[:24]!r}", flush=True)

    # -------- REPORT: L x depth accuracy grid per condition --------
    print("\n[niah] " + "=" * 60, flush=True)
    for cond in conds:
        print(f"[niah] {cond.upper()} accuracy grid (rows=L, cols=depth):", flush=True)
        header = "        " + "".join(f"d={d:<6.1f}" for d in depths)
        print("[niah] " + header, flush=True)
        for L in lengths:
            cells = []
            for d in depths:
                h, n, _ = grid[(cond, L, d)]
                cells.append(f"{(h/n if n else float('nan')):<8.2f}")
            print(f"[niah] L={L:<6d} " + "".join(cells), flush=True)
        avg_s = sum(grid[(cond, L, d)][2] for L in lengths for d in depths) / max(
            1, sum(grid[(cond, L, d)][1] for L in lengths for d in depths))
        print(f"[niah]   avg {avg_s:.1f}s/sample", flush=True)
    print("[niah] " + "=" * 60, flush=True)


if __name__ == "__main__":
    main()

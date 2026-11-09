"""How much of an edit delta survives being written INTO the quantised n-gram row files,
versus an after-dequant overlay?  (minisgl df29117, deployed q4e-mxfp4 checkpoint.)

FACT (row_table.py:1001-1010 at df29117, serve.sh:1371-1398, ple_split.py:4-7): the deployed PLE
table is INT6 group-32 (120 B of codes + 5 fp16 scales = 130 B/row; 320,001,536 rows x 130 B =
38.74 GiB, which is the "38.7 GiB" the owner quotes). The MXFP4 PLE variant (uint8 E8M0 scales)
raises NotImplementedError in the fork. MXFP4 is measured too, as the brief assumed it.

Torch use: ONLY to read the real bf16 rows from the .pt fixture and to cross-check the numpy bf16
rounding against torch's cast. Every codec below is numpy-only, and the fork's own decoder
`row_table.dequant_int6_g32` is imported by file path in a process where `import torch` is blocked
(proves the engine-side decode is torch-free).

Paths compared, all ending in the serve's device cast fp32 -> bf16 (source.py:243-245):
  overlay_f32   : served = bf16(dec(base) + d)                  [hook after decode_raw, fp32 delta]
  overlay_bf16r : store r = bf16(dec(base) + d) as uint16 bits; served = r  (bit-identical to above)
  int6_keep     : rewrite only the 120 B codes, keep the row's 5 fp16 scales
  int6_rescale  : rewrite codes AND scales (130 B) from the target's group absmax
  mxfp4_keep / mxfp4_rescale : same for an MXFP4 (E2M1 + E8M0/32) table
Metrics on the EFFECTIVE delta d' = served_after - served_before:
  keep = <d',d>/|d|^2, err = |d'-d|/|d|, cos(d',d), lost = frac of nonzero delta elems with d'==0,
  coll = |d' on elements where d==0| / |d|  (collateral damage to untouched elements).
"""
import importlib.util
import os
import subprocess
import sys

import numpy as np

REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
SP = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
REAL = f"{SP}/nm/t1/real/real_ple_block.pt"
OUT = os.path.dirname(os.path.abspath(__file__))
G = 32


# ------------------------------------------------------------------ torch-free import proof
def torch_free_check():
    code = (
        "import sys, importlib.util\n"
        "sys.modules['torch'] = None  # any 'import torch' now raises ImportError\n"
        f"s = importlib.util.spec_from_file_location('rt', '{REPO}/weights/row_table.py')\n"
        "m = importlib.util.module_from_spec(s); sys.modules['rt'] = m; s.loader.exec_module(m)\n"
        "import numpy as np\n"
        "p = np.zeros((1,120), np.uint8); p[:] = 0x82; sc = np.ones((1,5), np.float16)\n"
        "print('row_table imported with torch blocked; dequant ok:', m.dequant_int6_g32(p, sc).shape)\n"
    )
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def load_row_table():
    spec = importlib.util.spec_from_file_location("row_table_df29117", f"{REPO}/weights/row_table.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m  # dataclasses look the module up by name
    spec.loader.exec_module(m)
    return m


rt = load_row_table()


# ------------------------------------------------------------------ numpy codecs
def bf16_round(x):
    """fp32 -> bf16 (round-to-nearest-even) returned as fp32. No NaN/inf in this data."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16 << 16
    return u.astype(np.uint32).view(np.float32)


def bf16_bits(x):
    return (bf16_round(x).view(np.uint32) >> 16).astype(np.uint16)


def bits_to_f32(b):
    return (b.astype(np.uint32) << 16).view(np.float32)


def int6_scales(x):
    n, d = x.shape
    s = (np.abs(x.reshape(n, d // G, G)).max(-1) / 31.0).astype(np.float16)
    s[s == 0] = np.float16(6e-8)
    return s


def int6_encode(x, s):
    n, d = x.shape
    q = np.rint(x.reshape(n, d // G, G) / s.astype(np.float32)[..., None])
    codes = np.clip(q, -31, 31).astype(np.int32) + rt.INT6_CODE_OFFSET
    c = codes.reshape(n, -1, 4).astype(np.uint32)
    w = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    return np.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], -1).astype(np.uint8).reshape(n, -1)


def int6_decode(packed, s):  # the FORK's decoder, verbatim
    return rt.dequant_int6_g32(packed, s)


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)  # quant/mxfp4.py FP4_E2M1_LUT
E2M1_LUT = np.concatenate([E2M1, -E2M1]).astype(np.float32)


def mx_scales(x):
    """OCP MX: shared_exp = floor(log2(amax)) - emax_elem(E2M1)=2, stored E8M0 = exp + 127."""
    n, d = x.shape
    amax = np.abs(x.reshape(n, d // G, G)).max(-1)
    e = np.floor(np.log2(np.maximum(amax, 2.0 ** -126))) - 2
    return np.clip(e + 127, 0, 254).astype(np.uint8)


def mx_encode(x, e8):
    n, d = x.shape
    sc = np.exp2(e8.astype(np.float32) - 127)[..., None]
    v = x.reshape(n, d // G, G) / sc
    mag = np.abs(v)
    idx = np.abs(mag[..., None] - E2M1).argmin(-1)  # nearest (ties -> lower code; fine for a bound)
    code = (idx | (np.signbit(v) << 3)).astype(np.uint8).reshape(n, d)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).astype(np.uint8)  # even element in LOW nibble


def mx_decode(packed, e8):
    n = packed.shape[0]
    lo, hi = packed & 0xF, packed >> 4
    codes = np.stack([lo, hi], -1).reshape(n, -1)
    vals = E2M1_LUT[codes].reshape(n, -1, G)
    return (vals * np.exp2(e8.astype(np.float32) - 127)[..., None]).reshape(n, -1)


# ------------------------------------------------------------------ data
def real_rows():
    import torch  # fixture reading only

    d = torch.load(REAL, map_location="cpu", weights_only=False)
    rows = torch.cat([d[k] for k in ("rows_head0_start", "rows_head7_start",
                                     "rows_head8_start", "rows_head15_start")]).float()
    # cross-check numpy bf16 RNE against torch on random fp32
    r = torch.randn(200000) * 0.01
    assert np.array_equal(bf16_round(r.numpy()), r.to(torch.bfloat16).float().numpy()), "bf16 RNE mismatch"
    return rows.numpy().astype(np.float32)


def metrics(d, de):
    nz = d != 0
    dd = (d * d).sum(1)
    keep = (de * d).sum(1) / dd
    err = np.linalg.norm(de - d, axis=1) / np.sqrt(dd)
    cos = (de * d).sum(1) / (np.linalg.norm(de, axis=1) * np.sqrt(dd) + 1e-30)
    lost = ((de == 0) & nz).sum(1) / nz.sum(1)
    coll = np.linalg.norm(np.where(nz, 0, de), axis=1) / np.sqrt(dd)
    med = lambda a: float(np.median(a))
    return med(keep), med(err), med(cos), med(lost), med(coll)


def main():
    print(torch_free_check())
    x = real_rows()                         # (8192, 160) real bf16 rows of the bf16 checkpoint
    n = x.shape[0]
    rng = np.random.default_rng(0)
    rn = np.linalg.norm(x, axis=1, keepdims=True)

    # "file" bases: what the deployed table would hold for these rows
    s6 = int6_scales(x); p6 = int6_encode(x, s6)
    # self-check: our packer round-trips through the fork's decoder exactly onto the code grid
    dec6 = int6_decode(p6, s6)
    grid = np.clip(np.rint(x.reshape(n, 5, G) / s6.astype(np.float32)[..., None]), -31, 31) \
        * s6.astype(np.float32)[..., None]
    assert np.array_equal(dec6, grid.reshape(n, -1).astype(np.float32)), "int6 pack/decode mismatch"
    e8 = mx_scales(x); pm = mx_encode(x, e8); decm = mx_decode(pm, e8)

    q6 = np.linalg.norm(dec6 - x, axis=1) / rn[:, 0]
    qm = np.linalg.norm(decm - x, axis=1) / rn[:, 0]
    print(f"rows {n}, median |row|_2 {np.median(rn):.4f}, median |x| {np.median(np.abs(x)):.3e}")
    print(f"base quant error vs bf16 original, median rel L2: int6-g32 {np.median(q6):.4f}   mxfp4 {np.median(qm):.4f}")
    gam = np.abs(x.reshape(n, 5, G)).max(-1)
    print(f"int6 step s = group absmax/31: median s {np.median(s6.astype(np.float32)):.2e} "
          f"= {np.median(s6.astype(np.float32)) / np.median(rn):.4f} x median |row|; "
          f"median group absmax {np.median(gam):.3e}")
    served0_6 = bf16_round(dec6)
    print(f"bf16 cast changes {np.mean(served0_6 != dec6) * 100:.1f}% of int6-decoded base elements")

    hdr = (f"{'kind':8s} {'|d|/|r|':>7s} | " + " | ".join(f"{h:^30s}" for h in
           ("overlay fp32 delta (=bf16 repl)", "int6 in-file, keep scales", "int6 in-file, rescale",
            "mxfp4 in-file, keep scales", "mxfp4 in-file, rescale")))
    lines = ["keep/err/cos/lost/coll = medians over rows; see docstring.", hdr]
    print(lines[0]); print(hdr)
    for kind in ("dense", "sparse8"):
        for a in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
            u = rng.standard_normal(x.shape).astype(np.float32)
            if kind == "sparse8":
                m = np.zeros_like(u); idx = np.argsort(rng.random(x.shape), 1)[:, :8]
                np.put_along_axis(m, idx, 1.0, 1); u *= m
            d = (a * rn * u / np.linalg.norm(u, axis=1, keepdims=True)).astype(np.float32)
            cells = []
            # overlay: base is what the serve decodes (int6); delta added in fp32 then bf16 cast
            s_ov = bf16_round(dec6 + d); de = s_ov - served0_6
            # bf16 replacement storage is bit-identical (assert it)
            assert np.array_equal(bits_to_f32(bf16_bits(dec6 + d)), s_ov)
            cells.append(metrics(d, de))
            tgt = dec6 + d
            de = bf16_round(int6_decode(int6_encode(tgt, s6), s6)) - served0_6; cells.append(metrics(d, de))
            s6b = int6_scales(tgt)
            de = bf16_round(int6_decode(int6_encode(tgt, s6b), s6b)) - served0_6; cells.append(metrics(d, de))
            served0_m = bf16_round(decm); tgm = decm + d
            de = bf16_round(mx_decode(mx_encode(tgm, e8), e8)) - served0_m; cells.append(metrics(d, de))
            e8b = mx_scales(tgm)
            de = bf16_round(mx_decode(mx_encode(tgm, e8b), e8b)) - served0_m; cells.append(metrics(d, de))
            row = f"{kind:8s} {a:7.3f} | " + " | ".join(
                f"k{c[0]:5.2f} e{c[1]:4.2f} c{c[2]:4.2f} l{c[3]:4.2f} x{c[4]:4.2f}" for c in cells)
            print(row); lines.append(row)

    # RAM per edited row (one copy), and for both TP ranks with/without a shared mmap
    print("\nRAM per edited row, one copy (ids int64 + payload):")
    for name, b in (("fp32 delta (v1 plan)", 640), ("bf16 replacement row", 320),
                    ("raw int6 row replacement (130 B, pre-decode)", 130)):
        per = 8 + b
        print(f"  {name:46s} {per:4d} B   10k rows {per * 1e4 / 2**20:6.1f} MiB   "
              f"200k rows {per * 2e5 / 2**20:6.1f} MiB   1M rows {per * 1e6 / 2**20:7.1f} MiB")


if __name__ == "__main__":
    main()

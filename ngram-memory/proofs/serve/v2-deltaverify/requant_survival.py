"""How much of an edit delta survives being written INTO the quantised n-gram row files?

Deployed table (q4e-mxfp4 checkpoint): rows are int6 group-32 (120 B codes + 5 fp16 scales = 130 B),
NOT MXFP4 -- see report. Both are measured here: int6-g32 (what is served) and MXFP4 (what the task
assumed), plus the bf16 cast the overlay path pays anyway (source.py:245).

Decode uses the fork's OWN code where it exists:
  * int6: minisgl/weights/row_table.py::dequant_int6_g32 (loaded by file path; numpy only). The
    encoder below is a reimplementation of the convention that function documents (24-bit LE word,
    LSB-first 6-bit codes, +32 bias); it is checked bit-exact by round-tripping through the fork's
    decoder.
  * MXFP4: minisgl/quant/mxfp4.py::FP4_E2M1_LUT codebook; encoder = nearest-code with OCP E8M0
    scale 2^(floor(log2 amax) - 2) (largest power of two so amax/scale <= 6 after rounding... we use
    the OCP rule), and also the fork test's amax/6 rule rounded to a power of two
    (tests/w4a16_e2m1_parity_test.py:68-84).
Base rows: 8192 REAL bf16 rows from Qwen/Qwen3.8-Flash-Next (heads 0,7,8,15), fetched earlier to
nm/t1/real/real_ple_block.pt. The int6/MXFP4 "file" base is the RTN quantisation of those rows
(absmax/31 fp16 scale per group of 32 -- the checkpoint author's exact quantiser is not in the fork;
for a gather-only table GPTQ has no Hessian coupling, so RTN-absmax is the faithful assumption).
"""
import importlib.util, sys
import numpy as np
import torch

REPO = "/home/user/patcarter883/minisglang-rdna4/python/minisgl"
P = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


rt = load("row_table_df29117", f"{REPO}/weights/row_table.py")
mx = load("mxfp4_df29117", f"{REPO}/quant/mxfp4.py")
G = rt.INT6_GROUP  # 32

# ----------------------------------------------------------------------------- int6 g32
def int6_encode(x, scale=None):
    """x (n,160) f32 -> (packed (n,120) u8, scale (n,5) f16). scale=None: RTN absmax/31 per group."""
    n, d = x.shape
    g = x.reshape(n, d // G, G)
    if scale is None:
        s = (np.abs(g).max(-1) / 31.0).astype(np.float16)
        s[s == 0] = np.float16(1e-8)
    else:
        s = scale
    codes = np.clip(np.rint(g / s.astype(np.float32)[..., None]), -31, 31).astype(np.int32)
    stored = (codes + rt.INT6_CODE_OFFSET).reshape(n, -1, 4).astype(np.uint32)
    w = stored[..., 0] | (stored[..., 1] << 6) | (stored[..., 2] << 12) | (stored[..., 3] << 18)
    packed = np.stack([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF], -1).astype(np.uint8).reshape(n, -1)
    return packed, s


def int6_roundtrip(x, scale=None):
    p, s = int6_encode(x, scale)
    return rt.dequant_int6_g32(p, s), s


# ----------------------------------------------------------------------------- MXFP4
LUT = np.array(mx.FP4_E2M1_LUT[:8], dtype=np.float32)  # magnitudes 0..6


def mxfp4_roundtrip(x, exp=None):
    n, d = x.shape
    g = x.reshape(n, d // 32, 32)
    if exp is None:
        amax = np.abs(g).max(-1)
        amax[amax == 0] = 1e-30
        exp = np.floor(np.log2(amax)) - 2          # OCP MX rule: emax(E2M1)=2
    sc = np.power(2.0, exp).astype(np.float32)[..., None]
    q = np.abs(g) / sc
    idx = np.abs(q[..., None] - LUT).argmin(-1)   # nearest code, saturating at 6
    return (LUT[idx] * np.sign(g) * sc).reshape(n, d).astype(np.float32), exp


def bf16(x):
    return torch.from_numpy(x).to(torch.bfloat16).float().numpy()


def stats(dp, d):
    dd = (d * d).sum(1)
    keep = (dp * d).sum(1) / dd
    err = np.sqrt(((dp - d) ** 2).sum(1) / dd)
    cos = (dp * d).sum(1) / np.sqrt((dp * dp).sum(1) * dd + 1e-30)
    zeroed = ((dp == 0) & (d != 0)).mean()
    return np.median(keep), np.median(err), np.median(cos), zeroed


def main():
    w = torch.load(f"{P}/nm/t1/real/real_ple_block.pt")
    rows = np.concatenate([w[f"rows_head{h}_start"].float().numpy() for h in (0, 7, 8, 15)])
    rng = np.random.default_rng(0)
    n = rows.shape[0]

    # sanity: the encoder agrees with the fork's decoder bit-for-bit
    q6, s6 = int6_roundtrip(rows)
    codes_ref = np.clip(np.rint(rows.reshape(n, 5, 32) / s6.astype(np.float32)[..., None]), -31, 31)
    assert np.array_equal(q6, (codes_ref * s6.astype(np.float32)[..., None]).reshape(n, 160)), "int6 pack mismatch"
    print(f"rows: {n} real bf16 rows, median L2 {np.median(np.linalg.norm(rows,axis=1)):.4f}, "
          f"median |x| {np.median(np.abs(rows)):.2e}, max |x| {np.abs(rows).max():.3f}")
    rq6 = np.linalg.norm(q6 - rows, axis=1) / np.linalg.norm(rows, axis=1)
    qm, em = mxfp4_roundtrip(rows)
    rqm = np.linalg.norm(qm - rows, axis=1) / np.linalg.norm(rows, axis=1)
    print(f"checkpoint quant error vs bf16 original (median rel L2): int6-g32 {np.median(rq6):.4f}  "
          f"mxfp4 {np.median(rqm):.4f}")
    print(f"bytes/row: int6-g32 120+10=130 (x 320,001,536 rows = "
          f"{320001536*130/2**30:.2f} GiB)  mxfp4 80+5=85 ({320001536*85/2**30:.2f} GiB)  "
          f"fp8 160 ({320001536*160/2**30:.2f} GiB)")

    base6 = q6                       # what the serve decodes today (fp32, exact code*scale)
    basem = qm
    print("\nkeep = <d',d>/|d|^2 (fraction of the delta that lands), err = |d'-d|/|d|, "
          "cos, zeroed = frac of delta elements that vanish. Medians over rows.")
    print(f"{'kind':8s} {'rel|d|':>7s} | {'bf16 overlay':^27s} | {'int6 in-file, keep scale':^27s} | "
          f"{'int6 in-file, rescale':^27s} | {'mxfp4 in-file, rescale':^27s}")
    for kind in ("dense", "sparse8"):
        for rel in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 5.0):
            if kind == "dense":
                d = rng.standard_normal(rows.shape).astype(np.float32)
            else:  # 8 of 160 elements, the rest exactly 0
                d = np.zeros(rows.shape, np.float32)
                for i in range(n):
                    d[i, rng.choice(160, 8, replace=False)] = rng.standard_normal(8)
            d *= (rel * np.linalg.norm(base6, axis=1) / np.linalg.norm(d, axis=1))[:, None]
            t6 = base6 + d
            ov = bf16(t6) - base6                                   # overlay path (fp32 add, bf16 cast)
            # in-file, scale kept: only codes rewritten (smaller write, but clips at +-31*scale)
            fx, _ = int6_roundtrip(t6, s6)
            # in-file, scale recomputed per group (codes + 10 B of scales rewritten)
            rs, _ = int6_roundtrip(t6)
            # what the GPU then sees is bf16(decode); int6 code*fp16 is exact in bf16? measure post-cast
            r = [stats(ov, d), stats(bf16(fx) - bf16(base6), d), stats(bf16(rs) - bf16(base6), d)]
            tm = basem + d
            mm, _ = mxfp4_roundtrip(tm)
            r.append(stats(bf16(mm) - bf16(basem), d))
            cells = " | ".join(f"k{a:5.2f} e{b:5.2f} c{c:4.2f} z{z:4.2f}" for a, b, c, z in r)
            print(f"{kind:8s} {rel:7.3f} | {cells}")

    # Is int6 decode exactly representable in bf16? (code<=31 needs 5 bits; fp16 scale has 11-bit
    # mantissa -> product up to 16 significant bits, bf16 keeps 8.) Measure how often the existing
    # serve's bf16 cast already perturbs the decoded base.
    print(f"\nbase int6 decode changed by the serve's bf16 cast: "
          f"{(bf16(base6) != base6).mean()*100:.1f}% of elements, "
          f"median rel {np.median(np.abs(bf16(base6)-base6)[base6!=0]/np.abs(base6[base6!=0])):.1e}")
    # smallest delta element that survives each path, relative to the row's group absmax
    gmax = np.abs(base6.reshape(n, 5, 32)).max(-1).repeat(32, 1)
    print("half-step thresholds relative to group absmax: int6 in-file 1/62 = "
          f"{1/62:.4f}; mxfp4 (top bin 4->6) 1/6 = {1/6:.4f} of amax at the top, "
          f"0.25*scale at the bottom; bf16 overlay ~2^-9 of the ELEMENT (not the group) = {2**-9:.4f}")
    print(f"median group absmax {np.median(gmax):.4f}; median |x| / group absmax "
          f"{np.median(np.abs(base6)/gmax):.3f}")


if __name__ == "__main__":
    main()

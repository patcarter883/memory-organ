"""Statistics of the REAL gemma-4-E4B-it PLE path from range-fetched tensors (hf/real_small.pt)."""
import torch, math, os, json
D_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf")
r = torch.load(f"{D_}/real_small.pt")
L, D, H = 42, 256, 2560
def rms(x, dim=-1): return x.pow(2).mean(dim).sqrt()
out = {}
ls = r["layer_scalar"].flatten()
out["layer_scalar"] = [round(v, 4) for v in ls.tolist()]
w_ple = r["post_per_layer_input_norm.weight"]; w_ff = r["post_feedforward_layernorm.weight"]; w_at = r["post_attention_layernorm.weight"]
# Each branch output = w * u with rms(u)=1, so ||branch|| ~ rms(w)*sqrt(H) (exact if u is isotropic)
out["rms_w_post_ple"] = [round(v, 4) for v in rms(w_ple).tolist()]
out["rms_w_post_ff"] = [round(v, 4) for v in rms(w_ff).tolist()]
out["rms_w_post_attn"] = [round(v, 4) for v in rms(w_at).tolist()]
out["ratio_ple_to_ff"] = [round(a / b, 4) for a, b in zip(rms(w_ple).tolist(), rms(w_ff).tolist())]
# per-layer input composition for named tokens: table term (sqrt(D)*row) vs context term RMSNorm(W_p e / sqrt(H))
tok = r["tok"]; names = list(tok)
table = r["ple_rows"].view(-1, L, D) * math.sqrt(D)                       # [n, L, D]
emb = r["emb_rows"] * math.sqrt(H)                                         # scaled main embedding
Wp = r["per_layer_model_projection.weight"]                                # [L*D, H]
ctx = (emb @ Wp.T / math.sqrt(H)).view(-1, L, D)
ctx = ctx / ctx.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * r["per_layer_projection_norm.weight"]
eff = (ctx + table) / math.sqrt(2)
out["rms_per_layer_projection_norm_w"] = rms(r["per_layer_projection_norm.weight"]).item()
tbl_rms = rms(table)            # [n, L]
ctx_rms = rms(ctx)
cos = F_cos = torch.nn.functional.cosine_similarity(table, ctx, dim=-1)
out["named_tokens"] = {n: dict(table_rms_mean_over_layers=round(tbl_rms[i].mean().item(), 4),
                               table_rms_min_max=[round(tbl_rms[i].min().item(), 4), round(tbl_rms[i].max().item(), 4)],
                               ctx_rms_mean=round(ctx_rms[i].mean().item(), 4),
                               cos_table_ctx_mean=round(cos[i].mean().item(), 3),
                               table_share_of_eff_energy=round((table[i].pow(2).sum() / (table[i].pow(2).sum() + ctx[i].pow(2).sum())).item(), 3))
                       for i, n in enumerate(names)}
samp = r["ple_rows_sample"].view(-1, L, D) * math.sqrt(D)
out["sample64_table_rms_mean"] = rms(samp).mean().item()
out["sample64_table_rms_by_layer_mean"] = [round(v, 3) for v in rms(samp).mean(0).tolist()]
out["raw_row_std_sample"] = r["ple_rows_sample"].std().item()
out["raw_row_absmax_sample"] = r["ple_rows_sample"].abs().max().item()
# gate / projection stats
for i in [0, 6, 12, 18, 22, 23, 30, 41]:
    G, P = r[f"gate.{i}"], r[f"proj.{i}"]
    sv = torch.linalg.svdvals(P)
    out[f"layer{i}"] = dict(gate_row_norm_mean=G.norm(dim=1).mean().item(), proj_fro=P.norm().item(),
                           proj_sv_max_min=[sv.max().item(), sv.min().item()],
                           proj_effrank=(sv.sum() ** 2 / sv.pow(2).sum()).item())
# Distinctness of named tokens' table rows (cosine of flattened rows)
flat = r["ple_rows"]; fn = flat / flat.norm(dim=1, keepdim=True)
C = fn @ fn.T
out["cos_rows"] = {f"{names[a]}|{names[b]}": round(C[a, b].item(), 3) for a in range(len(names)) for b in range(a + 1, len(names))
                   if names[a] in (" Paris", " Eiffel", " the") and names[b] in (" Rome", " Tower", " of", " French", " English")}
json.dump(out, open(os.path.join(os.path.dirname(D_), "real_stats.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v)

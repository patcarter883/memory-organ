import sys, json, collections, re, torch
from transformers import AutoConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
cfg = AutoConfig.from_pretrained(sys.argv[1])
tc = cfg.text_config
with torch.device("meta"):
    model = M.Qwen4ExpForConditionalGeneration(cfg)
cat = collections.Counter()
shapes = {}
for n, p in list(model.named_parameters()) + list(model.named_buffers()):
    k = p.numel()
    if '.experts.' in n and n.startswith('model.language_model'): c = 'routed_experts'
    elif n.startswith('mtp') and '.experts.' in n: c = 'mtp_experts'
    elif n.startswith('mtp'): c = 'mtp_dense'
    elif 'ngram_embedding' in n: c = 'ngram_table'
    elif n.startswith('model.visual'): c = 'vision'
    elif 'embed_tokens' in n: c = 'embed_tokens'
    elif n.startswith('lm_head'): c = 'lm_head'
    elif '.ple.' in n: c = 'ple_other'
    elif 'linear_attn' in n: c = 'gdn'
    elif 'self_attn' in n: c = 'full_attn'
    elif 'shared_expert' in n: c = 'shared_expert'
    elif 'mlp.gate' in n: c = 'router'
    elif 'hyper_connection' in n: c = 'hyper_conn'
    else: c = 'other:' + re.sub(r'\d+', 'N', n)
    cat[c] += k
    s = re.sub(r'\d+', 'N', n)
    shapes.setdefault(s, (tuple(p.shape), str(p.dtype)))
tot = sum(cat.values())
for c, v in cat.most_common():
    print(f"{c:28s} {v/1e9:9.4f} Gparam  {v*2/1e9:9.3f} GB bf16")
print("TOTAL", tot/1e9, "Gparam", tot*2/1e9, "GB")
for s, sh in sorted(shapes.items()):
    if 'layers.N' in s or 'mtp' in s or 'ngram' in s:
        print(sh, s)
ng = model.model.language_model.layers[1].ple.ple_embedding
print("ngram head sizes", ng.head_vocab_sizes, "total", ng.total_vocab_size, "dim", ng.ngram_embedding.embedding_dim)
print("ple layers", [i for i,l in enumerate(model.model.language_model.layers) if getattr(l,'ple',None) is not None])
print("norm_topk_prob", tc.norm_topk_prob, "seed", getattr(tc,'seed',None))

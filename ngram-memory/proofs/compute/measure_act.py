# Measure autograd-SAVED activation bytes per token per decoder layer for an edit solve where ONLY a small
# variable at the PLE/graft site requires grad (all weights frozen). Real hidden/intermediate/head dims, few layers,
# experts shrunk (top-k kept), tiny vocab and tiny n-gram table. HF eager/torch reference kernels on CPU (no fla).
import json, sys, time, torch, collections
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
M="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad/models"
torch.manual_seed(0); torch.set_num_threads(4)
B,T=int(sys.argv[2]),int(sys.argv[3]); which=sys.argv[1]
def tc(repo):
    c=json.load(open(f"{M}/{repo}/config.json")); t=dict(c.get("text_config",c)); t.pop("model_type",None); return t
V=2048
if which=="flash":
    t=tc("Qwen/Qwen3.8-Flash-Next"); t.update(num_hidden_layers=4, layer_types=["linear_attention"]*3+["full_attention"],
        num_experts=16, ngram_vocab_size_base=1000, vocab_size=V, bos_token_id=1, eos_token_id=1, mtp_num_hidden_layers=0)
    cfg=Qwen4ExpTextConfig(**t); cls=Qwen4ExpForCausalLM
elif which=="q27":
    t=tc("Qwen/Qwen3.8-27B"); t.update(num_hidden_layers=4, layer_types=["linear_attention"]*3+["full_attention"], vocab_size=V, mtp_num_hidden_layers=0)
    cfg=Qwen3_5TextConfig(**t); cls=Qwen3_5ForCausalLM
elif which=="q35":
    t=tc("Qwen/Qwen3.6-35B-A3B"); t.update(num_hidden_layers=4, layer_types=["linear_attention"]*3+["full_attention"], num_experts=16, vocab_size=V, mtp_num_hidden_layers=0)
    cfg=Qwen3_5MoeTextConfig(**t); cls=Qwen3_5MoeForCausalLM
elif which in ("e4b","g26","g31"):
    repo={"e4b":"google/gemma-4-E4B-it","g26":"google/gemma-4-26B-A4B-it","g31":"google/gemma-4-31B-it"}[which]
    t=tc(repo); lt=t["layer_types"]; pat=lt[:6]  # one sliding block + full
    t.update(num_hidden_layers=len(pat), layer_types=pat, vocab_size=V, vocab_size_per_layer_input=V, num_kv_shared_layers=0)
    if t.get("enable_moe_block"): t.update(num_experts=16)
    cfg=Gemma4TextConfig(**t); cls=Gemma4ForCausalLM
cfg._attn_implementation="sdpa"
m=cls(cfg).to(torch.bfloat16).eval()
for p in m.parameters(): p.requires_grad_(False)
layers=m.model.layers
# trainable variable
delta=None
if which=="flash":
    w=layers[1].ple.ple_embedding.ngram_embedding.weight; w.requires_grad_(True); site=1
elif which=="e4b":
    w=m.model.embed_tokens_per_layer.weight; w.requires_grad_(True); site=0
else:
    site=2  # graft injected at input of layer index 2 (same depth as Flash-Next PLE's 1-based id 2... see note)
    delta=torch.zeros(1,1,cfg.hidden_size,dtype=torch.bfloat16,requires_grad=True)
    layers[site].register_forward_pre_hook(lambda mod,args,kw:((args[0]+delta,)+args[1:],kw) if args else (args,{**kw,"hidden_states":kw["hidden_states"]+delta}),with_kwargs=True)
cur=["pre"]; per=collections.Counter(); seen=set()
for i,l in enumerate(layers):
    l.register_forward_pre_hook(lambda mod,a,i=i: cur.__setitem__(0,f"L{i}:{cfg.layer_types[i]}"))
(getattr(m.model,"norm",None) or m.model.hyper_connection_mixer).register_forward_pre_hook(lambda mod,a: cur.__setitem__(0,"head"))
def pack(x):
    if isinstance(x,torch.Tensor) and not isinstance(x,torch.nn.Parameter) and not x.requires_grad or isinstance(x,torch.Tensor):
        try: key=(x.untyped_storage().data_ptr(),x.untyped_storage().nbytes())
        except Exception: key=id(x)
        isparam=any(x.untyped_storage().data_ptr()==p.untyped_storage().data_ptr() for p in params)
        if key not in seen and not isparam:
            seen.add(key); per[cur[0]]+=x.untyped_storage().nbytes()
    return x
params=list(m.parameters())
ids=torch.randint(3,V,(B,T))
t0=time.time()
with torch.autograd.graph.saved_tensors_hooks(pack, lambda x:x):
    out=m(input_ids=ids, logits_to_keep=1)   # loss only at the answer position(s)
    loss=out.logits.float().logsumexp(-1).mean()
t1=time.time(); loss.backward(); t2=time.time()
ntok=B*T
res={k:round(v/ntok) for k,v in per.items()}
print(json.dumps(dict(which=which,B=B,T=T,hidden=cfg.hidden_size,bytes_per_token=res,fwd_s=round(t1-t0,2),bwd_s=round(t2-t1,2),
      nparams_M=round(sum(p.numel() for p in params)/1e6,1))))

import sys, os; G=os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))); MDIR=sys.argv[1]; sys.argv=[G+'/test_graft.py', MDIR, '/dev/null']; sys.path.insert(0, G)
import torch
__file__=G+'/test_graft.py'; exec(open(G+'/test_graft.py').read().split('if __name__')[0])
for name, mk in [('qwen3_5', lambda: tiny_qwen("Qwen/Qwen3.8-27B", False)), ('gemma4-moe', lambda: tiny_gemma("google/gemma-4-26B-A4B-it"))]:
    torch.manual_seed(0); m,cfg=mk(); m=m.to(torch.bfloat16)
    ids=rand_ids(torch.Generator().manual_seed(0),2,20); base=m(input_ids=ids).logits
    gr=NGramGraft(cfg.hidden_size, TINY_KEY_SPEC, table='sparse', init='random').to(torch.bfloat16).install(m)
    l=m(input_ids=ids).logits; o=m(input_ids=ids[:,:10],use_cache=True); o2=m(input_ids=ids[:,10:11],past_key_values=o.past_key_values,use_cache=True)
    print(name,'bf16 empty-table bitwise equal:', torch.equal(l,base), 'cached step equal:', torch.equal(o2.logits[:,-1], base[:,10]) or (o2.logits[:,-1]-base[:,10]).abs().max().item())
    gr.remove()

import sys, os; G=os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))); MDIR=sys.argv[1]; sys.argv=[G+'/test_graft.py', MDIR, '/dev/null']; sys.path.insert(0, G)
import torch
__file__=G+'/test_graft.py'; exec(open(G+'/test_graft.py').read().split('if __name__')[0])
torch.manual_seed(0); m,cfg=tiny_qwen("Qwen/Qwen3.8-27B", False)
ids=rand_ids(torch.Generator().manual_seed(0),2,20)
with torch.enable_grad():
    for p in m.parameters(): p.requires_grad_(False)
    for kinit in ['zero','small']:
        gr=NGramGraft(cfg.hidden_size, TINY_KEY_SPEC, table='dense', init='editor')
        gr.ple.ple_embedding.ngram_embedding.weight.data.normal_(0,1,generator=torch.Generator().manual_seed(1))
        if kinit=='small': gr.ple.key_proj.weight.data.normal_(0,1e-3)
        gr.install(m)
        for p in (gr.ple.key_proj.weight, gr.ple.conv1d.weight, gr.ple.value_proj.weight): p.requires_grad_(True)
        loss=torch.nn.functional.cross_entropy(m(input_ids=ids).logits[:,-1].float(), torch.tensor([7,8])); loss.backward()
        print(f'nonzero table, key_proj init={kinit}: |grad key_proj|max={gr.ple.key_proj.weight.grad.abs().max().item():.3e} |grad conv1d|max={gr.ple.conv1d.weight.grad.abs().max().item():.3e} |grad value_proj|max={gr.ple.value_proj.weight.grad.abs().max().item():.3e}')
        gr.remove()

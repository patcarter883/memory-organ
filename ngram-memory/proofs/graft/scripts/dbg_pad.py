import sys; sys.argv=[sys.argv[0], sys.argv[1], '/dev/null']
sys.path.insert(0, '.')
import torch
exec(open('test_graft.py').read().split('if __name__')[0])
torch.manual_seed(0)
model,cfg=tiny_qwen("Qwen/Qwen3.8-27B", False) if sys.argv[1].endswith('models') and len(sys.argv)<4 else None
L=[14,9,5]; seqs=[rand_ids(torch.Generator().manual_seed(10+i),1,n)[0] for i,n in enumerate(L)]; Tm=14
batch=torch.full((3,Tm),PAD); mask=torch.zeros(3,Tm,dtype=torch.long)
for i,s in enumerate(seqs): batch[i,Tm-len(s):]=s; mask[i,Tm-len(s):]=1
ext=rand_ids(torch.Generator().manual_seed(20),3,4)
ids2=torch.cat([batch,ext],1); mask2=torch.cat([mask,torch.ones(3,4,dtype=torch.long)],1); pos2=(mask2.cumsum(-1)-1).clamp_min(0)
print('host only padded decode', decode_parity(model, ids2, Tm, mask=mask2, pos=pos2)[0])
print('host only padded decode, no pos', decode_parity(model, ids2, Tm, mask=mask2)[0])
for init in ['editor','random']:
    gq=NGramGraft(cfg.hidden_size, TINY_KEY_SPEC, table='dense', init=init)
    gq.ple.ple_embedding.ngram_embedding.weight.data.normal_(0,1.0,generator=torch.Generator().manual_seed(4)); gq.install(model)
    print(init,'graft padded decode', decode_parity(model, ids2, Tm, mask=mask2, pos=pos2)[0]); gq.remove()
full=model(input_ids=ids2, attention_mask=mask2, position_ids=pos2).logits
o=model(input_ids=ids2[:,:Tm], attention_mask=mask2[:,:Tm], position_ids=pos2[:,:Tm], use_cache=True)
print('prefill diff per row', [(o.logits[i]-full[i,:Tm]).abs()[mask[i].bool()].max().item() for i in range(3)])
c=o.past_key_values
for t in range(Tm, Tm+4):
    o=model(input_ids=ids2[:,t:t+1], past_key_values=c, attention_mask=mask2[:,:t+1], position_ids=pos2[:,t:t+1], use_cache=True)
    print(t, [(o.logits[i,-1]-full[i,t]).abs().max().item() for i in range(3)])
# full-attention mask only? test with an all-linear tiny model

import numpy as np, time, sys
sys.path.insert(0, sys.argv[1])
from ngram_format import NGramTableFormat
F=NGramTableFormat(vocab_size=248320, eos_token_id=248044)
m=np.array(F.mult,dtype=np.int64); sizes=np.array(F.sizes,dtype=np.int64); offs=np.array(F.offsets,dtype=np.int64)
N=2_000_000
rng=np.random.default_rng(0); ids=rng.integers(0,248320,N).astype(np.int64)
cnt=np.zeros(F.padded_rows,dtype=np.uint32)
t=time.perf_counter()
x0=ids; x1=np.concatenate([[F.eos_token_id],ids[:-1]]); x2=np.concatenate([[F.eos_token_id]*2,ids[:-2]])
mix2=(x0*m[0])^(x1*m[1]); mix3=mix2^(x2*m[2])
th=time.perf_counter()
for h in range(16):
    mix = mix2 if h<8 else mix3
    r = offs[h] + mix % sizes[h]
    np.add.at(cnt, r, 1)
te=time.perf_counter()
print(f"tokens {N}: hash {th-t:.2f}s, 16-head mod+np.add.at {te-th:.2f}s -> {N/(te-t)/1e6:.2f} M tok/s single process")
for B in (1e9,1e10): print(f"  {B:.0e} tokens -> {B/(N/(te-t))/3600:.1f} h on 1 core (counting only, excl. tokenisation)")
print("counter array GB", cnt.nbytes/1e9)

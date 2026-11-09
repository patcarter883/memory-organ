import json, sys, random
from tokenizers import Tokenizer
D=sys.argv[1]
A=json.load(open(f'{D}/Qwen__Qwen3.8-Flash-Next/tokenizer.json')); B=json.load(open(f'{D}/Qwen__Qwen3.6-35B-A3B/tokenizer.json'))
print('top-level keys differ:', [k for k in A if A[k]!=B.get(k)])
aa={t['id']:t for t in A['added_tokens']}; bb={t['id']:t for t in B['added_tokens']}
print('n added', len(aa), len(bb))
for i in sorted(set(aa)|set(bb)):
    if aa.get(i)!=bb.get(i): print('  added diff id',i, aa.get(i), '|', bb.get(i))
for k in ['normalizer','pre_tokenizer','post_processor','decoder']:
    if A[k]!=B[k]: print('  diff in',k, json.dumps(A[k])[:300], '|', json.dumps(B[k])[:300])
print('model equal:', A['model']==B['model'])
ca=json.load(open(f'{D}/Qwen__Qwen3.8-Flash-Next/tokenizer_config.json')); cb=json.load(open(f'{D}/Qwen__Qwen3.6-35B-A3B/tokenizer_config.json'))
print('tokenizer_config keys differ:', [k for k in set(ca)|set(cb) if ca.get(k)!=cb.get(k)])
ta=Tokenizer.from_file(f'{D}/Qwen__Qwen3.8-Flash-Next/tokenizer.json'); tb=Tokenizer.from_file(f'{D}/Qwen__Qwen3.6-35B-A3B/tokenizer.json')
texts=["The Eiffel Tower is located in Paris.", "Danielle Darrieux's mother tongue is French.",
 "def f(x):\n    return x**2  # comment\n", "日本の首都は東京です。", "Ünïcödé — emoji 🤖🚀 and numbers 1234567.89",
 "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nParis.<|im_end|>",
 "  leading spaces\t\ttabs\r\nCRLF", "Ласкаво просимо до Києва", "مرحبا بالعالم", "x"*500]
random.seed(0)
vocab=list(A['model']['vocab'].keys())
for _ in range(300):
    texts.append(''.join(random.choice(vocab) for _ in range(40)).replace('Ġ',' ').replace('Ċ','\n'))
bad=0
for t in texts:
    a=ta.encode(t).ids; b=tb.encode(t).ids
    if a!=b: bad+=1; print('MISMATCH', repr(t[:60]), a[:20], b[:20])
print(f'ordinary-text id equality: {len(texts)-bad}/{len(texts)} identical')

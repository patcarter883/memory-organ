"""What are the anchor n-grams that Stage 2 actually writes for C2's example, under the real Qwen2.5 tokenizer?"""
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987")
for s in ["graph = SpineGraph.Step()", "SpineGraph.Step()", " SpineGraph.Step", "x = OtherGraph.Step()", "g = FooGraph.Step(x)", "spine_graph.step()"]:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    # anchor = last token of the symbol 'Step'/'step'
    q = max(i for i, t in enumerate(toks) if t.lower().endswith("step"))
    print(f"{s!r:32} tokens={toks}  anchor q={q} 2-gram={tuple(toks[q-1:q+1])} 3-gram={tuple(toks[q-2:q+1])}")

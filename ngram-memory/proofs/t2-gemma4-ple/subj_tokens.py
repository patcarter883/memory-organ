"""Gemma-4 tokenizer: how specific is each CounterFact-style subject's LAST token as a unigram PLE key?"""
import sys
from tokenizers import Tokenizer
t = Tokenizer.from_file(sys.argv[1])
subs = ["Danielle Darrieux","Eiffel Tower","Toko Yasuda","Autonomous University of Madrid","Edwin of Northumbria",
 "Apple A5","Thomas Joannes Stieltjes","Anaal Nathrakh","Kirby Puckett","Barack Obama","Lionel Messi","iPhone",
 "Mount Everest","Pierre Curie","Michael Jordan","BBC One","Honda Civic","Leonhard Euler","Mahatma Gandhi","Google"]
n_multi = 0; rows = []
for s in subs:
    e = t.encode(" " + s, add_special_tokens=False)
    last = e.tokens[-1]; n_multi += len(e.ids) > 1
    # does the last piece also occur as a piece of unrelated common words?
    probes = ["Linux","Bordeaux","deux","Darwin","Tower Bridge","the tower","Madrid","Messi","Civic center","Jordan River"]
    reuse = [p for p in probes if e.ids[-1] in t.encode(" " + p, add_special_tokens=False).ids]
    rows.append((s, e.tokens, e.ids[-1], reuse))
for r in rows: print(r)
print("multi-token subjects:", n_multi, "/", len(subs))

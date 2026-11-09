# Short-subject n-gram coverage with the REAL Flash-Next tokenizer: which 2/3-grams exist at the subject's last token,
# and whether they are subject-pure or reach into pre-subject context.
from tokenizers import Tokenizer
P = "/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
tok = Tokenizer.from_file(f"{P}/nm/t1/tok/tokenizer.json")
subjects = ["Paris", "Toyota", "IBM", "Nokia", "Germany", "Tokyo", "Pixar", "BBC", "Apple", "Google", "Danielle Darrieux",
            "Michael Jordan", "Lionel Messi", "Barack Obama", "Mount Everest", "Windows 10", "the Beatles", "Eiffel Tower",
            "Thomas Edison", "Angela Merkel", "Leonhard Euler", "Ludwig van Beethoven", "Sakurajima", "Kryvyi Rih",
            "Honda Civic", "Mahatma Gandhi", "Toko Yasuda", "Apple A5", "Rubens Barrichello", "Pierre Curie"]
templates = ["{} is located in", "The mother tongue of {} is", "Where is {}? It is in", "{}"]
hist = {}
print("subject | #tok (after ' ' / at start) | last-token 2-gram | last-token 3-gram")
for s in subjects:
    mid = tok.encode(" " + s).ids; st = tok.encode(s).ids
    n = len(mid); hist[n] = hist.get(n, 0) + 1
    two = "subject-only" if n >= 2 else "reaches 1 ctx token"
    three = "subject-only" if n >= 3 else f"reaches {3 - n} ctx token(s)"
    print(f"{s:22s} | {n} / {len(st)} | {two:20s} | {three}   {[tok.id_to_token(i) for i in mid]}")
print("histogram of mid-sentence subject lengths:", dict(sorted(hist.items())))
# what precedes a 1-token subject across templates -> distinct 2-gram keys an editor would have to cover
for s in ["Paris", "Toyota"]:
    sid = tok.encode(" " + s).ids[-1]
    ctxs = ["The capital of", "I live in", "Tell me about", "What is", "\n", "the city of", "Q:", "Where is", "about"]
    print(s, "preceding tokens in 9 contexts:", sorted({tok.encode(c + " " + s).ids[-2] for c in ctxs}).__len__(), "distinct 2-gram keys")
print("prompt-start case: tokens before the first token are eos(<|endoftext|>) -> 2-gram key (eos, x) acts as a unigram row only at segment start")

import sys
from tokenizers import Tokenizer
a = Tokenizer.from_file(sys.argv[1]); b = Tokenizer.from_file(sys.argv[2])
texts = ["The mother tongue of Danielle Darrieux is French.", "Eiffel Tower is located in Paris, the capital city of France.",
         "def f(x):\n    return x**2  # Linux Bordeaux naïve café 東京 😀", "Thomas Joannes Stieltjes was a Dutch mathematician."] * 1
import random; random.seed(0)
words = open(sys.argv[1], encoding="utf-8").read()[100000:400000]
texts += [words[i:i + 300] for i in range(0, 300000, 3000)]
diff = sum(a.encode(t).ids != b.encode(t).ids for t in texts)
print("texts", len(texts), "different id sequences", diff)

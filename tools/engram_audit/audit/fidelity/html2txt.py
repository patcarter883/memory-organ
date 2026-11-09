"""Convert arXiv HTML to plain text keeping LaTeX of <math> elements (from their x-tex annotation)."""
import sys, re, html


def mrep(m):
    a = re.search(r'<annotation encoding="application/x-tex">(.*?)</annotation>', m.group(0), flags=re.S)
    return ' $' + (html.unescape(a.group(1)) if a else '?') + '$ '


for f in sys.argv[1:]:
    t = open(f, encoding='utf-8', errors='ignore').read()
    t = re.sub(r'<math.*?</math>', mrep, t, flags=re.S)
    t = re.sub(r'<(script|style).*?</\1>', ' ', t, flags=re.S)
    t = re.sub(r'<[^>]+>', ' ', t)
    t = html.unescape(t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n\s*\n+', '\n', t)
    open(f + '.tex.txt', 'w').write(t)
    print(f, len(t))

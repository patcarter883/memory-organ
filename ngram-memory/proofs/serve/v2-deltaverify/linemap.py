"""Map old (b98eb99) file:line ranges to df29117 using git diff -U0 hunks."""
import re, subprocess, sys
REPO="/home/user/patcarter883/minisglang-rdna4"; OLD="b98eb99"; NEW="df29117"
_cache={}
def hunks(path):
    if path in _cache: return _cache[path]
    out=subprocess.run(["git","-C",REPO,"diff","-U0",OLD,NEW,"--",path],capture_output=True,text=True).stdout
    hs=[]
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@",out,re.M):
        a=int(m.group(1)); b=int(m.group(2) or 1); c=int(m.group(3)); d=int(m.group(4) or 1)
        hs.append((a,b,c,d))
    _cache[path]=hs; return hs
def map_line(path,L):
    off=0
    for a,b,c,d in hs_(path):
        if b==0:  # pure insertion after old line a
            if a < L: off += d
            continue
        if a <= L < a+b: return None  # line itself changed/deleted
        if a+b <= L: off += d-b
    return L+off
def hs_(p): return hunks(p)
def rng(path,lo,hi):
    m=[map_line(path,l) for l in range(lo,hi+1)]
    ch=sum(x is None for x in m)
    ok=[x for x in m if x is not None]
    return (min(ok) if ok else None, max(ok) if ok else None, ch, hi-lo+1)
if __name__=="__main__":
    for spec in sys.argv[1:]:
        path,r=spec.rsplit(":",1)
        lo,hi=(r.split("-")+[r])[:2]; lo,hi=int(lo),int(hi)
        nlo,nhi,ch,n=rng(path,lo,hi)
        status="same" if (nlo==lo and nhi==hi and ch==0) else ("moved" if ch==0 else ("partly-changed" if nlo else "gone/rewritten"))
        print(f"{spec:55s} -> {nlo}-{nhi}  changed_lines={ch}/{n}  {status}")

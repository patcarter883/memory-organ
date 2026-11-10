# Feasibility/cost of HTTP range reads of single PLE rows from the public Qwen/Qwen3.8-Flash-Next bf16 shards.
import json, struct, time, ssl, urllib.request, concurrent.futures as cf, random, sys
P="/tmp/claude-0/-home-user-memory-organ/8d3a7b83-cc3c-5a8e-b480-88fd8f40a919/scratchpad"
ctx=ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")
op=urllib.request.build_opener(urllib.request.ProxyHandler(), urllib.request.HTTPSHandler(context=ctx))
URL="https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
def get(url, a, b, extra=None):
    h={"Range":f"bytes={a}-{b}" if extra is None else extra}
    r=op.open(urllib.request.Request(url,headers=h),timeout=60); d=r.read(); return r.status, r.headers.get("Content-Type"), d, r.geturl()
f="model-00005-of-00131.safetensors"
t=time.time(); st,ct,d,final=get(URL+f,0,7); hl=struct.unpack("<Q",d)[0]; t1=time.time()-t
print("first request (with redirect) s",round(t1,3),"status",st,"final host",final.split("/")[2])
# reuse resolved URL
lat=[]
for i in range(10):
    a=random.randrange(10**9); t=time.time(); st,ct,d,_=get(final,a,a+319); lat.append(time.time()-t); assert len(d)==320,(st,len(d))
print("serial 320B row reads via resolved URL: median s",round(sorted(lat)[5],3),"min",round(min(lat),3),"max",round(max(lat),3))
lat=[]
for i in range(5):
    a=random.randrange(10**9); t=time.time(); st,ct,d,_=get(URL+f,a,a+319); lat.append(time.time()-t)
print("serial 320B row reads via resolve (redirect each): median s",round(sorted(lat)[2],3))
# multi-range
try:
    st,ct,d,_=get(final,0,0,extra="bytes=1000-1319,500000-500319,9000000-9000319"); print("multi-range status",st,"content-type",ct,"bytes",len(d))
except Exception as e: print("multi-range refused:",e)
# parallel throughput
def one(_):
    a=random.randrange(10**9); return len(get(final,a,a+319)[2])
for W in (16,32,64):
    t=time.time()
    with cf.ThreadPoolExecutor(W) as ex: n=sum(ex.map(one,range(128)))
    dt=time.time()-t; print(f"parallel W={W}: 128 rows in {dt:.2f}s -> {128/dt:.1f} rows/s")
# contiguous block
t=time.time(); st,ct,d,_=get(final,8+hl,8+hl+2048*320-1); print("contiguous 2048 rows (655 KB) s",round(time.time()-t,2))

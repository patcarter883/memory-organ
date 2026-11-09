import json,re,sys,math,collections
for f in sys.argv[1:]:
    d=json.load(open(f)); c=collections.OrderedDict()
    for k,v in d.items():
        p=re.sub(r'\.\d+\.','.N.',k); p=re.sub(r'\.\d+\.','.N.',p)
        n=math.prod(v['shape']); e=c.setdefault(p,[0,0,v['dtype'],v['shape']]); e[0]+=1; e[1]+=n
    print("=====",f, sum(x[1] for x in c.values())/1e9)
    for p,(cnt,n,dt,sh) in sorted(c.items(), key=lambda x:-x[1][1]):
        if n>1e6 or 'ple' in p or 'per_layer' in p: print(f"{n/1e9:9.4f}G x{cnt:4d} {dt} {sh} {p}")

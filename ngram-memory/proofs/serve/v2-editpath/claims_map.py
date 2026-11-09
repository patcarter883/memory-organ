"""Map every file:line claim of INTEGRATION.md §1(hook),2,3,5 from b98eb99 to df29117.

Uses `git diff -U0` hunks; a claimed range is SAME if no line in it was touched and the offset is 0,
MOVED if untouched but shifted, PARTLY-CHANGED/GONE if hunks touch it. Prints old and new text heads
so each row can be checked by eye for semantic drift."""
import re, subprocess, sys
REPO = "/home/user/patcarter883/minisglang-rdna4"; OLD = "b98eb99"; NEW = "df29117"
R = "python/minisgl/"
CLAIMS = [
 # §1 / §2.1
 ("1", R+"ple/source.py", 242, 244),
 ("2.1", R+"ple/hashing.py", 56, 68), ("2.1", R+"ple/hashing.py", 61, 61), ("2.1", R+"ple/hashing.py", 71, 92),
 ("2.1", R+"ple/hashing.py", 95, 131), ("2.1", R+"ple/hashing.py", 134, 160), ("2.1", R+"ple/hashing.py", 254, 272),
 ("2.1", R+"weights/row_table.py", 936, 954), ("2.1", R+"models/config.py", 663, 664), ("2.1", R+"models/config.py", 1086, 1096),
 ("2.1", R+"ple/state.py", 56, 58), ("2.1", R+"ple/state.py", 136, 150), ("2.1", R+"weights/row_table.py", 361, 378),
 ("2.1", R+"weights/row_table.py", 1048, 1053), ("2.1", R+"ple/runtime.py", 407, 419), ("2.1", R+"ple/hashing.py", 11, 13),
 # §2.2
 ("2.2", R+"weights/row_table.py", 19, 22), ("2.2", R+"weights/row_table.py", 236, 238), ("2.2", R+"weights/row_table.py", 961, 961),
 ("2.2", R+"weights/row_table.py", 1088, 1102), ("2.2", R+"ple/source.py", 49, 52), ("2.2", R+"ple/source.py", 85, 91),
 ("2.2", R+"weights/row_table.py", 796, 909), ("2.2", R+"ple/source.py", 75, 75), ("2.2", R+"ple/source.py", 96, 100),
 ("2.2", R+"weights/row_table.py", 387, 408), ("2.2", R+"ple/source.py", 213, 229), ("2.2", R+"weights/row_table.py", 675, 722),
 ("2.2", R+"weights/row_table.py", 307, 307), ("2.2", R+"ple/source.py", 73, 73), ("2.2", R+"weights/row_table.py", 290, 290),
 ("2.2", R+"scheduler/scheduler.py", 2048, 2077), ("2.2", R+"ple/source.py", 197, 211), ("2.2", R+"ple/source.py", 110, 134),
 ("2.2", R+"ple/source.py", 231, 246), ("2.2", R+"ple/source.py", 169, 187), ("2.2", R+"ple/source.py", 248, 252),
 # §2.3
 ("2.3", R+"scheduler/scheduler.py", 2922, 3003), ("2.3", R+"ple/runtime.py", 128, 177), ("2.3", R+"scheduler/scheduler.py", 1344, 1350),
 ("2.3", R+"models/qwen4exp.py", 420, 442), ("2.3", R+"models/qwen4exp.py", 542, 543), ("2.3", R+"models/qwen4exp.py", 14, 14),
 ("2.3", R+"models/qwen4exp.py", 521, 527), ("2.3", R+"models/qwen4exp.py", 373, 418),
 # §2.4
 ("2.4", R+"ple/state.py", 50, 52), ("2.4", R+"ple/runtime.py", 220, 260), ("2.4", R+"scheduler/scheduler.py", 3154, 3155),
 ("2.4", R+"scheduler/scheduler.py", 5005, 5013), ("2.4", R+"models/qwen4exp.py", 409, 410), ("2.4", R+"ple/runtime.py", 262, 299),
 ("2.4", R+"ple/state.py", 106, 131), ("2.4", R+"models/qwen4exp.py", 784, 789), ("2.4", R+"spec/mtp.py", 41, 45),
 ("2.4", R+"spec/mtp.py", 143, 147), ("2.4", R+"spec/mtp.py", 196, 212), ("2.4", R+"spec/proposer.py", 88, 100),
 # §2.5
 ("2.5", R+"ple/graph_capture.py", 60, 101), ("2.5", R+"ple/graph_capture.py", 105, 131), ("2.5", R+"ple/graph_capture.py", 132, 147),
 ("2.5", R+"scheduler/scheduler.py", 1260, 1260), ("2.5", R+"scheduler/scheduler.py", 3629, 3629), ("2.5", R+"scheduler/scheduler.py", 3789, 3789),
 ("2.5", R+"engine/graph.py", 383, 389),
 # §3
 ("3.1", R+"ple/source.py", 245, 245), ("3.2", R+"ple/source.py", 102, 102), ("3.2", R+"ple/source.py", 227, 229),
 ("3.2", R+"ple/source.py", 243, 243), ("3.2", R+"ple/runtime.py", 420, 428), ("3.2", R+"core.py", 138, 138),
 ("3.2", R+"scheduler/scheduler.py", 1596, 1613), ("3.2", R+"scheduler/scheduler.py", 2861, 2888), ("3.2", R+"scheduler/io.py", 119, 151),
 ("3.2", R+"server/api_server.py", 3528, 3537), ("3.2", R+"cam/runtime.py", 206, 220), ("3.2", R+"tokenizer/server.py", 39, 47),
 ("3.4", R+"engine/config.py", 89, 117), ("3.4", R+"scheduler/scheduler.py", 430, 441), ("3.4", R+"kvcache/composite_state.py", 24, 71),
 ("3.4", R+"ple/state.py", 68, 85), ("3.4", R+"kvcache/radix_cache.py", 199, 219), ("3.4", R+"kvcache/radix_cache.py", 299, 312),
 ("3.4", R+"scheduler/cache.py", 116, 120), ("3.4", R+"scheduler/scheduler.py", 1768, 1800), ("3.4", R+"kvcache/radix_cache.py", 221, 236),
 ("3.4", R+"kvcache/radix_cache.py", 362, 363), ("3.4", R+"kvcache/radix_cache.py", 314, 360), ("3.4", R+"kvcache/radix_cache.py", 351, 351),
 ("3.4", R+"scheduler/scheduler.py", 1738, 1766), ("3.4", R+"kvcache/radix_cache.py", 175, 197), ("3.4", R+"kvcache/radix_cache.py", 178, 190),
 ("3.4", R+"kvcache/radix_cache.py", 352, 352), ("3.4", R+"scheduler/cache.py", 110, 113), ("3.5", R+"ple/graph_capture.py", 132, 147),
 # §5
 ("5", R+"ple/source.py", 243, 245), ("5", R+"core.py", 29, 34), ("5", R+"server/api_server.py", 375, 380),
 ("5", R+"server/api_server.py", 1485, 1489), ("5", R+"weights/cpu_tier.py", 38, 40),
]
_h = {}
def hunks(p):
    if p not in _h:
        out = subprocess.run(["git", "-C", REPO, "diff", "-U0", OLD, NEW, "--", p], capture_output=True, text=True).stdout
        _h[p] = [(int(m[1]), int(m[2] or 1), int(m[3]), int(m[4] or 1))
                 for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", out, re.M)]
    return _h[p]
def map_line(p, L):
    off = 0
    for a, b, c, d in hunks(p):
        if b == 0:
            if a < L: off += d
            continue
        if a <= L < a + b: return None
        if a + b <= L: off += d - b
    return L + off
def show(rev, p, lo, hi):
    txt = subprocess.run(["git", "-C", REPO, "show", f"{rev}:{p}"], capture_output=True, text=True).stdout.splitlines()
    return txt[lo-1:hi]
if __name__ == "__main__":
    verbose = "-v" in sys.argv
    for sec, p, lo, hi in CLAIMS:
        m = [map_line(p, l) for l in range(lo, hi + 1)]
        ch = sum(x is None for x in m); ok = [x for x in m if x is not None]
        nlo, nhi = (min(ok), max(ok)) if ok else (None, None)
        st = "SAME" if (ch == 0 and nlo == lo) else ("MOVED" if ch == 0 else ("PARTLY-CHANGED" if ok else "GONE/REWRITTEN"))
        short = p.replace(R, "R/")
        print(f"§{sec:4s} {short}:{lo}-{hi} -> {nlo}-{nhi}  touched={ch}/{hi-lo+1}  {st}")
        if verbose:
            o = show(OLD, p, lo, min(hi, lo + 2)); n = show(NEW, p, nlo or lo, min(nhi or hi, (nlo or lo) + 2))
            for x in o: print("   old|", x[:110])
            for x in n: print("   new|", x[:110])

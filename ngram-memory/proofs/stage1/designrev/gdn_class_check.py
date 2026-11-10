"""Does minisgl.gdn.hf_patch.patch_qwen3_5_gdn (matches type name == 'Qwen3_5GatedDeltaNet') cover the
qwen3_5_moe host? Compare the HF GDN classes' source, modulo the class-name prefix."""
import inspect, re, sys
from transformers.models.qwen3_5 import modeling_qwen3_5 as a
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as b
sa = inspect.getsource(a.Qwen3_5GatedDeltaNet); sb = inspect.getsource(b.Qwen3_5MoeGatedDeltaNet)
norm = lambda s: re.sub(r"Qwen3_5Moe", "Qwen3_5", s)
print("moe class name:", b.Qwen3_5MoeGatedDeltaNet.__name__)
print("source identical modulo 'Moe' prefix:", norm(sa) == norm(sb))
patch_src = open("/home/user/patcarter883/minisglang-rdna4/python/minisgl/gdn/hf_patch.py").read()
print("hf_patch matches by exact name 'Qwen3_5GatedDeltaNet':", 'type(child).__name__ == "Qwen3_5GatedDeltaNet"' in patch_src)
print("=> patch_qwen3_5_gdn on a qwen3_5_moe model would replace 0 layers unless the name test is widened")
from transformers.models.qwen4_exp import modeling_qwen4_exp as c
names = [n for n in dir(c) if n.endswith("GatedDeltaNet")]
print("qwen4_exp GDN classes:", names)
for n in names:
    print(n, "identical to Qwen3_5GatedDeltaNet modulo prefix:", re.sub(n.replace("GatedDeltaNet",""), "Qwen3_5", inspect.getsource(getattr(c, n))) == sa)

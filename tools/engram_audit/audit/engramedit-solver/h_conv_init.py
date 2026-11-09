"""Check what nn.init.dirac_ does to the depthwise short_conv and the effective lag under Gemini's causal padding."""
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(2)
d, K, T = 6, 4, 6
for groups_arg in [1, d]:
    conv = nn.Conv1d(d, d, K, groups=d, bias=False)
    nn.init.dirac_(conv.weight, groups=groups_arg)
    x = torch.zeros(1, T, d); x[0, 2] = 1.0           # impulse at t=2 on every channel
    y = conv(F.pad(x.transpose(1, 2), (K - 1, 0))).transpose(1, 2)   # exactly Gemini's forward_fusion conv path
    print(f"dirac_(groups={groups_arg}): nonzero channels={int((conv.weight.abs().sum((1,2))>0).sum())}/{d}; "
          f"impulse at t=2 -> output nonzero at t={sorted(set(torch.nonzero(y[0])[:,0].tolist()))}, channels={sorted(set(torch.nonzero(y[0])[:,1].tolist()))}")

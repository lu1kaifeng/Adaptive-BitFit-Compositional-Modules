import math

import torch
import torch.nn as nn
import torch.nn.functional as F

s=100.0
m=0.05

def get_logits(fc,x):
    W = F.normalize(fc.weight, dim=1)

    x = F.normalize(x, dim=-1)

    wf = F.linear(x, W)
    return wf
def loss(logits, labels):
    wf = logits
    numerator = s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m)
    excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
    denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
    L = numerator - torch.log(denominator)
    return -torch.mean(L)
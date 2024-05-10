import math

import torch
import torch.nn as nn
import torch.nn.functional as F

s = 100
m =0.05
easy_margin= False
cos_m = math.cos(m)
sin_m = math.sin(m)
th = math.cos(math.pi - m)
mm = math.sin(math.pi - m) * m

def get_logits(fc,x):
    cosine = F.linear(F.normalize(x), F.normalize(fc.weight))
    return cosine
def loss(cosine, label):
    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-9, 1))
    phi = cosine * cos_m - sine * sin_m
    if easy_margin:
        phi = torch.where(cosine > 0, phi, cosine)
    else:
        phi = torch.where(cosine > th, phi, cosine - mm)
    # --------------------------- convert label to one-hot ---------------------------
    # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
    one_hot = torch.zeros(cosine.size(), device='cuda')
    one_hot.scatter_(1, label.view(-1, 1).long(), 1)
    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
    output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
    output *= s
    # print(output)

    return F.cross_entropy(output,label)
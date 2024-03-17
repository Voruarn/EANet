import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

from network.EANet import EANet


if __name__ == "__main__":
    print('Test EANet !')
    model=EANet()
    model.eval()
    batch_size=1
    input=torch.rand(batch_size,3,512,512)
    flops, params = profile(model, inputs=(input, ))

    GFLOPs=10**9
    Million=10**6
    print('FLOPs:{:.2f}G'.format((flops/GFLOPs)/batch_size))

    print('params:{:.2f}M'.format(params/Million))

 



"""
EANet
FLOPs:30.10G
params:25.24M
"""




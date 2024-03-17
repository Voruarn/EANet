import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from network.EANet import EANet

if __name__ == "__main__":
    print('Hello')

    input=torch.rand(2,3,256,256).cuda()
    model = EANet().cuda()
    model.eval()
    output=model(input)
    for x in output:
        print('x.shape:',x.shape)












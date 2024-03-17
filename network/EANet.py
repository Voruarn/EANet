import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import *
from .init_weights import init_weights
from .ResNet import resnet18, resnet34, resnet50
from .poolformer_cus import PoolFormer, DecBlk


class EANet(nn.Module):
    # EANet
    # backbone:  resnet50,
    def __init__(self, backbone='resnet50', mid_ch=64, **kwargs):
        super(EANet, self).__init__()      

        self.encoder=None
        enc_dims=[64, 128, 320, 512]
        if backbone=='resnet50':
            enc_dims=[256, 512, 1024, 2048]  
            self.encoder=eval(backbone)()
       
        out_ch=1
        # Encoder
        self.eside1=Conv(enc_dims[0], mid_ch)
        self.eside2=Conv(enc_dims[1], mid_ch)
        self.eside3=Conv(enc_dims[2], mid_ch)
        self.eside4=Conv(enc_dims[3], mid_ch)

        # EDFAM
        self.edfam1=EDFAM(mid_ch,mid_ch)
        self.edfam2=EDFAM(mid_ch,mid_ch)
        self.edfam3=EDFAM(mid_ch,mid_ch)
        self.edfam4=EDFAM(mid_ch,mid_ch)

        # ASSAM
        self.assam1=ASSAM(mid_ch)
        self.assam2=ASSAM(mid_ch)
        self.assam3=ASSAM(mid_ch)
        self.assam4=ASSAM(mid_ch)

        # Decoder
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dec1=DecBlk(mid_ch)
        self.dec2=DecBlk(mid_ch)
        self.dec3=DecBlk(mid_ch)
        self.dec4=DecBlk(mid_ch)

        self.dside1 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside2 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside3 = nn.Conv2d(mid_ch,out_ch,3,padding=1)
        self.dside4 = nn.Conv2d(mid_ch,out_ch,3,padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        # encoder
        outs = self.encoder(inputs)
        c1, c2, c3, c4 = outs
    
        c1=self.eside1(c1)
        c2=self.eside2(c2)
        c3=self.eside3(c3)
        c4=self.eside4(c4)

        ce1=self.edfam1(c1)
        ce2=self.edfam2(c2)
        ce3=self.edfam3(c3)
        ce4=self.edfam4(c4)

        ca1=self.assam1(in1=ce1, in2=ce2)
        ca2=self.assam2(in1=ce2, in2=ce1, in3=ce3)
        ca3=self.assam3(in1=ce3, in2=ce2, in3=ce4)
        ca4=self.assam4(in1=ce4, in2=ce3)

        # Decoder
        up4=self.dec4(ca4)

        up3=self.upsample2(up4) + ca3
        up3=self.dec3(up3)

        up2=self.upsample2(up3) + ca2
        up2=self.dec2(up2)

        up1=self.upsample2(up2) + ca1
        up1=self.dec1(up1)


        d1=self.dside1(up1)
        d2=self.dside2(up2)
        d3=self.dside3(up3)
        d4=self.dside4(up4)
      
        S1 = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=True)
        S2 = F.interpolate(d2, size=(H, W), mode='bilinear', align_corners=True)
        S3 = F.interpolate(d3, size=(H, W), mode='bilinear', align_corners=True)
        S4 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=True)

        return S1,S2,S3,S4, torch.sigmoid(S1),torch.sigmoid(S2),torch.sigmoid(S3),torch.sigmoid(S4)





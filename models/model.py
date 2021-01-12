# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:22:57 2021

@author: ajs3
"""

import torch
import torch.nn as nn
import numpy as np



class Conv(nn.Module):
    def __init__(self,c1,c2,k=1,s=1,act = True):
        super(Conv,self).__init__()
        self.conv = nn.Conv2d(c1, c2, k,s,k//2,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.active = nn.Hardswish() if act else nn.Identity()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.active(x)
        return x
    
class Focus(nn.Module):
    def __init__(self,c1,c2):
        super(Focus,self).__init__()
        self.c1 = c1
        self.c2 = c2
        self.conv = Conv(c1*4,c2)
        
    def forward(self,x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x
        
class BottleneckCSP(nn.Module):
    def __init__(self,c1,shotCut = True):
        super(BottleneckCSP,self).__init__()
        
        self.conv1 = Conv(c1,c1//2)
        self.shot1 = Conv(c1//2,c1//2)
        self.shot2 = Conv(c1//2,c1//2)
        self.conv2 = nn.Conv2d(c1//2, c1//2, 1,1,bias=False)
        self.conv3 = nn.Conv2d(c1, c1//2, 1,1,bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.conv4 = Conv(c1,c1)
        self.shotCut = shotCut
        
    def forward(self,x):
        CSP1 = self.conv1(x)
        
        CSP1 = CSP1 + self.shot2(self.shot1(CSP1)) if self.shotCut else self.shot2(self.shot1(CSP1))
        CSP1 = self.conv2(CSP1)
        CSP2 = self.conv3(x)
        out = torch.cat([CSP1,CSP2],1)
        out = self.conv4(self.act(self.bn(out)))
        return out
        
        
            
        
class SPP(nn.Module):
    def __init__(self,c1,k=[5,9,13]):
        super(SPP,self).__init__()
        self.conv1 = Conv(c1,c1//2)
        self.spp1 = nn.MaxPool2d(k[0],stride=1,padding=k[0]//2)
        self.spp2 = nn.MaxPool2d(k[1],stride=1,padding=k[1]//2)
        self.spp3 = nn.MaxPool2d(k[2],stride=1,padding=k[2]//2)
        self.conv2 = Conv(c1*2,c1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = torch.cat([x,self.spp1(x),self.spp2(x),self.spp3(x)],1)
        x = self.conv2(x)
        return x
        
        

class yoloV5s(nn.Module):
    def __init__(self):
        super(yoloV5s,self).__init__()
        self.backbone = nn.Sequential(
                        Focus(3,32),
                        Conv(32,64,3,2),
                        BottleneckCSP(64),
                        Conv(64,128,3,2),
                        BottleneckCSP(128),
                        BottleneckCSP(128),
                        BottleneckCSP(128),
                        Conv(128,256,3,2),
                        BottleneckCSP(256),
                        BottleneckCSP(256),
                        BottleneckCSP(256),
                        Conv(256,512,3,2),
                        SPP(512),
                        BottleneckCSP(512,False)            
            
                    )
    def forward(self,x):
        out = self.backbone(x)
        return out
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 640, 640)
    model = yoloV5s()
    print(model(x).shape)

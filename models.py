import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_models import *

class BiCModule(nn.Module):
    def __init__(self, in_feat):
        super(BiCModule, self).__init__()

        self.down = Downsample(in_feat,in_feat)
        self.up = TransposedUpsample(in_feat, in_feat)

        self.conv1 = nn.Conv2d(in_feat,in_feat, 1)
        self.conv2 = nn.Conv2d(in_feat,in_feat, 1)
        self.conv3 = nn.Conv2d(in_feat,in_feat, 1)

    def forward(self,x,xprev,p):

        x = self.conv1(x)

        p = self.up(p)

        xprev = self.conv2(xprev)
        xprev = self.down(xprev)

        out = torch.cat([x,p,xprev],dim=1)

        return out

class SimCSPSPPF(nn.Module):
    def __init__(self):
        super(SimCSPSPPF,self).__init__()

    def forward(self,x):

        return out

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()

    def forward(self,x):

        return out


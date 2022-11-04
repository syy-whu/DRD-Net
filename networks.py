#PyTorch lib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
 
class rain_residual_block(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64):
        super(rain_residual_block, self).__init__()
        self.rain_residual_block_head = nn.Sequential(
            nn.Conv2d(64, feature_dim, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.PReLU(feature_dim),
            nn.Conv2d(64, feature_dim, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(feature_dim),
            # nn.functional.adaptive_avg_pool2d((2,2))
        )
        self.rain_residual_block_body = nn.Sequential(
            nn.Linear(64, 4, False),
            nn.ReLU(),
            nn.Linear(4, 64, False),
            nn.Sigmoid()

        )

    def forward(self,input):
        x1 = self.rain_residual_block_head(input)
        se = F.adaptive_avg_pool2d(x1, (1, 1))
        se = self.rain_residual_block_body(se.view(se.size(0), se.size(1) * se.size(2) * se.size(3)))
        se = se.unsqueeze(2).unsqueeze(3)
        x1 = torch.mul(x1, se)
        out = torch.add(x1, input)
        return out

class RainNetwork(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64,n_blocks=16):
        super(RainNetwork, self).__init__()
        self.rain_net_head=nn.Sequential(
            nn.Conv2d(3, feature_dim, (3, 3), stride=(1,1), padding=1),
            nn.PReLU(feature_dim)
        )
        rain_residual_blocks = []
        for _ in range(n_blocks):
            rain_residual_blocks += [rain_residual_block()]
        self.rain_residual_blocks = nn.Sequential(*rain_residual_blocks)

        self.rain_net_body = nn.Sequential(
            nn.Conv2d(64, feature_dim, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(feature_dim),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(64, 3, (3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, input):
        x=self.rain_net_head(input)
        x0=x
        x=self.rain_residual_blocks(x)
        x=self.rain_net_body(x)
        x=torch.add(x,x0)
        x=self.tail(x)
        return  x

class empty_block(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64):
        super(empty_block, self).__init__()
        self.conv1= nn.Sequential(nn.Conv2d(64, feature_dim, (3, 3), stride=(1, 1), padding=1))
        self.conv2 = nn.Sequential(nn.Conv2d(64, feature_dim, (3, 3), stride=(1, 1), padding=3,dilation=3))
        self.conv3 = nn.Sequential(nn.Conv2d(64, feature_dim, (3, 3), stride=(1, 1), padding=5,dilation=5))
        self.conv4 = nn.Sequential(nn.Conv2d(192, feature_dim, (1, 1), stride=(1, 1), padding=0))


    def forward(self, input):
        x1=self.conv1(input)
        x2=self.conv2(input)
        x3=self.conv3(input)
        x=torch.cat((x1,x2,x3),1)
        # temp=torch.cat((x1,x2),3)
        # x=torch.cat((temp,x3),3)
        out=self.conv4(x)
        return out
#####
class back_residual_block(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64):
        super(back_residual_block, self).__init__()
        self.back_residual_block_body = nn.Sequential(
            empty_block(),
            nn.BatchNorm2d(feature_dim),
            nn.PReLU(feature_dim),
            empty_block(),
            nn.BatchNorm2d(feature_dim)
            # nn.functional.adaptive_avg_pool2d((2,2))
        )

    def forward(self,input):
        x = self.back_residual_block_body(input)
        m = torch.add(x,input)
        return m




class BackNetwork(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64,n_blocks=16):
        super(BackNetwork, self).__init__()
        self.back_net_head=nn.Sequential(
            nn.Conv2d(3, feature_dim, (3, 3), stride=(1,1), padding=1),
            nn.PReLU(feature_dim)
        )

        back_residual_blocks = []
        for _ in range(n_blocks):
            back_residual_blocks += [back_residual_block()]
        self.back_residual_blocks = nn.Sequential(*back_residual_blocks)

        self.back_net_body = nn.Sequential(
            nn.Conv2d(64, feature_dim, (3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(feature_dim),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(64, 3, (3, 3), stride=(1, 1), padding=1)
        )

    def forward(self, input):
        x = self.back_net_head(input)
        x0 = x
        x = self.back_residual_blocks(x)
        x = self.back_net_body(x)
        x = torch.add(x, x0)
        x = self.tail(x)
        return x

class DRDNetwork(nn.Module):
    def __init__(self, input_channel_num=3, feature_dim=64,is_train=True):
        super(DRDNetwork, self).__init__()
        self.rain_net=nn.Sequential(RainNetwork())
        self.back_net=nn.Sequential(BackNetwork())
        self.is_train=is_train
        # self.DRNMnetwork = detailnonlocal(3,128,128)
    def forward(self, input):
        Rain=self.rain_net(input)
        out1=torch.sub(input,Rain)
        # back_in=torch.add(out1,input)
        # back=self.back_net(back_in)
        back = self.back_net(input)
        out=torch.add(out1,back)
        # out = self.DRNMnetwork(out1,back)
        if self.is_train:
            return out1,out
        else:
            return out1,out,back





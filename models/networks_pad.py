import torch
from torch import nn
import torch.nn.functional as F

class semantic_feature_decoder(nn.Module):
    def __init__(self, feature_out_dim, feature_out_H, feature_out_W):
        super().__init__()
        self.feature_out_H = feature_out_H
        self.feature_out_W = feature_out_W
        
        #self.conv1x1 = nn.Conv2d(512, feature_out_dim, kernel_size=1).cuda()
        #self.conv1x1 = nn.Conv2d(32, feature_out_dim, kernel_size=1).cuda()
        #self.conv1x1 = nn.Conv2d(8, feature_out_dim, kernel_size=1).cuda()

        # self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1).cuda()
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).cuda()
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1).cuda()
        # self.conv4 = nn.Conv2d(256, feature_out_dim, kernel_size=1).cuda()

        #self.conv1 = nn.Conv2d(32, 512, kernel_size=3, padding=1).cuda()
        #self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1).cuda()
        #self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1).cuda()

        #self.In1 = nn.InstanceNorm2d(64)  
        #self.In2 = nn.InstanceNorm2d(128)
        #self.In3 = nn.InstanceNorm2d(256)
        
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((feature_out_H, feature_out_W))
        self.conv1 = nn.Conv2d(16, feature_out_dim, kernel_size=1).cuda()

    def forward(self, x):
        # x = self.In1(F.relu(self.conv1(x)))
        # x = self.In2(F.relu(self.conv2(x)))
        # x = self.In3(F.relu(self.conv3(x)))
        # x = self.conv4(x)
        #print(self.conv1x1.weight)
        #x = F.elu(self.conv1(x))
      
        if self.feature_out_H == self.feature_out_W:
            #padh = max(x.shape[1], x.shape[2]) - x.shape[1] # 480 - 360 = 120
            #padw = max(x.shape[1], x.shape[2]) - x.shape[2] # 480 - 480 = 0
            scale = self.feature_out_H * 1.0 / max(x.shape[1], x.shape[2])
            newh, neww = x.shape[1] * scale, x.shape[2] * scale
            neww = int(neww + 0.5)
            newh = int(newh + 0.5)
            x = F.interpolate(x.unsqueeze(0), size=(newh, neww), mode='bilinear', align_corners=True).squeeze(0)
            padh = x.shape[2] - x.shape[1]
            padw = x.shape[2] - x.shape[2]
            x = F.pad(x,(0, padw, 0, padh))
            #x = self.adaptive_pool(x)
            x = self.conv1(x)
        else:
            #x = self.adaptive_pool(x)
            x = self.conv1(x)
        
        #x = self.adaptive_pool(x)
        #x = self.conv1(x)

        return x

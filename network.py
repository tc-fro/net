import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from net.PVT_V2 import pvt_v2_b1, pvt_v2_b3, pvt_v2_b5
from timm.models.vision_transformer import Block
from nets.lgffm import LGFFM1, LGFFM
from nets.efm import EFM
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class ConvBNR1(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1, dilation=1, bias=False):
        super(ConvBNR1, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation*(kernel_size//2), dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class MSFM(nn.Module):
    def __init__(self, num_classes=1):
        super(MSFM, self).__init__()

        # 用1x1卷积减少维度
        self.conv1x1_p4 = nn.Conv2d(512, 256, kernel_size=1)

        # 用于多尺度融合的卷积
        self.conv_p3 = nn.Conv2d(320+256, 256, kernel_size=3, padding=1)
        self.conv_p2 = nn.Conv2d(256+128, 256, kernel_size=3, padding=1)
        self.conv_p1 = nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1)


        self.block = nn.Sequential(
            ConvBNR(256 , 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))


    def forward(self, p4, p3, p2, p1):
        # 位置编码

        p4 = F.relu(self.conv1x1_p4(p4))

        # 多尺度融合
        heatmap_p4_up_p3 = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_fused = torch.cat([p3, heatmap_p4_up_p3], dim=1)
        p3_optimized = F.relu(self.conv_p3(p3_fused))


        heatmap_p3_up_p2 = F.interpolate(p3_optimized, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p2_fused = torch.cat([p2, heatmap_p3_up_p2], dim=1)
        p2_optimized = F.relu(self.conv_p2(p2_fused))


        heatmap_p2_up_p1 = F.interpolate(p2_optimized, size=p1.shape[2:], mode='bilinear', align_corners=False)
        p1_fused = torch.cat([p1, heatmap_p2_up_p1], dim=1)
        p1_optimized = F.relu(self.conv_p1(p1_fused))


        # 最终目标分割
        segmentation_output = self.block(p1_optimized)

        return segmentation_output, p4, p3_optimized, p2_optimized, p1_optimized



class Net(nn.Module):
    def __init__(self, fun_str = 'pvt_v2_b3'):
        super(Net, self).__init__()

        self.backbone, embedding_dims = eval(fun_str)()

        self.msfm = MSFM()

        self.efm1 = EFM(64)
        self.efm2 = EFM(128)
        self.efm3 = EFM(320)
        self.efm4 = EFM(512)

        self.reduce1 = Conv1x1(64, 64)
        self.reduce2 = Conv1x1(128, 128)
        self.reduce3 = Conv1x1(320, 256)
        self.reduce4 = Conv1x1(512, 256)

        self.lgffm1 = LGFFM(128, 64)
        self.lgffm2 = LGFFM(256, 128)
        self.lgffm3 = LGFFM(256, 256)
        self.lgffm4 = LGFFM1(256, 256)

        self.predictor1 = nn.Conv2d(64, 1, 1)
        self.predictor2 = nn.Conv2d(128, 1, 1)
        self.predictor3 = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        edge, p4, p3, p2, p1= self.msfm(x4, x3, x2, x1)
        edge_att = torch.sigmoid(edge)

        x1a = self.efm1(x1, edge_att)
        x2a = self.efm2(x2, edge_att)
        x3a = self.efm3(x3, edge_att)
        x4a = self.efm4(x4, edge_att)

        x1r = self.reduce1(x1a)
        x2r = self.reduce2(x2a)
        x3r = self.reduce3(x3a)
        x4r = self.reduce4(x4a)

        x4f = self.lgffm4(x4r, p4)
        x34 = self.lgffm3(x3r, x4f, p3)
        x234 = self.lgffm2(x2r, x34, p2)
        x1234 = self.lgffm1(x1r, x234, p1)

        o3 = self.predictor3(x34)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(x234)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False)
        o1 = self.predictor1(x1234)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return o3, o2, o1, oe

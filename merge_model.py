import torch
import torch.nn as nn
import torch.nn.functional as F

from module.CoordinateAtten import CoordAttention
from module.cbam import CBAM

class DoubleConv(nn.Module):
    """双卷积块（Conv+BN+ReLU）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)
        self.coord_att = CoordAttention(out_channels)

        # 注意力权重融合（可学习）
        self.alpha = nn.Parameter(torch.ones(1))  # CBAM权重
        self.beta = nn.Parameter(torch.ones(1))   # CoordAttention权重

    def forward(self, x):
        x = self.double_conv(x)
        x_cbam = self.cbam(x)
        x_coord = self.coord_att(x)
        x = (self.alpha * x_cbam + self.beta * x_coord) / (self.alpha + self.beta)
        return x

class Down(nn.Module):
    """下采样块（MaxPool+DoubleConv）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样块（Upsample+Concat+DoubleConv）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """输出卷积"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class UNetMerge(nn.Module):
    """轻量化UNet用于低光照增强"""
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetMerge, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(768, 256)  # 512+256=768
        self.up2 = Up(384, 128)   # 256+128=384
        self.up3 = Up(192, 64)    # 128+64=192
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)  # 输出归一化到[0,1]
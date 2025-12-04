import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 沿宽度池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 沿高度池化
        
        mid_channels = in_channels // reduction
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.fc_h = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.fc_w = nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        # 沿高度/宽度提取坐标特征
        x_h = self.pool_h(x)  # (b,c,h,1)
        x_w = self.pool_w(x).permute(0,1,3,2)  # (b,c,w,1)
        
        # 融合坐标特征
        x_cat = torch.cat([x_h, x_w], dim=2)  # (b,c,h+w,1)
        x_cat = self.fc(x_cat)
        
        # 拆分并还原尺寸
        x_h, x_w = torch.split(x_cat, [h, w], dim=2)
        x_w = x_w.permute(0,1,3,2)
        
        # 生成坐标注意力权重
        att_h = self.sigmoid(self.fc_h(x_h))
        att_w = self.sigmoid(self.fc_w(x_w))
        
        # 加权增强
        out = x * att_h * att_w
        return out
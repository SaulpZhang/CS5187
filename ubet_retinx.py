import torch
import torch.nn as nn
import torch.nn.functional as F

class RetinexBranch(nn.Module):
    """Retinex分解分支：输出亮度层和反射层"""
    def __init__(self, in_channels=3):
        super().__init__()
        # 亮度层提取（输出1通道，代表全局亮度）
        self.illumination = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # 亮度层归一化到[0,1]
        )
        # 反射层提取（输出3通道，保留细节）
        self.reflectance = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()  # 反射层归一化到[0,1]
        )

    def forward(self, x):
        illu = self.illumination(x)    # 亮度层 (B,1,H,W)
        refl = self.reflectance(x)     # 反射层 (B,3,H,W)
        # Retinex重构：图像=反射层×亮度层（模拟物理规律）
        recon = refl * illu.repeat(1,3,1,1)  # 亮度层扩展到3通道
        return illu, refl, recon

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

    def forward(self, x):
        return self.double_conv(x)

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

class RetinexUNet(nn.Module):
    """融合Retinex分支的UNet"""
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 1. 新增Retinex分解分支
        self.retinex_branch = RetinexBranch(n_channels)
        
        # 2. UNet主分支（复用之前的模块，可结合CBAM）
        self.inc = DoubleConv(n_channels + 3, 64)  # 输入：原图 + Retinex重构图（共6通道）
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(768, 256)
        self.up2 = Up(384, 128)
        self.up3 = Up(192, 64)
        
        # 3. 融合Retinex亮度层的输出层
        self.outc = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )
        # 亮度调整层（学习最优的亮度增益）
        self.illu_adjust = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step1: Retinex分解
        illu, refl, recon = self.retinex_branch(x)
        
        # Step2: 调整亮度层（提升低光照区域亮度）
        illu_enhanced = self.illu_adjust(illu) + 0.5  # 增益偏移，避免过暗
        
        # Step3: UNet主分支输入：原图 + Retinex重构图（融合物理先验）
        unet_input = torch.cat([x, recon], dim=1)  # (B,6,H,W)
        x1 = self.inc(unet_input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # Step4: 融合亮度层输出最终增强图
        unet_out = self.outc(x)
        final_out = unet_out * illu_enhanced.repeat(1,3,1,1)  # 结合调整后的亮度层
        
        return torch.sigmoid(final_out)
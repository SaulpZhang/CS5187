import torch.optim as optim
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch

import torch.nn as nn
import torch.nn.functional as F
from mymodel import UNet
from dataload import load_data

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

train_loader, test_loader = load_data()

# 损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失（适合像素级回归）
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for low_imgs, high_imgs in tqdm(loader, desc='Training'):
        low_imgs = low_imgs.to(device)
        high_imgs = high_imgs.to(device)
        
        # 前向传播
        outputs = model(low_imgs)
        loss = criterion(outputs, high_imgs)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# 测试函数（计算PSNR/SSIM）
def test_epoch(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for low_imgs, high_imgs in tqdm(loader, desc='Testing'):
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)
            
            # 推理
            outputs = model(low_imgs)
            
            # 转为numpy（0-1 → 0-255）
            outputs_np = outputs.cpu().squeeze().permute(1,2,0).numpy() * 255
            high_np = high_imgs.cpu().squeeze().permute(1,2,0).numpy() * 255
            
            # 计算指标
            total_psnr += psnr(high_np, outputs_np, data_range=255)
            total_ssim += ssim(high_np, outputs_np, channel_axis=2, data_range=255)
    
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    return avg_psnr, avg_ssim

# 开始训练
num_epochs = 100
best_psnr = 0.0

for epoch in range(num_epochs):
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    test_psnr, test_ssim = test_epoch(model, test_loader, device)
    
    # 学习率衰减
    scheduler.step()
    
    # 保存最优模型
    if test_psnr > best_psnr:
        best_psnr = test_psnr
        torch.save(model.state_dict(), 'unet_lol_best.pth')
    
    print(f'Train Loss: {train_loss:.4f}, Test PSNR: {test_psnr:.2f}, Test SSIM: {test_ssim:.4f}')
    print(f'Best PSNR: {best_psnr:.2f}')
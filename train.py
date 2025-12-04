import torch.optim as optim
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch

import torch.nn as nn
import torch.nn.functional as F
from mymodel import UNet
from unet_cbam import UNetCBAM
from ubet_retinx import RetinexUNet
from dataload import load_data
import argparse
import os
import json
import random

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
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

        # # 转为numpy（0-1 → 0-255）
        # for i in range(outputs.size(0)):
        #     outputs_np = outputs[i].cpu().permute(1,2,0).numpy() * 255
        #     high_np = high_imgs[i].cpu().permute(1,2,0).numpy() * 255
        #     # 计算指标
        #     total_psnr += psnr(high_np, outputs_np, data_range=255)
        #     total_ssim += ssim(high_np, outputs_np, channel_axis=2, data_range=255)
    
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    avg_loss = total_loss / len(loader)
    return avg_loss, avg_psnr, avg_ssim

# 测试函数（计算PSNR/SSIM）
def test_epoch(model, loader,criterion, device):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for low_imgs, high_imgs in tqdm(loader, desc='Testing'):
            low_imgs = low_imgs.to(device)
            high_imgs = high_imgs.to(device)
            
            # 推理
            outputs = model(low_imgs)
            loss = criterion(outputs, high_imgs)
            total_loss += loss.item()

            # 转为numpy（0-1 → 0-255）
            outputs_np = outputs.cpu().squeeze().permute(1,2,0).numpy() * 255
            high_np = high_imgs.cpu().squeeze().permute(1,2,0).numpy() * 255
            
            # 计算指标
            total_psnr += psnr(high_np, outputs_np, data_range=255)
            total_ssim += ssim(high_np, outputs_np, channel_axis=2, data_range=255)
    
    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    avg_loss = total_loss / len(loader)
    return avg_loss, avg_psnr, avg_ssim

def get_model(args):
    if args.model == 'unet':
        return UNet()
    elif args.model == 'unet_cbam':
        return UNetCBAM()
    elif args.model == 'unet_retinx':
        return RetinexUNet()
    else:
        return UNet()

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def main(args):
    os.makedirs('./result/best', exist_ok=True)
    os.makedirs('./result/checkpoints', exist_ok=True)
    # 设备配置
    device = get_device()
    model = get_model(args).to(device)
    if args.c != 0:
        model.load_state_dict(torch.load(f'./result/checkpoints/unet_lol_epoch_{args.c}.pth', map_location=torch.device(device)))

    print(f"device: {device}, model: {model.__class__.__name__}")
    train_loader, test_loader = load_data()

    # 损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失（适合像素级回归）
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 学习率衰减



    # 开始训练
    num_epochs = args.epochs
    best_psnr = 0.0
    train_losses = []
    train_PSNRs = []
    train_SSIMs = []

    test_losses = []
    test_PSNRs = []
    test_SSIMs = []

    if args.c != 0:
        with open('./result/train_stats.json', 'r') as f:
            stats = json.load(f)
            train_losses = stats['train_losses'][:args.c]
            train_PSNRs = stats['train_PSNRs'][:args.c]
            train_SSIMs = stats['train_SSIMs'][:args.c]
            test_losses = stats['test_losses'][:args.c]
            test_PSNRs = stats['test_PSNRs'][:args.c]
            test_SSIMs = stats['test_SSIMs'][:args.c]
            best_psnr = max(test_PSNRs)


    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        train_loss, train_psnr, train_ssim = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_psnr, test_ssim = test_epoch(model, test_loader, criterion, device)
        
        train_losses.append(float(train_loss))
        train_PSNRs.append(float(train_psnr))
        train_SSIMs.append(float(train_ssim))

        test_losses.append(float(test_loss))
        test_PSNRs.append(float(test_psnr))
        test_SSIMs.append(float(test_ssim))

        # 学习率衰减
        scheduler.step()
        
        # 保存最优模型
        if test_psnr > best_psnr:
            best_psnr = test_psnr
            torch.save(model.state_dict(), './result/best/unet_lol_best.pth')
        
        if (epoch + 1) % args.save_step == 0:
            torch.save(model.state_dict(), f'./result/checkpoints/unet_lol_epoch_{epoch+1}.pth')
            with open(f'./result/checkpoints/train_stats_epoch_{epoch+1}.json', 'w') as f:
                params = {
                    'train_losses': train_losses,
                    'train_PSNRs': train_PSNRs,
                    'train_SSIMs': train_SSIMs,
                    'test_losses': test_losses,
                    'test_PSNRs': test_PSNRs,
                    'test_SSIMs': test_SSIMs
                }
                json.dump(params, f)
        
        print(f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test PSNR: {test_psnr:.2f}, Test SSIM: {test_ssim:.4f}')
        print(f'Best PSNR: {best_psnr:.2f}')

def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet on LOL Dataset')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--model', type=str, default='unet', help='model architecture to use')
    parser.add_argument('--save_step', type=int, default=10, help='save model every n epochs')
    parser.add_argument('--c', type=int, default=0, help='continue training from checkpoint')
    return parser.parse_args()

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    args = parse_args()
    main(args)

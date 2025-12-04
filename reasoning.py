from mymodel import UNet
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataload import get_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载最优模型
model = UNet().to(device)
model.load_state_dict(torch.load('./result/unet/result/best/unet_lol_best.pth', map_location=torch.device(device)))
model.eval()
transform = get_transform()
# 单张图像推理
def infer_image(model, img_path, transform, device):
    # 读取图像
    low_img = cv2.imread(img_path)
    low_img_rgb = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
    
    # 预处理
    img_tensor = transform(low_img_rgb).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(img_tensor)
    
    # 后处理
    output_np = output.cpu().squeeze().permute(1,2,0).numpy()
    output_np = (output_np * 255).astype(np.uint8)
    
    return low_img_rgb, output_np

# 测试推理
if __name__ == "__main__":
    test_img_path = './lol_dataset/eval15/low/1.png'
    low_img, enhanced_img = infer_image(model, test_img_path, transform, device)
    
    # 可视化
    plt.figure(figsize=(18,6))
    plt.subplot(131), plt.imshow(low_img), plt.title('Low Light')
    plt.subplot(132), plt.imshow(enhanced_img), plt.title('UNet Enhanced')
    
    # 显示参考图像
    ref_img = cv2.imread('./lol_dataset/eval15/high/1.png')
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    plt.subplot(133), plt.imshow(ref_img), plt.title('Reference')
    plt.show()
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LOLDataset(Dataset):
    """LOL数据集加载类"""
    def __init__(self, root_dir, phase='our485', transform=None):
        self.root_dir = root_dir
        self.phase = phase
        self.low_dir = os.path.join(root_dir, phase, 'low')
        self.high_dir = os.path.join(root_dir, phase, 'high')
        self.image_names = os.listdir(self.low_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 读取图像（RGB格式）
        img_name = self.image_names[idx]
        low_img = cv2.imread(os.path.join(self.low_dir, img_name))
        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.imread(os.path.join(self.high_dir, img_name))
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

        # 数据变换
        if self.transform:
            low_img = self.transform(low_img)
            high_img = self.transform(high_img)
        
        return low_img, high_img



# 加载数据集
def load_data(rootDir = './lol_dataset'):
    # 数据变换
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # 统一尺寸
        transforms.ToTensor(),  # 转为Tensor并归一化到[0,1]
    ])

    train_dataset = LOLDataset(root_dir=rootDir, phase='our485', transform=transform)
    test_dataset = LOLDataset(root_dir=rootDir, phase='eval15', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = load_data()
    for low_img, high_img in train_loader:
        print(f'Low light image batch shape: {low_img.shape}')
        print(f'High light image batch shape: {high_img.shape}')
        break
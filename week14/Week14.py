import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

"""
    Dataset các bạn download tại:
    https://drive.google.com/drive/folders/15wG2QgWU8dKs-NeoLI48TxzFWdydg7Jj?usp=drive_link
    Các bạn hãy tạo class Dataset cho bộ data này mà không dùng ImageFolder nhé
"""
class AnimalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(os.listdir(root))
        self.classes_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            if os.path.isdir(cls_path):
                images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.endswith('.jpg')]
                for img in images:
                    self.samples.append((img, self.classes_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def show_image(dataset, index):
    image, label = dataset[index]
    plt.imshow(image.permute(1, 2, 0))  # Chuyển từ [C, H, W] → [H, W, C]
    plt.title(f"Label: {dataset.classes[label]}")
    plt.axis('off')
    plt.show()

animal_dataset = AnimalDataset(root="./animals", transform=transform)
print(f"Total samples in AnimalDataset: {len(animal_dataset)}")
image, lable = animal_dataset[0]
print(f"Image shape: {image.shape}, Label: {lable}")
show_image(animal_dataset, 10)
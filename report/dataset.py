import os
import cv2
import torch
from torch.utils.data import Dataset

class HCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.images = [f for f in os.listdir(image_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = cv2.resize(img, (self.img_size, self.img_size)) / 255.0
        mask = cv2.resize(mask, (self.img_size, self.img_size)) / 255.0

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)   # (1,H,W)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) # (1,H,W)

        return img, mask

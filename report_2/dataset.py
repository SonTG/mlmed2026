import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import os

IMG_SIZE = 256

class FetalDataset(Dataset):
    def __init__(self, csv_path, img_dir="train_set/images", mask_dir="train_set/masks_filled"):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]

        base_name = os.path.splitext(filename)[0]
        mask_name = f"{base_name}_Annotation.png"

        img_path  = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        mask = mask[None, :, :]

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            row["pixel size(mm)"],
            row["head circumference (mm)"],
            filename
    )

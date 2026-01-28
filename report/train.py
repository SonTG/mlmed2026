import torch
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm
from dataset import HCDataset
from model import AttentionUNet
from loss import BCEDiceLoss

IMAGE_DIR = "training_set/images"
MASK_DIR = "training_set/annotations"
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = HCDataset(IMAGE_DIR, MASK_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = AttentionUNet().to(device)
criterion = BCEDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def dice_coeff(pred, target, smooth=1.0):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1.0):
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

EPOCHS = 20
history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_dice, total_iou = 0, 0, 0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", unit="batch"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)

        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coeff(preds, masks).item()
        total_iou += iou_score(preds, masks).item()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_dice = total_dice / len(train_loader)
    avg_train_iou = total_iou / len(train_loader)

    model.eval()
    val_loss, val_dice, val_iou = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", unit="batch"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)

            loss = criterion(preds, masks)
            val_loss += loss.item()
            val_dice += dice_coeff(preds, masks).item()
            val_iou += iou_score(preds, masks).item()

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}, IoU: {avg_train_iou:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}")

    history.append({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "train_dice": avg_train_dice,
        "train_iou": avg_train_iou,
        "val_loss": avg_val_loss,
        "val_dice": avg_val_dice,
        "val_iou": avg_val_iou
    })


torch.save(model.state_dict(), "attention_unet_hc.pth")
print("Model saved as attention_unet_hc.pth")

df = pd.DataFrame(history)
df.to_csv("training_metrics.csv", index=False)
print("Metrics saved to training_metrics.csv")

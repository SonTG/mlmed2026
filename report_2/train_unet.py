import torch
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.metrics import IoU, Fscore

from dataset import FetalDataset
from tqdm import tqdm
import time

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FetalDataset("train_set/training_set_pixel_size_and_HC.csv")
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

dice = smp.losses.DiceLoss(mode="binary")
bce  = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(pred, target):
    return dice(pred, target) + bce(pred, target)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

iou_metric = IoU(threshold=0.5)
dice_metric = Fscore(threshold=0.5) 

best_val_loss = float("inf")

for epoch in range(20):
    start_time = time.time()
    model.train()
    train_loss = 0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1:02d} [TRAIN]",
        leave=False
    )

    for imgs, masks, _, _, _ in train_bar:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": optimizer.param_groups[0]["lr"]
        })

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    val_iou, val_dice = 0, 0

    val_bar = tqdm(
        val_loader,
        desc=f"Epoch {epoch+1:02d} [VAL]  ",
        leave=False
    )

    with torch.no_grad():
        for imgs, masks, _, _, _ in val_bar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            val_loss += loss.item()
            val_iou  += iou_metric(preds, masks).item()
            val_dice += dice_metric(preds, masks).item()

            val_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

    val_loss /= len(val_loader)
    val_iou  /= len(val_loader)
    val_dice /= len(val_loader)

    scheduler.step(val_loss)

    epoch_time = time.time() - start_time


    print(
        f"Epoch {epoch+1:02d} | "
        f"Time {epoch_time:.1f}s | "
        f"Train Loss {train_loss:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"IoU {val_iou:.4f} | "
        f"Dice {val_dice:.4f} | "
        f"LR {optimizer.param_groups[0]['lr']:.1e}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_unet_hc.pth")
        print("Saved best model")

print("Training finished")

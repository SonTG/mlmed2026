import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')


# pip install pandas matplotlib pillow
# pip install scikit-learn


# Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow version:", tf.__version__)


base_dir = Path("..")
data_dir = base_dir / "data" / "Infection Segmentation Data" / "Infection Segmentation Data"

train_dir = data_dir / "Train"
val_dir = data_dir / "Val"
test_dir = data_dir / "Test"

print("Train dir:", train_dir)
print("Val dir:  ", val_dir)
print("Test dir: ", test_dir)

print("Train classes:", [p.name for p in sorted(train_dir.iterdir())])



def collect_pairs(split_dir: Path) -> pd.DataFrame:
    pairs = []
    for class_dir in sorted(split_dir.iterdir()): 
        images_dir = class_dir / "images"
        masks_dir = class_dir / "infection masks"
        if not images_dir.exists() or not masks_dir.exists():
            continue

        for img_path in images_dir.glob("*.png"):
            mask_path = masks_dir / img_path.name
            if mask_path.exists():
                pairs.append(
                    {
                        "class": class_dir.name,
                        "image": img_path,
                        "mask": mask_path,
                    }
                )
    return pd.DataFrame(pairs)

train_df = collect_pairs(train_dir)
val_df = collect_pairs(val_dir)

print("Train samples:", len(train_df))
print("Val samples:  ", len(val_df))
train_df.head()



N_SHOW = 3

fig, axes = plt.subplots(N_SHOW, 3, figsize=(9, 3 * N_SHOW))

for i in range(N_SHOW):
    row = train_df.sample(1, random_state=42 + i).iloc[0]
    img = np.array(Image.open(row["image"]).convert("L"))
    mask = np.array(Image.open(row["mask"]).convert("L"))

    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title(f"Image ({row['class']})")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Infection mask")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(img, cmap="gray")
    axes[i, 2].imshow(mask, cmap="jet", alpha=0.4)
    axes[i, 2].set_title("Overlay")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()


# ## U-net




IMG_SIZE = (256, 256)  
BATCH_SIZE = 4         

def load_pair(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMG_SIZE, method="nearest")
    mask = tf.cast(mask > 127, tf.float32)  # 0/1 mask

    return img, mask


def df_to_dataset(df: pd.DataFrame, shuffle=True):
    images = df["image"].astype(str).values
    masks = df["mask"].astype(str).values

    ds = tf.data.Dataset.from_tensor_slices((images, masks))
    ds = ds.map(
        lambda im, ma: load_pair(im, ma),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = df_to_dataset(train_df, shuffle=True)
val_ds = df_to_dataset(val_df, shuffle=False)

train_ds


# simple U-Net 

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

inputs = layers.Input(shape=IMG_SIZE + (1,))

c1 = conv_block(inputs, 32)
p1 = layers.MaxPooling2D()(c1)

c2 = conv_block(p1, 64)
p2 = layers.MaxPooling2D()(c2)

c3 = conv_block(p2, 128)
p3 = layers.MaxPooling2D()(c3)

b = conv_block(p3, 256)


u3 = layers.UpSampling2D()(b)
u3 = layers.Concatenate()([u3, c3])
cd3 = conv_block(u3, 128)

u2 = layers.UpSampling2D()(cd3)
u2 = layers.Concatenate()([u2, c2])
cd2 = conv_block(u2, 64)

u1 = layers.UpSampling2D()(cd2)
u1 = layers.Concatenate()([u1, c1])
cd1 = conv_block(u1, 32)

outputs = layers.Conv2D(1, 1, activation="sigmoid")(cd1)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],  
)

model.summary()


EPOCHS = 30  

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)



plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Val loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train acc")
plt.plot(history.history["val_accuracy"], label="Val acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Pixel accuracy")

plt.tight_layout()
plt.show()



val_batch = next(iter(val_ds))
imgs, true_masks = val_batch
pred_masks = model.predict(imgs)

N_SHOW = min(3, imgs.shape[0])

fig, axes = plt.subplots(N_SHOW, 3, figsize=(9, 3 * N_SHOW))

for i in range(N_SHOW):
    img = imgs[i, ..., 0]
    true_mask = true_masks[i, ..., 0]
    pred_mask = pred_masks[i, ..., 0]

    axes[i, 0].imshow(img, cmap="gray")
    axes[i, 0].set_title("Image")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(true_mask, cmap="gray")
    axes[i, 1].set_title("True infection mask")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(img, cmap="gray")
    axes[i, 2].imshow(pred_mask > 0.5, cmap="jet", alpha=0.4)
    axes[i, 2].set_title("Predicted mask (overlay)")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()


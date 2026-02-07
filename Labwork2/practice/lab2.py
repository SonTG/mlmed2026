
# ## Load the data


import pandas as pd
from pathlib import Path

data_dir = Path('../data')
train_csv = data_dir / 'training_set_pixel_size_and_HC.csv'
test_csv = data_dir / 'test_set_pixel_size.csv'

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
print('train:', train_df.shape, '| test:', test_df.shape)


train_img_dir = data_dir / 'training_set' / 'training_set'
test_img_dir = data_dir / 'test_set' / 'test_set'


def check_images(df, img_dir, name):
    missing = [fn for fn in df['filename'] if not (img_dir / fn).exists()]
    n_ok = len(df) - len(missing)
    print(f'{name}: {n_ok}/{len(df)} images ok' + (f' ({len(missing)} missing)' if missing else ''))
    return len(missing)

check_images(train_df, train_img_dir, 'train')
check_images(test_df, test_img_dir, 'test')

# drop rows where image is missing
train_df = train_df[train_df['filename'].apply(lambda fn: (train_img_dir / fn).exists())].reset_index(drop=True)
test_df = test_df[test_df['filename'].apply(lambda fn: (test_img_dir / fn).exists())].reset_index(drop=True)
print('after filter: train', len(train_df), '| test', len(test_df))


# ##  Quick look at the data


print(train_df.shape, train_df.columns.tolist())
print('missing:', train_df.isnull().sum().sum())
train_df.head()


train_df.describe()


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
train_df['head circumference (mm)'].hist(ax=ax[0], bins=40, edgecolor='black', alpha=0.7)
ax[0].set_title('HC (mm)')
train_df['pixel size(mm)'].hist(ax=ax[1], bins=40, edgecolor='black', alpha=0.7)
ax[1].set_title('pixel size (mm)')
plt.tight_layout()
plt.show()


# ##  Train/val split and load images


from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
train_idx, val_idx = train_test_split(train_df.index, test_size=0.2, random_state=RANDOM_STATE)
train_sub = train_df.loc[train_idx].reset_index(drop=True)
val_sub = train_df.loc[val_idx].reset_index(drop=True)
print(len(train_sub), 'train |', len(val_sub), 'val')


import numpy as np
from PIL import Image

IMG_SIZE = 128

def load_image(path):
    img = Image.open(path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0

def load_data(df, img_dir):
    X, y = [], []
    for _, row in df.iterrows():
        fp = img_dir / row['filename']
        if not fp.exists():
            continue
        X.append(load_image(fp))
        y.append(row['head circumference (mm)'])
    return np.stack(X)[..., np.newaxis], np.array(y, dtype=np.float32)

X_train, y_train = load_data(train_sub, train_img_dir)
X_val, y_val = load_data(val_sub, train_img_dir)
print('X_train', X_train.shape, '| X_val', X_val.shape)


# ##  Model simple CNN

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BATCH_SIZE = 32
EPOCHS = 20

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPool2D(2),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model()
model.summary()


history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# Plot training curves (loss and MAE) 
import matplotlib.pyplot as plt
from pathlib import Path

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

epochs_range = range(1, len(history.history['loss']) + 1)
axes[0].plot(epochs_range, history.history['loss'], label='Train loss (MSE)', marker='o', markersize=3)
axes[0].plot(epochs_range, history.history['val_loss'], label='Val loss (MSE)', marker='s', markersize=3)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].set_title('Loss (MSE)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history.history['mae'], label='Train MAE (mm)', marker='o', markersize=3)
axes[1].plot(epochs_range, history.history['val_mae'], label='Val MAE (mm)', marker='s', markersize=3)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE (mm)')
axes[1].set_title('Mean Absolute Error')
axes[1].legend()
axes[1].grid(True, alpha=0.3)


from sklearn.metrics import mean_squared_error, r2_score

loss, mae = model.evaluate(X_val, y_val)
y_pred_val = model.predict(X_val, verbose=0).flatten()
mse = mean_squared_error(y_val, y_pred_val)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred_val)

print('val MAE (mm):', round(mae, 4))
print('val MSE:', round(mse, 4), '| RMSE:', round(rmse, 4), '| R²:', round(r2, 4))


# ## see result



np.random.seed(RANDOM_STATE)
n_show = 6
idx = np.random.choice(len(X_val), size=n_show, replace=False)

fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
for j, k in enumerate(idx):
    img = X_val[k].squeeze()
    true_hc = y_val[k]
    pred_hc = model.predict(X_val[k:k+1], verbose=0).flatten()[0]
    err = abs(pred_hc - true_hc)
    axes[0, j].imshow(img, cmap='gray')
    axes[0, j].set_title(f'#{j+1}')
    axes[0, j].axis('off')
    axes[1, j].imshow(img, cmap='gray')
    axes[1, j].set_title(f'true {true_hc:.0f} | pred {pred_hc:.0f} | err {err:.0f} mm', fontsize=9)
    axes[1, j].axis('off')
axes[0, 0].set_ylabel('original')
axes[1, 0].set_ylabel('prediction')
plt.tight_layout()
plt.show()



#  3 samples at full resolution 
n_orig = 3
fig2, axes2 = plt.subplots(1, n_orig, figsize=(4 * n_orig, 4))
for j in range(n_orig):
    k = idx[j]
    fn = val_sub.iloc[k]['filename']
    img_orig = np.array(Image.open(train_img_dir / fn).convert('L'))
    pred_hc = model.predict(X_val[k:k+1], verbose=0).flatten()[0]
    axes2[j].imshow(img_orig, cmap='gray')
    axes2[j].set_title(f'{fn}\ntrue {y_val[k]:.0f} | pred {pred_hc:.0f} mm')
    axes2[j].axis('off')
plt.tight_layout()
plt.show()


# ##  Save model 


model_path = data_dir / 'hc18_regression_model.keras'
model.save(model_path)
print('model saved:', model_path)




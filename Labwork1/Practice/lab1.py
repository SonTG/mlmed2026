# P1: data checking

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('../Dataset/mitbih_train.csv', header=None)
test_df = pd.read_csv('../Dataset/mitbih_test.csv', header=None)

print("training set shape:", train_df.shape)
print("test set shape:", test_df.shape)
train_df.head()

## check 

X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:, -1].values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.iloc[:, -1].values

print(f"X_train: {X_train.shape} (samples, timesteps)")
print(f"y_train: {y_train.shape} (labels)")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


class_names = {
    0: 'N - Non-ectopic',
    1: 'S - Supraventricular ectopic',
    2: 'V - Ventricular ectopic',
    3: 'F - Fusion',
    4: 'Q - Unknown'
}

# count classes in training
unique_train, counts_train = np.unique(y_train, return_counts=True)
train_dist = dict(zip(unique_train, counts_train))

print("training set class distribution:")
for class_id in sorted(unique_train):
    count = train_dist[class_id]
    pct = (count / len(y_train)) * 100
    print(f"{class_names[class_id]:35s}: {count:6d} ({pct:5.2f}%)")

# count classes in test
unique_test, counts_test = np.unique(y_test, return_counts=True)
test_dist = dict(zip(unique_test, counts_test))

print("\ntest set class distribution:")
for class_id in sorted(unique_test):
    count = test_dist[class_id]
    pct = (count / len(y_test)) * 100
    print(f"{class_names[class_id]:35s}: {count:6d} ({pct:5.2f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# training
train_counts = [train_dist.get(i, 0) for i in range(5)]
axes[0].bar(range(5), train_counts, color=['blue', 'red', 'green', 'orange', 'purple'])
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Count')
axes[0].set_title('Training Set')
axes[0].set_xticks(range(5))
axes[0].set_xticklabels(['N', 'S', 'V', 'F', 'Q'])

# test
test_counts = [test_dist.get(i, 0) for i in range(5)]
axes[1].bar(range(5), test_counts, color=['blue', 'red', 'green', 'orange', 'purple'])
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Count')
axes[1].set_title('Test Set')
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(['N', 'S', 'V', 'F', 'Q'])

plt.tight_layout()
plt.show()

# Check class imbalance
max_count = max(train_counts)
print("\nClass imbalance ratio:")
for i, count in enumerate(train_counts):
    ratio = max_count / count if count > 0 else 0
    print(f"{class_names[i]:35s}: {ratio:.2f}x")


# plot 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for class_id in range(5):
    class_idx = np.where(y_train == class_id)[0]
    if len(class_idx) > 0:
        signal = X_train[class_idx[0]]
        axes[class_id].plot(signal, linewidth=2)
        axes[class_id].set_title(class_names[class_id])
        axes[class_id].set_xlabel('Time Steps')
        axes[class_id].set_ylabel('Amplitude')
        axes[class_id].grid(True, alpha=0.3)

axes[5].axis('off')
plt.suptitle('ECG Signals from Each Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

## static
stats = []

for class_id in range(5):
    class_idx = np.where(y_train == class_id)[0]
    if len(class_idx) > 0:
        signals = X_train[class_idx]
        stats.append({
            'Class': class_names[class_id],
            'Count': len(class_idx),
            'Mean': np.mean(signals),
            'Std': np.std(signals),
            'Min': np.min(signals),
            'Max': np.max(signals)
        })

stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))


# check   
print(f"Missing values - Train: {train_df.isnull().sum().sum()}, Test: {test_df.isnull().sum().sum()}")
print(f"Infinite values - Train: {np.isinf(X_train).sum()}, Test: {np.isinf(X_test).sum()}")

print(f"\nData types:")
print(f"X_train: {X_train.dtype}, y_train: {y_train.dtype}")
print(f"X_test: {X_test.dtype}, y_test: {y_test.dtype}")
print(f"\nUnique labels - Train: {np.unique(y_train)}, Test: {np.unique(y_test)}")

# P2 – Preprocessing 

## cast data types

import numpy as np

X_train_full = X_train.astype('float32')
y_train_full = y_train.astype('int64')
X_test_float = X_test.astype('float32')
y_test_int = y_test.astype('int64')

print("train full shape:", X_train_full.shape, y_train_full.shape)
print("test shape:      ", X_test_float.shape, y_test_int.shape)
print("dtypes:")
print("  X_train_full:", X_train_full.dtype)
print("  y_train_full:", y_train_full.dtype)
print("  X_test_float:", X_test_float.dtype)
print("  y_test_int:  ", y_test_int.dtype)

## original data

from collections import Counter

orig_counts = Counter(y_train_full)
print("Original training labels (before SMOTE):")
for cls in sorted(orig_counts.keys()):
    print(f"  Class {cls}: {orig_counts[cls]}")

## smote method

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train_full, y_train_full)

sm_counts = Counter(y_train_sm)
print("class distribution after smote:")
for cls in sorted(sm_counts.keys()):
    print(f"  class {cls}: {sm_counts[cls]}")

print("\nbalanced train shape:", X_train_sm.shape, y_train_sm.shape)

## train / validation split data.

from sklearn.model_selection import train_test_split

X_train_cnn, X_val_cnn, y_train_cnn, y_val_cnn = train_test_split(
    X_train_sm.astype('float32'),
    y_train_sm.astype('int64'),
    test_size=0.2,
    random_state=42,
    stratify=y_train_sm,
)

print("shapes after split (balanced data):")
print("  X_train_cnn:", X_train_cnn.shape)
print("  y_train_cnn:", y_train_cnn.shape)
print("  X_val_cnn:  ", X_val_cnn.shape)
print("  y_val_cnn:  ", y_val_cnn.shape)

# check
import numpy as np

def print_label_distribution(name, y):
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    print(name)
    for c, cnt in zip(unique, counts):
        pct = cnt / total * 100
        print(f"  Class {c}: {cnt:6d} ({pct:5.2f}%)")
    print()

print_label_distribution("test labels (after smote)", y_train_cnn)
print_label_distribution("val labels (after smote )", y_val_cnn)

## 1D 

X_train_cnn = X_train_cnn[..., np.newaxis]
X_val_cnn   = X_val_cnn[..., np.newaxis]
X_test_cnn  = X_test_float[..., np.newaxis]

print("final shapes for CNN:")
print("  X_train_cnn:", X_train_cnn.shape)
print("  X_val_cnn:  ", X_val_cnn.shape)
print("  X_test_cnn: ", X_test_cnn.shape)


X_train_cnn = X_train_cnn[..., np.newaxis]
X_val_cnn   = X_val_cnn[..., np.newaxis]
X_test_cnn  = X_test_float[..., np.newaxis]

print("Final shapes for CNN:")
print("  X_train_cnn:", X_train_cnn.shape)
print("  X_val_cnn:  ", X_val_cnn.shape)
print("  X_test_cnn: ", X_test_cnn.shape)

# P3 Model 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

#CNN

num_classes = 5
input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2])  # (187, 1)

model = keras.Sequential([
    layers.Input(shape=input_shape),

    layers.Conv1D(32, kernel_size=5, padding='same', activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
    layers.MaxPooling1D(pool_size=2),

    layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
    layers.GlobalMaxPooling1D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

model.summary()

## compile , train

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
]

history = model.fit(
    X_train_cnn, y_train_cnn,
    validation_data=(X_val_cnn, y_val_cnn),
    epochs=40,
    batch_size=128,
    verbose=1,
)

print("Training done.")

# plot 
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# evaluate 

y_prob = model.predict(X_test_cnn, batch_size=256, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

test_acc = accuracy_score(y_test_int, y_pred)
print(f"test accuracy: {test_acc:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test_int,
    y_pred,
    target_names=[class_names[i] for i in range(5)]
))

# confusion matrix
cm = confusion_matrix(y_test_int, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['0','1','2','3','4'],
            yticklabels=['0','1','2','3','4'])
plt.title('Confusion Matrix (Test)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# P4, compare

paper_accuracy = 93.4  

print(f"(kachuee et al., 2018): {paper_accuracy}% accuracy")
print(f"model:                    {test_acc*100:.2f}% accuracy")
print(f"difference:                    {abs(test_acc*100 - paper_accuracy):.2f}%")

#  P5: Save Model 

model.save('../model.h5')
print("Model saved to: ../model.h5")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train_df = pd.read_csv('C:/Users/quang/ipynb code/mlmed/dataset/heartbeat/mitbih_train.csv', header=None)
test_df = pd.read_csv('C:/Users/quang/ipynb code/mlmed/dataset/heartbeat/mitbih_test.csv', header=None)

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

X_train_original = train_df.iloc[:, :-1].values
y_train_original = train_df.iloc[:, -1].values

X_test_full = test_df.iloc[:, :-1].values
y_test_full = test_df.iloc[:, -1].values

print(f"\nOriginal training set size: {len(X_train_original)}")
print(f"Original test set size: {len(X_test_full)}")

print("GENERATING FIGURE 1: Original Training Dataset Distribution")


class_names = ['Normal', 'Supraventricular', 'Ventricular', 'Fusion', 'Unknown']
unique, counts = np.unique(y_train_original, return_counts=True)

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(unique)), counts, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Frequency of each class in the dataset', fontsize=14, fontweight='bold')
plt.xticks(range(len(unique)), class_names, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

for i, (bar, count) in enumerate(zip(bars, counts)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
             str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figure1_original_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: figure1_original_distribution.png")
plt.close()

print("\n" + "=" * 80)
print("BALANCING DATASET")
print("=" * 80)

target_samples = 17500
train_data = np.column_stack((X_train_original, y_train_original))

balanced_data = []
for class_label in range(5):
    class_data = train_data[train_data[:, -1] == class_label]
    
    if len(class_data) < target_samples:
        class_data_resampled = resample(class_data, 
                                       replace=True, 
                                       n_samples=target_samples, 
                                       random_state=42)
    else:
        class_data_resampled = resample(class_data, 
                                       replace=False, 
                                       n_samples=target_samples, 
                                       random_state=42)
    
    balanced_data.append(class_data_resampled)
    print(f"Class {class_label} ({class_names[class_label]}): {len(class_data)} -> {len(class_data_resampled)}")

balanced_data = np.vstack(balanced_data)
np.random.shuffle(balanced_data)

X_train = balanced_data[:, :-1]
y_train = balanced_data[:, -1]

print(f"\nBalanced training set size: {len(X_train)}")
print("GENERATING FIGURE 2: Balanced Training Dataset Distribution")


unique_balanced, counts_balanced = np.unique(y_train, return_counts=True)

plt.figure(figsize=(10, 6))
bars = plt.bar(range(len(unique_balanced)), counts_balanced, color='coral', edgecolor='black', alpha=0.7)
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.ylabel('Frequency', fontsize=12, fontweight='bold')
plt.title('Frequency of each class in the dataset after balancing', fontsize=14, fontweight='bold')
plt.xticks(range(len(unique_balanced)), class_names, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.ylim(0, max(counts_balanced) * 1.1)
for i, (bar, count) in enumerate(zip(bars, counts_balanced)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
             str(count), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_balanced_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: figure2_balanced_distribution.png")
plt.close()

print("SPLITTING TEST SET")

X_val, X_test, y_val, y_test = train_test_split(
    X_test_full, y_test_full, test_size=0.5, random_state=42, stratify=y_test_full
)

print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")

print("GENERATING FIGURE 3: Validation and Test Dataset Distribution")


unique_val, counts_val = np.unique(y_val, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

bars1 = ax1.bar(range(len(unique_val)), counts_val, color='lightgreen', edgecolor='black', alpha=0.7)
ax1.set_xlabel('Class', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('Valid Dataset', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(unique_val)))
ax1.set_xticklabels(class_names, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bar, count in zip(bars1, counts_val):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

bars2 = ax2.bar(range(len(unique_test)), counts_test, color='lightblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Class', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Test Dataset', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(unique_test)))
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar, count in zip(bars2, counts_test):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
             str(count), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Frequency of each class in the valid and test dataset', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figure3_valid_test_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: figure3_valid_test_distribution.png")
plt.close()
print("NORMALIZING DATA")

X_train_norm = X_train / 255.0
X_val_norm = X_val / 255.0
X_test_norm = X_test / 255.0

print("Data normalized")


print("MODEL 1: RANDOM FOREST")

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
rf_model.fit(X_train_norm, y_train)

rf_pred = rf_model.predict(X_test_norm)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest Test Accuracy: {rf_accuracy:.4f}")



print("MODEL 2: SIMPLE CNN")
X_train_cnn = X_train_norm.reshape(-1, 187, 1)
X_val_cnn = X_val_norm.reshape(-1, 187, 1)
X_test_cnn = X_test_norm.reshape(-1, 187, 1)
y_train_cat = keras.utils.to_categorical(y_train, 5)
y_val_cat = keras.utils.to_categorical(y_val, 5)
y_test_cat = keras.utils.to_categorical(y_test, 5)

def build_simple_cnn():
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(187, 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(5, activation='softmax')
    ])
    return model

cnn_model = build_simple_cnn()
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(cnn_model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

cnn_history = cnn_model.fit(
    X_train_cnn, y_train_cat,
    validation_data=(X_val_cnn, y_val_cat),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

cnn_pred = cnn_model.predict(X_test_cnn)
cnn_pred_classes = np.argmax(cnn_pred, axis=1)
cnn_accuracy = accuracy_score(y_test, cnn_pred_classes)
print(f"\nSimple CNN Test Accuracy: {cnn_accuracy:.4f}")
print("MODEL 3: CNN WITH RESIDUAL CONNECTION")

def residual_block(x, filters, kernel_size=3):
    """Residual block with skip connection"""
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    return x

def build_cnn_residual():
    inputs = layers.Input(shape=(187, 1))
    
    # Initial conv layer
    x = layers.Conv1D(64, 7, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(5, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

cnn_res_model = build_cnn_residual()
cnn_res_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(cnn_res_model.summary())

cnn_res_history = cnn_res_model.fit(
    X_train_cnn, y_train_cat,
    validation_data=(X_val_cnn, y_val_cat),
    epochs=50,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

cnn_res_pred = cnn_res_model.predict(X_test_cnn)
cnn_res_pred_classes = np.argmax(cnn_res_pred, axis=1)
cnn_res_accuracy = accuracy_score(y_test, cnn_res_pred_classes)
print(f"\nCNN with Residual Connection Test Accuracy: {cnn_res_accuracy:.4f}")

print("GENERATING FIGURE 4 (TABLE 1): Loss Track Plots")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Loss track of the Deep Learning models', fontsize=16, fontweight='bold')

models_history = [
    (cnn_history, 'CNN', axes[0, 0]),
    (cnn_res_history, 'CNN with Residual Connection', axes[0, 1])
]

for history, name, ax in models_history:
    ax.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure4_loss_tracks.png', dpi=300, bbox_inches='tight')
print("Saved: figure4_loss_tracks.png")
plt.close()



print("GENERATING FIGURE 5 (TABLE 2): Confusion Matrices")


predictions = [
    (rf_pred, 'Random Forest', rf_accuracy),
    (cnn_pred_classes, 'CNN', cnn_accuracy),
    (cnn_res_pred_classes, 'CNN with Residual Connection', cnn_res_accuracy)
]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Confusion Matrices of all Models', fontsize=16, fontweight='bold')

axes = axes.flatten()

for idx, (pred, name, acc) in enumerate(predictions):
    cm = confusion_matrix(y_test, pred)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[idx], cbar_kws={'label': 'Count'})
    
    axes[idx].set_title(f'{name}\nAccuracy: {acc:.4f}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Actual', fontsize=10, fontweight='bold')
    axes[idx].tick_params(axis='both', labelsize=8)

axes[3].axis('off')

plt.tight_layout()
plt.savefig('figure5_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("Saved: figure5_confusion_matrices.png")
plt.close()


print("FINAL SUMMARY")

print("\n ALL FIGURES GENERATED:")

print("figure1_original_distribution.png - Original training dataset class distribution")
print("figure2_balanced_distribution.png - Balanced training dataset class distribution")
print("figure3_valid_test_distribution.png - Validation and test dataset distributions")
print("figure4_loss_tracks.png - Training/validation loss for all DL models")
print("figure5_confusion_matrices.png - Confusion matrices for all 5 models")

print("\nMODEL PERFORMANCE SUMMARY:")
print(f"Random Forest:                     {rf_accuracy:.4f}")
print(f"Simple CNN:                        {cnn_accuracy:.4f}")
print(f"CNN with Residual Connection:      {cnn_res_accuracy:.4f}")

best_model = max(predictions, key=lambda x: x[2])
print(f"\nBEST MODEL: {best_model[1]} (Accuracy: {best_model[2]:.4f})")

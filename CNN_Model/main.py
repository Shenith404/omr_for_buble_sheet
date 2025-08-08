import os
import shutil
import random
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from PIL import Image
import pickle

# Set dataset paths
dataset_path = "dataset/"
train_path = "dataset_split/train/"
val_path = "dataset_split/val/"
test_path = "dataset_split/test/"

# Ensure dataset is split
def split_dataset():
    print("Splitting dataset into train, validation, and test sets...")
    split_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    
    for path in [train_path, val_path, test_path]:
        for category in os.listdir(dataset_path):
            os.makedirs(os.path.join(path, category), exist_ok=True)
    
    for category in os.listdir(dataset_path):
        images = os.listdir(os.path.join(dataset_path, category))
        random.shuffle(images)
        
        train_idx = int(len(images) * split_ratios["train"])
        val_idx = int(len(images) * (split_ratios["train"] + split_ratios["val"]))
        
        for i, img in enumerate(images):
            src_path = os.path.join(dataset_path, category, img)
            
            if i < train_idx:
                dst_path = os.path.join(train_path, category, img)
            elif i < val_idx:
                dst_path = os.path.join(val_path, category, img)
            else:
                dst_path = os.path.join(test_path, category, img)
            
            shutil.copy(src_path, dst_path)
    print("Dataset split completed.")

# Split dataset only if not already split
if (not os.path.exists(train_path)) or (not os.path.exists(val_path)) or (not os.path.exists(test_path)):
    split_dataset()
else:
    print("Dataset already split, skipping splitting.")

# Data Augmentation - for training only
print("Setting up data augmentation for training set...")
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,  
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation and Test data generators - only rescaling
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_path, 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical', 
    color_mode='grayscale'
)

val_generator = val_datagen.flow_from_directory(
    val_path, 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical', 
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_path, 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical', 
    shuffle=False, 
    color_mode='grayscale'
)

# Dynamically calculate class distribution based on the dataset
print("Calculating class distribution from the dataset...")
labels_count = {}
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):
        labels_count[category] = len(os.listdir(category_path))

# Print class distribution
print("Class distribution:")
print(labels_count)

# Plot class distribution before handling imbalance
plt.figure(figsize=(10, 5))
plt.bar(labels_count.keys(), labels_count.values(), color='blue')
plt.title("Class Distribution Before Handling Imbalance")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.show()

# Handle Class Imbalance
print("Computing class weights to handle imbalance...")
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(list(labels_count.keys())),
    y=np.array(sum([[k] * v for k, v in labels_count.items()], []))  # Flatten label counts
)

class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weights_dict)

# Simulate effective sample counts after applying class weights
effective_samples = {cls: count * weight for (cls, count), weight in zip(labels_count.items(), class_weights)}
print("Effective sample counts after applying class weights:")
print(effective_samples)

# Plot effective sample counts after handling imbalance
plt.figure(figsize=(10, 5))
plt.bar(effective_samples.keys(), effective_samples.values(), color='green')
plt.title("Effective Sample Counts After Handling Imbalance")
plt.xlabel("Class")
plt.ylabel("Effective Number of Samples")
plt.xticks(rotation=45)
plt.show()

# Build Improved CNN Model
print("Building the CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),  # Grayscale input
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(labels_count), activation='softmax')  # adjust output units dynamically
])

# Compile Model
print("Compiling the model...")
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_model.keras", 
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, checkpoint, reduce_lr]

# Train Model with Class Weights and callbacks
print("Training the model...")
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Evaluate Model on Test Set
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# === Save Training History ===
with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("✅ Training history saved as 'training_history.pkl'")

# === Load training history back (optional) ===
with open("training_history.pkl", "rb") as f:
    history_dict = pickle.load(f)

# Simple smoothing function for better plot visualization
def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

val_acc_smoothed = smooth_curve(history_dict['val_accuracy'])
val_loss_smoothed = smooth_curve(history_dict['val_loss'])

# Create output directory for graphs
output_dir = "training_graphs"
os.makedirs(output_dir, exist_ok=True)

# === Plot and Save Accuracy Graph ===
plt.figure(figsize=(10, 5))
plt.plot(history_dict['accuracy'], label='Train Accuracy')
plt.plot(val_acc_smoothed, label='Validation Accuracy (Smoothed)')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
accuracy_path = os.path.join(output_dir, "accuracy_graph.png")
plt.savefig(accuracy_path)
plt.close()
print(f"✅ Accuracy graph saved at: {accuracy_path}")

# === Plot and Save Loss Graph ===
plt.figure(figsize=(10, 5))
plt.plot(history_dict['loss'], label='Train Loss')
plt.plot(val_loss_smoothed, label='Validation Loss (Smoothed)')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
loss_path = os.path.join(output_dir, "loss_graph.png")
plt.savefig(loss_path)
plt.close()
print(f"✅ Loss graph saved at: {loss_path}")

# Save final model
print("Saving the final model...")
model.save("bubble_cnn_model.keras")
print("✅ Model saved as 'bubble_cnn_model.keras'")

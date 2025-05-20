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
from PIL import Image

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

# Split dataset
split_dataset()

# Data Augmentation
print("Setting up data augmentation...")
datagen = ImageDataGenerator(
    rescale=1.0/255.0,  # Normalize pixel values
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    train_path, 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical', 
    color_mode='grayscale'
)

val_generator = datagen.flow_from_directory(
    val_path, 
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='categorical', 
    color_mode='grayscale'
)

test_generator = datagen.flow_from_directory(
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
    if os.path.isdir(category_path):  # Check if it's a directory (class)
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
    Dense(3, activation='softmax')  # 4 output classes
])

# Compile Model
print("Compiling the model...")
model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model with Class Weights
print("Training the model...")
history = model.fit(
    train_generator, 
    epochs=100,  # Increased epochs
    validation_data=val_generator, 
    class_weight=class_weights_dict
)

# Evaluate Model on Test Set
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Save Model
print("Saving the model...")
model.save("bubble_cnn_model.keras")


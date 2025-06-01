import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Load trained model
MODEL_PATH = "bubble_cnn_model copy.keras"
# IMG_PATH = "crossImg.jpg"

model = load_model(MODEL_PATH)

# Check model input shape
print("Model expects input shape:", model.input_shape)  # Debugging step

# Load image as grayscale (1 channel)
# img2 = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

# Resize image to match model input size (64x64)
TARGET_SIZE = (64, 64)
# Class labels
class_labels = ["Crossed_Bubble", "Empty_Bubble", "Other", "Shaded_Bubble"]

def classify_bubble(image):
    img = cv2.resize(image, TARGET_SIZE)

    # Convert to array and normalize
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (grayscale)


    # Predict
    prediction = model.predict(img_array)
    label = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return label, confidence

# label, confidence = classify_bubble(img2)

# print(f"Predicted Label: {label}, Confidence: {confidence:.2f}%")

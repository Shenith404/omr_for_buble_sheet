import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Load trained model
MODEL_PATH = "bubble_cnn_model.keras"
model = load_model(MODEL_PATH)

# Class labels
class_labels = [
    "Crossed_Bubble", 
    "Empty_Bubble", 
    "Other", 
    "Shaded_Bubble"
]

def classify_bubble(image):
    """Classify a bubble image using the trained model"""
    try:
        # Resize and preprocess
        img = cv2.resize(image, (64, 64))
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        
        # Predict
        prediction = model.predict(img_array)
        label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        return label, confidence
        
    except Exception as e:
        print(f"Classification error: {e}")
        return "Error", 0.0
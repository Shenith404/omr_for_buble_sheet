# model.py
import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
def get_model_path():
    # For development
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)
    return os.path.join(base_path, "models", "model_adpthresh_2.h5")

model = load_model(get_model_path())

# Class labels
class_labels = ['cross_sheets_adpthresh', 'empty_sheets_adpthresh', 'shaded_sheets_adpthresh']


def classify_bubble(image):
    """Classify a single bubble image using the trained model"""
    try:
        img = cv2.resize(image, (64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)       # batch dimension
        img_array = np.expand_dims(img_array, axis=-1)      # channel dimension
        prediction = model.predict(img_array)
        label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return label, confidence
    except Exception as e:
        print(f"Classification error: {e}")
        return "Error", 0.0

def classify_batch(images):
    """
    Classify a list of bubble images using the trained model.
    Returns a list of (label, confidence) for each image.
    """
    try:
        preprocessed = []
        for img in images:
            resized = cv2.resize(img, (64, 64))
            arr = img_to_array(resized) / 255.0
            arr = np.expand_dims(arr, axis=-1)  # Add channel dimension
            preprocessed.append(arr)

        batch_input = np.array(preprocessed)
        predictions = model.predict(batch_input)

        results = []
        for pred in predictions:
            label = class_labels[np.argmax(pred)]
            confidence = np.max(pred) * 100
            results.append((label, confidence))
        return results

    except Exception as e:
        print(f"Batch classification error: {e}")
        return [("Error", 0.0)] * len(images)

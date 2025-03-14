
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
MODEL_PATH = "bubble_cnn_model.keras"
model = load_model(MODEL_PATH)

# Class labels
class_labels = ["Cross_Removed_Bubble", "Crossed_Bubble", "Empty_Bubble", "Other", "Shaded_Bubble"]

def classify_bubble(image):
    """Classifies the uploaded bubble image."""
    # image = image.convert("L")  # Convert image to grayscale
    img = image.resize((64, 64))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)  # Ensure correct shape for grayscale
    
    prediction = model.predict(img_array)
    label = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return label, confidence

# Streamlit UI
st.title("Bubble Sheet Classifier")
st.write("Upload an image of a bubble sheet to classify its category.")

uploaded_file = st.file_uploader("Upload a Bubble Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    label, confidence = classify_bubble(image)
    st.write(f"Prediction: {label} ({confidence:.2f}% Confidence)")

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Set up page config
st.set_page_config(page_title="Bubble Classification", layout="wide")

# Load the trained model
@st.cache_resource
def load_bubble_model():
    return load_model("bubble_cnn_model.keras")

model = load_bubble_model()

# Define class labels
class_labels = ["Crossed_Bubble", "Empty_Bubble", "Other", "Shaded_Bubble"]

# Function to classify multiple images and return labels
def process_and_classify_multiple(images):
    img_arrays = []
    for image in images:
        img = image.resize((64, 64))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Grayscale channel
        img_arrays.append(img_array)
    img_arrays = np.array(img_arrays)
    predictions = model.predict(img_arrays)
    result_labels = [class_labels[np.argmax(pred)] for pred in predictions]
    return result_labels

# Title
st.title("🔍 Bubble Image Classification")
st.write("Upload one or more grayscale bubble images to classify them.")

# File uploader allowing multiple images
uploaded_files = st.file_uploader("Upload Bubble Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Process and display result array
if uploaded_files:
    images = [Image.open(file).convert("L") for file in uploaded_files]
    result_labels = process_and_classify_multiple(images)

    # ✅ Display single-line result array
    st.subheader("🧾 Classification Results:")
    # st.write(result_labels)
    st.text(str(result_labels))


import os
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash")


def extract_digits_from_image(image_input):
    """
    Extract registration numbers in format: EG_20XX_XXXX
    Supports file path, OpenCV image, or PIL image.
    """

    try:
        # --- Convert input to PIL image ---
        if isinstance(image_input, str):
            img_cv = cv2.imread(image_input)
            if img_cv is None:
                raise ValueError(f"Cannot read image: {image_input}")
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

        elif isinstance(image_input, np.ndarray):
            img_rgb = (
                cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                if image_input.ndim == 3 and image_input.shape[2] == 3
                else image_input
            )
            img_pil = Image.fromarray(img_rgb)

        elif isinstance(image_input, Image.Image):
            img_pil = image_input
        else:
            raise TypeError("Unsupported image input type.")

        # --- Prompt for Gemini ---
        prompt ="""
Extract all text form this image.
first two digits are E and G
last 8 digits are numbers, give the result as one string


"""

        # --- Call Gemini ---
        response = model.generate_content([prompt, img_pil])
        print("Gemini response:", response.text if response and response.text else "No response text")

        return {
            "success": True,
            "digit_sequence": response.text.strip() if response and response.text else "",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

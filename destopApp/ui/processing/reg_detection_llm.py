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
model = genai.GenerativeModel("models/gemma-3-27b-it")


def extract_digits_from_image(image_input):
 

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
Extract registration numbers from the following text.  Each registration number must adhere to the format EG/XXXX/YYYY, where:

*   It always begins with "EG".
*   The third character is a forward slash ("/").
*   The next four characters represent the first part of the registration number (XXXX).
*   Another forward slash ("/") separates the two parts.
*   The final four characters represent the second part of the registration number (YYYY).

Return a list of valid registration numbers in the format "EG_XXXX_YYYY".  Do not include any partial or invalid numbers.

For example, if the input text contains "EG/1234/5678", the output should be EG_1234_5678. don give any other things
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

import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_number_v3(img_array, border_crop=10):
    """
    Detect numbers from a numpy array (H x W x 3 or H x W).
    Handles inconsistent backgrounds, low-contrast text, and broken characters.
    Optionally crops 'border_crop' pixels from each side of the image.
    """
    try:
        if img_array is None:
            print("Error: Input array is None")
            return None

        # Convert to grayscale if image has 3 channels
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()

        height, width = gray.shape

        # --- 1. Optional border crop ---
        if border_crop > 0:
            gray = gray[border_crop:height-border_crop, border_crop:width-border_crop]
            height, width = gray.shape  # update dimensions

        # --- 2. Trim borders and upscale ---
        x_start, y_start = 2, 2
        borderless_gray = gray[y_start:, x_start:]

        scale_factor = 3
        resized_img = cv2.resize(borderless_gray, None,
                                 fx=scale_factor, fy=scale_factor,
                                 interpolation=cv2.INTER_CUBIC)

        # --- 3. Morphological closing and adaptive thresholding ---
        kernel = np.ones((3, 3), np.uint8)
        closed_img = cv2.morphologyEx(resized_img, cv2.MORPH_CLOSE, kernel)

        processed_img = cv2.adaptiveThreshold(closed_img, 255,
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 21, 5)

        # --- 4. Tesseract OCR ---
        custom_config = r'--psm 10 -c tessedit_char_whitelist=0123456789'
        pil_image = Image.fromarray(processed_img)
        text = pytesseract.image_to_string(pil_image, config=custom_config)

        # --- 5. Post-processing ---
        detected_number = ''.join(filter(str.isdigit, text.strip()))

        return detected_number if detected_number else None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Example usage with a loaded image as array:
if __name__ == '__main__':
    img = cv2.imread('28.png')
    detected_number = detect_number_v3(img)
    print(f"Detected Number: {detected_number}")

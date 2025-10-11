import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_multi_digit_high_threshold(img):
    """
    Detects multi-digit numbers from an image array using a high binary threshold,
    dilation, and sharpening. If the image is larger than 80x80, shaves off 10px border.
    Converts to grayscale only if necessary.
    """
    try:
        if img is None or not isinstance(img, np.ndarray):
            print("Error: Invalid image input.")
            return None

        full_crop_img = img

        if len(full_crop_img.shape) == 3:
            gray = cv2.cvtColor(full_crop_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = full_crop_img

        borderless_gray = gray[2:, 2:]

        scale_factor = 3
        resized_img = cv2.resize(borderless_gray, None,
                                 fx=scale_factor, fy=scale_factor,
                                 interpolation=cv2.INTER_CUBIC)

        HIGH_THRESHOLD_VALUE = 220
        _, processed_img = cv2.threshold(resized_img, HIGH_THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        # --- Step 1: Shave off fixed border if large enough ---
        h, w = processed_img.shape
        if h > 80 and w > 80:
            processed_img = processed_img[10:h-10, 10:w-10]

        # --- Step 2: Dilation (Thickens digits) ---
        dilation_kernel = np.ones((2, 2), np.uint8)
        dilated_img = cv2.dilate(processed_img, dilation_kernel, iterations=1)

        # --- Step 3: Sharpening (Enhances edges) ---
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        sharpened_img = cv2.filter2D(dilated_img, -1, sharpen_kernel)

        custom_config = r'--psm 13 -c tessedit_char_whitelist=0123456789'
        pil_image = Image.fromarray(sharpened_img)
        text = pytesseract.image_to_string(pil_image, config=custom_config)

        detected_number = text.strip()

        if detected_number and detected_number.isdigit():
            return detected_number
        else:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    img = cv2.imread('54.png')
    detected_number = detect_multi_digit_high_threshold(img)
    print(f"Detected Number: {detected_number}")

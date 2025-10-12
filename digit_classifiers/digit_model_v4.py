import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_number_v4(img_array, border_crop=0):
    """
    Final robust detection with parameter tuning to eliminate noise-based misclassifications 
    (like 274) and improve detection of faint single digits (like 1 and 7).
    Takes a NumPy image array (H x W x 3 or H x W) as input.
    """
    try:
        if img_array is None:
            print("Error: Input image array is None")
            return None

        # --- 0. Optional Border Crop ---
        if border_crop > 0:
            h, w = img_array.shape[:2]
            img_array = img_array[border_crop:h-border_crop, border_crop:w-border_crop]

        # Convert to grayscale if image has 3 channels
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()

        height, width = gray.shape

        # --- 1. Border Trim and Upscaling ---
        x_start, y_start = 2, 2
        borderless_gray = gray[y_start:, x_start:]

        scale_factor = 3
        resized_img = cv2.resize(borderless_gray, None,
                                 fx=scale_factor, fy=scale_factor,
                                 interpolation=cv2.INTER_CUBIC)

        # --- 2. Morphological Closing and Binarization ---
        kernel = np.ones((5, 5), np.uint8)
        closed_img = cv2.morphologyEx(resized_img, cv2.MORPH_CLOSE, kernel)
        _, processed_img = cv2.threshold(closed_img, 127, 255, cv2.THRESH_BINARY)

        processed_img = cv2.GaussianBlur(processed_img, ksize=(3, 3), sigmaX=0)
        processed_img = cv2.GaussianBlur(processed_img, ksize=(3, 3), sigmaX=0)

        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        processed_img = cv2.filter2D(processed_img, -1, sharpen_kernel)
        processed_img = cv2.filter2D(processed_img, -1, sharpen_kernel)
        processed_img = cv2.filter2D(processed_img, -1, sharpen_kernel)

        dilate_kernel = np.ones((2, 2), np.uint8) 
        processed_img = cv2.dilate(processed_img, dilate_kernel, iterations=1)

        # cv2.imshow("To OCR", processed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # --- 3. Tesseract OCR ---
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
        pil_image = Image.fromarray(processed_img)
        text = pytesseract.image_to_string(pil_image, config=custom_config)

        # --- 4. Post-processing ---
        detected_number = ''.join(filter(str.isdigit, text))
        # print("Found: ", detected_number)

        if detected_number:
            return detected_number
        else:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


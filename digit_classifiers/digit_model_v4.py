import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_number_v4(img_array, border_crop=0):
    try:
        if img_array is None:
            print("Error: Input image array is None")
            return None
        
        # --- 0. Optional Border Crop ---
        if border_crop > 0:
            h, w = img_array.shape[:2]
            img_array = img_array[border_crop:h-border_crop, border_crop:w-border_crop]
        
        # Convert to grayscale if image has 3 channels
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()
        
        # --- 1. Upscaling ---
        scale_factor = 4
        resized_img = cv2.resize(gray, None,
                                 fx=scale_factor, fy=scale_factor,
                                 interpolation=cv2.INTER_CUBIC)
        
        # --- 2. Gentler preprocessing ---
        # Apply gentle denoising
        denoised = cv2.fastNlMeansDenoising(resized_img, h=10)
        
        # Use adaptive thresholding for better handling of varying contrast
        # OR use THRESH_BINARY_INV if your digits are dark on light background
        _, processed_img = cv2.threshold(denoised, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Lighter sharpening (only once)
        sharpen_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        processed_img = cv2.filter2D(processed_img, -1, sharpen_kernel)
        
        # Optional: Add small dilation only if digits are too thin
        # dilate_kernel = np.ones((2,2), np.uint8) 
        # processed_img = cv2.dilate(processed_img, dilate_kernel, iterations=1)
        
        # Uncomment to debug
        # cv2.imshow("To OCR", processed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        # --- 3. Tesseract OCR with multiple PSM modes ---
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
        pil_image = Image.fromarray(processed_img)
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        
        # Try alternative PSM if first attempt fails
        if not text.strip():
            custom_config = r'--psm 7 -c tessedit_char_whitelist=0123456789'
            text = pytesseract.image_to_string(pil_image, config=custom_config)
        
        # --- 4. Post-processing ---
        detected_number = ''.join(filter(str.isdigit, text))
        detected_number = detected_number[:2]
        
        if detected_number:
            return detected_number
        else:
            return None
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
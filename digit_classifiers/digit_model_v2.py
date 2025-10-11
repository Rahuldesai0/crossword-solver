import cv2
import pytesseract
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detect_digit_with_preprocessing(img, border_crop=10):
    """
    Takes a cv2 image array, crops borders, applies inversion and thresholding,
    and uses Tesseract to extract a single digit.
    
    Args:
        img (np.ndarray): OpenCV image array.
        border_crop (int): Number of pixels to remove from each side.
        
    Returns:
        str: The single detected digit, or None if detection fails.
    """
    try:
        if img is None:
            print("Error: Provided image is None")
            return None

        # Crop border
        h, w = img.shape[:2]
        cropped_img = img[border_crop:h-border_crop, border_crop:w-border_crop]

        # Convert to grayscale if color
        if len(cropped_img.shape) == 3:
            gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_img

        # Apply binary inversion threshold
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Optional: small dilation
        kernel = np.ones((1, 1), np.uint8)
        processed_img = cv2.dilate(thresh, kernel, iterations=1)

        # Save processed image for debugging
        # output_path = "processed_for_tesseract.png"
        # cv2.imwrite(output_path, processed_img)
        # print(f"DEBUG: Saved processed image to {output_path}")

        # Tesseract config: single character, digits only
        custom_config = r'--psm 10 -c tessedit_char_whitelist=0123456789'
        pil_image = Image.fromarray(processed_img)
        text = pytesseract.image_to_string(pil_image, config=custom_config)

        detected_digit = text.strip()
        if detected_digit and detected_digit.isdigit():
            return detected_digit[0]
        else:
            # print(f"OCR failed. Raw output: '{text}'")
            return None

    except Exception as e:
        print(f"An error occurred during detection: {e}")
        return None


if __name__ == "__main__":
    test_image_path = r"C:\Users\Ahaan\Programming\Projects\crossword-solver\cells\cell_0_0.png"
    img = cv2.imread(test_image_path, cv2.IMREAD_UNCHANGED)

    detected_number = detect_digit_with_preprocessing(img, border_crop=12)
    print(f"Detected Digit: {detected_number}")

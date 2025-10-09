import cv2
import pytesseract
from pytesseract import Output
import numpy as np 
import sys
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def ocr_image(image, tesseract_cmd=None, lang='eng', min_size=50, debug=False, output_image_path=None):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    

    # Load image
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image}")
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise TypeError("Input must be a file path or a NumPy array")

    # Resize so minimum dimension is min_size
    img = 255 - img
    h, w = img.shape[:2]
    scale = max(1, min_size / min(h, w))
    if scale > 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

    # Grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if debug:
        cv2.imshow("Gray for OCR", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # OCR
    custom_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'  # psm 6 treats as a uniform block of text
    text = pytesseract.image_to_string(gray, config=custom_config, lang=lang)

    if output_image_path:
        cv2.imwrite(output_image_path, gray)

    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_simple.py <image_path> [tesseract_cmd] [output_image]")
        sys.exit(1)

    img_path = sys.argv[1]
    t_cmd = sys.argv[2] if len(sys.argv) >= 3 else None
    out_img = sys.argv[3] if len(sys.argv) >= 4 else "ocr_output.png"

    text = ocr_image(img_path, tesseract_cmd=t_cmd, debug=True, output_image_path=None)
    print("Recognized text:")
    print(text)
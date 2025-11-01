import cv2
import pytesseract
import numpy as np
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def detect_number_v5(img_array, border_crop=0):
    try:
        import cv2
        import numpy as np
        from PIL import Image
        import pytesseract
        cv2.imshow("To input", img_array)

        if img_array is None:
            print("Error: Input image array is None")
            return None

        if border_crop > 0:
            h, w = img_array.shape[:2]
            img_array = img_array[border_crop:h-border_crop, border_crop:w-border_crop]

        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_array.copy()

        scale_factor = 3
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        avg = np.mean(gray)
        th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        white_frac = np.mean(th == 255)
        if white_frac > 0.6:
            th = cv2.bitwise_not(th)

        def morph_reconstruction(marker, mask, kernel):
            prev = np.zeros_like(marker)
            curr = marker.copy()
            while True:
                curr = cv2.dilate(curr, kernel)
                curr = cv2.min(curr, mask)
                if np.array_equal(curr, prev):
                    break
                prev = curr.copy()
            return curr

        k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        seed = cv2.erode(th, k3, iterations=1)
        opened = morph_reconstruction(seed, th, k3)

        seed2 = cv2.dilate(opened, k3, iterations=1)
        closed = morph_reconstruction(seed2, opened, k3)
        proc = closed

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(proc, connectivity=8)
        h, w = proc.shape
        area_thresh = max(20, (h * w) // 2000)
        final_mask = np.zeros_like(proc)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            ww = stats[i, cv2.CC_STAT_WIDTH]
            hh = stats[i, cv2.CC_STAT_HEIGHT]
            if area < area_thresh:
                continue
            ar = ww / float(max(1, hh))
            if ar < 0.15 and hh < 10:
                continue
            final_mask[labels == i] = 255

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        processed_img = final_mask

        pil_image = Image.fromarray(processed_img)
        custom_config = r'--psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(pil_image, config=custom_config)

        detected_number = ''.join(filter(str.isdigit, text))[:2]
        if detected_number:
            return detected_number
        else:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

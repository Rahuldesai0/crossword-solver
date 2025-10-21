import cv2 
import numpy as np
from skimage.morphology import skeletonize
from ocr import ocr_image
from digit_classifiers.digit_model_v2 import detect_digit_with_preprocessing as detect_number_v2
from digit_classifiers.digit_model_v3 import detect_number_v3
from digit_classifiers.digit_model_v4 import detect_number_v4
from digit_classifiers.digit_model_multi import detect_multi_digit_high_threshold
import os 
import concurrent.futures

detect = detect_number_v2

def identify_numbers(
    cropped_img,
    cell_size,
    rows,
    cols,
    border_thickness,
    grid,
    debug=False,
    hardcode=False,
    save=False,
    border_crop=0
):
    print("Identifying numbers....")

    height, width = cropped_img.shape[:2]
    result = {}

    save_dir = "cells"
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cell_size -= border_thickness / 4

    def process_cell(i, j):
        if grid and grid[i][j] == 0:
            return None

        x = j * cell_size
        y = i * cell_size
        w = h = cell_size

        if x + w > width or y + h > height:
            return None

        cell = cropped_img[int(y+border_thickness*1.5):int(y+h), int(x+border_thickness*1.5):int(x+w)]
        gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if len(cell.shape) == 3 else cell.copy()
        _, binary_cell = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        if save:
            cv2.imwrite(os.path.join(save_dir, f"cell_{i}_{j}.png"), binary_cell)

        coords = cv2.findNonZero(binary_cell)
        if coords is None:
            if debug:
                print(f"Empty cell at ({i},{j})")
            return None

        x_, y_, w_, h_ = cv2.boundingRect(coords)
        padding = 3
        digit_crop = binary_cell[y_:y_+h_, x_:x_+w_]
        digit_crop = cv2.copyMakeBorder(digit_crop, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

        digit = detect(digit_crop, border_crop=border_crop)

        if debug and digit is not None:
            cv2.imshow("Cropped Digit", digit_crop)
            print(f"Digit at cell ({i},{j}) detected: {digit}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if digit is not None:
            if isinstance(digit, list):
                digit = digit[0]
            return digit, (i, j)
        return None

    # Run cells in parallel using threads
    args = [(i, j) for i in range(rows) for j in range(cols)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for res in executor.map(lambda p: process_cell(*p), args):
            if res is not None:
                digit, pos = res
                result[digit] = pos

    if hardcode:
        return {
            1:(0,0), 2:(0,1), 3:(0,2), 4:(0,3), 5:(0,4), 6:(0,5), 7:(0,7), 8:(0,8), 9:(0,9),
            10:(1,0), 11:(1,7), 12:(2,0), 13:(2,6), 14:(3,0), 15:(3,5), 16:(4,4),
            17:(5,0), 18:(5,1), 19:(5,2), 20:(5,3), 21:(6,0), 22:(6,6), 23:(6,7), 24:(6,8), 25:(6,9),
            26:(7,0), 27:(7,5), 28:(8,0), 29:(8,4), 30:(9,0), 31:(9,4)
        }

    return result

def preprocess_digit(img_bin, digit_size=28, final_size=(28, 56), sharpen=True, skeletonize_digit=True, debug=False):
    img = img_bin.copy()
    if img.max() > 1 and img.max() <= 255:
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros(final_size, dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    x_off = (size - w) // 2
    y_off = (size - h) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    resized = cv2.resize(square, (digit_size, digit_size), interpolation=cv2.INTER_AREA)

    # Erosion to remove very small isolated pixels
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    eroded = cv2.erode(resized, kernel_erode, iterations=1)

    # Morphological closing to fill small holes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    morphed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel_close)

    if sharpen:
        kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]], dtype=np.float32)
        morphed = cv2.filter2D(morphed, -1, kernel_sharp)
        morphed = np.clip(morphed, 0, 255).astype(np.uint8)

    if skeletonize_digit:
        bool_img = morphed > 0
        skeleton = skeletonize(bool_img)
        morphed = (skeleton.astype(np.uint8) * 255)

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        morphed = cv2.dilate(morphed, kernel_dilate, iterations=2)

    final = np.zeros(final_size, dtype=np.uint8)
    x_pad = (final_size[1] - digit_size) // 2
    final[:, x_pad:x_pad+digit_size] = morphed

    if debug:
        print(f"preprocess_digit: orig {img_bin.shape}, bbox {(w,h)}, digit_size {digit_size}, final_size {final_size}")

    return final


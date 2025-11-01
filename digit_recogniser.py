import cv2 
import numpy as np
from skimage.morphology import skeletonize
from ocr import ocr_image
from digit_classifiers.digit_model_v2 import detect_digit_with_preprocessing as detect_number_v2
from digit_classifiers.digit_model_v3 import detect_number_v3
from digit_classifiers.digit_model_v4 import detect_number_v4
from digit_classifiers.digit_model_v5 import detect_number_v5
from digit_classifiers.digit_model_multi import detect_multi_digit_high_threshold
import os 
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

detect = detect_number_v4

def identify_numbers_parallel(
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

def identify_numbers_serial(
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
    import os 

    result = {}
    save_dir = "cells"
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cell_size -= border_thickness / 4
    for i in range(rows):
        for j in range(cols):
            if grid and grid[i][j] == 0:
                continue  # skip this cell

            # Extract cell from full image
            x = j * cell_size
            y = i * cell_size
            w = cell_size
            h = cell_size

            if x + w > width or y + h > height:
                continue  # out of bounds

            cell = cropped_img[int(y+border_thickness*1.5):int(y+h), int(x+border_thickness*1.5):int(x+w)]

            # h_cell, w_cell = cell.shape[:2]
            # cropped_w = int(w_cell * 0.45)
            # cropped_h = int(h_cell * 0.32)
            # cell = cell[0:cropped_h, 0:cropped_w]

            # Convert to grayscale if needed
            if len(cell.shape) == 3:
                gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            else:
                gray_cell = cell.copy()

            # Threshold to binary (invert so digit is white)
            _, binary_cell = cv2.threshold(
                gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            if save:
                cell_filename = os.path.join(save_dir, f"cell_{i}_{j}.png")
                cv2.imwrite(cell_filename, binary_cell)

            # Remove white border by cropping to digit bounding box
            coords = cv2.findNonZero(binary_cell)
            if coords is not None:
                c = cv2.boundingRect(coords)
                x_, y_, w_, h_ = c
            else:
                # nothing found in this cell
                if debug:
                    print(f"Empty cell at ({i}, {j}) - no nonzero pixels")
                continue

            # if w_ < cell_size//2 and h_ < cell_size//2:

            # Small padding so strokes aren’t cut
            padding = 3
            x_start = max(0, x_)
            y_start = max(0, y_)
            x_end = min(binary_cell.shape[1], x_ + w_ )
            y_end = min(binary_cell.shape[0], y_ + h_ )

            digit_crop = binary_cell[y_start:y_end, x_start:x_end]
            digit_crop = cv2.copyMakeBorder(
                digit_crop,
                top=padding,
                bottom=padding,
                left=padding,
                right=padding,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

            # processed = preprocess_digit(digit_crop, digit_size=32, final_size=(32, 32), debug=debug)
            digit = detect(digit_crop, border_crop=border_crop)

            if debug:
                # cv2.imshow("Original Cell", cell)
                # cv2.imshow("Binary Cell", binary_cell)
                cv2.imshow("Cropped Digit", digit_crop)
                print(f"Digit size: {digit_crop.shape}")
                print(f"Digit at cell ({i}, {j}) detected: {digit}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if digit is not None:
                if isinstance(digit, str):
                    digit = int(digit)
                    result[digit] = (i, j)
                elif isinstance(digit, int):
                    result[digit] = (i, j)
                elif isinstance(digit, list):
                    result[digit[0]] = (i, j)
            # else:
            #     if debug:
            #         print(f"Empty cell at ({i}, {j})")
            #         cv2.imshow("Original Cell", cell)
            #         cv2.imshow("Empty Cell", binary_cell)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()


    if hardcode:
        # results for img1.jpg
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

def identify_numbers_serial_v2(
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
    import os 
    result = {}
    save_dir = "cells"
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Don't modify cell_size - keep original for calculations
    original_cell_size = cell_size

    for i in range(rows):
        for j in range(cols):
            if grid and grid[i][j] == 0:
                continue  # skip this cell
            
            # Calculate cell position with border offset
            x = int(j * original_cell_size + border_thickness)
            y = int(i * original_cell_size + border_thickness)
            w = int(original_cell_size - border_thickness * 2)
            h = int(original_cell_size - border_thickness * 2)
            
            # Bounds checking
            if x + w > width or y + h > height or x < 0 or y < 0:
                continue
            
            # Extract cell from full image
            cell = cropped_img[y:y+h, x:x+w]
            
            if cell.size == 0:
                continue
            
            # Convert to grayscale if needed
            if len(cell.shape) == 3:
                gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            else:
                gray_cell = cell.copy()
            
            # Threshold to binary (invert so digit is white)
            _, binary_cell = cv2.threshold(
                gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            
            if save:
                cell_filename = os.path.join(save_dir, f"cell_{i}_{j}.png")
                cv2.imwrite(cell_filename, binary_cell)
            
            # Find digit bounding box
            coords = cv2.findNonZero(binary_cell)
            if coords is None:
                if debug:
                    print(f"Empty cell at ({i}, {j}) - no nonzero pixels")
                continue
            
            # Get bounding rectangle of the digit
            x_, y_, w_, h_ = cv2.boundingRect(coords)
            
            # Crop to the digit with small padding
            padding = 3
            x_start = max(0, x_ - padding)
            y_start = max(0, y_ - padding)
            x_end = min(binary_cell.shape[1], x_ + w_ + padding)
            y_end = min(binary_cell.shape[0], y_ + h_ + padding)
            
            # Extract just the digit region
            digit_crop = binary_cell[y_start:y_end, x_start:x_end]
            
            if digit_crop.size == 0:
                continue
            
            # Add uniform border around the cropped digit
            digit_crop = cv2.copyMakeBorder(
                digit_crop,
                top=padding,
                bottom=padding,
                left=padding,
                right=padding,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )
            
            # Detect digit
            digit = detect(digit_crop, border_crop=border_crop)
            
            if debug:
                cv2.imshow("Original Cell", cell)
                cv2.imshow("Binary Cell", binary_cell)
                cv2.imshow("Cropped Digit", digit_crop)
                print(f"Cell ({i},{j}): Digit={digit}, BBox=({x_},{y_},{w_},{h_}), CropShape={digit_crop.shape}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            if digit is not None:
                # Normalize digit to integer
                if isinstance(digit, str):
                    digit = int(digit)
                elif isinstance(digit, list):
                    digit = digit[0]
                
                if isinstance(digit, int):
                    result[digit] = (i, j)
            # else:
            #     if debug:
            #         print(f"Empty cell at ({i}, {j})")
            #         cv2.imshow("Original Cell", cell)
            #         cv2.imshow("Empty Cell", binary_cell)
            #         cv2.waitKey(0)
            #         cv2.destroyAllWindows()


    if hardcode:
        # results for img1.jpg
        return {
            1:(0,0), 2:(0,1), 3:(0,2), 4:(0,3), 5:(0,4), 6:(0,5), 7:(0,7), 8:(0,8), 9:(0,9),
            10:(1,0), 11:(1,7), 12:(2,0), 13:(2,6), 14:(3,0), 15:(3,5), 16:(4,4),
            17:(5,0), 18:(5,1), 19:(5,2), 20:(5,3), 21:(6,0), 22:(6,6), 23:(6,7), 24:(6,8), 25:(6,9),
            26:(7,0), 27:(7,5), 28:(8,0), 29:(8,4), 30:(9,0), 31:(9,4)
        }

    return result    

def identify_numbers_parallel_v2(
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

    original_cell_size = cell_size

    # Worker function for one cell
    def process_cell(i, j):
        try:
            if grid and grid[i][j] == 0:
                return None

            x = int(j * original_cell_size + border_thickness)
            y = int(i * original_cell_size + border_thickness)
            w = int(original_cell_size - border_thickness * 2)
            h = int(original_cell_size - border_thickness * 2)

            if x + w > width or y + h > height or x < 0 or y < 0:
                return None

            cell = cropped_img[y:y+h, x:x+w]
            if cell.size == 0:
                return None

            if len(cell.shape) == 3:
                gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            else:
                gray_cell = cell.copy()

            _, binary_cell = cv2.threshold(
                gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            if save:
                cv2.imwrite(os.path.join(save_dir, f"cell_{i}_{j}.png"), binary_cell)

            coords = cv2.findNonZero(binary_cell)
            if coords is None:
                if debug:
                    print(f"Empty cell at ({i}, {j}) - no nonzero pixels")
                return None

            x_, y_, w_, h_ = cv2.boundingRect(coords)
            padding = 3
            x_start = max(0, x_ - padding)
            y_start = max(0, y_ - padding)
            x_end = min(binary_cell.shape[1], x_ + w_ + padding)
            y_end = min(binary_cell.shape[0], y_ + h_ + padding)
            digit_crop = binary_cell[y_start:y_end, x_start:x_end]

            if digit_crop.size == 0:
                return None

            digit_crop = cv2.copyMakeBorder(
                digit_crop, top=padding, bottom=padding, left=padding, right=padding,
                borderType=cv2.BORDER_CONSTANT, value=0
            )

            digit = detect(digit_crop, border_crop=border_crop)

            if debug:
                cv2.imshow("Original Cell", cell)
                cv2.imshow("Binary Cell", binary_cell)
                cv2.imshow("Cropped Digit", digit_crop)
                print(f"Cell ({i},{j}): Digit={digit}, BBox=({x_},{y_},{w_},{h_}), CropShape={digit_crop.shape}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if digit is not None:
                if isinstance(digit, str):
                    digit = int(digit)
                elif isinstance(digit, list):
                    digit = digit[0]

                if isinstance(digit, int):
                    return (digit, (i, j))
            return None

        except Exception as e:
            if debug:
                print(f"Error in cell ({i},{j}): {e}")
            return None

    # Parallel execution
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(rows):
            for j in range(cols):
                futures.append(executor.submit(process_cell, i, j))

        for f in as_completed(futures):
            res = f.result()
            if res:
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
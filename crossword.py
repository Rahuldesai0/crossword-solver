import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from llm.github_model_solver import GitHubModelSolver
from llm.hf_solver import HuggingFaceModelSolver
from ocr import ocr_image

model_choice = 1  # set 1 or 2

model = None
try:
    if model_choice == 1:
        from two_digit_classifier.model3.two_digit.model import CNN
        model = CNN(num_classes=100)
        model_path = './two_digit_classifier/model3/two_digit/model_combined.pth'
    elif model_choice == 2:
        from two_digit_classifier.model2.two_digit.model import CNN
        model = CNN(num_classes=100)
        model_path = './two_digit_classifier/model2/two_digit/model1.pth'
    else:
        raise ValueError("Invalid model_choice. Use 1 or 2.")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
except Exception as e:
    print("Warning: could not load model")
    print(f"Model load error: {e}")
    exit(-1)

"""Crossword Grid Identification"""
def preProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 1)
    return cv2.adaptiveThreshold(blur, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

def estimate_cell_size(cropped_img):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 0.8 < w / h < 1.2 and 10 < w < 200:
            cell_rects.append((x, y, w, h))

    if not cell_rects:
        return None

    # Sort by top to bottom, then left to right
    cell_rects = sorted(cell_rects, key=lambda b: (b[1], b[0]))

    # Cluster into rows
    rows = []
    current_row = []
    row_y_thresh = 10  # vertical tolerance

    for rect in cell_rects:
        x, y, w, h = rect
        if not current_row:
            current_row.append(rect)
        else:
            last_y = current_row[-1][1]
            if abs(y - last_y) < row_y_thresh:
                current_row.append(rect)
            else:
                rows.append(current_row)
                current_row = [rect]
    if current_row:
        rows.append(current_row)

    # Estimate cell size as average w/h
    all_ws = [w for row in rows for (_, _, w, _) in row]
    all_hs = [h for row in rows for (_, _, _, h) in row]
    avg_w = int(np.mean(all_ws))
    avg_h = int(np.mean(all_hs))

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows)

    return avg_w, avg_h, num_rows, num_cols

def estimate_border_thickness(binary_img, cell_w, cell_h):
    h, w = binary_img.shape
    cell_w = int(cell_w)
    cell_h = int(cell_h)
    border_samples = []

    for y in range(0, h - cell_h + 1, cell_h):
        for x in range(0, w - cell_w + 1, cell_w):
            cx = x + cell_w // 2
            cy = y + cell_h // 2

            if binary_img[cy, cx] == 0:  # white cell center
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    count = 0
                    i = 1
                    while i <= max(cell_w, cell_h) // 2:
                        nx = cx + dx * i
                        ny = cy + dy * i

                        if not (0 <= nx < w and 0 <= ny < h):
                            break

                        if binary_img[ny, nx] == 255:
                            count += 1
                            i += 1
                        elif binary_img[ny, nx] == 0:
                            if count > 0:
                                border_samples.append(count)
                            break
                        else:
                            break

    return int(np.median(border_samples)) if border_samples else -1

def classify_grid(cropped_img, num_rows, num_cols, debug=False, resize=None):
    if resize:
        cropped_img = cv2.resize(cropped_img, resize, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    grid = []
    debug_img = cropped_img.copy()

    cell_h = cropped_img.shape[0] // num_rows
    cell_w = cropped_img.shape[1] // num_cols

    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            cx = x1 + cell_w // 2
            cy = y1 + cell_h // 2
            pixel = gray[cy, cx]

            value = 0 if pixel < 150 else 1
            row.append(value)

            if debug:
                color = (0, 0, 255) if value == 0 else (0, 255, 0)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 1)
                cv2.circle(debug_img, (cx, cy), 2, color, -1)
        grid.append(row)

    return debug_img, grid, cell_w, cell_h



"""Corner Digit Detection"""
def identify_numbers(cropped_img, cell_size, rows, cols, border_thickness, grid, debug=False, hardcode=False, ocr=False, save=False):
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

            if ocr:
                text = predict_image(binary_cell, model=model, ocr=ocr)
                print("Text result from OCR: ", text)
                return 

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

            if w_ < cell_size//2 and h_ < cell_size//2:

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

                # processed_just_increase = cv2.resize(digit_crop, (28, 28), interpolation=cv2.INTER_AREA)
                # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                # processed_just_increase = cv2.morphologyEx(processed_just_increase, cv2.MORPH_OPEN, kernel)
                # processed_just_increase = cv2.morphologyEx(processed_just_increase, cv2.MORPH_CLOSE, kernel)
                # _, processed_just_increase = cv2.threshold(processed_just_increase, 67, 255, cv2.THRESH_BINARY)
                # processed = processed_just_increase

                # --- New preprocessing: morphological clean-up + resize to 16x16 ---
                processed = preprocess_digit(digit_crop, digit_size=32, final_size=(32, 32), debug=debug)
                # Predict (predict_image will up/downscale as necessary for the model)
                if model is not None:
                    digit = predict_image(processed, model)
                else:
                    digit = None
                    if debug:
                        print("Model not available, skipping prediction. Processed image shown for inspection.")

                if debug:
                    # cv2.imshow("Original Cell", cell)
                    # cv2.imshow("Binary Cell", binary_cell)
                    cv2.imshow("Cropped Digit", digit_crop)
                    cv2.imshow("Processed Digit (16x16)", processed)
                    print(f"Digit size: {digit_crop.shape}")
                    print(f"Digit at cell ({i}, {j}) detected: {digit}")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                
                result[(i, j)] = digit[0]

            else:
                if debug:
                    print(f"Empty cell at ({i}, {j})")
                    cv2.imshow("Original Cell", cell)
                    cv2.imshow("Empty Cell", binary_cell)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()


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

def predict_image(img_np, model, model_choice=1, min_size=10, ocr=False):
    if ocr:
        text, data, img = ocr_image(img_np)
        return text

    target_h = 28
    target_w = 56 if model_choice == 2 else 28

    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img_np.copy()

    img_resized = cv2.resize(img_gray, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_arr = 255.0 - img_resized.astype(np.float32)

    # Threshold to binary for connected components
    _, img_bin = cv2.threshold(img_arr, 10, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin.astype(np.uint8))

    # If all components are too small, return -1
    large_components = [s for i, s in enumerate(stats[1:], start=1) if s[cv2.CC_STAT_AREA] >= min_size]
    if len(large_components) == 0:
        return [-1]

    img_tensor = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        values, indices = torch.topk(outputs, k=5, dim=1)

    return indices[0].cpu().tolist()

def compute_lengths_and_intersections(grid, numbers, hints):
    rows, cols = len(grid), len(grid[0])

    def get_length_and_coords(start, direction):
        r, c = start
        coords = []
        if direction == "across":
            while c < cols and grid[r][c] != 0:
                coords.append((r, c))
                c += 1
        else:
            while r < rows and grid[r][c] != 0:
                coords.append((r, c))
                r += 1
        return len(coords), coords

    clue_coords = {}
    result = {}

    for num, dirs in hints.items():
        result[num] = {}
        clue_coords[num] = {}
        for direction, hint_text in dirs.items():
            length, coords = get_length_and_coords(numbers[num], direction)
            clue_coords[num][direction] = coords
            result[num][direction] = {
                "hint": hint_text,
                "length": length,
                "intersections": []
            }

    # Compute intersections
    for num1, dirs1 in clue_coords.items():
        for dir1, coords1 in dirs1.items():
            intersections = []
            for num2, dirs2 in clue_coords.items():
                if num1 == num2:
                    continue
                for dir2, coords2 in dirs2.items():
                    if dir1 == dir2:
                        continue
                    overlap = set(coords1) & set(coords2)
                    for r, c in overlap:
                        intersections.append((num2, dir2, (r, c)))
            result[num1][dir1]["intersections"] = intersections

    return result

def overlay_grid(grid_img, grid, numbers, final_hints, result):
    # Convert solution keys to integers
    for direction in ["across", "down"]:
        if direction in result.get("solutions", {}):
            result["solutions"][direction] = {int(k): v for k, v in result["solutions"][direction].items()}

    overlay_img = grid_img.copy()
    rows, cols = len(grid), len(grid[0])
    cell_h = overlay_img.shape[0] // rows
    cell_w = overlay_img.shape[1] // cols

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(cell_w, cell_h) / 30
    thickness = 1

    filled = [[None for _ in range(cols)] for _ in range(rows)]

    total_letters = 0
    conflict_count = 0

    for direction in ['across', 'down']:
        for num, word in result['solutions'].get(direction, {}).items():
            if num not in numbers:
                continue
            start_r, start_c = numbers[num]
            for idx, char in enumerate(word):
                r, c = start_r, start_c
                if direction == 'across':
                    c += idx
                else:
                    r += idx

                if 0 <= r < rows and 0 <= c < cols and grid[r][c] != 0:
                    total_letters += 1
                    if filled[r][c] is None:
                        x1 = c * cell_w
                        y1 = r * cell_h

                        text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
                        text_x = x1 + (cell_w - text_size[0]) // 2
                        text_y = y1 + (cell_h + text_size[1]) // 2

                        cv2.putText(overlay_img, char, (text_x, text_y),
                                    font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
                        filled[r][c] = char
                    elif filled[r][c] != char:
                        conflict_count += 1

    conflict_percentage = (conflict_count / total_letters) * 100 if total_letters > 0 else 0
    print(f"Total letters: {total_letters}, Conflicts: {conflict_count}, Conflict %: {conflict_percentage:.2f}%")
    return overlay_img, conflict_percentage



if __name__ == "__main__":
    path = "img3.jpg"
    img = cv2.imread(path)
    if img is None:
        print("Error loading image.")
        exit()

    # 1. Preprocess
    thresh = preProcess(img)

    # 2. Find largest contour (grid)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        exit()

    largest = max(contours, key=cv2.contourArea)

    # 3. Isolate grid
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    isolated = cv2.bitwise_and(img, mask_rgb)

    # 4. Crop grid (add 2 px padding)
    x, y, w, h = cv2.boundingRect(largest)
    cropped_isolated = isolated[y:y+h+2, x:x+w+2]

    W, H = cropped_isolated.shape[:2]
    W -= 2
    H -= 2

    # 5. Grayscale and binarize cropped image
    gray_cropped = cv2.cvtColor(cropped_isolated, cv2.COLOR_BGR2GRAY)
    binary_cropped = cv2.adaptiveThreshold(gray_cropped, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 2)

    # 6. Estimate cell size
    cell_w, cell_h, rows, cols = estimate_cell_size(cropped_isolated)

    # 7. Estimate border thickness
    border_thickness = estimate_border_thickness(binary_cropped, cell_w, cell_h)

    cell_w += border_thickness
    cell_h += border_thickness
    rows = round(W/cell_w)
    cols = round(H/cell_h)
    print(f"Estimated Rows: {rows}, Columns: {cols}, Cell W: {cell_w:.2f}, Cell H: {cell_h:.2f}")
    print("Estimated Border Thickness:", border_thickness)
    grid_img, grid, cell_w_ind, cell_h_ind = classify_grid(cropped_isolated, rows, cols, debug=True, resize=None)

    for r in grid:
        print(r)
    0
    # 8. Show images
    # cv2.imshow("Original", img)
    # cv2.imshow("Cropped Isolated", cropped_isolated)
    cv2.imshow("Binary Cropped", binary_cropped)
    cv2.imshow("Grid classification", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    cell_size = grid_img.shape[0] // rows  # assuming square cells; same as height / rows
    border_thickness = 2  # adjust if needed

    numbers = identify_numbers(
        cropped_img=cropped_isolated,
        cell_size=cell_size,
        rows=rows,
        cols=cols,
        border_thickness=border_thickness,
        grid=grid,
        debug=True,
        hardcode=True,
        ocr=False,
        save=False
    )

    # Now we have grid numbers
    # hints = {
    #     1: {"across": "Computer on a cafe table, probably", "down": "Speech mannerism"},
    #     2: {"down": "Cambodia’s continent"},
    #     3: {"down": "Hunger twinge"},
    #     4: {"down": "Little kid"},
    #     5: {"down": "Twelve minus eleven"},
    #     6: {"down": "Wormhole in a sci-fi movie, e.g."},
    #     7: {"across": "“Sherlock” network", "down": "General Electric or General Mills"},
    #     8: {"down": "Ship’s lower area"},
    #     9: {"down": "Printer type"},
    #     10: {"across": "Forget it!"},
    #     11: {"across": "City home to the Copacabana"},
    #     12: {"across": "Tough pitch to hit"},
    #     13: {"down": "Disallow"},
    #     14: {"across": "Kindle display"},
    #     15: {"across": "Dance for duos"},
    #     16: {"across": "Allergen from a pet", "down": "Kalahari or Gobi"},
    #     17: {"across": "Actor Craig of “Casino Royale”", "down": "“Same here!”"},
    #     18: {"down": "Farewell, to François"},
    #     19: {"down": "Song for nine voices"},
    #     20: {"down": "Sort or kind"},
    #     21: {"across": "Objects of worship"},
    #     22: {"across": "Greatly impresses", "down": "“Take a Chance on Me” singers"},
    #     23: {"down": "Halloween decorations"},
    #     24: {"down": "Model Macpherson"},
    #     25: {"down": "Downhill racer"},
    #     26: {"across": "Pixie that responds to clapping"},
    #     27: {"down": "Feel remorse"},
    #     28: {"across": "Golfer’s peg"},
    #     29: {"across": "Demolition site debris"},
    #     30: {"across": "Away from home"},
    #     31: {"across": "Kidded around"}
    # }

    # final_hints = compute_lengths_and_intersections(grid, numbers, hints)
    # # print(final_hints)


    # print("JSON Result: ")
    # # solver = HuggingFaceModelSolver(final_hints, model_name="Qwen/Qwen3-VL-235B-A22B-Instruct:novita")
    # solver = GitHubModelSolver(final_hints, model_name="openai/gpt-5")
    # result = solver.solve()
    # print(result)

    # # for dummy
    # solved_img, conflict = overlay_grid(cropped_isolated, grid, numbers, final_hints, result)
    # print("Conflict: ", conflict)
    # cv2.imshow("Solved Crossword", solved_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

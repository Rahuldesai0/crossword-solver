import cv2
import numpy as np

def preProcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 1)
    return cv2.adaptiveThreshold(blur, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)

# Estimate cell size
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



def classify_grid(cropped_img, cell_w, cell_h, num_rows, num_cols, debug=False):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    grid = []
    debug_img = cropped_img.copy()

    for r in range(num_rows):
        row = []
        for c in range(num_cols):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h

            if x2 > gray.shape[1] or y2 > gray.shape[0]:
                row.append(-1)
                continue

            cx = x1 + cell_w // 2
            cy = y1 + cell_h // 2
            pixel = gray[cy, cx]

            value = 0 if pixel < 50 else 1
            row.append(value)

            # Debug: Draw cell border and classification
            if debug:
                color = (0, 0, 255) if value == 0 else (0, 255, 0)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 1)
                cv2.circle(debug_img, (cx, cy), 2, color, -1)
        grid.append(row)

    return debug_img, grid

def identify_numbers(cropped_img, cell_size, rows, cols, border_thickness, grid, debug=False):
    height, width = cropped_img.shape[:2]

    for i in range(rows):
        for j in range(cols):
            if grid and grid[i][j] == 0:
                continue  # skip this cell

            x = j * cell_size + border_thickness
            y = i * cell_size + border_thickness
            w = cell_size - 3 * border_thickness
            h = cell_size - 3 * border_thickness

            if x + w > width or y + h > height:
                continue  # out of bounds

            cell = cropped_img[y:y+h, x:x+w]

            # Convert to grayscale if needed
            if len(cell.shape) == 3:
                gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            else:
                gray_cell = cell.copy()

            # Threshold to binary
            _, binary_cell = cv2.threshold(gray_cell, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find black pixel locations
            coords = cv2.findNonZero(binary_cell)
            if coords is not None:
                x_, y_, w_, h_ = cv2.boundingRect(coords)
                padding = 2
                x_start = max(0, x_ - padding)
                y_start = max(0, y_ - padding)
                x_end = min(binary_cell.shape[1], x_ + w_ + padding)
                y_end = min(binary_cell.shape[0], y_ + h_ + padding)
                digit_crop = binary_cell[y_start:y_end, x_start:x_end]

                if debug:
                    cv2.imshow("Original Cell", cell)
                    cv2.imshow("Digit", digit_crop)
                    cv2.imshow("Cell", binary_cell)
                    print(f"Digit at cell ({i}, {j})")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                if debug:
                    cv2.imshow("Empty Cell", binary_cell)
                    print(f"Empty cell at ({i}, {j})")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()



if __name__ == "__main__":
    path = input("Enter image path: ")
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
    grid_img, grid = classify_grid(cropped_isolated, cell_w, cell_h, rows, cols, debug=True)
    for r in grid:
        print(r)
    
    # 8. Show images
    cv2.imshow("Original", img)
    cv2.imshow("Cropped Isolated", cropped_isolated)
    cv2.imshow("Binary Cropped", binary_cropped)
    cv2.imshow("Grid classification", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    numbers = identify_numbers(cropped_isolated, cell_w, rows, cols, border_thickness, grid, debug=True)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

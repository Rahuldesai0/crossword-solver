import cv2 
import numpy as np 

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

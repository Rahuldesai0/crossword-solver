import cv2
import numpy as np
from grid_classifier import preProcess, estimate_cell_size, estimate_border_thickness, classify_grid
from digit_recogniser import identify_numbers_parallel, identify_numbers_serial, identify_numbers_serial_v2, identify_numbers_parallel_v2
from generate_json import compute_lengths_and_intersections, compute_lengths_and_intersections_parallel
from llm.solvers import GitHubModelSolver, HuggingFaceModelSolver, GeminiSolver
from overlay_grid import overlay_grid, overlay_grid_parallel
import json

import time 

path = input("Enter path: ")
solve = True if input("Solve? (yes/no): ") == 'yes' else False
debugGrid = True if input("Debug grid? (yes/no): ") == 'yes' else False
debugNumbers = True if input("Debug numbers? (yes/no): ") == 'yes' else False
border_crop=10
if solve:
    use_solver = input("Which solver?: ")

img = cv2.imread(path)
if img is None:
    print("Error loading image.")
    exit()

# Preprocess
thresh = preProcess(img)

# Find largest contour (grid)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No contours found.")
    exit()

largest = max(contours, key=cv2.contourArea)

# Isolate grid
mask = np.zeros(thresh.shape, dtype=np.uint8)
cv2.drawContours(mask, [largest], -1, 255, cv2.FILLED)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
isolated = cv2.bitwise_and(img, mask_rgb)

# Crop grid (add 2 px padding)
x, y, w, h = cv2.boundingRect(largest)
cropped_isolated = isolated[y:y+h+2, x:x+w+2]

W, H = cropped_isolated.shape[:2]
W -= 2
H -= 2

# Grayscale and binarize cropped image
gray_cropped = cv2.cvtColor(cropped_isolated, cv2.COLOR_BGR2GRAY)
binary_cropped = cv2.adaptiveThreshold(gray_cropped, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Estimate cell size
cell_w, cell_h, rows, cols = estimate_cell_size(cropped_isolated)

# Estimate border thickness
border_thickness = estimate_border_thickness(binary_cropped, cell_w, cell_h)

cell_w += border_thickness
cell_h += border_thickness
rows = round(W/cell_w)
cols = round(H/cell_h)
print(f"Estimated Rows: {rows}, Columns: {cols}, Cell W: {cell_w:.2f}, Cell H: {cell_h:.2f}")
print("Estimated Border Thickness:", border_thickness)
grid_img, grid, cell_w_ind, cell_h_ind = classify_grid(cropped_isolated, rows, cols, debug=True, resize=None)
print(len(grid_img), len(grid_img[0]))


# Show images
if debugGrid:
    for r in grid:
        print(r)
    cv2.imshow("Original", img)
    cv2.imshow("Cropped Isolated", cropped_isolated)
    cv2.imshow("Binary Cropped", binary_cropped)
    cv2.imshow("Grid classification", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


cell_size = grid_img.shape[0] // rows  # assuming square cells; same as height / rows
border_thickness = 2  # adjust if needed

start_time = time.time()
numbers = identify_numbers_parallel(
    cropped_img=cropped_isolated,
    cell_size=cell_size,
    rows=rows,
    cols=cols,
    border_thickness=border_thickness,
    grid=grid,
    debug=debugNumbers,
    hardcode=False,
    save=False,
    border_crop=border_crop
)
end_time = time.time()
print(f"identify_numbers took {end_time - start_time:.4f} seconds")

print(numbers)
print(f"Identified: {len(numbers)} numbers")
numbers = {str(k): v for k, v in numbers.items()}

with open('./json/img3_hints.json', 'r', encoding='utf-8') as f:
    hints = json.load(f)

hints = {str(k): v for k, v in hints.items()}
# Step 2: Compute lengths and intersections
start_time = time.time()
final_hints = compute_lengths_and_intersections(grid, numbers, hints)
end_time = time.time()
print(f"compute_lengths_and_intersections took {end_time - start_time:.4f} seconds")
print(final_hints)

if solve:
    if use_solver == "huggingface":
        solver = HuggingFaceModelSolver(final_hints, model_name="Qwen/Qwen3-VL-235B-A22B-Instruct:novita")
    elif use_solver == "github":
        solver = GitHubModelSolver(final_hints, model_name="openai/gpt-4o")
    elif use_solver == "gemini":
        solver = GeminiSolver(final_hints)
    else:
        raise ValueError("Invalid solver type specified")

    # For testing
    result = solver.solve()

    # Step 3: Overlay grid
    start_time = time.time()
    solved_img, conflict = overlay_grid(cropped_isolated, grid, numbers, final_hints, result)
    end_time = time.time()
    print(f"overlay_grid took {end_time - start_time:.4f} seconds")

    print("Conflict: ", conflict)
    cv2.imshow("Solved Crossword", solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

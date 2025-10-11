import cv2
import numpy as np
from grid_classifier import preProcess, estimate_cell_size, estimate_border_thickness, classify_grid
from digit_recogniser import identify_numbers
from generate_json import compute_lengths_and_intersections
from llm.solvers import GitHubModelSolver, HuggingFaceModelSolver, GeminiSolver
from overlay_grid import overlay_grid
import json

solve = True
debug = False

path = "test/img3.jpg"
use_solver = "github"
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

for r in grid:
    print(r)
0
# Show images
if debug:
    cv2.imshow("Original", img)
    cv2.imshow("Cropped Isolated", cropped_isolated)
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
    debug=False,
    hardcode=False,
    save=False
)

print(numbers)
if solve:
    with open('./json/img3_hints.json', 'r', encoding='utf-8') as f:
        hints = json.load(f)

    final_hints = compute_lengths_and_intersections(grid, numbers, hints)
    # print(final_hints)


    print("JSON Result: ")
    if use_solver == "huggingface":
        solver = HuggingFaceModelSolver(final_hints, model_name="Qwen/Qwen3-VL-235B-A22B-Instruct:novita")
    elif use_solver == "github":
        solver = GitHubModelSolver(final_hints, model_name="openai/gpt-5")
    elif use_solver == "gemini":
        solver = GeminiSolver(final_hints)
    else:
        raise ValueError("Invalid solver type specified")

    result = solver.solve()
    print(result)
    # for dummy
    solved_img, conflict = overlay_grid(cropped_isolated, grid, numbers, final_hints, result)
    print("Conflict: ", conflict)
    cv2.imshow("Solved Crossword", solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.waitKey(0)
    cv2.destroyAllWindows()

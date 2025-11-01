import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

def overlay_grid_parallel(grid_img, grid, numbers, final_hints, result):
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

    def draw_word(direction, num, word):
        nonlocal total_letters, conflict_count
        if num not in numbers:
            return
        start_r, start_c = numbers[num]
        local_conflicts = 0
        local_letters = 0

        for idx, char in enumerate(word):
            r, c = start_r, start_c
            if direction == 'across':
                c += idx
            else:
                r += idx

            if 0 <= r < rows and 0 <= c < cols and grid[r][c] != 0:
                local_letters += 1
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
                    local_conflicts += 1

        return local_letters, local_conflicts

    tasks = []
    with ThreadPoolExecutor() as executor:
        for direction in ['across', 'down']:
            for num, word in result['solutions'].get(direction, {}).items():
                tasks.append(executor.submit(draw_word, direction, num, word))

        for task in tasks:
            res = task.result()
            if res:
                letters, conflicts = res
                total_letters += letters
                conflict_count += conflicts

    conflict_percentage = (conflict_count / total_letters) * 100 if total_letters > 0 else 0
    print(f"Total letters: {total_letters}, Conflicts: {conflict_count}, Conflict %: {conflict_percentage:.2f}%")
    return overlay_img, conflict_percentage

def overlay_grid(grid_img, grid, numbers, final_hints, result):
    # Ensure numbers and solution keys are integers
    numbers = {int(k): v for k, v in numbers.items()}
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

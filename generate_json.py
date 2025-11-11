from concurrent.futures import ThreadPoolExecutor

def compute_lengths_and_intersections(grid, numbers, hints):
    rows, cols = len(grid), len(grid[0])
    hints = {int(k): v for k, v in hints.items()}
    numbers = {int(k): v for k, v in numbers.items()}

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

    # Compute intersections textually
    for num1, dirs1 in clue_coords.items():
        for dir1, coords1 in dirs1.items():
            intersections = []
            for num2, dirs2 in clue_coords.items():
                if num1 == num2:
                    continue
                for dir2, coords2 in dirs2.items():
                    if dir1 == dir2:
                        continue
                    for idx1, pos1 in enumerate(coords1):
                        if pos1 in coords2:
                            idx2 = coords2.index(pos1)
                            intersections.append(
                                f"Intersects with {num2} ({dir2}) at position {idx1+1} (this) and position {idx2+1} (that)"
                            )
            result[num1][dir1]["intersections"] = intersections

    return result
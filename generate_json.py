from concurrent.futures import ThreadPoolExecutor

def compute_lengths_and_intersections_parallel(grid, numbers, hints):
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

    print(numbers)
    for num, dirs in hints.items():
        result[num] = {}
        clue_coords[num] = {}
        for direction, hint_text in dirs.items():
            try:
                value = numbers[num]
            except KeyError:
                try:
                    value = numbers[str(num)]
                except KeyError:
                    print("Error: Number keys do not exist in the dictionary.")
                    exit(1)

                
            length, coords = get_length_and_coords(value, direction)
            clue_coords[num][direction] = coords
            result[num][direction] = {
                "hint": hint_text,
                "length": length,
                "intersections": []
            }

    def compute_intersections_for_clue(clue_item):
        num1, dir1, coords1 = clue_item
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
        return num1, dir1, intersections

    tasks = [(num1, dir1, coords1)
             for num1, dirs1 in clue_coords.items()
             for dir1, coords1 in dirs1.items()]

    with ThreadPoolExecutor() as executor:
        for num1, dir1, intersections in executor.map(compute_intersections_for_clue, tasks):
            result[num1][dir1]["intersections"] = intersections

    return result

def compute_lengths_and_intersections(grid, numbers, hints):
    rows, cols = len(grid), len(grid[0])

    # Convert hints and numbers keys to int
    hints = {int(k): v for k, v in hints.items()}
    numbers = {int(k): v for k,v in numbers.items()}

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

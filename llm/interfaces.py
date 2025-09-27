from abc import ABC, abstractmethod

class CrosswordSolverModel(ABC):
    def __init__(self, input_json):
        self.input = input_json
        self.prompt = f"""
You are given a crossword puzzle in JSON format.
The crossword consists of a grid and a set of clues (across and down).

Your task:

1. Solve all the clues.
2. Fill the answers into the correct positions in the grid.
3. Return only a JSON object containing:

* `"solutions"`: all solved answers keyed by their clue number, separated into `"across"` and `"down"`.
* `"grid_filled"`: the final grid with all letters filled in (uppercase A–Z, `#` for black squares).

Rules:

* Do not output explanations, reasoning, or extra text.
* Output strictly valid JSON that matches this schema:

```json
{{
  "solutions": {{
    "across": {{ "clue_number": "ANSWER" }},
    "down":   {{ "clue_number": "ANSWER" }}
  }},
  "grid_filled": [
    ["A", "B", ".", "#", "C", "D"],
    ["E", "#", "F", ".", ".", "."]
  ]
}}
````

Here is the crossword puzzle input:

```json
{self.input}
```

Return **only** the JSON.
"""

    @abstractmethod
    def solve(self, input):
        pass
from abc import ABC, abstractmethod
import os

class CrosswordSolverModel(ABC):
    def __init__(self, input_json):
        self.input_json = input_json

        prompt_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
        with open(prompt_path, "r") as f:
            self.prompt = f.read()

        self.full_prompt = (
            f"{self.prompt}\n\n"
            f"Here is the crossword puzzle input:\n```json\n{self.input_json}\n```\n"
        )

    @abstractmethod
    def solve(self):
        pass

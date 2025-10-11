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

    def remove_backticks(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    @abstractmethod
    def solve(self):
        pass

import os
import time
from .interfaces import CrosswordSolverModel
import google.generativeai as genai
from dotenv import load_dotenv
import json

class GeminiSolver(CrosswordSolverModel):
    def __init__(self, input_json, model_name="gemini-2.5-pro"):
        super().__init__(input_json)
        self.model_name = model_name
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
    
    def solve(self):
        print(f"Solving with {self.model_name}: ")
        start_time = time.time()
        
        response = self.model.generate_content(
            self.full_prompt + "Do not add backticks either, just the JSON."
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"Time taken: {elapsed:.2f} seconds")

        text = response.text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return eval(text)
            except Exception:
                return text

if __name__ == '__main__':
    input_json = """
    {
    "1down": {
        "length": 4,
        "intersections": [["3across", 6], ["5across", 2]],
        "hint": "To disappear gradually"
    },
    "2down": {
        "length": 4,
        "intersections": [["5across", 7], ["4across", 10]],
        "hint": "To change from a solid to a liquid"
    },
    "3across": {
        "length": 8,
        "intersections": [["1down", 2], ["5across", 4]],
        "hint": "Neighbour of Laos and Cambodia"
    },
    "4across": {
        "length": 13,
        "intersections": [["2down", 4]],
        "hint": "The stargazers' favourite pastime, connecting the dots in the night sky"
    },
    "5across": {
        "length": 7,
        "intersections": [["1down", 4], ["2down", 2]],
        "hint": "Turn waste into new materials or products"
    }
    }
    """

    fsolver = GeminiSolver(input_json=input_json, model_name="gemini-2.5-flash")
    fresult = fsolver.solve()
    print("Flash Result:", fresult)

    psolver = GeminiSolver(input_json=input_json, model_name="gemini-2.5-pro")
    presult = psolver.solve()
    print("Pro Result:", presult)

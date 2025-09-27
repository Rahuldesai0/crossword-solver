import os
import requests
from .interfaces import CrosswordSolverModel
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

class GitHubModelSolver(CrosswordSolverModel):
    def __init__(self, input_json, model_name="openai/gpt-4.1"):
        super().__init__(input_json)
        self.model_name = model_name
        self.api_key = os.environ.get("GITHUB_MODELS_TOKEN")  
        
    def solve(self):
        if not self.api_key:
            raise ValueError("Missing GITHUB_MODELS_TOKEN in environment variables")

        url = "https://models.github.ai/inference/chat/completions"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.api_key}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": self.prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code != 200:
            raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")

        result = response.json()
        # Extract model output
        return result["choices"][0]["message"]["content"]



input_json = """
{
  "grid": [
    ["#", ".", ".", "#"],
    [".", ".", ".", "."],
    [".", "#", ".", "."],
    ["#", ".", ".", "#"]
  ],
  "clues": {
    "across": {
      "1": "Opposite of down",
      "2": "The first letter of the alphabet"
    },
    "down": {
      "1": "Not yes",
      "3": "Hot drink made from leaves"
    }
  }
}

"""

# solver = GitHubModelSolver(input_json=input_json, model_name="openai/gpt-4.1")
solver = GitHubModelSolver(input_json=input_json, model_name="deepseek/deepseek-v3-0324")
print(solver.solve())
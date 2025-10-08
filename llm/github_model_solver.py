import os
import requests
from .interfaces import CrosswordSolverModel
from dotenv import load_dotenv
import time

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

class GitHubModelSolver(CrosswordSolverModel):
    def __init__(self, input_json, model_name="openai/gpt-4.1"):
        super().__init__(input_json)
        self.model_name = model_name
        dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
        load_dotenv(dotenv_path)
        self.api_key = os.environ.get("GITHUB_MODELS_TOKEN")
        if not self.api_key:
            raise ValueError("Missing GITHUB_MODELS_TOKEN in environment variables")

    def solve(self):
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
                {
                    "role": "user",
                    "content": self.full_prompt
                }
            ]
        }

        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        elapsed = time.time() - start_time
        print(f"Request completed in {elapsed:.2f} seconds, status code: {response.status_code}")

        if response.status_code != 200:
            raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        try:
            # Attempt to parse JSON if model returned a JSON string
            return content if isinstance(content, dict) else eval(content)
        except Exception:
            # Return raw content if parsing fails
            return content



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
      "intersections": [["1down", 2], ["5across", 4],
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


  print("Select a model to use:")
  print("1. OpenAI GPT-5")
  print("2. DeepSeek v3")
  print("3. LLaMA")
  print("4. Grok")

  choice = input("Enter the number corresponding to the model: ").strip()

  if choice == "1":
      solver = GitHubModelSolver(input_json=input_json, model_name="openai/gpt-5")
  elif choice == "2":
      solver = GitHubModelSolver(input_json=input_json, model_name="deepseek/DeepSeek-R1-0528")
  elif choice == "3":
      solver = GitHubModelSolver(input_json=input_json, model_name="meta/Llama-4-Scout-17B-16E-Instruct")
  elif choice == "4":
      solver = GitHubModelSolver(input_json=input_json, model_name="xAI/grok-3-mini")
  else:
      print("Invalid choice, defaulting to OpenAI GPT-5")
      solver = GitHubModelSolver(input_json=input_json, model_name="openai/gpt-5")

  result = solver.solve()
  print(result)
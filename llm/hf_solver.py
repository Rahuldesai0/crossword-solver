import os
import requests
from .interfaces import CrosswordSolverModel
from dotenv import load_dotenv
import json

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN in environment variables")

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


class HuggingFaceModelSolver(CrosswordSolverModel):
    def __init__(self, input_json, model_name="Qwen/Qwen3-VL-235B-A22B-Instruct:novita"):
        super().__init__(input_json)
        self.model_name = model_name

    def solve(self):
        print("Solving with Hugging Face Chat API...")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Solve this crossword puzzle:\n{self.full_prompt}"
                        }
                    ]
                }
            ]
        }

        response = self._make_request_with_retry(payload)
        return self._parse_response(response)

    def _make_request_with_retry(self, payload, max_retries=3):
        import time
        for attempt in range(max_retries):
            try:
                response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
                if response.status_code == 503:
                    print(f"Model loading, waiting 20s... (attempt {attempt+1})")
                    time.sleep(20)
                    continue
                elif response.status_code == 429:
                    print(f"Rate limited, waiting 60s... (attempt {attempt+1})")
                    time.sleep(60)
                    continue
                elif response.status_code != 200:
                    raise RuntimeError(f"Hugging Face API error {response.status_code}: {response.text}")
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Request failed after {max_retries} attempts: {e}")
                print(f"Request failed, retrying... (attempt {attempt+1})")
                time.sleep(5)

    def _parse_response(self, result):
        try:
            msg_content = result["choices"][0]["message"]["content"]

            # if content is a string, parse it
            if isinstance(msg_content, str):
                try:
                    parsed = json.loads(msg_content)
                    return parsed
                except json.JSONDecodeError:
                    return msg_content  # fallback to raw string

            # if content is a list of dicts 
            elif isinstance(msg_content, list) and len(msg_content) > 0:
                if "text" in msg_content[0]:
                    return msg_content[0]["text"]
            
            # fallback: return string
            return str(msg_content)

        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected response format: {result}. Error: {e}")


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

if __name__ == '__main__':
    models = [
        'Qwen/Qwen3-VL-235B-A22B-Instruct:novita',
        'Kwaipilot/KAT-Dev',
        'zai-org/GLM-4.5',
        'deepcogito/cogito-v2-preview-llama-109B-MoE'
    ]
    model = HuggingFaceModelSolver(input_json, models[3])
    print(model.solve())

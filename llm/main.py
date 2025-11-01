from llm.solvers import solve_in_parallel, pick_best_result
import json, sys
from concurrent.futures import TimeoutError

def example(input_json):
    pairs = [
        ('github', 'openai/gpt-4.1'),
        # ('github', 'deepseek/DeepSeek-R1-0528'),
        ('hf', 'Qwen/Qwen3-VL-235B-A22B-Instruct:novita'),
        ('hf', 'Kwaipilot/KAT-Dev')
    ]

    try:
        results = solve_in_parallel(input_json, pairs, max_workers=4, timeout=300)  # 5 minutes
    except TimeoutError:
        print("Execution exceeded 5 minutes. Exiting.")
        sys.exit(1)

    best_key, best = pick_best_result(results)
    return best_key, best


if __name__ == '__main__':
    with open("./json/img3_intersections.json") as f:
        input_json = json.load(f)

    best_key, best  = example(input_json)
    print(best)

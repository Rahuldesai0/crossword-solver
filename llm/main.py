from llm.solvers import solve_in_parallel, pick_best_result
import json, sys
from concurrent.futures import TimeoutError

def solve(input_json, grid, numbers):
    pairs = [
        ('github', 'openai/gpt-4.1'),
        ('github', 'openai/gpt-4o'),
        ('github', 'microsoft/phi-4'),
        ('hf', 'Qwen/Qwen3-VL-235B-A22B-Instruct:novita'),
        ('hf', 'google/gemma-2-2b-it'),
    ]
    
    pairs_more_time = [
        ('github', 'openai/gpt-5-mini'),
        ('github', 'deepseek/DeepSeek-R1-0528'),
        ('gemini', 'gemini-2.5-pro')
    ]

    try:
        results = solve_in_parallel(input_json, pairs, grid, numbers, max_workers=4, timeout=300)  # 5 minutes
    except TimeoutError:
        print("Execution exceeded 5 minutes. Exiting.")
        sys.exit(1)

    best_key, best = pick_best_result(results)
    return best_key, best


if __name__ == '__main__':
    with open("./json/img3_intersections.json") as f:
        input_json = json.load(f)

    best_key, best  = solve(input_json)
    print(best)

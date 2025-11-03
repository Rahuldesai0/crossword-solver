from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any, Dict

from llm.github_model_solver import GitHubModelSolver
from llm.gemini_solver import GeminiSolver
from llm.hf_solver import HuggingFaceModelSolver

def solve_in_parallel(input_json: str, solvers_and_models: List[Tuple[str, str]], 
                      grid, numbers,
                      max_workers: int = 4, timeout: int = 300) -> Dict[str, Any]:

    def make_solver(name: str, model: str):
        if name == 'gemini':
            return GeminiSolver(input_json=input_json, model_name=model)
        if name == 'github':
            return GitHubModelSolver(input_json=input_json, model_name=model)
        if name == 'hf':
            return HuggingFaceModelSolver(input_json=input_json, model_name=model)
        raise ValueError(f"Unknown solver: {name}")

    def compute_conflict_percentage(result):
        rows, cols = len(grid), len(grid[0])
        filled = [[None for _ in range(cols)] for _ in range(rows)]
        total_letters = 0
        conflict_count = 0

        for direction in ['across', 'down']:
            for num, word in result.get('solutions', {}).get(direction, {}).items():
                if num not in numbers:
                    continue
                start_r, start_c = numbers[num]
                for idx, char in enumerate(word):
                    r, c = start_r, start_c
                    if direction == 'across':
                        c += idx
                    else:
                        r += idx

                    if 0 <= r < rows and 0 <= c < cols and grid[r][c] != 0:
                        total_letters += 1
                        if filled[r][c] is None:
                            filled[r][c] = char
                        elif filled[r][c] != char:
                            conflict_count += 1

        return (conflict_count / total_letters) * 100 if total_letters > 0 else 100.0

    results: Dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        future_map = {}
        for name, model in solvers_and_models:
            solver = make_solver(name, model)
            key = f"{name}:{model}"
            future = exe.submit(solver.solve)
            future_map[future] = key

        for fut in as_completed(future_map):
            key = future_map[fut]
            try:
                res = fut.result(timeout=timeout)
                if isinstance(res, dict) and 'solutions' in res:
                    res['conflict_percentage'] = compute_conflict_percentage(res)
                else:
                    res = {'output': res, 'conflict_percentage': 100.0}
                print(f"Result obtained from {key} (conflict={res['conflict_percentage']:.2f}%)")
            except TimeoutError:
                res = {'error': f"__TIMEOUT__ after {timeout}s", 'conflict_percentage': 100.0}
                print(f"Timeout for {key}")
            except Exception as e:
                res = {'error': f"__ERROR__:{e}", 'conflict_percentage': 100.0}
                print(f"Error in {key}: {e}")
            results[key] = res

    return results


def pick_best_result(results) -> Tuple[str, Any]:
    best_key, best_value = None, None
    lowest_conflict = float('inf')

    for k, v in results.items():
        if isinstance(v, dict) and 'conflict_percentage' in v:
            if v['conflict_percentage'] < lowest_conflict:
                best_key, best_value = k, v
                lowest_conflict = v['conflict_percentage']

    if best_key is not None:
        return best_key, best_value

    first_key = next(iter(results))
    return first_key, results[first_key]

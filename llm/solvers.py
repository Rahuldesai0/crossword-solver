from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any, Dict

from llm.github_model_solver import GitHubModelSolver
from llm.gemini_solver import GeminiSolver
from llm.hf_solver import HuggingFaceModelSolver


def _is_structured(result: Any) -> bool:
	# prefer dict/list (parsed JSON-like) results
	return isinstance(result, (dict, list))

def solve_in_parallel(input_json: str, solvers_and_models: List[Tuple[str, str]], 
                      max_workers: int = 4, timeout: int = 300) -> Dict[str, Any]:
    def make_solver(name: str, model: str):
        if name == 'gemini':
            return GeminiSolver(input_json=input_json, model_name=model)
        if name == 'github':
            return GitHubModelSolver(input_json=input_json, model_name=model)
        if name == 'hf':
            return HuggingFaceModelSolver(input_json=input_json, model_name=model)
        raise ValueError(f"Unknown solver: {name}")

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
                print(f"Result obtained from {key}")  # <-- print when a result is ready
            except TimeoutError:
                res = f"__TIMEOUT__ after {timeout}s"
                print(f"Timeout for {key}")
            except Exception as e:
                res = f"__ERROR__:{e}"
                print(f"Error in {key}: {e}")
            results[key] = res

    return results


def pick_best_result(results: Dict[str, Any]) -> Tuple[str, Any]:
    # will need to check % conflict, and smallest among all
	"""Simple heuristic to pick a best result: prefer structured (dict/list), otherwise first non-error string."""
	# prefer structured
	for k, v in results.items():
		if _is_structured(v):
			return k, v

	# fallback: first non-error result
	for k, v in results.items():
		if not (isinstance(v, str) and v.startswith("__ERROR__:")):
			return k, v

	# if all errored, return the first error
	first_key = next(iter(results))
	return first_key, results[first_key]#

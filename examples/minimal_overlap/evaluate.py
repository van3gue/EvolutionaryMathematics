"""
Evaluator for Erdos' minimal overlap problem (n=100).
"""

import os
import argparse
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval


def _true_worst_case_overlap(a_list, n):
    """Independently recompute the worst-case overlap from A alone."""
    N = 2 * n
    a_set = set(a_list)
    b_set = set(range(N)) - a_set
    worst = 0
    for k in range(1, N):
        overlap_up = sum(1 for b in b_set if (b + k) in a_set)
        if overlap_up > worst:
            worst = overlap_up
        overlap_down = sum(1 for b in b_set if (b - k) in a_set)
        if overlap_down > worst:
            worst = overlap_down
    return worst


def adapted_validate_split(
    run_output: Tuple[List[int], int, int],
) -> Tuple[bool, Optional[str]]:
    """Validates that A is a legal split and the reported overlap is honest."""
    if not isinstance(run_output, (tuple, list)) or len(run_output) != 3:
        return False, "Expected (a, worst, n) tuple from run_minimal_overlap."

    a, worst, n = run_output

    if not isinstance(n, int) or n <= 0:
        return False, f"Invalid n: {n}"
    N = 2 * n

    if not isinstance(a, (list, tuple)):
        return False, f"Expected a to be a list/tuple, got {type(a)}"
    a_list = list(a)

    if len(a_list) != n:
        return False, f"Expected |A| = {n}, got {len(a_list)}"

    if len(set(a_list)) != len(a_list):
        return False, "A contains duplicate elements."

    for x in a_list:
        if not isinstance(x, int) or x < 0 or x >= N:
            return False, f"Element {x} out of range [0, {N})."

    true_worst = _true_worst_case_overlap(a_list, n)
    if true_worst != worst:
        return False, (
            f"Reported worst-case overlap ({worst}) does not match "
            f"recomputed value ({true_worst})."
        )

    return True, "Valid split; reported overlap confirmed correct."


def get_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def aggregate_metrics(results) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    a, worst, n = results[0]

    metrics = {
        "combined_score": float(n - worst),
        "public": {
            "worst_case_overlap": int(worst),
            "n": int(n),
        },
        "private": {
            "a_sample": sorted(a)[:20],
        },
    }
    return metrics


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_minimal_overlap",
        num_runs=1,
        get_experiment_kwargs=get_kwargs,
        validate_fn=adapted_validate_split,
        aggregate_metrics_fn=aggregate_metrics,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal overlap evaluator using shinka.eval"
    )
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)

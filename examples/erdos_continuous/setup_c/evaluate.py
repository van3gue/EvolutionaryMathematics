"""
Evaluator for Setup C (structure search). Primary fitness is -C_max;
the number of tied maximizing shifts is tracked as a secondary/diagnostic
metric only (per the spec) - it does not directly affect combined_score.
"""

import os
import argparse
from typing import Tuple, Optional, Dict, Any

import numpy as np

from shinka.core import run_shinka_eval
from examples.erdos_continuous.common import worst_case_overlap, validate_heights

KNOWN_RECORD = 0.380871
MIN_N, MAX_N = 256, 4096


def adapted_validate(run_output: Tuple[np.ndarray, float, int, int]) -> Tuple[bool, Optional[str]]:
    if not isinstance(run_output, (tuple, list)) or len(run_output) != 4:
        return False, "Expected (h, c_max, n, num_tied) tuple from run_setup_c."

    h, c_max, n, num_tied = run_output
    if not isinstance(h, np.ndarray):
        h = np.array(h)

    ok, msg = validate_heights(h)
    if not ok:
        return False, msg

    if len(h) != n:
        return False, f"len(h)={len(h)} does not match reported n={n}."

    if not (MIN_N <= n <= MAX_N):
        return False, f"n={n} outside allowed range [{MIN_N}, {MAX_N}]."

    true_c = worst_case_overlap(h)
    if abs(true_c - c_max) > 1e-6:
        return False, (
            f"Reported C_max ({c_max:.8f}) does not match recomputed "
            f"value ({true_c:.8f})."
        )

    return True, "Valid construction; reported C_max confirmed correct."


def get_kwargs(run_index: int) -> Dict[str, Any]:
    return {}


def aggregate_metrics(results) -> Dict[str, Any]:
    if not results:
        return {"combined_score": -1.0, "error": "No results to aggregate"}

    h, c_max, n, num_tied = results[0]
    true_c = worst_case_overlap(np.array(h))

    metrics = {
        "combined_score": float(-true_c),
        "public": {
            "worst_case_overlap_bound": float(true_c),
            "n_steps": int(n),
            "gap_to_known_record": float(true_c - KNOWN_RECORD),
            "num_tied_shifts": int(num_tied),  # secondary/diagnostic only
        },
        "private": {},
    }
    return metrics


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    os.makedirs(results_dir, exist_ok=True)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_setup_c",
        num_runs=1,
        get_experiment_kwargs=get_kwargs,
        validate_fn=adapted_validate,
        aggregate_metrics_fn=aggregate_metrics,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")
    print("Metrics:", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)

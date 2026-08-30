"""
Evaluator for Setup A (SOTA refinement) of the continuous Erdos minimum
overlap problem. Independently re-verifies every claimed score - never
trusts the evolved program's self-reported C_max.
"""

import os
import argparse
from typing import Tuple, Optional, Dict, Any

import numpy as np

from shinka.core import run_shinka_eval
from examples.erdos_continuous.common import worst_case_overlap, validate_heights

KNOWN_RECORD = 0.380871  # most recent published bound at time of writing


def adapted_validate(run_output: Tuple[np.ndarray, float, int]) -> Tuple[bool, Optional[str]]:
    if not isinstance(run_output, (tuple, list)) or len(run_output) != 3:
        return False, "Expected (h, c_max, n) tuple from run_setup_a."

    h, c_max, n = run_output
    if not isinstance(h, np.ndarray):
        h = np.array(h)

    ok, msg = validate_heights(h)
    if not ok:
        return False, msg

    if len(h) != n:
        return False, f"len(h)={len(h)} does not match reported n={n}."

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

    h, c_max, n = results[0]
    true_c = worst_case_overlap(np.array(h))

    metrics = {
        # Higher is better in ShinkaEvolve's convention, but SMALLER C_max
        # is better here, so the score is the negation of the true bound.
        "combined_score": float(-true_c),
        "public": {
            "worst_case_overlap_bound": float(true_c),
            "n_steps": int(n),
            "gap_to_known_record": float(true_c - KNOWN_RECORD),
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
        experiment_fn_name="run_setup_a",
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

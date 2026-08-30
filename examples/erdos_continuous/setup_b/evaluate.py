"""
Evaluator for Setup B (optimizer evolution). Fitness is -C_max, with a
penalty applied if the returned construction's integral constraint is
violated beyond a tight tolerance (per the spec: "penalty if integral(f)
deviates > 1e-12") - this discourages the evolved optimizer from cheating
by ignoring the constraint to chase a lower raw C_max.
"""

import os
import argparse
from typing import Tuple, Optional, Dict, Any

import numpy as np

from shinka.core import run_shinka_eval
from examples.erdos_continuous.common import worst_case_overlap, validate_heights

KNOWN_RECORD = 0.380871
INTEGRAL_TOL = 1e-12
PENALTY_WEIGHT = 50.0  # large enough that constraint violations never pay off


def adapted_validate(run_output: Tuple[np.ndarray, float, int]) -> Tuple[bool, Optional[str]]:
    if not isinstance(run_output, (tuple, list)) or len(run_output) != 3:
        return False, "Expected (h, c_max, n) tuple from run_setup_b."

    h, c_max, n = run_output
    if not isinstance(h, np.ndarray):
        h = np.array(h)

    # Use a looser tolerance for basic legality (0<=h<=1) than for the
    # integral penalty, which is handled separately/continuously below.
    ok, msg = validate_heights(h, tol=1e-6)
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
    h = np.array(h)
    true_c = worst_case_overlap(h)

    w = 2.0 / n
    integral_error = abs(float(np.sum(h) * w) - 1.0)
    penalty = PENALTY_WEIGHT * max(0.0, integral_error - INTEGRAL_TOL)

    score = -true_c - penalty

    metrics = {
        "combined_score": float(score),
        "public": {
            "worst_case_overlap_bound": float(true_c),
            "n_steps": int(n),
            "gap_to_known_record": float(true_c - KNOWN_RECORD),
            "integral_error": float(integral_error),
            "penalty_applied": float(penalty),
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
        experiment_fn_name="run_setup_b",
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

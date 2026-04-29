"""
Three-point triangulation comparison for reach detection metrics.

Compares three metric sets -- pre_dlc, best_post_dlc, and experiment --
and produces a structured verdict for each metric: whether the experiment
recovers pre-DLC quality, improves on the best post-DLC state, regresses,
or shows no change.

Usage::

    from mousereach.improvement.reach_detection.triangulate import triangulate

    verdict = triangulate(pre_dlc, best_post_dlc, experiment)
    # verdict is a dict with per-metric rows and overall summary

CLI::

    python -m mousereach.improvement.reach_detection.triangulate \\
        --pre-dlc path/to/pre_dlc_scalars.json \\
        --best-post-dlc path/to/best_post_dlc_scalars.json \\
        --experiment path/to/experiment_scalars.json
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Metrics where HIGHER is better
_HIGHER_IS_BETTER = {
    "n_matched",
    "median_coverage",
    "mean_coverage",
    "frac_coverage_gte_1",
    "frac_apex_included",
    "frac_anchor_start_ok",
    "frac_anchor_end_ok",
    "frac_both_anchors_ok",
    "n_perfect_videos",
    "recall",
}

# Metrics where LOWER is better
_LOWER_IS_BETTER = {
    "n_fp",
    "n_fn",
    "fn_rate_pct",
    "mean_abs_start_delta",
    "mean_abs_end_delta",
}

# Tolerance for "no change" verdict (relative)
_NO_CHANGE_TOL = 0.005


@dataclass
class MetricVerdict:
    """Verdict for a single metric across the three points.

    Attributes
    ----------
    metric : str
        Name of the metric.
    pre_dlc : float or None
        Value from the pre-DLC baseline.
    best_post_dlc : float or None
        Value from the best post-DLC snapshot.
    experiment : float or None
        Value from the experiment being evaluated.
    verdict : str
        One of: "experiment recovers pre_dlc",
        "experiment improves on best_post_dlc", "experiment regresses",
        "no change", "insufficient data".
    direction : str
        "higher_is_better", "lower_is_better", or "unknown".
    """
    metric: str
    pre_dlc: Optional[float]
    best_post_dlc: Optional[float]
    experiment: Optional[float]
    verdict: str
    direction: str


def _classify_direction(metric_name: str) -> str:
    """Return the improvement direction for a metric."""
    if metric_name in _HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric_name in _LOWER_IS_BETTER:
        return "lower_is_better"
    return "unknown"


def _is_better(val: float, ref: float, direction: str) -> bool:
    """Return True if val is strictly better than ref."""
    if direction == "higher_is_better":
        return val > ref * (1 + _NO_CHANGE_TOL)
    elif direction == "lower_is_better":
        return val < ref * (1 - _NO_CHANGE_TOL)
    return False


def _is_worse(val: float, ref: float, direction: str) -> bool:
    """Return True if val is strictly worse than ref."""
    if direction == "higher_is_better":
        return val < ref * (1 - _NO_CHANGE_TOL)
    elif direction == "lower_is_better":
        return val > ref * (1 + _NO_CHANGE_TOL)
    return False


def triangulate(
    pre_dlc_metrics: Dict[str, Any],
    best_post_dlc_metrics: Dict[str, Any],
    experiment_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare three metric sets and produce per-metric verdicts.

    Parameters
    ----------
    pre_dlc_metrics : dict
        Flat dict of metric_name -> numeric value for the pre-DLC baseline.
    best_post_dlc_metrics : dict
        Flat dict for the best post-DLC snapshot (current production).
    experiment_metrics : dict
        Flat dict for the new experiment being evaluated.

    Returns
    -------
    dict
        Keys:
        - "verdicts": list of MetricVerdict dicts
        - "summary": {"n_improved", "n_recovered", "n_regressed", "n_no_change",
                       "n_insufficient", "overall_verdict"}
    """
    # Collect all metric names across all three sets
    all_keys = sorted(
        set(pre_dlc_metrics.keys())
        | set(best_post_dlc_metrics.keys())
        | set(experiment_metrics.keys())
    )

    verdicts: List[MetricVerdict] = []
    n_improved = 0
    n_recovered = 0
    n_regressed = 0
    n_no_change = 0
    n_insufficient = 0

    for key in all_keys:
        pre = pre_dlc_metrics.get(key)
        best = best_post_dlc_metrics.get(key)
        exp = experiment_metrics.get(key)
        direction = _classify_direction(key)

        # Skip non-numeric or None values
        if not all(isinstance(v, (int, float)) for v in [pre, best, exp] if v is not None):
            continue

        if exp is None or direction == "unknown":
            verdicts.append(MetricVerdict(
                metric=key, pre_dlc=pre, best_post_dlc=best,
                experiment=exp, verdict="insufficient data",
                direction=direction,
            ))
            n_insufficient += 1
            continue

        if pre is None and best is None:
            verdicts.append(MetricVerdict(
                metric=key, pre_dlc=pre, best_post_dlc=best,
                experiment=exp, verdict="insufficient data",
                direction=direction,
            ))
            n_insufficient += 1
            continue

        # Determine verdict
        improves_on_best = best is not None and _is_better(exp, best, direction)
        recovers_pre = pre is not None and not _is_worse(exp, pre, direction)
        regresses_from_best = best is not None and _is_worse(exp, best, direction)

        if improves_on_best:
            verdict = "experiment improves on best_post_dlc"
            n_improved += 1
        elif regresses_from_best:
            if recovers_pre:
                verdict = "experiment recovers pre_dlc"
                n_recovered += 1
            else:
                verdict = "experiment regresses"
                n_regressed += 1
        else:
            verdict = "no change"
            n_no_change += 1

        verdicts.append(MetricVerdict(
            metric=key, pre_dlc=pre, best_post_dlc=best,
            experiment=exp, verdict=verdict, direction=direction,
        ))

    # Overall verdict
    if n_regressed > 0:
        overall = "regression detected"
    elif n_improved > 0:
        overall = "net improvement"
    elif n_recovered > 0:
        overall = "recovery to pre-DLC level"
    else:
        overall = "no change"

    return {
        "verdicts": [asdict(v) for v in verdicts],
        "summary": {
            "n_improved": n_improved,
            "n_recovered": n_recovered,
            "n_regressed": n_regressed,
            "n_no_change": n_no_change,
            "n_insufficient": n_insufficient,
            "overall_verdict": overall,
        },
    }


def print_triangulation_table(result: Dict[str, Any]) -> None:
    """Print a human-readable triangulation table to stdout.

    ASCII-only output safe for Windows console.
    """
    verdicts = result["verdicts"]
    summary = result["summary"]

    # Header
    print("")
    print("=" * 100)
    print("THREE-POINT TRIANGULATION COMPARISON")
    print("=" * 100)
    print("")
    print(
        f"{'Metric':<30} {'Pre-DLC':>12} {'Best Post':>12} "
        f"{'Experiment':>12} {'Verdict':<30}"
    )
    print("-" * 100)

    for v in verdicts:
        pre_str = f"{v['pre_dlc']:.4f}" if v["pre_dlc"] is not None else "N/A"
        best_str = f"{v['best_post_dlc']:.4f}" if v["best_post_dlc"] is not None else "N/A"
        exp_str = f"{v['experiment']:.4f}" if v["experiment"] is not None else "N/A"
        print(
            f"{v['metric']:<30} {pre_str:>12} {best_str:>12} "
            f"{exp_str:>12} {v['verdict']:<30}"
        )

    print("-" * 100)
    print(f"\nSummary: {summary['n_improved']} improved, "
          f"{summary['n_recovered']} recovered, "
          f"{summary['n_regressed']} regressed, "
          f"{summary['n_no_change']} no change, "
          f"{summary['n_insufficient']} insufficient data")
    print(f"Overall verdict: {summary['overall_verdict']}")
    print("")


def _load_flat_metrics(path: Path) -> Dict[str, Any]:
    """Load metrics from a scalars.json or completeness_scalars.json.

    Flattens nested 'total' and 'aggregates' dicts into top-level keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    flat: Dict[str, Any] = {}
    for key, val in data.items():
        if isinstance(val, dict) and key in ("total", "aggregates",
                                              "exhaustive", "all"):
            for subkey, subval in val.items():
                if isinstance(subval, (int, float)):
                    flat[subkey] = subval
        elif isinstance(val, (int, float)):
            flat[key] = val
    return flat


def main() -> None:
    """CLI entry point for triangulation comparison."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Three-point triangulation comparison for reach detection metrics."
    )
    parser.add_argument(
        "--pre-dlc", required=True, type=Path,
        help="Path to pre-DLC scalars JSON",
    )
    parser.add_argument(
        "--best-post-dlc", required=True, type=Path,
        help="Path to best post-DLC scalars JSON",
    )
    parser.add_argument(
        "--experiment", required=True, type=Path,
        help="Path to experiment scalars JSON",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional path to write the verdict JSON",
    )
    args = parser.parse_args()

    pre = _load_flat_metrics(args.pre_dlc)
    best = _load_flat_metrics(args.best_post_dlc)
    exp = _load_flat_metrics(args.experiment)

    result = triangulate(pre, best, exp)
    print_triangulation_table(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Verdict written to {args.output}")


if __name__ == "__main__":
    main()

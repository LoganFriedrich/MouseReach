#!/usr/bin/env python
"""Per-pass diagnosis: re-runs detection with each pass individually to measure contribution."""

import sys
import os
from pathlib import Path

# Ensure we pick up the local editable install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mousereach.reach.core.spatial_refiner import (
    SpatialRefiner, SpatialRefinerConfig,
    AbsorbedReachSplitter, LateEndTrimmer,
    ShortFalsePositiveFilter, EarlyStartCorrector,
)


def run_with_config(label, pass_names):
    """Re-run detection with specific passes, then eval."""
    cfg = SpatialRefinerConfig()

    pass_map = {
        'splitter': lambda: AbsorbedReachSplitter(cfg),
        'trimmer': lambda: LateEndTrimmer(cfg),
        'fp_filter': lambda: ShortFalsePositiveFilter(cfg),
        'start_corrector': lambda: EarlyStartCorrector(cfg),
    }

    desired_passes = [pass_map[n]() for n in pass_names] if pass_names else []

    # Monkey-patch SpatialRefiner.__init__ to set desired passes
    orig_init = SpatialRefiner.__init__

    def patched_init(self, config=None, hand_points=None):
        orig_init(self, config=config, hand_points=hand_points)
        self._passes = list(desired_passes)  # Override passes

    SpatialRefiner.__init__ = patched_init

    try:
        from mousereach.reach.core import process_batch
        from mousereach.config import require_processing_root
        from mousereach.eval.collect_results import collect_all

        processing_dir = require_processing_root() / "Processing"

        # Re-run detection
        process_batch(processing_dir, verbose=False)

        # Collect results
        results = collect_all(processing_dir)

        if results.reach_results:
            import numpy as np
            mean_p = np.mean([r.precision * 100 for r in results.reach_results])
            mean_r = np.mean([r.recall * 100 for r in results.reach_results])
            mean_f1 = np.mean([r.f1 * 100 for r in results.reach_results])

            all_start = results.all_start_errors
            all_end = results.all_end_errors
            start_acc = (sum(1 for e in all_start if abs(e) <= 2) / len(all_start) * 100
                         if all_start else 0)
            end_acc = (sum(1 for e in all_end if abs(e) <= 2) / len(all_end) * 100
                       if all_end else 0)

            total_tp = sum(r.tp for r in results.reach_results)
            total_fp = sum(r.fp for r in results.reach_results)
            total_fn = sum(r.fn for r in results.reach_results)

            print(f"{label:30s}  P={mean_p:5.1f}%  R={mean_r:5.1f}%  F1={mean_f1:5.1f}%  "
                  f"Start±2={start_acc:5.1f}%  End±2={end_acc:5.1f}%  "
                  f"TP={total_tp} FP={total_fp} FN={total_fn}")
        else:
            print(f"{label:30s}  NO RESULTS")
    finally:
        SpatialRefiner.__init__ = orig_init


def run_no_refiner(label):
    """Disable refiner entirely."""
    # Monkey-patch ReachDetector to set _spatial_refiner = None
    from mousereach.reach.core.reach_detector import ReachDetector
    orig_init = ReachDetector.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        self._spatial_refiner = None

    ReachDetector.__init__ = patched_init

    try:
        from mousereach.reach.core import process_batch
        from mousereach.config import require_processing_root
        from mousereach.eval.collect_results import collect_all

        processing_dir = require_processing_root() / "Processing"

        process_batch(processing_dir, verbose=False)
        results = collect_all(processing_dir)

        if results.reach_results:
            import numpy as np
            mean_p = np.mean([r.precision * 100 for r in results.reach_results])
            mean_r = np.mean([r.recall * 100 for r in results.reach_results])
            mean_f1 = np.mean([r.f1 * 100 for r in results.reach_results])

            all_start = results.all_start_errors
            all_end = results.all_end_errors
            start_acc = (sum(1 for e in all_start if abs(e) <= 2) / len(all_start) * 100
                         if all_start else 0)
            end_acc = (sum(1 for e in all_end if abs(e) <= 2) / len(all_end) * 100
                       if all_end else 0)

            total_tp = sum(r.tp for r in results.reach_results)
            total_fp = sum(r.fp for r in results.reach_results)
            total_fn = sum(r.fn for r in results.reach_results)

            print(f"{label:30s}  P={mean_p:5.1f}%  R={mean_r:5.1f}%  F1={mean_f1:5.1f}%  "
                  f"Start±2={start_acc:5.1f}%  End±2={end_acc:5.1f}%  "
                  f"TP={total_tp} FP={total_fp} FN={total_fn}")
        else:
            print(f"{label:30s}  NO RESULTS")
    finally:
        ReachDetector.__init__ = orig_init


if __name__ == '__main__':
    import functools
    print = functools.partial(print, flush=True)

    print("=" * 120)
    print("Per-pass diagnosis (recalibrated thresholds)")
    print("=" * 120)

    print("\n>>> Running NO_REFINER baseline...")
    run_no_refiner("NO_REFINER")
    print("\n>>> Running PASS1_ONLY...")
    run_with_config("PASS1_ONLY (Splitter)", ['splitter'])
    print("\n>>> Running PASS2_ONLY...")
    run_with_config("PASS2_ONLY (Trimmer)", ['trimmer'])
    print("\n>>> Running PASS3_ONLY...")
    run_with_config("PASS3_ONLY (FP Filter)", ['fp_filter'])
    print("\n>>> Running PASS4_ONLY...")
    run_with_config("PASS4_ONLY (Start Corr)", ['start_corrector'])
    print("\n>>> Running ALL_PASSES...")
    run_with_config("ALL_PASSES", ['splitter', 'trimmer', 'fp_filter', 'start_corrector'])

    print("=" * 120)

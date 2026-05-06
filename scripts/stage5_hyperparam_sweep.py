"""Hyperparameter sweep for Stage 5 (displaced_sa) to find the best
trust on class-matched commits. We freeze Stages 1-4 (already locked
at 100% trust) and only vary Stage 5's parameters.

Reports: top-N configurations by full trust (class + OKF±3 + IFR±3),
plus the ceiling -- max achievable trust across all configs. If max
is < 100%, the approach is not viable as currently structured and
needs editing.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.outcomes.v6_cascade.stage_1_pellet_stable_untouched import (
    Stage1PelletStableUntouched)
from mousereach.outcomes.v6_cascade.stage_2_paw_never_in_pellet_area import (
    Stage2PawNeverInPelletArea)
from mousereach.outcomes.v6_cascade.stage_3_pellet_returns_to_pillar import (
    Stage3PelletReturnsToPillar)
from mousereach.outcomes.v6_cascade.stage_4_pellet_off_pillar_throughout import (
    Stage4PelletOffPillarThroughout)
from mousereach.outcomes.v6_cascade.stage_5_pellet_displaced_to_sa import (
    Stage5PelletDisplacedToSA)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.outcomes.v6_cascade.trust_calibrator import calibrate_stage
from mousereach.reach.v8.features import load_dlc_h5

import importlib.util
spec = importlib.util.spec_from_file_location(
    "runner",
    str(Path(__file__).resolve().parents[1] / "scripts" / "restart_phase_e_stage45_validate.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def main():
    print("Loading segments + GT ...", flush=True)
    seg_inputs, gt_lookup = runner.build_seg_inputs_and_gt()
    print(f"  Loaded {len(seg_inputs)} segments")

    # Run S1 + S2 + S3 + S4 once
    inputs = seg_inputs
    consumed = set()
    for stage in [
        Stage1PelletStableUntouched(commit_frac=0.95, commit_distance_radii=1.5),
        Stage2PawNeverInPelletArea(),
        Stage3PelletReturnsToPillar(),
        Stage4PelletOffPillarThroughout(),
    ]:
        stage_in = [s for s in inputs if (s.video_id, s.segment_num) not in consumed]
        cal = calibrate_stage(stage=stage, seg_inputs=stage_in, gt_lookup=gt_lookup,
                              okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
        for c in cal.cases:
            if c.decision in ("commit", "triage"):
                consumed.add((c.video_id, c.segment_num))
    s5_inputs = [s for s in inputs if (s.video_id, s.segment_num) not in consumed]
    print(f"  Stage 5 input: {len(s5_inputs)} segments")
    print()

    # Sweep configs
    sweep_grid = list(product(
        [2.5, 3.0, 3.5, 4.0],     # pellet_off_pillar_radii
        [3, 5, 8, 12],             # post_bout_settling_frames
        [2, 3, 4],                 # pellet_off_pillar_sustained_frames
        [1, 2, 3],                 # pillar_visible_sustained_frames
    ))
    print(f"Sweeping {len(sweep_grid)} configurations ...")
    print()

    results = []
    for i, (off_r, settling, pell_sust, pill_sust) in enumerate(sweep_grid):
        s5 = Stage5PelletDisplacedToSA(
            pellet_off_pillar_radii=off_r,
            post_bout_settling_frames=settling,
            pellet_off_pillar_sustained_frames=pell_sust,
            pillar_visible_sustained_frames=pill_sust,
        )
        cal = calibrate_stage(stage=s5, seg_inputs=s5_inputs, gt_lookup=gt_lookup,
                              okf_tolerance=3, ifr_tolerance=3, transition_zone_half=5)
        n_commits = sum(1 for c in cal.cases if c.decision == "commit")
        n_triage = sum(1 for c in cal.cases if c.decision == "triage")
        # Full trust: class match AND okf within tol AND ifr within tol
        n_full_trust = 0
        n_class_match = 0
        for c in cal.cases:
            if c.decision != "commit":
                continue
            if c.class_match:
                n_class_match += 1
                if c.okf_within_tol and c.ifr_within_tol:
                    n_full_trust += 1
        full_trust_pct = (100 * n_full_trust / max(n_commits, 1))
        results.append({
            "off_r": off_r, "settling": settling,
            "pell_sust": pell_sust, "pill_sust": pill_sust,
            "n_commits": n_commits,
            "n_triage": n_triage,
            "n_class_match": n_class_match,
            "n_full_trust": n_full_trust,
            "full_trust_pct": full_trust_pct,
        })
        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(sweep_grid)} done", flush=True)

    print()
    print("=" * 80)
    print("TOP 20 CONFIGS BY FULL TRUST %")
    print("=" * 80)
    results.sort(key=lambda r: (-r["full_trust_pct"], -r["n_full_trust"]))
    print(f'{"off_r":>5s} {"sett":>5s} {"pell":>5s} {"pill":>5s} '
          f'{"commits":>8s} {"triage":>7s} {"cls_ok":>7s} {"trust":>7s} {"trust%":>7s}')
    for r in results[:20]:
        print(f'{r["off_r"]:>5.1f} {r["settling"]:>5d} {r["pell_sust"]:>5d} {r["pill_sust"]:>5d} '
              f'{r["n_commits"]:>8d} {r["n_triage"]:>7d} {r["n_class_match"]:>7d} '
              f'{r["n_full_trust"]:>7d} {r["full_trust_pct"]:>7.1f}')
    print()
    print(f"Best full-trust achievable: {results[0]['full_trust_pct']:.1f}% "
          f"({results[0]['n_full_trust']}/{results[0]['n_commits']} commits)")

    # Also report best by absolute count of correct commits
    print()
    results.sort(key=lambda r: -r["n_full_trust"])
    print("TOP 5 CONFIGS BY ABSOLUTE COUNT OF FULLY-TRUSTED COMMITS")
    print(f'{"off_r":>5s} {"sett":>5s} {"pell":>5s} {"pill":>5s} '
          f'{"commits":>8s} {"trust":>7s} {"trust%":>7s}')
    for r in results[:5]:
        print(f'{r["off_r"]:>5.1f} {r["settling"]:>5d} {r["pell_sust"]:>5d} {r["pill_sust"]:>5d} '
              f'{r["n_commits"]:>8d} {r["n_full_trust"]:>7d} {r["full_trust_pct"]:>7.1f}')


if __name__ == "__main__":
    main()

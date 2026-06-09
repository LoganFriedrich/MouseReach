"""
Generate v8.0.4 FP/FN review manifests for the 47-corpus exhaustive videos
that did NOT already have a manifest (the ~33 newly-exhaustive / never-
manifested videos). Separate output dir; the existing calibration_loocv +
holdout_2026_05_11 manifest sets are left UNTOUCHED.

WHY (2026-06-09): Logan's exhaustive pass made all 47 corpus videos exhaustive.
Only 14 (the old LOOCV calibration set) had review manifests. This generates
manifests for the rest so the FP/FN review widget has full 47-corpus coverage.
Per Logan: keep these SEPARATE from the calibration/holdout sets (new dir).

SOURCES (guideline-compliant)
  - Algo:  production detect_reaches_v8 (v8.0.4, bundled model v8.0.0_bsw_w0.8,
           all postprocess applied) -- INFERENCE, not a retrain -- on the
           model-3.1 DLC. The detector never trained on these videos (they were
           non-exhaustive / held out at training), so this is out-of-sample.
  - GT:    LIVE canonical validation_runs/DLC_2026_03_27/gt (never a snapshot)
           -- per feedback_never_pull_gt_from_snapshots.
  - Topology + manifest schema + matcher: reused VERBATIM from
           generate_fpfn_review_manifests_v8_0_3_new_topology.py (imported, not
           edited) so the widget reads an identical schema, with matcher-aware
           topology -- per feedback_pair_legacy_with_topology.
  - Stamps gt_last_modified_at + manifest_generated_at + detector_version.

OUTPUT
  fpfn_review_manifests/v8.0.4/newly_exhaustive/<video_id>.json

GUARDRAILS
  - Does NOT call the imported generator's main()/archive_existing_manifests --
    only build_manifests + write_manifests, pointed at the NEW dir. Existing
    manifest dirs are never moved or overwritten.
  - DLC + GT read from Y: canonical only (feedback_y_drive_only).
  - Run with --limit 1 first to validate one manifest before the full set.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
sys.path.insert(0, SRC)

# Import the canonical v8.0.3-new-topology generator as a module (no edits to it)
_GEN_PATH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach\scripts"
    r"\generate_fpfn_review_manifests_v8_0_3_new_topology.py"
)
_spec = importlib.util.spec_from_file_location("_gen_v803_topo", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

from mousereach.reach.v8 import detect_reaches_v8
from mousereach.reach.v8.features import load_dlc_h5

# --- paths (Y: canonical) ---
GT_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt")
DLC_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\Processing\updated dlc model 3.1")
EXISTING_CAL_MANIFESTS = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests\calibration_loocv")

LABEL = "newly_exhaustive"
SNAPSHOT_NAME = "v8.0.4_production_inference_on_model31_dlc"

# --- override imported-module globals so we write to a NEW dir + correct GT root ---
gen.OUTPUT_ROOT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests\v8.0.4")
gen.DETECTOR_VERSION = "v8.0.4"
gen.GT_ROOTS[LABEL] = GT_DIR          # used by _load_live_gt_reaches + _load_gt_boundaries


def target_videos():
    """47-corpus videos lacking a calibration_loocv manifest."""
    corpus47 = sorted(p.name.replace("_unified_ground_truth.json", "")
                      for p in GT_DIR.glob("*_unified_ground_truth.json"))
    have = {p.stem for p in EXISTING_CAL_MANIFESTS.glob("*.json")}
    return [v for v in corpus47 if v not in have]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N target videos (validation). 0 = all.")
    args = ap.parse_args()

    vids = target_videos()
    if args.limit:
        vids = vids[:args.limit]
    print(f"GT (live):  {GT_DIR}")
    print(f"DLC:        {DLC_DIR}")
    print(f"Output dir: {gen.OUTPUT_ROOT / LABEL}")
    print(f"Target videos lacking a manifest: {len(vids)}")
    print("=" * 70)

    per_video = {}
    for vid in vids:
        h5 = sorted(DLC_DIR.glob(f"{vid}DLC_*.h5"))
        if not h5:
            print(f"  [skip: no model-3.1 DLC] {vid}")
            continue
        algos = detect_reaches_v8(load_dlc_h5(h5[0]))
        gts, last_mod = gen._load_live_gt_reaches(LABEL, vid)
        per_video[vid] = {
            "algos": sorted(set((int(s), int(e)) for s, e in algos)),
            "gts": gts,
            "gt_last_modified_at": last_mod,
        }
        print(f"  {vid:32} algo={len(per_video[vid]['algos']):>4}  gt={len(gts):>4}  gt_last_mod={last_mod}")

    print("=" * 70)
    print("Building manifests (reusing canonical topology/schema)...")
    manifests = gen.build_manifests(per_video, LABEL, SNAPSHOT_NAME)
    gen._print_summary(manifests, LABEL)
    n = gen.write_manifests(manifests, LABEL)
    print(f"\nWrote {n} manifests to {gen.OUTPUT_ROOT / LABEL}/")


if __name__ == "__main__":
    main()

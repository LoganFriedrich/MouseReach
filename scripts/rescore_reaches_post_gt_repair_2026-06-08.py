"""
Rescore reaches after the 2026-06-08 seg_num/dedup GT repair.

NOT a retrain: runs the production v8.0.4 reach detector (detect_reaches_v8,
bundled model v8.0.0_bsw_w0.8.joblib) as inference on each video, then scores
via the canonical compute_reach_detection_metrics against (a) the pre-repair
backup GT and (b) the repaired GT. The (a)->(b) delta isolates the GT-repair
effect and validates the pipeline reproduces the known headline on old GT.

Notes
- Canonical scorer auto-discovers EXHAUSTIVE-reach-GT videos only, so the
  dedup on non-exhaustive videos (e.g. 20250701_CNT0110_P2, 49 dups) does not
  enter any headline metric. Only the generalization corpus (all exhaustive,
  4 dups in 2 videos) can move.
- 47-corpus exhaustive scoring here is the PRODUCTION model run IN-SAMPLE
  (the bundle was trained on those 20 videos); it is NOT the LOOCV headline
  and is shown only to confirm the exhaustive GT was untouched by the repair.
- Generalization is out-of-sample -> reproduces the holdout methodology.
- Imports from Y: source (C: runtime postprocess.py is stale / pre-apex-split).
"""
from __future__ import annotations

import sys
sys.path.insert(0, r"Y:\2_Connectome\Behavior\MouseReach\src")

import json
import tempfile
from pathlib import Path

from mousereach.reach.v8.features import load_dlc_h5
from mousereach.reach.v8 import detect_reaches_v8
from mousereach.improvement.reach_detection.metrics import compute_reach_detection_metrics

STAMP = "20260608_142022"
CORPORA = [
    ("47-corpus (exhaustive, IN-SAMPLE -- not the LOOCV headline)",
     Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt"),
     Path(rf"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\gt_backup_pre_seg_dedup_{STAMP}"),
     Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\Processing\updated dlc model 3.1")),
    ("generalization (exhaustive, OUT-OF-SAMPLE -- holdout methodology)",
     Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\gt"),
     Path(rf"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\gt_backup_pre_seg_dedup_{STAMP}"),
     Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations\generalization_test_2026-05-11\dlc")),
]


def infer_algo_dir(gt_dir, dlc_dir):
    """Run detect_reaches_v8 on every video that has DLC, write algo reaches
    JSON to a temp dir in the format the scorer expects. Returns the dir."""
    tmp = Path(tempfile.mkdtemp(prefix="rescore_algo_"))
    for gt in sorted(gt_dir.glob("*_unified_ground_truth.json")):
        vid = gt.name.replace("_unified_ground_truth.json", "")
        h5 = sorted(dlc_dir.glob(f"{vid}DLC_*.h5"))
        if not h5:
            continue
        spans = detect_reaches_v8(load_dlc_h5(h5[0]))
        data = {"segments": [{"reaches": [
            {"start_frame": int(s), "end_frame": int(e)} for (s, e) in spans]}]}
        (tmp / f"{vid}_reaches.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp


def headline(scalars):
    """Schema-agnostic: show scalar fields; summarize lists/dicts."""
    out = {}
    for k, v in scalars.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, list):
            out[k] = f"<list len={len(v)}>"
        elif isinstance(v, dict):
            out[k] = {kk: vv for kk, vv in v.items()
                      if isinstance(vv, (int, float, str, bool)) or vv is None}
        else:
            out[k] = f"<{type(v).__name__}>"
    return out


def main():
    for label, repaired_gt, backup_gt, dlc_dir in CORPORA:
        print("=" * 96)
        print(label)
        print("=" * 96)
        algo_dir = infer_algo_dir(repaired_gt, dlc_dir)
        for tag, gtd in (("PRE-repair (backup GT)", backup_gt), ("POST-repair (current GT)", repaired_gt)):
            out = Path(tempfile.mkdtemp(prefix="rescore_out_"))
            scalars = compute_reach_detection_metrics(gt_dir=gtd, algo_dir=algo_dir, output_dir=out)
            print(f"\n--- {tag}  gt={gtd}")
            print(json.dumps(headline(scalars), indent=2, default=str))
        print()


if __name__ == "__main__":
    main()

"""For each segment that Stage 2 (was Stage 1) commits as untouched,
report whether new Stage 1 commits and what its features are. This
helps tune new Stage 1 thresholds."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mousereach.outcomes.v6_cascade.stage_1_pellet_position_never_changed import (
    Stage1PelletPositionNeverChanged)
from mousereach.outcomes.v6_cascade.stage_2_pellet_stable_untouched import (
    Stage2PelletStableUntouched)
from mousereach.outcomes.v6_cascade.stage_base import SegmentInput
from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory")

def find_dlc(vid):
    return next(DLC_DIR.glob(f"{vid}DLC_*.h5"))

def gt_reaches_for_segment(gt, sn):
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == sn:
            s = r.get("start_frame"); e = r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out

folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
train_pool_ids = folds["train_pool"]["video_ids"]

stage1_new = Stage1PelletPositionNeverChanged()
stage_old1 = Stage2PelletStableUntouched(commit_frac=0.95, commit_distance_radii=1.5)

n_old1_commits = 0
n_new1_commits_on_old1 = 0
defer_reason_counts = {}
for vid in train_pool_ids:
    dlc = load_dlc_h5(find_dlc(vid))
    gt = json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))
    gt_b = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    for i in range(len(gt_b) - 1):
        sn = i + 1
        seg = SegmentInput(
            video_id=vid, segment_num=sn,
            seg_start=gt_b[i], seg_end=gt_b[i + 1] - 1, dlc_df=dlc,
            reach_windows=gt_reaches_for_segment(gt, sn))
        old_d = stage_old1.decide(seg)
        if old_d.decision != "commit":
            continue
        n_old1_commits += 1
        new_d = stage1_new.decide(seg)
        if new_d.decision == "commit":
            n_new1_commits_on_old1 += 1
        else:
            # Bucket by short reason key
            key = new_d.reason.split("(")[0].strip()
            defer_reason_counts[key] = defer_reason_counts.get(key, 0) + 1
            # Print first 3 of each
            if defer_reason_counts[key] <= 3:
                print(f"  [{key}] {vid} seg {sn}")
                print(f"    new1 features: {new_d.features}")
                print(f"    new1 reason: {new_d.reason}")

print()
print(f"Total old-Stage-1 commits: {n_old1_commits}")
print(f"  Of which new Stage 1 also commits: {n_new1_commits_on_old1}")
print(f"  Of which new Stage 1 DEFERS:")
for r, n in sorted(defer_reason_counts.items(), key=lambda x: -x[1]):
    print(f"    {n:4d}: {r}")

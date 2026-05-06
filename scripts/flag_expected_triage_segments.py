"""Add `expected_triage` flag to specific GT segments that we know
the algo cannot reasonably classify (apparatus failure / data quality
issues / DLC label-switch / etc.).

This is a manual-curation step parallel to GT outcome scoring: GT
outcome captures what physically happened (pellet was/wasn't moved),
expected_triage captures whether the algo should be expected to score
it or send it to human review.

Updates BOTH canonical GT (MouseReach_Pipeline/Processing/) and the
quarantine GT (Validation_Runs/.../gt/), with an audit trail.

Per memory rule `gt_correction_workflow.md`: when the segment is
clearly outside the algo's reasonable competence (and the user has
inspected and confirmed), edit the GT JSON in place with audit trail.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

CANONICAL_GT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Processing")
QUARANTINE_GT = Path(r"Y:\2_Connectome\Validation_Runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt")

# Segments to flag. Each entry: (video_id, segment_num, reason).
FLAG_LIST = [
    ("20250806_CNT0311_P2", 17,
     "abnormally short segment (169 frames vs ~1840 typical) -- end-of-video apparatus failure / cycling abort, not a real pellet trial. Cannot reasonably be classified by outcome algo; route to human review."),
    ("20250806_CNT0311_P2", 18,
     "abnormally short segment (111 frames vs ~1840 typical) -- end-of-video apparatus failure / cycling abort, not a real pellet trial."),
    ("20250806_CNT0311_P2", 19,
     "abnormally short segment (76 frames vs ~1840 typical) -- end-of-video apparatus failure / cycling abort."),
    ("20250806_CNT0311_P2", 20,
     "abnormally short segment (67 frames vs ~1840 typical) -- end-of-video apparatus failure / cycling abort."),
    ("20250806_CNT0312_P2", 18,
     "abnormally short segment (102 frames vs ~1840 typical) -- end-of-video apparatus failure / cycling abort."),
    ("20250806_CNT0312_P2", 19,
     "abnormally short segment (79 frames vs ~1840 typical) -- end-of-video apparatus failure / cycling abort."),
]

NOW = datetime.now(timezone.utc).isoformat()
USER = "FRIEDRICHL"


def update_gt(path: Path):
    if not path.exists():
        print(f"  SKIP (missing): {path}")
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    segments = data.get("outcomes", {}).get("segments", []) or []
    for video_id, sn, reason in FLAG_LIST:
        if data.get("video_name", "").startswith(video_id) or video_id in path.stem:
            for seg in segments:
                if seg.get("segment_num") == sn:
                    seg["expected_triage"] = True
                    seg["expected_triage_reason"] = reason
                    seg["expected_triage_flagged_by"] = USER
                    seg["expected_triage_flagged_at"] = NOW
                    changed += 1
    if changed:
        # Append audit trail to top-level
        data["last_modified_at"] = NOW
        data["last_modified_by"] = USER
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  UPDATED ({changed} segs): {path}")
        return True
    return False


def main():
    videos_to_update = sorted({v for v, _, _ in FLAG_LIST})
    for vid in videos_to_update:
        print(f"\n{vid}:")
        for d in [CANONICAL_GT, QUARANTINE_GT]:
            for fname in [f"{vid}_unified_ground_truth.json"]:
                update_gt(d / fname)


if __name__ == "__main__":
    main()

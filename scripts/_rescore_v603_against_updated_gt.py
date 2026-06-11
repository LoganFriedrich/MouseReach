"""Re-score v6.0.3 snapshot algo outputs against the live (updated) GT.

After this session's 4 GT edits:
  20251027_CNT0404_P4 s11: displaced_sa -> retrieved
  20250822_CNT0110_P2 s6:  uncertain    -> retrieved
  20250625_CNT0102_P4 s17: IFR cleared  (still untouched)
  20250625_CNT0102_P4 s19: IFR cleared  (still untouched)

Writes metrics into a sidecar dir so the original snapshot stays intact.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.outcome.metrics import compute_outcome_metrics

SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots\outcome"
    r"\v6.0.3_fix_b_retrieved_rescue_2026-06-02"
)
ALGO_DIR = SNAPSHOT / "algo_outputs"
RESCORE_DIR = SNAPSHOT / "metrics_rescored_against_gt_2026-06-02"
GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\gt"
)


def main():
    RESCORE_DIR.mkdir(parents=True, exist_ok=True)
    video_ids = sorted({
        p.stem.replace("_unified_ground_truth", "")
        for p in GT_DIR.glob("*_unified_ground_truth.json")
    })
    print(f"Re-scoring v6.0.3 algo outputs against current GT ({len(video_ids)} videos)")
    print(f"Output: {RESCORE_DIR}")
    compute_outcome_metrics(
        gt_dir=GT_DIR,
        algo_dir=ALGO_DIR,
        output_dir=RESCORE_DIR,
        video_ids=video_ids,
        reaches_dir=ALGO_DIR,
    )
    print("Done.")


if __name__ == "__main__":
    main()

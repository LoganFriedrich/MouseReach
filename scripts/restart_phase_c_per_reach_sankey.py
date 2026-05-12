"""
Phase C extension: build the per-REACH Sankey.

Combines v8 reach predictions with v5 outcome predictions to produce
the per-reach confusion matrix that includes `absent` categories
(algo reaches with no GT match, GT reaches with no algo match).

Procedure:
  1. Train one v8 reach model on all exhaustive videos in train_pool,
     predict reaches on all train_pool videos. Save as
     <out>/algo_reaches/{video}_reaches.json with the schema
     compute_per_reach_confusion expects.
  2. Convert v5 LOOCV outcome predictions (already in
     outcome_per_segment.csv) into per-video
     <out>/algo_outcomes/{video}_pellet_outcomes.json files with the
     schema compute_per_reach_confusion expects.
  3. Call compute_per_reach_confusion + run_sankey, writing to
     <out>/v5+v8_per_reach_sankey/.

The v8 model here is in-sample on train_pool for predicting reaches
(less honest than LOOCV) but the v5 outcome predictions are still
LOOCV-held-out per video. For pure held-out reach predictions we'd
need a saved v8 LOOCV artifact, which the prior runs didn't persist.
We can rerun v8 LOOCV with prediction saving as a follow-up if needed
to make the reach side fully held-out too.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.features import feature_columns as reach_feature_columns
from mousereach.reach.v8.postprocess import probabilities_to_reaches


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
GT_DIR = QUARANTINE / "gt"
ALGO_INPUT_DIR = QUARANTINE / "algo_outputs"   # has segments JSONs (segmenter output)

OUT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\outcome\v5.0.0_dev_per_reach_sankey"
)
OUT_REACHES = OUT_ROOT / "algo_reaches"
OUT_OUTCOMES = OUT_ROOT / "algo_outcomes"

V5_SNAPSHOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\outcome\v5.0.0_dev_initial_loocv"
)


def train_v8_global() -> HistGradientBoostingClassifier:
    """Train one v8 reach model on all exhaustive train_pool videos."""
    print("Loading reach feature dataset ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    feat_cols = reach_feature_columns()

    train = df.loc[df["exhaustive"]]
    X = train[feat_cols].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int8)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    sw = np.where(y == 1, len(y) / (2.0 * n_pos),
                  len(y) / (2.0 * n_neg)).astype(np.float32)
    print(f"  Training on {len(X)} frames "
          f"({n_pos} in-reach, {n_neg} out-of-reach) ...", flush=True)
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        random_state=42, early_stopping=False)
    clf.fit(X, y, sample_weight=sw)
    return clf


def predict_v8_reaches_per_video(clf, train_pool_ids: List[str]) -> None:
    """Predict reaches per video and save as _reaches.json files."""
    OUT_REACHES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    feat_cols = reach_feature_columns()

    for vid in train_pool_ids:
        sub = df.loc[df["video_id"] == vid].sort_values("frame")
        Xv = sub[feat_cols].to_numpy(dtype=np.float32)
        proba = clf.predict_proba(Xv)[:, 1]
        algo_raw = probabilities_to_reaches(
            proba, threshold=0.5, merge_gap=2, min_span=3)

        # Group reaches by segment
        seg_path = ALGO_INPUT_DIR / f"{vid}_segments.json"
        seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
        boundaries = seg_data.get("boundaries", []) or []
        segments_index = []
        for i in range(len(boundaries) - 1):
            segments_index.append({
                "segment_num": i + 1,
                "start_frame": int(boundaries[i]),
                "end_frame": int(boundaries[i + 1]) - 1,
                "reaches": [],
            })

        for j, r in enumerate(algo_raw):
            for s in segments_index:
                if s["start_frame"] <= r.start_frame <= s["end_frame"]:
                    s["reaches"].append({
                        "reach_id": j,
                        "start_frame": int(r.start_frame),
                        "end_frame": int(r.end_frame),
                    })
                    break

        out = {
            "detector_version": "v8.0.0_dev",
            "video_name": vid,
            "segments": segments_index,
        }
        (OUT_REACHES / f"{vid}_reaches.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")


def write_v5_outcomes_per_video(train_pool_ids: List[str]) -> None:
    """Convert v5 LOOCV per-segment predictions to per-video pellet_outcomes JSONs.

    Causal reach attribution heuristic (placeholder for Phase D):
      - Touched segment (retrieved / displaced_sa): causal = last reach
        in the segment (per memory `gt_interaction_frame_human_process`,
        the GT interaction_frame is the LAST paw-over-pellet event).
      - Untouched / abnormal_exception: no causal reach.
    """
    OUT_OUTCOMES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(V5_SNAPSHOT / "metrics" / "outcome_per_segment.csv")

    for vid in train_pool_ids:
        sub = df.loc[df["video_id"] == vid].sort_values("segment_num")
        if len(sub) == 0:
            continue

        seg_path = ALGO_INPUT_DIR / f"{vid}_segments.json"
        seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
        boundaries = seg_data.get("boundaries", []) or []

        # Load v8 reaches for this video (already saved) so we can pick
        # a causal reach per touched segment.
        v8_path = OUT_REACHES / f"{vid}_reaches.json"
        v8_data = json.loads(v8_path.read_text(encoding="utf-8"))
        seg_reaches = {s["segment_num"]: s.get("reaches", []) for s in v8_data["segments"]}

        segs_out = []
        for _, row in sub.iterrows():
            sn = int(row["segment_num"])
            if sn - 1 < len(boundaries) - 1:
                start = int(boundaries[sn - 1])
                end = int(boundaries[sn]) - 1
            else:
                start = end = 0

            outcome = str(row["algo_outcome"])
            causal_id = None
            if outcome in ("retrieved", "displaced_sa", "displaced_outside"):
                rs = seg_reaches.get(sn, [])
                if rs:
                    last = max(rs, key=lambda r: r.get("end_frame", 0))
                    causal_id = last.get("reach_id")

            segs_out.append({
                "segment_num": sn,
                "start_frame": start,
                "end_frame": end,
                "outcome": outcome,
                "interaction_frame": None,
                "causal_reach_id": causal_id,
                "flagged_for_review": False,
                "confidence": 1.0,
            })

        out = {
            "detector_version": "v5.0.0_dev",
            "video_name": vid,
            "segments": segs_out,
        }
        (OUT_OUTCOMES / f"{vid}_pellet_outcomes.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8")


def build_per_reach_sankey(train_pool_ids: List[str]) -> None:
    """Use compute_per_reach_confusion to build the matrix, then call run_sankey."""
    from mousereach.improvement.outcome.metrics import compute_per_reach_confusion
    from mousereach.improvement.outcome._run_notebooks import run_sankey

    res = compute_per_reach_confusion(
        gt_dir=GT_DIR,
        algo_dir=OUT_OUTCOMES,
        reaches_dir=OUT_REACHES,
        video_ids=train_pool_ids,
    )

    # Wrap in scalars.json the run_sankey expects
    metrics_dir = OUT_ROOT / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scalars = {
        "n_videos": len(train_pool_ids),
        "n_reaches_universe": res["n_reaches_universe"],
        "outcome_label": {
            "strict_accuracy": None,
            "committed_accuracy": None,
            "abstention_rate": 0.0,
            "per_class": res["per_class"],
            "confusion_matrix": res["confusion_matrix"],
        },
    }
    (metrics_dir / "scalars.json").write_text(
        json.dumps(scalars, indent=2), encoding="utf-8")

    run_sankey(OUT_ROOT)

    print()
    print("=" * 70)
    print("PER-REACH CONFUSION MATRIX")
    print("=" * 70)
    print(f"Total reaches in universe: {res['n_reaches_universe']}")
    print()
    cm = res["confusion_matrix"]
    print("By GT class:")
    by_gt = defaultdict(lambda: defaultdict(int))
    for k, v in cm.items():
        a, b = k.split("__")
        by_gt[a][b] += v
    for gt_cls in sorted(by_gt):
        total = sum(by_gt[gt_cls].values())
        flows = sorted(by_gt[gt_cls].items(), key=lambda x: -x[1])
        flow_str = ", ".join(f"{algo}={n}" for algo, n in flows)
        print(f"  {gt_cls:>22s} (n={total}): {flow_str}")


def main():
    folds = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds["train_pool"]["video_ids"]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE C PER-REACH SANKEY")
    print("=" * 70)
    print()
    print("Step 1: Train one v8 reach model on all exhaustive train_pool videos ...")
    clf = train_v8_global()
    print()
    print("Step 2: Predict v8 reaches per video ...", flush=True)
    predict_v8_reaches_per_video(clf, train_pool_ids)
    print(f"  Saved {len(list(OUT_REACHES.glob('*_reaches.json')))} reach files", flush=True)
    print()
    print("Step 3: Convert v5 LOOCV per-segment outcomes to per-video JSONs ...")
    write_v5_outcomes_per_video(train_pool_ids)
    print(f"  Saved {len(list(OUT_OUTCOMES.glob('*_pellet_outcomes.json')))} outcome files")
    print()
    print("Step 4: Compute per-reach confusion + render Sankey ...")
    build_per_reach_sankey(train_pool_ids)


if __name__ == "__main__":
    main()

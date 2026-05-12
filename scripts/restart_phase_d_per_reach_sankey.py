"""
Phase D step 3: per-reach Sankey using v8 reaches + v5 outcomes +
v1 assignment classifier.

Fully integrated end-to-end:
  - v8 reach detector (in-sample on train_pool exhaustive videos --
    same artifacts as the prior per-reach Sankey)
  - v5 outcome detector (LOOCV per video)
  - v1 assignment classifier (trained on all GT reaches in train_pool;
    applied to v8 reaches)

Picks causal reach per touched segment by:
  1. For every v8 reach in the segment, compute v1 features.
  2. Score with the global v1 model.
  3. Pick max-prob reach as causal; emit causal_reach_id.

Then run the existing compute_per_reach_confusion + run_sankey path.

The v1 assignment classifier is trained globally (in-sample on all 37
train_pool videos) for this initial integration. A future iteration
should LOOCV the assignment classifier too for per-video honesty.
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

from mousereach.assignment.v1.features import (
    extract_reach_features, feature_columns as assign_feature_columns)
from mousereach.reach.v8.features import load_dlc_h5

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
GT_DIR = QUARANTINE / "gt"
ALGO_DIR = QUARANTINE / "algo_outputs"

# Artifacts from prior runs we'll reuse
PRIOR_PER_REACH = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\outcome\v5.0.0_dev_per_reach_sankey")
V8_REACHES_DIR = PRIOR_PER_REACH / "algo_reaches"
V5_OUTCOMES_DIR = PRIOR_PER_REACH / "algo_outcomes"

OUT_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\assignment\v1.0.0_dev_per_reach_sankey_with_triage")
OUT_OUTCOMES = OUT_ROOT / "algo_outcomes"

TRIAGE_THRESHOLD = 0.40


def train_assignment_global() -> HistGradientBoostingClassifier:
    print("Training assignment classifier globally on all train_pool GT reaches ...")
    df = pd.read_parquet(CORPUS_DIR / "phase_d_dataset" / "train_pool.parquet")
    feat_cols = assign_feature_columns()
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["causal"].to_numpy(dtype=np.int8)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    sw = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg)).astype(np.float32)
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4,
        random_state=42, early_stopping=False)
    clf.fit(X, y, sample_weight=sw)
    print(f"  Trained on {n} per-reach rows ({n_pos} causal, {n_neg} miss)")
    return clf


def assign_causal_for_video(
    vid: str, clf, dlc, v8_reaches_data: dict, v5_outcomes_data: dict,
    triage_threshold: float = 0.40,
) -> dict:
    """For each touched segment in v5 predictions, score v8 reaches and
    pick causal. If max causal probability across reaches in a touched
    segment < triage_threshold, mark the segment as `triaged` instead
    of committing to a causal reach.

    Returns updated v5 outcomes dict with causal_reach_id and (if
    triaged) outcome="triaged".
    """
    feat_cols = assign_feature_columns()

    v8_segs = {s["segment_num"]: s for s in v8_reaches_data["segments"]}
    new_segs = []

    for seg in v5_outcomes_data["segments"]:
        sn = seg["segment_num"]
        outcome = seg["outcome"]
        v8_seg = v8_segs.get(sn, {})
        v8_reaches = v8_seg.get("reaches", []) or []

        causal_id = None
        would_be_causal_id = None  # best-guess reach when triaged
        triaged = False
        max_proba = None

        if outcome in ("retrieved", "displaced_sa", "displaced_outside") and v8_reaches:
            seg_start = seg.get("start_frame", v8_seg.get("start_frame", 0))
            seg_end = seg.get("end_frame", v8_seg.get("end_frame", 0))
            ordered = sorted(v8_reaches, key=lambda r: r.get("start_frame", 0))
            n_reaches = len(ordered)

            feature_rows = []
            for order, r in enumerate(ordered):
                feats = extract_reach_features(
                    dlc_df=dlc,
                    reach_start=int(r["start_frame"]),
                    reach_end=int(r["end_frame"]),
                    seg_start=int(seg_start),
                    seg_end=int(seg_end),
                    reach_order=order,
                    n_reaches_in_segment=n_reaches,
                )
                feature_rows.append([feats[c] for c in feat_cols])
            X = np.array(feature_rows, dtype=np.float32)
            proba = clf.predict_proba(X)[:, 1]
            max_proba = float(proba.max())
            best = int(np.argmax(proba))
            best_reach_id = ordered[best]["reach_id"]
            if max_proba < triage_threshold:
                triaged = True
                would_be_causal_id = best_reach_id
            else:
                causal_id = best_reach_id
        elif outcome in ("retrieved", "displaced_sa", "displaced_outside") and not v8_reaches:
            # v5 says touched but v8 found no reaches -- triage as well
            triaged = True

        new_seg = dict(seg)
        if triaged:
            new_seg["outcome"] = "triaged"
            new_seg["causal_reach_id"] = None
            new_seg["would_be_causal_reach_id"] = would_be_causal_id
            new_seg["triage_max_proba"] = max_proba
        else:
            new_seg["causal_reach_id"] = causal_id
            new_seg["triage_max_proba"] = max_proba
        new_segs.append(new_seg)

    out = dict(v5_outcomes_data)
    out["segments"] = new_segs
    out["detector_version"] = "v5+v1_assignment"
    out["triage_threshold"] = triage_threshold
    return out


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_OUTCOMES.mkdir(exist_ok=True)
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds["train_pool"]["video_ids"]

    print("=" * 70)
    print("PHASE D PER-REACH SANKEY (v8 + v5 + v1 assignment)")
    print("=" * 70)
    print()

    clf = train_assignment_global()
    print()

    print(f"Applying assignment classifier to v8 reaches per video "
          f"(triage_threshold={TRIAGE_THRESHOLD}) ...", flush=True)
    n_triaged = 0
    n_committed_touched = 0
    for vid in train_pool_ids:
        v8_path = V8_REACHES_DIR / f"{vid}_reaches.json"
        v5_path = V5_OUTCOMES_DIR / f"{vid}_pellet_outcomes.json"
        if not v8_path.exists() or not v5_path.exists():
            print(f"  SKIP {vid}: missing prior artifact")
            continue
        dlc = load_dlc_h5(next((QUARANTINE / "dlc").glob(f"{vid}DLC_*.h5")))
        v8_data = json.loads(v8_path.read_text(encoding="utf-8"))
        v5_data = json.loads(v5_path.read_text(encoding="utf-8"))
        new_outcomes = assign_causal_for_video(
            vid, clf, dlc, v8_data, v5_data,
            triage_threshold=TRIAGE_THRESHOLD)
        (OUT_OUTCOMES / f"{vid}_pellet_outcomes.json").write_text(
            json.dumps(new_outcomes, indent=2), encoding="utf-8")
        for s in new_outcomes["segments"]:
            if s["outcome"] == "triaged":
                n_triaged += 1
            elif s["outcome"] in ("retrieved", "displaced_sa", "displaced_outside"):
                n_committed_touched += 1
    print(f"  Wrote {len(list(OUT_OUTCOMES.glob('*_pellet_outcomes.json')))} files")
    print(f"  Triaged: {n_triaged} segments  |  Committed touched: {n_committed_touched}")
    print()

    # Compute per-reach Sankey using updated outcomes (with causal_reach_id)
    from mousereach.improvement.outcome.metrics import compute_per_reach_confusion
    from mousereach.improvement.outcome._run_notebooks import run_sankey

    print("Computing per-reach confusion matrix ...", flush=True)
    res = compute_per_reach_confusion(
        gt_dir=GT_DIR,
        algo_dir=OUT_OUTCOMES,
        reaches_dir=V8_REACHES_DIR,
        video_ids=train_pool_ids,
    )

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
    print("PER-REACH CONFUSION (with assignment)")
    print("=" * 70)
    print(f"Total reaches in universe: {res['n_reaches_universe']}")
    cm = res["confusion_matrix"]
    by_gt = defaultdict(lambda: defaultdict(int))
    for k, v in cm.items():
        a, b = k.split("__")
        by_gt[a][b] += v
    print()
    print("By GT class:")
    for gt_cls in sorted(by_gt):
        total = sum(by_gt[gt_cls].values())
        flows = sorted(by_gt[gt_cls].items(), key=lambda x: -x[1])
        flow_str = ", ".join(f"{algo}={n}" for algo, n in flows)
        print(f"  {gt_cls:>22s} (n={total}): {flow_str}")
    print()
    print(f"Sankey: {OUT_ROOT / 'figures' / 'sankey.png'}")


if __name__ == "__main__":
    main()

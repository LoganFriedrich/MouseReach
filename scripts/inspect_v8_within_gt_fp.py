"""
v8 diagnostic: inspect raw probability series for within_gt FP cases.

Purpose: verify the mid-reach probability dip hypothesis. Two
boundary-related experiments (proba_smoothing rejected, BSW accepted
but with minimal FP movement) failed to materially address the 275
within_gt FPs identified in v8.0.0_dev_failure_mode_breakdown. Before
designing more experiments, see what's actually happening for these
cases.

For each chosen video, this script:
  1. Loads the train_pool.parquet (read-only)
  2. Identifies the LOOCV fold for that video (train = pool minus this
     video, val = this video)
  3. Trains the GBM on the fold (HistGradientBoostingClassifier, same
     hyperparams as baseline)
  4. Runs inference on the val video to recover the per-frame
     probability series
  5. Loads the within_gt FP cases for that video from the existing
     v8.0.0_dev_failure_mode_breakdown extended-schema aggregate
  6. For up to MAX_CASES_PER_VIDEO of those cases, generates a figure
     showing the probability series + GT reach window + matched algo
     reach (TP) + within_gt FP, all overlaid in a 50-frame window
     centered on the GT reach

NO existing module code modified.

Output:
  Improvement_Snapshots/reach_detection/v8.0.0_dev_within_gt_fp_inspection/
    figures/<video_id>_<reach_idx>.png      one per inspected case
    metrics/inspection_metadata.json        which cases inspected, where
    RESULTS.md                              prose summary

This is purely informational. No algorithm change. No accept/reject
decision attached to the output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.eval import (
    GTReach, AlgoReach, evaluate_reaches, summarize_results,
)
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.reach.v8.features import feature_columns


CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)
BREAKDOWN_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_failure_mode_breakdown"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_within_gt_fp_inspection"
)

# Top videos by within_gt FP count (from fp_breakdown.json):
#   20250812_CNT0301_P3: 72
#   20251022_CNT0413_P4: 47
#   20251010_CNT0308_P2: 35
TARGET_VIDEOS = [
    "20250812_CNT0301_P3",
    "20251022_CNT0413_P4",
    "20251010_CNT0308_P2",
]
MAX_CASES_PER_VIDEO = 4
WINDOW_PADDING = 25  # frames on each side of the GT reach in plots

THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3


def find_within_gt_fps_for_video(raw_results, video_id):
    """Return list of (fp_record, matched_gt_record, matched_tp_record)
    tuples for within_gt FPs in this video.

    A within_gt FP is one whose algo_start_frame falls between the
    gt_start_frame and gt_end_frame of some GT reach in the same video.
    """
    rs = [r for r in raw_results if r["video_id"] == video_id]
    gt_records = [r for r in rs if r["status"] in ("tp", "fn")]
    fp_records = [r for r in rs if r["status"] == "fp"]
    tp_records = [r for r in rs if r["status"] == "tp"]

    cases = []
    for fp in fp_records:
        fp_s = fp["algo_start_frame"]
        for gt in gt_records:
            if gt["gt_start_frame"] <= fp_s <= gt["gt_end_frame"]:
                # find the TP that matched this GT (by gt_index)
                matched_tp = None
                for tp in tp_records:
                    if tp["gt_index"] == gt["gt_index"]:
                        matched_tp = tp
                        break
                cases.append((fp, gt, matched_tp))
                break
    return cases


def train_fold_for_video(df, train_pool_ids, val_vid, feat_cols):
    """Train v8 GBM on a fold (val = val_vid, train = pool minus val_vid).
    Same hyperparams as baseline train_one_fold.
    """
    train_mask = df["video_id"].isin(train_pool_ids) & df["exhaustive"]
    train_mask &= df["video_id"] != val_vid
    train = df.loc[train_mask]

    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)

    n = len(y_train)
    n_pos = int(y_train.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        sample_weight = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)
    else:
        sample_weight = None

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        random_state=42, early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    return clf


def get_per_frame_proba(df, val_vid, clf, feat_cols):
    """Return (frames, proba) arrays for val_vid in frame-sorted order."""
    val = df.loc[df["video_id"] == val_vid].sort_values("frame")
    Xv = val[feat_cols].to_numpy(dtype=np.float32)
    proba = clf.predict_proba(Xv)[:, 1]
    frames = val["frame"].to_numpy()
    return frames, proba


def plot_case(video_id, fp, gt, matched_tp, frames, proba, fig_dir):
    """Plot a single within_gt FP case showing probability series with
    GT/TP/FP overlaid.
    """
    gt_s = gt["gt_start_frame"]
    gt_e = gt["gt_end_frame"]
    fp_s = fp["algo_start_frame"]
    fp_e = fp["algo_end_frame"]

    plot_lo = max(0, gt_s - WINDOW_PADDING)
    plot_hi = gt_e + WINDOW_PADDING

    # Extract probability for the window
    mask = (frames >= plot_lo) & (frames <= plot_hi)
    f = frames[mask]
    p = proba[mask]

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)

    # GT reach window
    ax.axvspan(gt_s, gt_e + 0.5, alpha=0.18, color="#2E7D32",
               label=f"GT reach [{gt_s},{gt_e}]")

    # Matched TP reach window
    if matched_tp is not None:
        tp_s = matched_tp["algo_start_frame"]
        tp_e = matched_tp["algo_end_frame"]
        ax.axvspan(tp_s, tp_e + 0.5, alpha=0.18, color="#1976D2",
                   label=f"matched TP algo [{tp_s},{tp_e}] "
                         f"(start_delta={matched_tp['start_delta']}, "
                         f"span_delta={matched_tp['span_delta']})")

    # within_gt FP reach window
    ax.axvspan(fp_s, fp_e + 0.5, alpha=0.30, color="#D32F2F",
               label=f"within_gt FP [{fp_s},{fp_e}]")

    # Probability series
    ax.plot(f, p, color="#212121", linewidth=1.4, marker=".",
            markersize=4, label="GBM proba")

    # Threshold line
    ax.axhline(THRESHOLD, color="#9E9E9E", linewidth=1, linestyle="--",
               label=f"threshold={THRESHOLD}")

    ax.set_xlim(plot_lo, plot_hi)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("frame")
    ax.set_ylabel("p(in-reach)")
    ax.set_title(f"{video_id}  --  within_gt FP inspection (gt_idx={gt['gt_index']})",
                 fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = fig_dir / f"{video_id}_gt{gt['gt_index']}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    print("=" * 70)
    print("v8 WITHIN_GT FP INSPECTION")
    print("=" * 70)
    print()

    print("Loading aggregate from failure_mode_breakdown ...", flush=True)
    agg = json.loads(
        (BREAKDOWN_DIR / "metrics" / "loocv_aggregate.json").read_text(
            encoding="utf-8"))
    raw = agg["raw_results"]
    print(f"  {len(raw)} events total")

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    feat_cols = feature_columns()
    print()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = SNAPSHOT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    inspected_cases = []

    for vid in TARGET_VIDEOS:
        print(f"--- video: {vid} ---", flush=True)
        cases = find_within_gt_fps_for_video(raw, vid)
        print(f"  {len(cases)} within_gt FPs in this video; "
              f"plotting up to {MAX_CASES_PER_VIDEO}")

        if not cases:
            print("  (none, skipping)")
            continue

        # Train fold for this video
        print(f"  training fold (excl {vid}) ...", flush=True)
        clf = train_fold_for_video(df, train_pool_ids, vid, feat_cols)

        # Get probability series
        frames, proba = get_per_frame_proba(df, vid, clf, feat_cols)
        print(f"  proba shape={proba.shape}; "
              f"min={proba.min():.3f} mean={proba.mean():.3f} max={proba.max():.3f}")

        # Plot up to MAX_CASES_PER_VIDEO cases, spread across the video
        # (sample evenly by GT index to get diversity)
        if len(cases) > MAX_CASES_PER_VIDEO:
            step = len(cases) / MAX_CASES_PER_VIDEO
            chosen = [cases[int(step * i)] for i in range(MAX_CASES_PER_VIDEO)]
        else:
            chosen = cases

        for fp, gt, matched_tp in chosen:
            out = plot_case(vid, fp, gt, matched_tp,
                            frames, proba, figures_dir)
            print(f"    wrote {out.name}")
            inspected_cases.append({
                "video_id": vid,
                "gt_index": gt["gt_index"],
                "gt_start_frame": gt["gt_start_frame"],
                "gt_end_frame": gt["gt_end_frame"],
                "matched_tp_start": matched_tp["algo_start_frame"] if matched_tp else None,
                "matched_tp_end": matched_tp["algo_end_frame"] if matched_tp else None,
                "matched_tp_start_delta": matched_tp["start_delta"] if matched_tp else None,
                "matched_tp_span_delta": matched_tp["span_delta"] if matched_tp else None,
                "fp_start_frame": fp["algo_start_frame"],
                "fp_end_frame": fp["algo_end_frame"],
                "figure_path": str(out),
            })

        print()

    # Save metadata
    (metrics_dir / "inspection_metadata.json").write_text(
        json.dumps({
            "inspected_videos": TARGET_VIDEOS,
            "max_cases_per_video": MAX_CASES_PER_VIDEO,
            "window_padding_frames": WINDOW_PADDING,
            "n_cases_inspected": len(inspected_cases),
            "cases": inspected_cases,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote metadata: {metrics_dir / 'inspection_metadata.json'}")
    print(f"Wrote {len(inspected_cases)} figures to: {figures_dir}")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()

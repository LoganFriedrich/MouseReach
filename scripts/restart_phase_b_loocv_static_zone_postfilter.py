"""
v8 dev experiment: static-zone TRIM post-filter on top of BSW b=1 w=0.8.

Replaces an earlier drop-based draft. Drops whole reaches when the paw
fails the zone check throw away valid detections; we want to TRIM the
over-extending tail (and any leading frames where the paw is confirmed
out of the zone) instead. The target is the within_gt FP class
identified in the 2026-05-04 within_gt FP inspection: algo extends past
GT_end into paw retraction motion. When the paw retracts back through
the slit it leaves the zone, which is the signal to trim.

================================================================
ALGORITHM
================================================================

For each algo reach window [algo_start, algo_end]:

  1. Compute a per-frame "in zone" classification:
       - True     iff paw is visible AND zone defined AND paw position
                   inside the zone bounds at that frame
       - False    iff paw is visible AND zone defined AND paw position
                   outside the zone bounds
       - Unknown  otherwise (paw not visible, or zone undefined at that
                   frame -- e.g. landmarks were low-confidence and the
                   rolling-median had no value within the window)

  2. Trim leading False frames forward (new_start += 1 while the current
     start frame is confirmed-out-of-zone). Stop at the first non-False
     frame (in-zone or unknown). Unknown frames are KEPT -- absence of
     evidence is not evidence of absence.

  3. Trim trailing False frames backward symmetrically.

  4. If the trimmed window has fewer than MIN_SPAN_AFTER_TRIM frames,
     drop the reach (no meaningful detection survived).

  5. Sequentially apply a nose-at-slit gate on the TRIMMED window. If
     fewer than T2 of trimmed-reach frames have the nose within
     NOSE_PROXIMITY_PX of the slit centre, drop the reach (mouse was
     not in reaching posture during the residual window).

The action is principled: cut over-extending tails and over-extending
heads, but only on confirmed evidence the paw was outside the zone.
Real reaches with intermittent DLC dropouts will see no boundary
change. Within_gt FPs with paw-retraction tails will have those tails
trimmed off.

================================================================
PRE-EXPERIMENT CHECKLIST (per pre_experiment_checklist.md)
================================================================

1. Cumulative-stacking check (verified 2026-05-18):
   - v8.0.0 production already integrates BSW b=1, w=0.8 (commit
     79f217f, pipeline_versions.json reach_detector: 8.0.0).
   - Comparison baseline: production v8.0.0 LOOCV (BSW w=0.8 result
     TP=1935 / FP=330 / FN=440 / exact_start=84.08%).
   - Stacked improvements applied: BSW b=1 w=0.8 inline (identical to
     restart_phase_b_loocv_boundary_sample_weight_w08.py).

2. Existing-module-modification check:
   - Existing module code modified: NO. All zone/trim/nose logic is
     inline in this runner; no edits under src/mousereach/.

3. Assumption check (unverified hypotheses):
   - HYP: Trimming the trailing out-of-zone frames reduces within_gt
     FPs by converting "algo over-extends past GT_end" cases into
     properly bounded TPs. This is the targeted mechanism per the
     2026-05-04 inspection.
   - HYP: Real reaches keep the paw in-zone for the duration; brief
     out-of-zone single-frame blips inside a real reach do not
     propagate to the trim because trim acts only on the boundaries.
   - HYP: When the paw is unknown at the boundary (DLC dropout), the
     trim is conservative and leaves the boundary alone. This avoids
     over-trimming.
   - HYP: The nose-at-slit gate on the TRIMMED window is appropriate.
     T2=0.50 is a generous threshold from biomechanics; nose engagement
     during the residual reach window should be at least 50% if the
     mouse was actually reaching.
   - HYP: Expected magnitude is small but should preserve TP and reduce
     FP. Per the value-drift diagnostic, the within_gt FP class is ~52%
     of calibration FPs (173 events). Even partial trimming of those
     should reduce FP without TP loss.

4. FN-direction-reporting check:
   - First line of planned RESULTS.md prose:
     "FN vs cumulative best (v8.0.0 BSW w=0.8): [direction + magnitude];
      FN vs pure baseline (v8.0.0 no BSW): [direction + magnitude]."
   - Two-delta surfacing BEFORE any metric table.

5. Framework-not-adhoc check:
   - Output: Improvement_Snapshots/reach_detection/
     v8.0.0_dev_static_zone_trim_postfilter/
   - Canonical metrics layout (loocv_aggregate.json with extended
     schema + loocv_per_fold.json + filter_actions.json).
   - Canonical figure runner: render_v8_reach_figures.

6. Branch + tag check (deferred to user before run):
   - Tag: v8-pre-static-zone-trim-postfilter-2026-05-18
   - Branch: feature/v8-static-zone-trim-postfilter

7. Decision-rule check (vs cumulative best = BSW w=0.8 LOOCV):
   - REJECT if TP drops AND FN rises (vs cumulative best).
   - REJECT if exact-frame-start match rate drops > 0.3 pp.
   - ACCEPT if FN drops or TP rises with exact_start within 0.3 pp,
     AND the strict-rule FP count drops meaningfully (target: cut some
     fraction of the 173 within_gt FPs).
   - DO NOT retune the trim params, T2, or zone smoothing in response
     to results. If trimming hurts, the answer is "this design didn't
     work as specified", not "tune until it does."

================================================================
ZONE DEFINITION (locked 2026-05-15, unchanged from drop draft)
================================================================

  Top edge    = median(BOXL_y, BOXR_y) per frame  (slit line)
  Bottom edge = median(SABL_y, SABR_y) per frame  (SA bottom)
  Left edge   = min(SATL_x, SABL_x, BOXL_x) per frame  (outer envelope)
  Right edge  = max(SATR_x, SABR_x, BOXR_x) per frame  (outer envelope)

  Each landmark series smoothed with ZONE_SMOOTH_WIN rolling median
  centered, likelihood-gated at ZONE_LK_MIN, forward-filled across gaps.

================================================================
OUTPUT
================================================================

  Improvement_Snapshots/reach_detection/v8.0.0_dev_static_zone_trim_postfilter/
    metrics/
      loocv_aggregate.json
      loocv_per_fold.json
      filter_actions.json         # per-reach trim records + drops
    figures/
      reach_detection_summary.png # canonical
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.eval import (
    GTReach, AlgoReach, MatchResult, evaluate_reaches, summarize_results,
)
from mousereach.reach.v8.postprocess import probabilities_to_reaches
from mousereach.reach.v8.features import feature_columns
from mousereach.improvement.reach_detection.v8_figures import render_v8_reach_figures


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)

SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_dev_static_zone_trim_postfilter"
)


# ---------------------------------------------------------------------------
# Stacked-improvement params (BSW b=1 w=0.8, the cumulative-best baseline)
# ---------------------------------------------------------------------------

BOUNDARY_BUFFER = 1
BOUNDARY_WEIGHT = 0.8

# v8 postprocess params -- match production exactly
THRESHOLD = 0.5
MERGE_GAP = 2
MIN_SPAN = 3


# ---------------------------------------------------------------------------
# Trim-postfilter params -- fixed, not GT-calibrated
# ---------------------------------------------------------------------------

# After trim, drop reaches shorter than this many frames. Matches v8
# production's MIN_SPAN so the trim cannot produce reaches shorter than
# what the detector itself emits.
MIN_SPAN_AFTER_TRIM = MIN_SPAN

# Nose-at-slit gate threshold (applied to trimmed window). Justified
# from biomechanics: during the reach the nose should be near the slit
# for at least half the frames.
T2_NOSE_AT_SLIT = 0.50

# Zone smoothing + likelihood gating
ZONE_SMOOTH_WIN = 15
ZONE_LK_MIN = 0.5
PAW_LK_MIN = 0.5
NOSE_LK_MIN = 0.5
NOSE_PROXIMITY_PX = 25.0

# Landmark sets
PAW_BPS = ("RightHand", "RHLeft", "RHOut", "RHRight")
ZONE_LANDMARK_BPS = ("BOXL", "BOXR", "SATL", "SATR", "SABL", "SABR")


# ---------------------------------------------------------------------------
# Cumulative-stacking: BSW boundary weights (copied verbatim from
# restart_phase_b_loocv_boundary_sample_weight_w08.py)
# ---------------------------------------------------------------------------

def compute_boundary_weights(train_df, n_buffer=1, boundary_weight=0.5):
    sorted_df = train_df.sort_values(["video_id", "frame"])
    rid = sorted_df["reach_id"].to_numpy()
    vid = sorted_df["video_id"].to_numpy()
    n = len(sorted_df)

    transitions = np.zeros(n, dtype=bool)
    if n >= 2:
        same_video = vid[1:] == vid[:-1]
        rid_change = rid[1:] != rid[:-1]
        boundary_pairs = same_video & rid_change
        transitions[1:] |= boundary_pairs
        transitions[:-1] |= boundary_pairs

    dilated = transitions.copy()
    for d in range(1, n_buffer + 1):
        dilated[d:] |= transitions[:-d]
        dilated[:-d] |= transitions[d:]

    weights_sorted = np.ones(n, dtype=np.float32)
    weights_sorted[dilated] = boundary_weight
    weights_series = pd.Series(weights_sorted, index=sorted_df.index)
    return weights_series.reindex(train_df.index).to_numpy()


# ---------------------------------------------------------------------------
# Zone construction
# ---------------------------------------------------------------------------

def _gated_series(values: np.ndarray, lk: np.ndarray, lk_min: float) -> np.ndarray:
    out = values.astype(np.float64).copy()
    out[lk < lk_min] = np.nan
    return out


def _rolling_median_with_ffill(s: pd.Series, window: int) -> np.ndarray:
    smoothed = s.rolling(window=window, center=True, min_periods=1).median()
    smoothed = smoothed.ffill().bfill()
    return smoothed.to_numpy()


def compute_zone_per_frame(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    df_sorted = df.sort_values("frame").reset_index(drop=True)

    smoothed = {}
    for bp in ZONE_LANDMARK_BPS:
        x_col = f"{bp}_x"
        y_col = f"{bp}_y"
        lk_col = f"{bp}_lk"
        if x_col not in df_sorted.columns or lk_col not in df_sorted.columns:
            smoothed[f"{bp}_x"] = np.full(len(df_sorted), np.nan)
            smoothed[f"{bp}_y"] = np.full(len(df_sorted), np.nan)
            continue
        x = df_sorted[x_col].to_numpy()
        y = df_sorted[y_col].to_numpy()
        lk = df_sorted[lk_col].to_numpy()
        x_g = _gated_series(x, lk, ZONE_LK_MIN)
        y_g = _gated_series(y, lk, ZONE_LK_MIN)
        smoothed[f"{bp}_x"] = _rolling_median_with_ffill(pd.Series(x_g), ZONE_SMOOTH_WIN)
        smoothed[f"{bp}_y"] = _rolling_median_with_ffill(pd.Series(y_g), ZONE_SMOOTH_WIN)

    stacked_top = np.vstack([smoothed["BOXL_y"], smoothed["BOXR_y"]])
    top_y = np.nanmedian(stacked_top, axis=0)
    stacked_bot = np.vstack([smoothed["SABL_y"], smoothed["SABR_y"]])
    bottom_y = np.nanmedian(stacked_bot, axis=0)
    stacked_left = np.vstack([smoothed["SATL_x"], smoothed["SABL_x"], smoothed["BOXL_x"]])
    left_x = np.nanmin(stacked_left, axis=0)
    stacked_right = np.vstack([smoothed["SATR_x"], smoothed["SABR_x"], smoothed["BOXR_x"]])
    right_x = np.nanmax(stacked_right, axis=0)

    return {
        "frame_index": df_sorted["frame"].to_numpy(),
        "top_y": top_y,
        "bottom_y": bottom_y,
        "left_x": left_x,
        "right_x": right_x,
    }


# ---------------------------------------------------------------------------
# Per-frame paw centre + zone classification
# ---------------------------------------------------------------------------

def compute_paw_center(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_sorted = df.sort_values("frame").reset_index(drop=True)
    n = len(df_sorted)
    xs = []
    ys = []
    for bp in PAW_BPS:
        x_col = f"{bp}_x"; y_col = f"{bp}_y"; lk_col = f"{bp}_lk"
        if x_col not in df_sorted.columns:
            xs.append(np.full(n, np.nan)); ys.append(np.full(n, np.nan))
            continue
        x = df_sorted[x_col].to_numpy().astype(np.float64)
        y = df_sorted[y_col].to_numpy().astype(np.float64)
        lk = df_sorted[lk_col].to_numpy()
        mask = lk >= PAW_LK_MIN
        x = np.where(mask, x, np.nan)
        y = np.where(mask, y, np.nan)
        xs.append(x); ys.append(y)
    xs_arr = np.vstack(xs)
    ys_arr = np.vstack(ys)
    paw_x = np.nanmean(xs_arr, axis=0)
    paw_y = np.nanmean(ys_arr, axis=0)
    visible = ~np.isnan(paw_x)
    return paw_x, paw_y, visible


def classify_in_zone_per_frame(zone: Dict[str, np.ndarray],
                               paw_x: np.ndarray,
                               paw_y: np.ndarray,
                               paw_vis: np.ndarray) -> np.ndarray:
    """Per-frame classification:
        +1  = confirmed in zone (paw visible, zone defined, position inside)
        -1  = confirmed out of zone (paw visible, zone defined, position outside)
         0  = unknown (paw not visible OR zone not defined)
    """
    top_y = zone["top_y"]; bot_y = zone["bottom_y"]
    lx = zone["left_x"]; rx = zone["right_x"]
    zone_defined = ~(np.isnan(top_y) | np.isnan(bot_y) | np.isnan(lx) | np.isnan(rx))

    decidable = paw_vis & zone_defined
    inside = (paw_x >= lx) & (paw_x <= rx) & (paw_y >= top_y) & (paw_y <= bot_y)
    out = np.zeros_like(paw_x, dtype=np.int8)
    out[decidable & inside] = 1
    out[decidable & ~inside] = -1
    return out


# ---------------------------------------------------------------------------
# Nose-at-slit
# ---------------------------------------------------------------------------

def compute_nose_to_slit_dist(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df_sorted = df.sort_values("frame").reset_index(drop=True)
    n = len(df_sorted)
    if "Nose_x" not in df_sorted.columns:
        return np.full(n, np.nan), np.zeros(n, dtype=bool)
    nose_x = df_sorted["Nose_x"].to_numpy().astype(np.float64)
    nose_y = df_sorted["Nose_y"].to_numpy().astype(np.float64)
    nose_lk = df_sorted["Nose_lk"].to_numpy()
    nose_ok = nose_lk >= NOSE_LK_MIN
    nose_x[~nose_ok] = np.nan
    nose_y[~nose_ok] = np.nan

    def _smooth(bp: str) -> Tuple[np.ndarray, np.ndarray]:
        x = df_sorted[f"{bp}_x"].to_numpy().astype(np.float64)
        y = df_sorted[f"{bp}_y"].to_numpy().astype(np.float64)
        lk = df_sorted[f"{bp}_lk"].to_numpy()
        x_g = _gated_series(x, lk, ZONE_LK_MIN)
        y_g = _gated_series(y, lk, ZONE_LK_MIN)
        sx = _rolling_median_with_ffill(pd.Series(x_g), ZONE_SMOOTH_WIN)
        sy = _rolling_median_with_ffill(pd.Series(y_g), ZONE_SMOOTH_WIN)
        return sx, sy

    boxl_x, boxl_y = _smooth("BOXL")
    boxr_x, boxr_y = _smooth("BOXR")
    slit_x = np.nanmean(np.vstack([boxl_x, boxr_x]), axis=0)
    slit_y = np.nanmean(np.vstack([boxl_y, boxr_y]), axis=0)

    dist = np.sqrt((nose_x - slit_x) ** 2 + (nose_y - slit_y) ** 2)
    return dist, nose_ok


def nose_at_slit_ratio(start: int, end: int, nose_dist: np.ndarray,
                       nose_vis: np.ndarray
                       ) -> Tuple[Optional[float], int, int]:
    if end < start:
        return None, 0, 0
    sl = slice(start, end + 1)
    d = nose_dist[sl]; v = nose_vis[sl]
    valid = v & ~np.isnan(d)
    near = valid & (d < NOSE_PROXIMITY_PX)
    n_valid = int(valid.sum())
    n_near = int(near.sum())
    if n_valid == 0:
        return None, 0, 0
    return n_near / n_valid, n_near, n_valid


# ---------------------------------------------------------------------------
# Trim logic
# ---------------------------------------------------------------------------

def trim_to_zone(start: int, end: int, in_zone: np.ndarray
                 ) -> Tuple[Optional[int], Optional[int], int, int]:
    """Trim leading and trailing confirmed-out-of-zone frames.

    Stops at the first non-(-1) frame from each end. Unknown frames (0)
    are kept; in-zone frames (+1) are kept.

    Returns (new_start, new_end, frames_trimmed_start, frames_trimmed_end).
    If the trim consumes the entire window, returns (None, None, ...).
    """
    if end < start:
        return None, None, 0, 0
    new_start = start
    new_end = end
    while new_start <= new_end and in_zone[new_start] == -1:
        new_start += 1
    while new_end >= new_start and in_zone[new_end] == -1:
        new_end -= 1
    if new_end < new_start:
        return None, None, end - start + 1, 0
    return (new_start, new_end,
            new_start - start, end - new_end)


# ---------------------------------------------------------------------------
# Per-fold: train BSW w=0.8 inline, predict, apply trim+nose postfilter
# ---------------------------------------------------------------------------

def train_predict_and_filter(
    train_pool_df: pd.DataFrame,
    train_video_ids: List[str],
    val_vid: str,
    feat_cols: List[str],
):
    """One fold: train GBM with BSW b=1 w=0.8 on train_video_ids, predict on
    val_vid, apply zone-trim + nose post-filters, evaluate.

    Returns (summary, results, kept_algo_reaches, gt_reaches, filter_actions).
    """
    train_mask = train_pool_df["video_id"].isin(train_video_ids)
    train_mask &= train_pool_df["exhaustive"]
    train = train_pool_df.loc[train_mask]
    val = train_pool_df.loc[train_pool_df["video_id"] == val_vid]

    X_train = train[feat_cols].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int8)

    n = len(y_train)
    n_pos = int(y_train.sum())
    n_neg = n - n_pos
    if n_pos > 0 and n_neg > 0:
        w_pos = n / (2.0 * n_pos)
        w_neg = n / (2.0 * n_neg)
        class_w = np.where(y_train == 1, w_pos, w_neg).astype(np.float32)
    else:
        class_w = np.ones(n, dtype=np.float32)

    boundary_w = compute_boundary_weights(
        train, n_buffer=BOUNDARY_BUFFER, boundary_weight=BOUNDARY_WEIGHT)
    sample_weight = (class_w * boundary_w).astype(np.float32)

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=6,
        random_state=42, early_stopping=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)

    Xv = val[feat_cols].to_numpy(dtype=np.float32)
    proba = clf.predict_proba(Xv)[:, 1]

    raw_reaches = probabilities_to_reaches(
        proba, threshold=THRESHOLD, merge_gap=MERGE_GAP, min_span=MIN_SPAN)

    # Per-video zone, paw centre, nose distance
    zone = compute_zone_per_frame(val)
    paw_x, paw_y, paw_vis = compute_paw_center(val)
    in_zone = classify_in_zone_per_frame(zone, paw_x, paw_y, paw_vis)
    nose_dist, nose_vis = compute_nose_to_slit_dist(val)

    kept = []
    filter_actions = []
    for i, r in enumerate(raw_reaches):
        orig_s, orig_e = r.start_frame, r.end_frame

        new_s, new_e, trim_start, trim_end = trim_to_zone(orig_s, orig_e, in_zone)
        if new_s is None:
            filter_actions.append({
                "video_id": val_vid, "raw_index": i,
                "orig_start": orig_s, "orig_end": orig_e,
                "action": "dropped_no_in_zone_frames",
                "trim_start": trim_start, "trim_end": trim_end,
            })
            continue

        trimmed_span = new_e - new_s + 1
        if trimmed_span < MIN_SPAN_AFTER_TRIM:
            filter_actions.append({
                "video_id": val_vid, "raw_index": i,
                "orig_start": orig_s, "orig_end": orig_e,
                "new_start": new_s, "new_end": new_e,
                "action": "dropped_trimmed_too_short",
                "trim_start": trim_start, "trim_end": trim_end,
                "trimmed_span": trimmed_span,
            })
            continue

        # Nose gate on trimmed window
        nose_r, _, _ = nose_at_slit_ratio(new_s, new_e, nose_dist, nose_vis)
        if nose_r is not None and nose_r < T2_NOSE_AT_SLIT:
            filter_actions.append({
                "video_id": val_vid, "raw_index": i,
                "orig_start": orig_s, "orig_end": orig_e,
                "new_start": new_s, "new_end": new_e,
                "action": "dropped_nose_at_slit",
                "trim_start": trim_start, "trim_end": trim_end,
                "nose_at_slit_ratio": nose_r,
            })
            continue

        if (trim_start, trim_end) != (0, 0):
            filter_actions.append({
                "video_id": val_vid, "raw_index": i,
                "orig_start": orig_s, "orig_end": orig_e,
                "new_start": new_s, "new_end": new_e,
                "action": "trimmed",
                "trim_start": trim_start, "trim_end": trim_end,
                "nose_at_slit_ratio": nose_r,
            })

        kept.append((new_s, new_e))

    algo_reaches = [
        AlgoReach(start_frame=s, end_frame=e, video_id=val_vid, index=i)
        for i, (s, e) in enumerate(kept)
    ]

    sub = val.sort_values("frame")
    rid = sub["reach_id"].to_numpy()
    frames = sub["frame"].to_numpy()
    gt_reaches = []
    for ri in sorted(set(rid[rid >= 0].tolist())):
        m = rid == ri
        f = frames[m]
        gt_reaches.append(GTReach(
            start_frame=int(f.min()), end_frame=int(f.max()),
            video_id=val_vid, index=ri))

    results = evaluate_reaches(algo_reaches, gt_reaches, video_id=val_vid)
    summary = summarize_results(results)
    return summary, results, algo_reaches, gt_reaches, filter_actions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PHASE B LOOCV -- STATIC-ZONE TRIM POSTFILTER on top of BSW b=1 w=0.8")
    print(f"Trim acts on confirmed out-of-zone frames; nose gate T2={T2_NOSE_AT_SLIT}")
    print("=" * 70)
    print()

    print("Loading train_pool.parquet ...", flush=True)
    df = pd.read_parquet(CORPUS_DIR / "phase_b_dataset" / "train_pool.parquet")
    folds_def = json.loads(
        (CORPUS_DIR / "cv_folds.json").read_text(encoding="utf-8"))
    train_pool_ids = folds_def["train_pool"]["video_ids"]
    print(f"  Train pool: {len(train_pool_ids)} videos", flush=True)
    print()

    feat_cols = feature_columns()
    exh_set = set(df.loc[df["exhaustive"], "video_id"].unique().tolist())
    eligible_val = [v for v in train_pool_ids if v in exh_set]
    print(f"LOOCV exhaustive folds: {len(eligible_val)}")
    print()

    folds = []
    per_video_data = {}
    all_results_combined = []
    all_filter_actions = []

    for i, val_vid in enumerate(eligible_val):
        train_ids = [v for v in train_pool_ids if v != val_vid]
        print(f"  fold {i+1}/{len(eligible_val)}: val={val_vid}", flush=True)
        summary, results, algo_reaches, gt_reaches, filter_actions = \
            train_predict_and_filter(df, train_ids, val_vid, feat_cols)
        s = summary
        n_trim = sum(1 for a in filter_actions if a["action"] == "trimmed")
        n_drop_zone = sum(1 for a in filter_actions
                          if a["action"] in ("dropped_no_in_zone_frames",
                                             "dropped_trimmed_too_short"))
        n_drop_nose = sum(1 for a in filter_actions
                          if a["action"] == "dropped_nose_at_slit")
        sd_mean = s['tp_start_delta']['mean']
        sd_mean_str = f"{sd_mean:.3f}" if sd_mean is not None else "n/a"
        print(f"    TP={s['n_tp']:>4} FP={s['n_fp']:>4} FN={s['n_fn']:>4}  "
              f"start_delta median={s['tp_start_delta']['median']} "
              f"|median|={s['tp_start_delta']['abs_median']} "
              f"mean={sd_mean_str}  "
              f"span_delta median={s['tp_span_delta']['median']}  "
              f"trimmed={n_trim} dropped(zone)={n_drop_zone} dropped(nose)={n_drop_nose}",
              flush=True)
        folds.append({"val_video_ids": [val_vid], "summary": summary,
                      "n_trimmed": n_trim,
                      "n_dropped_zone": n_drop_zone,
                      "n_dropped_nose": n_drop_nose})
        per_video_data[val_vid] = (algo_reaches, gt_reaches)
        all_results_combined.extend(results)
        all_filter_actions.extend(filter_actions)

    print()
    agg = summarize_results(all_results_combined)
    print("=" * 70)
    print("AGGREGATE LOOCV (BSW b=1 w=0.8 + static-zone TRIM postfilter + nose gate)")
    print("=" * 70)
    sd_mean_a = agg['tp_start_delta']['mean']
    sp_mean_a = agg['tp_span_delta']['mean']
    sd_mean_a_s = f"{sd_mean_a:.3f}" if sd_mean_a is not None else "n/a"
    sp_mean_a_s = f"{sp_mean_a:.3f}" if sp_mean_a is not None else "n/a"
    print(f"  TP={agg['n_tp']}  FP={agg['n_fp']}  FN={agg['n_fn']}")
    print(f"  Start delta on TPs: median={agg['tp_start_delta']['median']}f  "
          f"|median|={agg['tp_start_delta']['abs_median']}f  "
          f"mean={sd_mean_a_s}  "
          f"range=[{agg['tp_start_delta']['min']},{agg['tp_start_delta']['max']}]")
    print(f"  Span delta on TPs:  median={agg['tp_span_delta']['median']}f  "
          f"|median|={agg['tp_span_delta']['abs_median']}f  "
          f"mean={sp_mean_a_s}  "
          f"range=[{agg['tp_span_delta']['min']},{agg['tp_span_delta']['max']}]")
    print()
    print("Compare against:")
    print("  Pure baseline:        TP=1918  FP=337  FN=457  exact_start=83.47%")
    print("  Cumulative best BSW:  TP=1935  FP=330  FN=440  exact_start=84.08%")
    print()
    total_trim = sum(1 for a in all_filter_actions if a["action"] == "trimmed")
    total_drop_zone = sum(1 for a in all_filter_actions
                          if a["action"] in ("dropped_no_in_zone_frames",
                                             "dropped_trimmed_too_short"))
    total_drop_nose = sum(1 for a in all_filter_actions
                          if a["action"] == "dropped_nose_at_slit")
    print(f"Postfilter actions across all folds:")
    print(f"  trimmed:                       {total_trim}")
    print(f"  dropped (no in-zone frames):   {sum(1 for a in all_filter_actions if a['action']=='dropped_no_in_zone_frames')}")
    print(f"  dropped (trimmed too short):   {sum(1 for a in all_filter_actions if a['action']=='dropped_trimmed_too_short')}")
    print(f"  dropped (nose-at-slit fail):   {total_drop_nose}")
    print()

    # Trim magnitude characterization
    trim_starts = [a["trim_start"] for a in all_filter_actions
                   if a.get("action") == "trimmed" and a.get("trim_start") is not None]
    trim_ends = [a["trim_end"] for a in all_filter_actions
                 if a.get("action") == "trimmed" and a.get("trim_end") is not None]
    if trim_starts:
        print(f"  trim_start magnitudes: median={int(np.median(trim_starts))}f  "
              f"mean={np.mean(trim_starts):.1f}  max={max(trim_starts)}")
    if trim_ends:
        print(f"  trim_end magnitudes:   median={int(np.median(trim_ends))}f  "
              f"mean={np.mean(trim_ends):.1f}  max={max(trim_ends)}")
    print()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    serialized_results = []
    for r in all_results_combined:
        record = {
            "status": r.status, "video_id": r.video_id,
            "gt_index": r.gt_index, "algo_index": r.algo_index,
            "start_delta": r.start_delta, "span_delta": r.span_delta,
        }
        algo_list, gt_list = per_video_data[r.video_id]
        if r.algo_index >= 0:
            record["algo_start_frame"] = algo_list[r.algo_index].start_frame
            record["algo_end_frame"] = algo_list[r.algo_index].end_frame
        else:
            record["algo_start_frame"] = -1
            record["algo_end_frame"] = -1
        if r.gt_index >= 0:
            record["gt_start_frame"] = gt_list[r.gt_index].start_frame
            record["gt_end_frame"] = gt_list[r.gt_index].end_frame
        else:
            record["gt_start_frame"] = -1
            record["gt_end_frame"] = -1
        serialized_results.append(record)

    (metrics_dir / "loocv_per_fold.json").write_text(
        json.dumps(folds, indent=2), encoding="utf-8")
    (metrics_dir / "loocv_aggregate.json").write_text(
        json.dumps({
            "n_folds": len(folds), "summary": agg,
            "raw_results": serialized_results,
            "merge_gap": MERGE_GAP,
            "boundary_buffer": BOUNDARY_BUFFER,
            "boundary_weight": BOUNDARY_WEIGHT,
            "min_span_after_trim": MIN_SPAN_AFTER_TRIM,
            "t2_nose_at_slit": T2_NOSE_AT_SLIT,
            "zone_smooth_win": ZONE_SMOOTH_WIN,
            "zone_lk_min": ZONE_LK_MIN,
            "nose_proximity_px": NOSE_PROXIMITY_PX,
            "schema_version": "extended_with_frame_positions",
        }, indent=2), encoding="utf-8")

    (metrics_dir / "filter_actions.json").write_text(
        json.dumps(all_filter_actions, indent=2), encoding="utf-8")

    render_v8_reach_figures(
        snapshot_dir=SNAPSHOT_DIR,
        raw_results=serialized_results,
        summary=agg,
        title_suffix=f" (LOOCV, BSW w=0.8 + zone TRIM + nose T2={T2_NOSE_AT_SLIT})",
    )

    print(f"Wrote: {metrics_dir / 'loocv_per_fold.json'}")
    print(f"Wrote: {metrics_dir / 'loocv_aggregate.json'}")
    print(f"Wrote: {metrics_dir / 'filter_actions.json'}")
    print(f"Wrote: {SNAPSHOT_DIR / 'figures' / 'reach_detection_summary.png'}")
    print()
    print("REMINDER: write RESULTS.md leading with FN delta vs cumulative best AND")
    print("vs pure baseline, BEFORE any metric table. Surface the trim magnitudes")
    print("and characterise the surviving FPs (within_gt class still present?).")


if __name__ == "__main__":
    main()

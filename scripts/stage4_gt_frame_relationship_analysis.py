"""
Stage 4 GT Frame Relationship Analysis
Characterises relationship between GT OKF/IFR and paw-past-y-line bouts
for ALL displaced_sa segments in the 37-video train pool.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

MOUSEREACH_SRC = "Y:/2_Connectome/Behavior/MouseReach/src"
if MOUSEREACH_SRC not in sys.path:
    sys.path.insert(0, MOUSEREACH_SRC)

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series_cleaned

GT_DIR = "Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/gt/"
DLC_DIR = "Y:/2_Connectome/Validation_Runs/DLC_2026_03_27/iterations/2026-04-28_outcome_v4.0.0_dev_walkthrough/dlc/"
CV_FOLDS_PATH = "Y:/2_Connectome/Behavior/MouseReach_Pipeline/Improvement_Snapshots/_corpus/2026-04-30_restart_inventory/cv_folds.json"
REPORT_PATH = ("Y:/2_Connectome/Behavior/MouseReach_Pipeline/Improvement_Snapshots/"
               "outcome/v6.0.0_dev_stage4_design/gt_frame_relationship_analysis.md")

PAW_BODYPARTS = ("RightHand", "RHLeft", "RHOut", "RHRight")
PAW_LK_THR = 0.5
TRANSITION_ZONE_HALF = 5


def load_train_pool(path):
    with open(path) as f:
        d = json.load(f)
    return d["train_pool"]["video_ids"]


def load_gt(video_id):
    path = os.path.join(GT_DIR, video_id + "_unified_ground_truth.json")
    with open(path) as f:
        return json.load(f)


def load_dlc(video_id):
    for fname in os.listdir(DLC_DIR):
        if fname.startswith(video_id) and fname.endswith(".h5"):
            path = os.path.join(DLC_DIR, fname)
            df = pd.read_hdf(path, key="df_with_missing")
            df.columns = ["{}_{}".format(bp, coord) for _scorer, bp, coord in df.columns]
            df = df.reset_index(drop=True)
            return df
    raise FileNotFoundError("No DLC h5 found for {} in {}".format(video_id, DLC_DIR))


def get_segments_from_gt(gt):
    boundaries = gt.get("segmentation", {}).get("boundaries", [])
    boundary_frames = sorted([b["frame"] for b in boundaries])
    outcome_by_seg = {}
    for s in gt.get("outcomes", {}).get("segments", []):
        outcome_by_seg[s["segment_num"]] = s
    segments = []
    for seg_num in range(len(boundary_frames) - 1):
        seg_start = boundary_frames[seg_num]
        seg_end = boundary_frames[seg_num + 1] - 1
        outcome_rec = outcome_by_seg.get(seg_num + 1)
        if outcome_rec is None:
            continue
        segments.append({
            "segment_num": seg_num + 1,
            "seg_start": seg_start,
            "seg_end": seg_end,
            "outcome": outcome_rec.get("outcome"),
            "interaction_frame": outcome_rec.get("interaction_frame"),
            "outcome_known_frame": outcome_rec.get("outcome_known_frame"),
        })
    return segments


def find_paw_past_y_line_bouts(paw_past_y):
    n = len(paw_past_y)
    bouts = []
    run_start = -1
    for i in range(n):
        if paw_past_y[i]:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0:
                bouts.append((run_start, i - 1))
                run_start = -1
    if run_start >= 0:
        bouts.append((run_start, n - 1))
    return bouts


def analyse_segment(dlc_df, seg_start, seg_end, gt_okf, gt_ifr):
    clean_end = seg_end - TRANSITION_ZONE_HALF
    if clean_end <= seg_start:
        return None
    sub_raw = dlc_df.iloc[seg_start:clean_end + 1].copy()
    n = len(sub_raw)
    if n == 0:
        return None

    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series_cleaned(sub)
    pillar_cy = geom["pillar_cy"].to_numpy()
    pillar_r = geom["pillar_r"].to_numpy()
    slit_y_line = pillar_cy + pillar_r

    paw_past_y = np.zeros(n, dtype=bool)
    for bp in PAW_BODYPARTS:
        ycol = bp + "_y"
        lkcol = bp + "_likelihood"
        if ycol not in sub.columns or lkcol not in sub.columns:
            continue
        paw_y = sub[ycol].to_numpy(dtype=float)
        paw_lk = sub[lkcol].to_numpy(dtype=float)
        paw_past_y |= (paw_y <= slit_y_line) & (paw_lk >= PAW_LK_THR)

    bouts_local = find_paw_past_y_line_bouts(paw_past_y)
    n_bouts = len(bouts_local)
    bouts_video = [(seg_start + bs, seg_start + be) for bs, be in bouts_local]

    n_bouts_before_okf = sum(1 for _, be in bouts_video if be <= gt_okf)

    causal_bout = None
    for b_start_v, b_end_v in reversed(bouts_video):
        if b_end_v <= gt_okf:
            causal_bout = (b_start_v, b_end_v)
            break

    if causal_bout is None:
        return {
            "n_bouts": n_bouts,
            "n_bouts_before_okf": n_bouts_before_okf,
            "degenerate": "no_bout_before_okf",
            "okf_minus_bout_middle": None,
            "okf_minus_bout_end": None,
            "ifr_minus_bout_start": None,
            "ifr_minus_bout_end": None,
            "ifr_position_within_bout": None,
            "bout_length": None,
            "ifr_inside_bout": None,
        }

    cb_start, cb_end = causal_bout
    cb_len = cb_end - cb_start + 1
    cb_middle = cb_start + (cb_len - 1) // 2

    okf_minus_bout_middle = gt_okf - cb_middle
    okf_minus_bout_end = gt_okf - cb_end
    ifr_minus_bout_start = gt_ifr - cb_start
    ifr_minus_bout_end = gt_ifr - cb_end

    if cb_len > 1:
        ifr_pos = (gt_ifr - cb_start) / (cb_end - cb_start)
    else:
        ifr_pos = 0.5

    ifr_inside_bout = bool(cb_start <= gt_ifr <= cb_end)

    return {
        "n_bouts": n_bouts,
        "n_bouts_before_okf": n_bouts_before_okf,
        "degenerate": None,
        "okf_minus_bout_middle": okf_minus_bout_middle,
        "okf_minus_bout_end": okf_minus_bout_end,
        "ifr_minus_bout_start": ifr_minus_bout_start,
        "ifr_minus_bout_end": ifr_minus_bout_end,
        "ifr_position_within_bout": ifr_pos,
        "bout_length": cb_len,
        "ifr_inside_bout": ifr_inside_bout,
    }


def write_report(df, good, deg, skipped_no_okf_or_ifr, skipped_no_dlc, n_displaced_total):
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]

    def ptable(col, label=None):
        vals = good[col].dropna().tolist()
        if not vals:
            return "  {}: no data\n".format(label or col)
        arr = np.array(vals)
        header = "  {} (n={}, mean={:.1f}, std={:.1f})".format(
            label or col, len(vals), arr.mean(), arr.std())
        row = "    " + "  ".join("p{}={:.1f}".format(p, np.percentile(arr, p)) for p in pcts)
        return header + "\n" + row + "\n"

    ifr_inside_n = int(good["ifr_inside_bout"].sum())
    ifr_inside_frac = ifr_inside_n / max(len(good), 1)
    okf_before_mid = int((good["okf_minus_bout_middle"] < 0).sum())
    okf_before_end = int((good["okf_minus_bout_end"] < 0).sum())
    ifr_after_end = int((good["ifr_minus_bout_end"] > 0).sum())
    ifr_pos = good["ifr_position_within_bout"].dropna()

    lines = []
    lines.append("# Stage 4 GT Frame Relationship Analysis")
    lines.append("## displaced_sa segments in train pool (37 videos)")
    lines.append("")
    lines.append("Generated: 2026-05-01")
    lines.append("")
    lines.append("## Dataset")
    lines.append("- Train pool: 37 videos")
    lines.append("- Total displaced_sa/displaced_outside segments found: {}".format(n_displaced_total))
    lines.append("- Analysed (OKF + IFR present, DLC available): {}".format(len(df)))
    lines.append("- Degenerate (no bout before OKF): {} ({:.1f}%)".format(
        len(deg), 100 * len(deg) / max(len(df), 1)))
    lines.append("- Good (causal bout identified): {}".format(len(good)))
    if skipped_no_okf_or_ifr:
        lines.append("- Skipped missing OKF/IFR: {}".format(len(skipped_no_okf_or_ifr)))
        for vid, seg_num in skipped_no_okf_or_ifr:
            lines.append("    {} seg {}".format(vid, seg_num))
    if skipped_no_dlc:
        lines.append("- Skipped missing DLC: {}".format(len(skipped_no_dlc)))
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    lines.append("**IFR inside causal bout:** {}/{} = {:.1f}%".format(
        ifr_inside_n, len(good), 100 * ifr_inside_frac))
    lines.append("**OKF < bout_middle (anomaly):** {} segments".format(okf_before_mid))
    lines.append("**OKF < bout_end (strong anomaly):** {} segments".format(okf_before_end))
    lines.append("**IFR > bout_end:** {} segments ({:.1f}%)".format(
        ifr_after_end, 100 * ifr_after_end / max(len(good), 1)))
    lines.append("")
    lines.append("## Percentile Distributions (good segments)")
    lines.append("")
    lines.append("### okf_minus_bout_middle  (GT_OKF - middle of causal bout)")
    lines.append("Primary anchor for OKF emit logic.")
    lines.append(ptable("okf_minus_bout_middle"))
    lines.append("### okf_minus_bout_end  (GT_OKF - end of causal bout)")
    lines.append("How many frames after paw leaves y-line does human mark OKF.")
    lines.append("p50 is the expected post-bout settling delay.")
    lines.append(ptable("okf_minus_bout_end"))
    lines.append("### ifr_minus_bout_start  (GT_IFR - start of causal bout)")
    lines.append(ptable("ifr_minus_bout_start"))
    lines.append("### ifr_minus_bout_end  (GT_IFR - end of causal bout)")
    lines.append("Negative = IFR is before bout end (IFR inside bout).")
    lines.append(ptable("ifr_minus_bout_end"))
    lines.append("### ifr_position_within_bout  (0.0=start, 1.0=end; >1.0=IFR after bout)")
    lines.append("Stage 4 emits IFR = bout_end. p50 near 1.0 means this is accurate.")
    lines.append(ptable("ifr_position_within_bout"))
    lines.append("### bout_length  (frames)")
    lines.append(ptable("bout_length"))
    lines.append("### n_bouts  (total bouts in clean zone)")
    lines.append(ptable("n_bouts"))
    lines.append("### n_bouts_before_okf")
    lines.append(ptable("n_bouts_before_okf"))
    lines.append("")
    lines.append("## IFR Position Breakdown")
    lines.append("- IFR in first half of bout (pos <= 0.5): {:.3f}".format((ifr_pos <= 0.5).mean()))
    lines.append("- IFR in second half (0.5 < pos <= 1.0): {:.3f}".format(
        ((ifr_pos > 0.5) & (ifr_pos <= 1.0)).mean()))
    lines.append("- IFR after bout end (pos > 1.0): {:.3f}".format((ifr_pos > 1.0).mean()))
    lines.append("- IFR before bout start (pos < 0.0): {:.3f}".format((ifr_pos < 0.0).mean()))
    lines.append("")
    lines.append("## Multi-Bout Segments")
    lines.append("- Segments with >1 bout total: {} ({:.1f}%)".format(
        int((good["n_bouts"] > 1).sum()), 100 * (good["n_bouts"] > 1).mean()))
    lines.append("- Segments with >1 bout before OKF: {} ({:.1f}%)".format(
        int((good["n_bouts_before_okf"] > 1).sum()),
        100 * (good["n_bouts_before_okf"] > 1).mean()))
    lines.append("")
    if len(deg) > 0:
        lines.append("## Degenerate Cases (no paw-past-y-line bout before OKF)")
        for _, row in deg.iterrows():
            lines.append("  - {} seg {}: outcome={} OKF={} IFR={} n_bouts={} seg=({}-{})".format(
                row["video_id"], row["segment_num"], row["outcome"],
                row["gt_okf"], row["gt_ifr"], row["n_bouts"],
                row["seg_start"], row["seg_end"]))
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- okf_minus_bout_end p50 is the expected post-bout settling delay the algo needs")
    lines.append("  to confirm displacement evidence after the paw retracts.")
    lines.append("- ifr_position_within_bout: bulk near 1.0 = Stage 4 IFR=bout_end is accurate.")
    lines.append("  bulk < 0.5 = bout_end over-estimates the contact frame.")
    lines.append("- IFR-inside-bout fraction is the primary sanity check: high fraction confirms")
    lines.append("  the bout-identification heuristic correctly finds the causal reach bout.")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    print("[OBJECTIVE] Characterise GT OKF/IFR vs paw-past-y-line bout timing for displaced_sa")
    train_pool = load_train_pool(CV_FOLDS_PATH)
    print("[DATA] train_pool: {} videos".format(len(train_pool)))

    rows = []
    skipped_no_dlc = []
    skipped_no_okf_or_ifr = []
    n_displaced_total = 0

    for vid in train_pool:
        gt = load_gt(vid)
        segments = get_segments_from_gt(gt)
        displaced = [s for s in segments
                     if s["outcome"] in ("displaced_sa", "displaced_outside")]
        if not displaced:
            continue

        try:
            dlc_df = load_dlc(vid)
        except FileNotFoundError as e:
            skipped_no_dlc.append(vid)
            print("  [!] No DLC for {}: {}".format(vid, e))
            continue

        for seg in displaced:
            n_displaced_total += 1
            gt_okf = seg["outcome_known_frame"]
            gt_ifr = seg["interaction_frame"]

            if gt_okf is None or gt_ifr is None:
                skipped_no_okf_or_ifr.append((vid, seg["segment_num"]))
                continue

            result = analyse_segment(dlc_df, seg["seg_start"], seg["seg_end"], gt_okf, gt_ifr)
            if result is None:
                continue

            rows.append({
                "video_id": vid,
                "segment_num": seg["segment_num"],
                "outcome": seg["outcome"],
                "seg_start": seg["seg_start"],
                "seg_end": seg["seg_end"],
                "gt_okf": gt_okf,
                "gt_ifr": gt_ifr,
                **result,
            })

        print("  {}: {} displaced segs".format(vid, len(displaced)))

    df = pd.DataFrame(rows)
    print("")
    print("[DATA] Total displaced segs: {}".format(n_displaced_total))
    print("[DATA] Analysed: {}".format(len(df)))
    print("[DATA] Skipped no OKF/IFR: {}".format(len(skipped_no_okf_or_ifr)))
    print("[DATA] Skipped no DLC: {}".format(len(skipped_no_dlc)))

    deg = df[df["degenerate"].notna()].copy()
    good = df[df["degenerate"].isna()].copy()

    print("")
    print("[DATA] Degenerate (no bout before OKF): {} ({:.1f}%)".format(
        len(deg), 100 * len(deg) / max(len(df), 1)))
    print("[DATA] Good (causal bout found): {} ({:.1f}%)".format(
        len(good), 100 * len(good) / max(len(df), 1)))

    num_cols = ["okf_minus_bout_middle", "okf_minus_bout_end",
                "ifr_minus_bout_start", "ifr_minus_bout_end",
                "ifr_position_within_bout", "bout_length",
                "n_bouts", "n_bouts_before_okf"]
    for col in num_cols:
        good[col] = pd.to_numeric(good[col], errors="coerce")

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print("")
    print("--- Percentile distributions (good segments) ---")
    for col in num_cols:
        vals = good[col].dropna().tolist()
        if not vals:
            continue
        arr = np.array(vals)
        prow = "  ".join("p{}={:.1f}".format(p, np.percentile(arr, p)) for p in pcts)
        print("  {} n={} mean={:.1f} std={:.1f}".format(col, len(vals), arr.mean(), arr.std()))
        print("    " + prow)

    ifr_inside_n = int(good["ifr_inside_bout"].sum())
    ifr_inside_frac = ifr_inside_n / max(len(good), 1)
    okf_before_mid = int((good["okf_minus_bout_middle"] < 0).sum())
    okf_before_end = int((good["okf_minus_bout_end"] < 0).sum())
    ifr_after_end = int((good["ifr_minus_bout_end"] > 0).sum())

    print("")
    print("[FINDING] IFR inside causal bout: {}/{} ({:.1f}%)".format(
        ifr_inside_n, len(good), 100 * ifr_inside_frac))
    print("[FINDING] OKF < bout_middle (anomaly): {}".format(okf_before_mid))
    print("[FINDING] OKF < bout_end (strong anomaly): {}".format(okf_before_end))
    print("[FINDING] IFR > bout_end: {} ({:.1f}%)".format(
        ifr_after_end, 100 * ifr_after_end / max(len(good), 1)))

    ifr_pos = good["ifr_position_within_bout"].dropna()
    print("")
    print("--- IFR position breakdown ---")
    print("  pos <= 0.5 (first half): {:.3f}".format((ifr_pos <= 0.5).mean()))
    print("  pos 0.5-1.0 (second half, inside bout): {:.3f}".format(
        ((ifr_pos > 0.5) & (ifr_pos <= 1.0)).mean()))
    print("  pos > 1.0 (IFR after bout end): {:.3f}".format((ifr_pos > 1.0).mean()))
    print("  pos < 0.0 (IFR before bout start): {:.3f}".format((ifr_pos < 0.0).mean()))

    print("")
    print("--- Multi-bout segments ---")
    print("  >1 bout total: {} ({:.1f}%)".format(
        int((good["n_bouts"] > 1).sum()), 100 * (good["n_bouts"] > 1).mean()))
    print("  >1 bout before OKF: {} ({:.1f}%)".format(
        int((good["n_bouts_before_okf"] > 1).sum()),
        100 * (good["n_bouts_before_okf"] > 1).mean()))

    if len(deg) > 0:
        print("")
        print("--- Degenerate cases (no bout before OKF) ---")
        for _, row in deg.iterrows():
            print("  {} seg {}: outcome={} OKF={} IFR={} n_bouts={} seg=({}-{})".format(
                row["video_id"], row["segment_num"], row["outcome"],
                row["gt_okf"], row["gt_ifr"], row["n_bouts"],
                row["seg_start"], row["seg_end"]))

    write_report(df, good, deg, skipped_no_okf_or_ifr, skipped_no_dlc, n_displaced_total)
    print("")
    print("Report written to: {}".format(REPORT_PATH))


if __name__ == "__main__":
    main()

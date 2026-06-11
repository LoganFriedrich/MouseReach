"""Apex double-hump generalization check across 4 paw-visibility mice.

Validates whether the CNT0413_P4 finding (MERGED algo spans show clean
double-hump trajectories) generalizes to the other paw-visibility mice
identified in the original merge-mechanism analysis:
  CNT0413_P4 (canonical, calibration)  -- already verified
  CNT0104_P3 (calibration)
  CNT0215_P4 (calibration)
  CNT0214_P1 (holdout)

For each video:
- Load v8.0.2 manifest
- Extract MERGED algo spans + TP algo spans
- Compute norm_pos trajectory using model 3.1 parquet (calibration) or
  DLC h5 + matching parquet schema (holdout)
- Detect peaks with the same parameters as the CNT0413_P4 diagnostic
- Report per-video MERGED detection rate and TP false-positive rate
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
HOLDOUT_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
MANIFEST_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests"
    r"\v8.0.2"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.2_dev_apex_double_hump_multi_video"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

VIDEOS = [
    ("20251022_CNT0413_P4", "calibration_loocv"),
    ("20250630_CNT0104_P3", "calibration_loocv"),
    ("20250710_CNT0215_P4", "calibration_loocv"),
    ("20250718_CNT0214_P1", "holdout_2026_05_11"),
]

# Peak detection parameters (same as CNT0413_P4 diagnostic)
PROMINENCE = 0.05
MIN_DISTANCE = 4
TROUGH_DEPTH_THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def load_norm_pos_calibration(video_id):
    """Load norm_pos for a calibration video from the parquet."""
    cols = ["video_id", "frame",
            "RightHand_x", "RightHand_y", "RHLeft_x", "RHLeft_y",
            "RHOut_x", "RHOut_y", "RHRight_x", "RHRight_y",
            "BOXL_x", "BOXL_y", "BOXR_x", "BOXR_y"]
    df = pd.read_parquet(CAL_PARQUET, columns=cols)
    df = df[df["video_id"] == video_id].sort_values("frame").reset_index(drop=True)
    if len(df) == 0:
        return None
    return _compute_norm_pos(df)


def load_norm_pos_holdout(video_id):
    """Load norm_pos for a holdout video from DLC h5."""
    dlc_path = HOLDOUT_DLC_DIR / f"{video_id}{DLC_SUFFIX}.h5"
    if not dlc_path.exists():
        return None
    df = pd.read_hdf(dlc_path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    # Need to remap likelihood columns and rename to match parquet schema
    rename = {}
    for kp in ("RightHand", "RHLeft", "RHOut", "RHRight", "BOXL", "BOXR"):
        # h5 columns are already named "{kp}_x", "{kp}_y", "{kp}_likelihood"
        pass
    # Add a fake "frame" column for compatibility
    df = df.reset_index(drop=True)
    df.insert(0, "frame", df.index)
    return _compute_norm_pos(df)


def _compute_norm_pos(df):
    """Common norm_pos computation. Expects raw position columns."""
    hand_x = smooth(np.mean([df[f"{kp}_x"].to_numpy() for kp in
                              ("RightHand", "RHLeft", "RHOut", "RHRight")], axis=0))
    hand_y = smooth(np.mean([df[f"{kp}_y"].to_numpy() for kp in
                              ("RightHand", "RHLeft", "RHOut", "RHRight")], axis=0))
    boxl_x = smooth(df["BOXL_x"].to_numpy())
    boxl_y = smooth(df["BOXL_y"].to_numpy())
    boxr_x = smooth(df["BOXR_x"].to_numpy())
    boxr_y = smooth(df["BOXR_y"].to_numpy())
    apparatus = np.sqrt((boxr_x - boxl_x) ** 2 + (boxr_y - boxl_y) ** 2)
    dist_boxl = np.sqrt((hand_x - boxl_x) ** 2 + (hand_y - boxl_y) ** 2)
    return dist_boxl / np.maximum(apparatus, 1e-3)


def detect_peaks(norm_pos, span_start, span_end):
    """Return (peaks_list, troughs_list). Each is list of (frame, value)."""
    if span_end < span_start or span_end >= len(norm_pos):
        return [], []
    sig = norm_pos[span_start:span_end + 1]
    if len(sig) < 3:
        return [], []
    peaks, _ = find_peaks(sig, prominence=PROMINENCE, distance=MIN_DISTANCE)
    peak_pairs = [(span_start + int(p), float(sig[p])) for p in peaks]
    troughs = []
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i + 1]
            if p2 - p1 < 2:
                continue
            between = sig[p1:p2 + 1]
            t_local = int(np.argmin(between))
            t_frame = span_start + p1 + t_local
            t_val = float(between[t_local])
            troughs.append((t_frame, t_val))
    return peak_pairs, troughs


def analyze_video(video_id, corpus):
    manifest_path = MANIFEST_ROOT / corpus / f"{video_id}.json"
    if not manifest_path.exists():
        return None
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = m["events"]

    # Load norm_pos
    if corpus == "calibration_loocv":
        norm_pos = load_norm_pos_calibration(video_id)
    else:
        norm_pos = load_norm_pos_holdout(video_id)
    if norm_pos is None:
        return None

    # MERGED algo spans (one per component)
    merged_algos = {}
    for e in events:
        if e["topology"] == "MERGED" and e.get("detector"):
            merged_algos[e["component_id"]] = (e["detector"]["start"],
                                                e["detector"]["end"])
    tp_algos = [(e["detector"]["start"], e["detector"]["end"])
                for e in events if e["topology"] == "TP" and e.get("detector")]

    merged_results = []
    for cid, (a_s, a_e) in merged_algos.items():
        peaks, troughs = detect_peaks(norm_pos, a_s, a_e)
        max_peak = max((p[1] for p in peaks), default=None)
        min_trough = min((t[1] for t in troughs), default=None)
        depth = (max_peak - min_trough) if (max_peak is not None and min_trough is not None) else None
        merged_results.append({
            "cid": cid, "algo_start": a_s, "algo_end": a_e,
            "algo_span": a_e - a_s + 1,
            "n_peaks": len(peaks), "max_peak": max_peak,
            "min_trough": min_trough, "trough_depth": depth,
        })

    tp_results = []
    for a_s, a_e in tp_algos:
        peaks, troughs = detect_peaks(norm_pos, a_s, a_e)
        max_peak = max((p[1] for p in peaks), default=None)
        min_trough = min((t[1] for t in troughs), default=None)
        depth = (max_peak - min_trough) if (max_peak is not None and min_trough is not None) else None
        tp_results.append({
            "algo_start": a_s, "algo_end": a_e,
            "algo_span": a_e - a_s + 1,
            "n_peaks": len(peaks),
            "max_peak": max_peak, "min_trough": min_trough, "trough_depth": depth,
        })

    return {
        "video": video_id, "corpus": corpus,
        "merged": merged_results, "tp": tp_results,
    }


def main():
    print("=" * 70)
    print(f"Apex double-hump generalization check across {len(VIDEOS)} videos")
    print(f"Parameters: prominence={PROMINENCE}, min_distance={MIN_DISTANCE}")
    print("=" * 70)
    print()

    all_results = {}
    for vid, corpus in VIDEOS:
        print(f"--- {vid} ({corpus}) ---", flush=True)
        r = analyze_video(vid, corpus)
        if r is None:
            print("  [skip] missing data")
            continue
        all_results[vid] = r
        n_merged = len(r["merged"])
        n_merged_2peaks = sum(1 for m in r["merged"] if m["n_peaks"] >= 2)
        n_tp = len(r["tp"])
        n_tp_2peaks = sum(1 for t in r["tp"] if t["n_peaks"] >= 2)
        print(f"  MERGED: {n_merged_2peaks}/{n_merged} with 2+ peaks ({100*n_merged_2peaks/max(n_merged,1):.0f}%)")
        print(f"  TP:     {n_tp_2peaks}/{n_tp} with 2+ peaks ({100*n_tp_2peaks/max(n_tp,1):.1f}%)")

        # Threshold sweep
        m_depths = [m["trough_depth"] for m in r["merged"]
                    if m["trough_depth"] is not None]
        t_depths = [t["trough_depth"] for t in r["tp"]
                    if t["trough_depth"] is not None]
        print(f"  trough_depth threshold sweep:")
        for thresh in TROUGH_DEPTH_THRESHOLDS:
            m_catch = sum(1 for d in m_depths if d >= thresh)
            t_split = sum(1 for d in t_depths if d >= thresh)
            print(f"    {thresh:.1f}: MERGED caught {m_catch}/{n_merged} ({100*m_catch/max(n_merged,1):.0f}%), "
                  f"TP false-split {t_split}/{n_tp} ({100*t_split/max(n_tp,1):.1f}%)")
        print()

    # ===== Summary table =====
    print("=" * 100)
    print("SUMMARY (all 4 videos)")
    print("=" * 100)
    print(f"{'Video':<24} {'n_MERGED':>9} {'2+pks':>6} {'n_TP':>5} {'2+pks':>6}  "
          f"{'M@0.4':>7} {'TPfs@0.4':>9} {'M@0.5':>7} {'TPfs@0.5':>9} {'M@0.6':>7} {'TPfs@0.6':>9}")
    print("-" * 100)
    for vid, _ in VIDEOS:
        if vid not in all_results: continue
        r = all_results[vid]
        n_m = len(r["merged"]); n_t = len(r["tp"])
        n_m2 = sum(1 for m in r["merged"] if m["n_peaks"] >= 2)
        n_t2 = sum(1 for t in r["tp"] if t["n_peaks"] >= 2)
        m_depths = [m["trough_depth"] for m in r["merged"] if m["trough_depth"] is not None]
        t_depths = [t["trough_depth"] for t in r["tp"] if t["trough_depth"] is not None]
        row = f"{vid:<24} {n_m:>9} {n_m2:>6} {n_t:>5} {n_t2:>6}"
        for thresh in (0.4, 0.5, 0.6):
            mc = sum(1 for d in m_depths if d >= thresh)
            ts = sum(1 for d in t_depths if d >= thresh)
            row += f"  {mc:>5}/{n_m:<2} {ts:>2}/{n_t:>4} ({100*ts/max(n_t,1):.1f}%)"
        print(row)

    # Aggregate across all 4
    all_merged = sum(len(r["merged"]) for r in all_results.values())
    all_tp = sum(len(r["tp"]) for r in all_results.values())
    print()
    print(f"AGGREGATE across {len(all_results)} videos:")
    for thresh in TROUGH_DEPTH_THRESHOLDS:
        mc = sum(1 for r in all_results.values()
                 for m in r["merged"] if m["trough_depth"] is not None and m["trough_depth"] >= thresh)
        ts = sum(1 for r in all_results.values()
                 for t in r["tp"] if t["trough_depth"] is not None and t["trough_depth"] >= thresh)
        print(f"  trough_depth >= {thresh}:  MERGED caught {mc}/{all_merged} ({100*mc/all_merged:.0f}%),  "
              f"TP false-split {ts}/{all_tp} ({100*ts/all_tp:.2f}%)")

    # Save
    out_data = {
        vid: {
            "corpus": r["corpus"],
            "merged": r["merged"],
            "tp_summary": {
                "n_total": len(r["tp"]),
                "n_with_2plus_peaks": sum(1 for t in r["tp"] if t["n_peaks"] >= 2),
                "false_split_at_thresh": {
                    str(th): sum(1 for t in r["tp"]
                                  if t["trough_depth"] is not None and t["trough_depth"] >= th)
                    for th in TROUGH_DEPTH_THRESHOLDS
                }
            }
        }
        for vid, r in all_results.items()
    }
    (OUT_DIR / "metrics" / "multi_video_results.json").write_text(
        json.dumps(out_data, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved: {OUT_DIR / 'metrics' / 'multi_video_results.json'}")


if __name__ == "__main__":
    main()

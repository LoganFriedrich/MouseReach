"""Apex double-hump check across ALL videos with MERGED in either corpus.

Extends the 4-video paw-visibility check to every video with at least
one MERGED component. Lets us see whether the double-hump signal:
  (a) generalizes beyond the paw-visibility class to the apparatus-quirk
      mergers (CNT0301_P3, CNT0308_P2, CNT0316_P3, etc.)
  (b) holds at low MERGED counts per video
  (c) maintains low TP false-split rate corpus-wide

Per-video output: MERGED catch rate at thresholds {0.4, 0.5, 0.6, 0.7},
TP false-split rate at same thresholds. Aggregate across all videos.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


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
    r"\reach_detection\v8.0.2_dev_apex_double_hump_all_videos"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PROMINENCE = 0.05
MIN_DISTANCE = 4
THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def compute_norm_pos(df):
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


def load_calibration_norm_pos_all():
    """Returns dict[video_id] -> norm_pos array."""
    cols = (["video_id", "frame"]
            + [f"{kp}_{ax}" for kp in ("RightHand", "RHLeft", "RHOut", "RHRight",
                                        "BOXL", "BOXR") for ax in ("x", "y")])
    df = pd.read_parquet(CAL_PARQUET, columns=cols)
    out = {}
    for vid, grp in df.groupby("video_id", sort=False):
        g = grp.sort_values("frame").reset_index(drop=True)
        out[vid] = compute_norm_pos(g)
    return out


def load_holdout_norm_pos(video_id):
    dlc_path = HOLDOUT_DLC_DIR / f"{video_id}{DLC_SUFFIX}.h5"
    if not dlc_path.exists():
        return None
    df = pd.read_hdf(dlc_path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    df = df.reset_index(drop=True)
    df.insert(0, "frame", df.index)
    return compute_norm_pos(df)


def analyze_video(video_id, corpus, norm_pos):
    """Return (per_merged_dicts, per_tp_dicts)."""
    manifest_path = MANIFEST_ROOT / corpus / f"{video_id}.json"
    if not manifest_path.exists():
        return [], []
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    events = m["events"]

    merged_algos = {}
    for e in events:
        if e["topology"] == "MERGED" and e.get("detector"):
            merged_algos[e["component_id"]] = (e["detector"]["start"],
                                                e["detector"]["end"])
    tp_algos = [(e["detector"]["start"], e["detector"]["end"])
                for e in events if e["topology"] == "TP" and e.get("detector")]

    merged_rows = []
    for cid, (a_s, a_e) in merged_algos.items():
        peaks, troughs = detect_peaks(norm_pos, a_s, a_e)
        max_peak = max((p[1] for p in peaks), default=None)
        min_trough = min((t[1] for t in troughs), default=None)
        depth = (max_peak - min_trough) if (max_peak is not None and min_trough is not None) else None
        merged_rows.append({
            "video": video_id, "corpus": corpus, "cid": cid,
            "algo_start": a_s, "algo_end": a_e,
            "n_peaks": len(peaks), "trough_depth": depth,
        })

    tp_rows = []
    for a_s, a_e in tp_algos:
        peaks, troughs = detect_peaks(norm_pos, a_s, a_e)
        max_peak = max((p[1] for p in peaks), default=None)
        min_trough = min((t[1] for t in troughs), default=None)
        depth = (max_peak - min_trough) if (max_peak is not None and min_trough is not None) else None
        tp_rows.append({
            "video": video_id, "corpus": corpus,
            "algo_start": a_s, "algo_end": a_e,
            "n_peaks": len(peaks), "trough_depth": depth,
        })
    return merged_rows, tp_rows


def main():
    print("=" * 70)
    print("Apex double-hump: ALL videos with MERGED")
    print("=" * 70)
    print()

    # Load calibration norm_pos for all videos at once
    print("Loading calibration parquet (all videos)...", flush=True)
    cal_norm_pos = load_calibration_norm_pos_all()
    print(f"  {len(cal_norm_pos)} calibration videos with norm_pos data")
    print()

    # Iterate manifests, only those with MERGED
    all_merged = []
    all_tp = []
    per_video = {}

    # Calibration
    print("Processing calibration manifests...", flush=True)
    for manifest_path in sorted((MANIFEST_ROOT / "calibration_loocv").glob("*.json")):
        vid = manifest_path.stem
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_merged_topo = m.get("topology_summary", {}).get("MERGED", 0)
        if n_merged_topo == 0:
            continue
        if vid not in cal_norm_pos:
            print(f"  [skip] {vid}: no parquet data")
            continue
        merged_rows, tp_rows = analyze_video(vid, "calibration_loocv", cal_norm_pos[vid])
        per_video[vid] = (merged_rows, tp_rows, "calibration_loocv")
        all_merged.extend(merged_rows)
        all_tp.extend(tp_rows)

    # Holdout
    print("Processing holdout manifests...", flush=True)
    for manifest_path in sorted((MANIFEST_ROOT / "holdout_2026_05_11").glob("*.json")):
        vid = manifest_path.stem
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        n_merged_topo = m.get("topology_summary", {}).get("MERGED", 0)
        if n_merged_topo == 0:
            continue
        norm_pos = load_holdout_norm_pos(vid)
        if norm_pos is None:
            print(f"  [skip] {vid}: no DLC h5")
            continue
        merged_rows, tp_rows = analyze_video(vid, "holdout_2026_05_11", norm_pos)
        per_video[vid] = (merged_rows, tp_rows, "holdout_2026_05_11")
        all_merged.extend(merged_rows)
        all_tp.extend(tp_rows)

    # ===== Per-video summary =====
    print()
    print("=" * 120)
    print("PER-VIDEO RESULTS")
    print("=" * 120)
    header = (f"{'video':<24} {'corpus':<20} {'n_MERGED':>9} {'2+pks':>6} {'n_TP':>5} {'2+pks':>6}  "
              f"{'M@0.4':>7} {'TPfs@0.4':>10} {'M@0.5':>7} {'TPfs@0.5':>10} {'M@0.6':>7} {'TPfs@0.6':>10}")
    print(header)
    print("-" * 120)
    for vid in sorted(per_video.keys()):
        merged_rows, tp_rows, corpus = per_video[vid]
        n_m = len(merged_rows); n_t = len(tp_rows)
        n_m2 = sum(1 for r in merged_rows if r["n_peaks"] >= 2)
        n_t2 = sum(1 for r in tp_rows if r["n_peaks"] >= 2)
        m_depths = [r["trough_depth"] for r in merged_rows if r["trough_depth"] is not None]
        t_depths = [r["trough_depth"] for r in tp_rows if r["trough_depth"] is not None]
        row = f"{vid:<24} {corpus:<20} {n_m:>9} {n_m2:>6} {n_t:>5} {n_t2:>6}"
        for thresh in (0.4, 0.5, 0.6):
            mc = sum(1 for d in m_depths if d >= thresh)
            ts = sum(1 for d in t_depths if d >= thresh)
            ts_pct = 100 * ts / max(n_t, 1)
            row += f"  {mc:>5}/{n_m:<2} {ts:>3}/{n_t:>4} ({ts_pct:>4.1f}%)"
        print(row)
    print()

    # ===== Aggregate =====
    n_all_m = len(all_merged)
    n_all_t = len(all_tp)
    print(f"AGGREGATE across {len(per_video)} videos:")
    print(f"  total MERGED: {n_all_m}")
    print(f"  total TP:     {n_all_t}")
    print()
    print(f"{'threshold':>10}  {'MERGED caught':>14}  {'%catch':>7}  {'TP false-split':>15}  {'%fs':>7}")
    for thresh in THRESHOLDS:
        mc = sum(1 for r in all_merged
                 if r["trough_depth"] is not None and r["trough_depth"] >= thresh)
        ts = sum(1 for r in all_tp
                 if r["trough_depth"] is not None and r["trough_depth"] >= thresh)
        print(f"  trough>={thresh:.1f}  {mc:>7} / {n_all_m:<3}  {100*mc/n_all_m:>6.1f}%  "
              f"{ts:>7} / {n_all_t:<4}  {100*ts/n_all_t:>6.2f}%")
    print()

    # ===== Per-corpus aggregate =====
    print("Per-corpus aggregate:")
    for corpus in ("calibration_loocv", "holdout_2026_05_11"):
        merged_corpus = [r for r in all_merged if r["corpus"] == corpus]
        tp_corpus = [r for r in all_tp if r["corpus"] == corpus]
        if not merged_corpus: continue
        n_m = len(merged_corpus); n_t = len(tp_corpus)
        print(f"\n  {corpus}: {n_m} MERGED, {n_t} TPs")
        for thresh in THRESHOLDS:
            mc = sum(1 for r in merged_corpus
                     if r["trough_depth"] is not None and r["trough_depth"] >= thresh)
            ts = sum(1 for r in tp_corpus
                     if r["trough_depth"] is not None and r["trough_depth"] >= thresh)
            print(f"    trough>={thresh:.1f}: M={mc}/{n_m} ({100*mc/n_m:.0f}%)  "
                  f"TPfs={ts}/{n_t} ({100*ts/n_t:.2f}%)")

    # Save per-event records
    out_csv_m = OUT_DIR / "metrics" / "all_merged_apex_features.csv"
    out_csv_t = OUT_DIR / "metrics" / "all_tp_apex_features.csv"
    pd.DataFrame(all_merged).to_csv(out_csv_m, index=False)
    pd.DataFrame(all_tp).to_csv(out_csv_t, index=False)
    print(f"\nSaved: {out_csv_m}")
    print(f"Saved: {out_csv_t}")


if __name__ == "__main__":
    main()

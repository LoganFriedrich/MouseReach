"""Apex-based split hypothesis diagnostic on CNT0413_P4.

Hypothesis: each real reach has a single apex (maximum extension). If
an algo's emitted reach span contains 2+ GT reaches (MERGED topology),
then the hand_x or hand-to-BoxL trajectory across that span should
contain 2+ local maxima separated by a detectable trough.

If true: a postprocess that detects 2+ peaks with a sufficiently deep
trough could split the merged span at the trough frame.

Risk: false splits on TPs where the trajectory has multiple peaks due
to jitter or genuine within-reach hesitations.

Procedure:
1. For each MERGED algo span on CNT0413_P4:
   - Extract hand-centroid trajectory (smoothed)
   - Compute norm_pos = dist(centroid -> BoxL) / apparatus_width
   - Find local maxima with prominence-based detection
   - Report n_peaks, max trough depth (peak - subsequent trough), peak frame indices
2. For a sample of TP algo spans (clean single reaches):
   - Same calculation
   - Report how often they have false 2+ peaks
3. Compare distributions: can we pick a peak-prominence threshold that
   catches MERGED spans without false-positive splits on TPs?
4. Visualize 6 merged + 6 TP spans side-by-side to verify by eye.
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


VIDEO_ID = "20251022_CNT0413_P4"
MANIFEST = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests"
    r"\v8.0.2\calibration_loocv\20251022_CNT0413_P4.json"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.2_dev_cnt0413_apex_double_hump"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)


def main():
    print("=" * 70)
    print(f"CNT0413_P4 apex double-hump diagnostic")
    print("=" * 70)

    # ===== Load manifest =====
    print("Loading manifest...", flush=True)
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    events = m["events"]

    # Group MERGED events by component_id to extract one algo span per component
    merged_algos: dict = {}
    for e in events:
        if e["topology"] == "MERGED" and e.get("detector") is not None:
            cid = e["component_id"]
            merged_algos[cid] = (e["detector"]["start"], e["detector"]["end"])

    # Collect GT reaches per MERGED component for ground-truth peak count
    merged_gts_per_cid: dict = defaultdict(list)
    for e in events:
        if e["topology"] == "MERGED" and e.get("gt") is not None:
            cid = e["component_id"]
            merged_gts_per_cid[cid].append((e["gt"]["start"], e["gt"]["end"]))

    print(f"  MERGED components: {len(merged_algos)}")

    # Collect TP algo spans (one peak expected)
    tp_algos = []
    for e in events:
        if e["topology"] == "TP" and e.get("detector") is not None:
            tp_algos.append((e["detector"]["start"], e["detector"]["end"]))
    print(f"  TP algo spans: {len(tp_algos)}")

    # ===== Load parquet for hand position =====
    print("Loading parquet (hand position + apparatus)...", flush=True)
    needed_cols = ["video_id", "frame",
                   "RightHand_x", "RightHand_y",
                   "RHLeft_x", "RHLeft_y",
                   "RHOut_x", "RHOut_y",
                   "RHRight_x", "RHRight_y",
                   "BOXL_x", "BOXL_y",
                   "BOXR_x", "BOXR_y"]
    df = pd.read_parquet(CAL_PARQUET, columns=needed_cols)
    df = df[df["video_id"] == VIDEO_ID].sort_values("frame").reset_index(drop=True)
    n_frames = len(df)
    print(f"  {n_frames} frames")

    def smooth(x, w=5):
        return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)

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
    norm_pos = dist_boxl / np.maximum(apparatus, 1e-3)

    # ===== Peak detection helper =====
    def detect_peaks(span_start, span_end, prominence=0.05, min_distance=4):
        """Detect peaks in norm_pos over the algo span.
        Returns: list of (peak_frame, peak_value), list of (trough_frame, trough_value) between peaks.
        """
        if span_end < span_start or span_end >= n_frames:
            return [], []
        sig = norm_pos[span_start:span_end + 1]
        if len(sig) < 3:
            return [], []
        # Find peaks
        peaks, props = find_peaks(sig, prominence=prominence, distance=min_distance)
        peak_pairs = [(span_start + int(p), float(sig[p])) for p in peaks]

        # Find troughs (peaks of -sig) between consecutive peaks
        troughs = []
        if len(peaks) >= 2:
            for i in range(len(peaks) - 1):
                p1, p2 = peaks[i], peaks[i + 1]
                if p2 - p1 < 2:
                    continue
                between = sig[p1:p2 + 1]
                trough_local = int(np.argmin(between))
                trough_frame = span_start + p1 + trough_local
                trough_val = float(between[trough_local])
                troughs.append((trough_frame, trough_val))
        return peak_pairs, troughs

    # ===== Analyze MERGED spans =====
    merged_results = []
    for cid in sorted(merged_algos.keys()):
        algo_s, algo_e = merged_algos[cid]
        gts = sorted(merged_gts_per_cid.get(cid, []))
        peaks, troughs = detect_peaks(algo_s, algo_e)
        merged_results.append({
            "cid": cid, "algo_start": algo_s, "algo_end": algo_e,
            "algo_span": algo_e - algo_s + 1,
            "n_gts": len(gts),
            "n_peaks_detected": len(peaks),
            "max_peak_value": max((p[1] for p in peaks), default=None),
            "min_trough_value": min((t[1] for t in troughs), default=None),
            "best_trough_depth": (max(p[1] for p in peaks) - min(t[1] for t in troughs)
                                   if peaks and troughs else None),
            "peak_frames": [p[0] for p in peaks],
            "trough_frames": [t[0] for t in troughs],
            "gts": gts,
        })

    # ===== Analyze TP spans =====
    tp_results = []
    for algo_s, algo_e in tp_algos:
        peaks, troughs = detect_peaks(algo_s, algo_e)
        tp_results.append({
            "algo_start": algo_s, "algo_end": algo_e,
            "algo_span": algo_e - algo_s + 1,
            "n_peaks_detected": len(peaks),
            "max_peak_value": max((p[1] for p in peaks), default=None),
            "min_trough_value": min((t[1] for t in troughs), default=None),
            "best_trough_depth": (max(p[1] for p in peaks) - min(t[1] for t in troughs)
                                   if peaks and troughs else None),
        })

    # ===== Headline summary =====
    print("\n=== MERGED SPANS (n=19) ===")
    print(f"{'cid':>4} {'algo_span':>10} {'n_gts':>6} {'n_peaks':>8} {'max_peak':>9} {'min_trough':>11} {'trough_depth':>13}")
    for r in merged_results:
        depth = r['best_trough_depth']
        depth_str = f"{depth:.3f}" if depth is not None else "n/a"
        max_peak = r['max_peak_value']
        max_peak_str = f"{max_peak:.3f}" if max_peak is not None else "n/a"
        min_trough = r['min_trough_value']
        min_trough_str = f"{min_trough:.3f}" if min_trough is not None else "n/a"
        print(f"{r['cid']:>4} {r['algo_span']:>10} {r['n_gts']:>6} {r['n_peaks_detected']:>8} {max_peak_str:>9} {min_trough_str:>11} {depth_str:>13}")

    print("\n=== TP SPANS (n={}) ===".format(len(tp_results)))
    # Aggregate
    tp_npeaks = [r["n_peaks_detected"] for r in tp_results]
    n_with_2plus = sum(1 for n in tp_npeaks if n >= 2)
    print(f"  n_peaks distribution:")
    from collections import Counter
    counter = Counter(tp_npeaks)
    for k in sorted(counter):
        bar = "#" * min(counter[k], 50)
        print(f"    n_peaks={k}: {counter[k]:>4}  {bar}")
    print(f"  TP spans with >=2 peaks detected: {n_with_2plus} / {len(tp_results)} ({100*n_with_2plus/len(tp_results):.0f}%)")

    # Among TPs with 2+ peaks, what's the trough depth?
    tp_depths = [r["best_trough_depth"] for r in tp_results if r["best_trough_depth"] is not None]
    if tp_depths:
        print(f"  TP trough depths (where 2+ peaks): n={len(tp_depths)}, median={np.median(tp_depths):.3f}, max={np.max(tp_depths):.3f}")

    # Now do MERGED's trough depths
    m_depths = [r["best_trough_depth"] for r in merged_results if r["best_trough_depth"] is not None]
    if m_depths:
        print(f"\n  MERGED trough depths: n={len(m_depths)}, median={np.median(m_depths):.3f}, max={np.max(m_depths):.3f}")
        print(f"  MERGED with no 2+ peaks: {sum(1 for r in merged_results if r['n_peaks_detected'] < 2)}/{len(merged_results)}")

    # Save per-merged details
    cdf = pd.DataFrame([
        {"cid": r["cid"], "algo_start": r["algo_start"], "algo_end": r["algo_end"],
         "algo_span": r["algo_span"], "n_gts": r["n_gts"],
         "n_peaks": r["n_peaks_detected"],
         "peak_frames": ",".join(map(str, r["peak_frames"])),
         "trough_frames": ",".join(map(str, r["trough_frames"])),
         "max_peak": r["max_peak_value"], "min_trough": r["min_trough_value"],
         "trough_depth": r["best_trough_depth"]}
        for r in merged_results])
    cdf.to_csv(OUT_DIR / "metrics" / "merged_peak_analysis.csv", index=False)
    pd.DataFrame(tp_results).to_csv(OUT_DIR / "metrics" / "tp_peak_analysis.csv", index=False)

    # ===== Visualize 6 MERGED + 6 TP examples =====
    n_per_class = 6
    sample_merged = merged_results[:n_per_class]
    # Pick 6 representative TP spans: small, medium, large by span
    tp_sorted = sorted(tp_results, key=lambda r: r["algo_span"])
    tp_idxs = [int(len(tp_sorted) * i / n_per_class) for i in range(n_per_class)]
    sample_tp = [tp_sorted[i] for i in tp_idxs]

    fig, axes = plt.subplots(n_per_class, 2, figsize=(14, 3 * n_per_class), squeeze=False)
    for col, (samples, title) in enumerate([
        (sample_merged, "MERGED"),
        (sample_tp, "TP (cleanly split)"),
    ]):
        for row, r in enumerate(samples):
            ax = axes[row, col]
            algo_s, algo_e = r["algo_start"], r["algo_end"]
            # Plot a window with buffer
            win_lo = max(0, algo_s - 5)
            win_hi = min(n_frames, algo_e + 6)
            frames = np.arange(win_lo, win_hi)
            ax.plot(frames, norm_pos[win_lo:win_hi], color="C0", lw=1.5)
            ax.axvspan(algo_s, algo_e, alpha=0.15, color="C1", label="algo span")
            # Mark detected peaks/troughs
            peaks, troughs = ([(p[0], p[1]) for p in zip(r.get("peak_frames", []) or [],
                                                          [norm_pos[f] for f in (r.get("peak_frames", []) or [])])],
                              [(t, norm_pos[t]) for t in r.get("trough_frames", []) or []])
            if peaks:
                ax.plot([p[0] for p in peaks], [p[1] for p in peaks],
                        "o", color="C3", markersize=8, label="peak")
            if troughs:
                ax.plot([t[0] for t in troughs], [t[1] for t in troughs],
                        "v", color="C2", markersize=8, label="trough")
            # Mark GT reach spans if available (only for MERGED)
            if "gts" in r:
                for gs, ge in r["gts"]:
                    ax.axvspan(gs, ge, alpha=0.2, color="C4")
            ax.set_xlabel("frame")
            ax.set_ylabel("norm_pos")
            ax.set_title(f"{title}: algo {algo_s}-{algo_e} (span {r['algo_span']})  "
                         f"peaks={r['n_peaks_detected']}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"{VIDEO_ID}: norm_pos trajectory in MERGED vs TP algo spans", fontsize=11)
    fig.tight_layout()
    out_fig = OUT_DIR / "figures" / "merged_vs_tp_trajectories.png"
    fig.savefig(out_fig, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure: {out_fig}")


if __name__ == "__main__":
    main()

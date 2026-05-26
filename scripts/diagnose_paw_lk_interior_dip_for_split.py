"""Diagnostic for split-at-low-lk interior gap postprocess (lever #2).

Question
--------
For the residual filtered MERGED events post-v8.0.4 (10 cal + 3 hol), does
the algo span have a sustained paw_mean_lk dip somewhere in the interior --
specifically at the inter-GT gap where the two real reaches meet?

If yes (and TPs don't show comparable dips), split-at-low-lk is a viable
postprocess. If no, it's dead (we'd be guessing where to split with no
feature signal).

Strategy
--------
1. Enumerate filtered MERGED events from v8.0.3 manifests (current
   production-state manifests, regen'd 2026-05-26 post-v8.0.4 ship).
   Filter = neither kinematically_excluded nor outside_gt_segmentation.
2. For each MERGED component, identify:
   - The algo span (the FP-kind event with topology=MERGED).
   - The absorbed GT reaches (FN-kind events in the same component_id).
   - The inter-GT gap (frames between gt[i].end and gt[i+1].start).
3. Load paw_mean_lk for that video (parquet for cal, DLC h5 for hol).
4. Compute:
   - min paw_lk inside the algo span (interior, excluding leading/trailing N frames)
   - longest sustained run of paw_lk < threshold inside algo span
   - paw_lk values specifically at the inter-GT gap frames
5. Control: same metrics on filtered TP events. The question is whether
   MERGEDs systematically have deeper/longer interior dips than TPs.

Thresholds to scan: T in {0.30, 0.40, 0.50, 0.60} x sustain_n in {2,3,4,5}.

Output
------
Per-MERGED event report (one line each) + summary tables comparing
MERGED vs TP dip-event prevalence at each threshold/sustain combo.
"""
from __future__ import annotations
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.reach.v8.postprocess import compute_paw_mean_lk
from mousereach.reach.v8.features import load_dlc_h5


# ---------- Paths ----------
MANIFEST_ROOT = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\fpfn_review_manifests\v8.0.3"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
HOL_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"

PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]


# ---------- Paw_lk loaders ----------

def load_cal_paw_lk_per_video() -> Dict[str, np.ndarray]:
    print("Loading calibration paw_mean_lk from parquet...", flush=True)
    df = pd.read_parquet(CAL_PARQUET, columns=["video_id", "frame"] + PARQUET_LK_COLS)
    df["paw_mean_lk"] = df[PARQUET_LK_COLS].to_numpy(dtype=np.float32).mean(axis=1)
    out = {}
    for vid, g in df.groupby("video_id", sort=False):
        g = g.sort_values("frame").reset_index(drop=True)
        mx = int(g["frame"].max())
        arr = np.full(mx + 1, np.nan, dtype=np.float32)
        arr[g["frame"].to_numpy()] = g["paw_mean_lk"].to_numpy()
        out[vid] = arr
    return out


def load_hol_paw_lk(vid: str) -> Optional[np.ndarray]:
    p = HOL_DLC_DIR / f"{vid}{DLC_SUFFIX}.h5"
    if not p.exists():
        return None
    dlc = load_dlc_h5(p)
    return compute_paw_mean_lk(dlc)


# ---------- MERGED extraction ----------

def collect_filtered_merged(manifest_dir: Path):
    """Returns list of MERGED component records:
       {video, corpus, algo_start, algo_end, gt_spans:[(s,e),...]}
    Filtered = neither flag set.
    """
    out = []
    for f in sorted(manifest_dir.glob("*.json")):
        d = json.load(open(f))
        vid = d["video_id"]
        corpus = d["corpus"]
        # group events by component_id where topology == MERGED
        comp = defaultdict(list)
        for ev in d.get("events", []):
            if ev.get("topology") == "MERGED":
                comp[ev["component_id"]].append(ev)
        for cid, evs in comp.items():
            # Drop if ANY event in the component is kex or outside_gt_seg
            if any(ev.get("kinematically_excluded") or ev.get("outside_gt_segmentation")
                   for ev in evs):
                continue
            # Find the algo span (the FP-kind event with non-null detector)
            algo_evs = [ev for ev in evs if ev["kind"] == "FP" and ev.get("detector")]
            gt_evs = [ev for ev in evs if ev["kind"] == "FN" and ev.get("gt")]
            # Also pull any TP events (these would be matched halves of an N:M; rare)
            tp_evs = [ev for ev in evs if ev["kind"] == "TP" and ev.get("detector")]
            if not algo_evs and not tp_evs:
                continue
            # The MERGED algo span: typically one FP. If there's also a TP,
            # the 1:N or N:M case has a matched half. We want the unmatched
            # algo's span as the merging emission. For simplicity take the
            # widest algo detector across FP+TP events in the component.
            algo_dets = [ev["detector"] for ev in algo_evs + tp_evs]
            algo_start = min(d_["start"] for d_ in algo_dets)
            algo_end = max(d_["end"] for d_ in algo_dets)
            gt_spans = sorted([(ev["gt"]["start"], ev["gt"]["end"]) for ev in gt_evs])
            # Also include matched GT from TP events for 1:N reporting
            matched_gts = sorted([(ev["gt"]["start"], ev["gt"]["end"])
                                  for ev in tp_evs if ev.get("gt")])
            all_gts = sorted(gt_spans + matched_gts)
            out.append({
                "video": vid,
                "corpus": corpus,
                "component_id": cid,
                "algo_start": algo_start,
                "algo_end": algo_end,
                "absorbed_gt_spans": gt_spans,
                "matched_gt_spans": matched_gts,
                "all_gt_spans": all_gts,
                "n_gts_total": len(all_gts),
            })
    return out


def collect_filtered_tps(manifest_dir: Path, max_per_video: Optional[int] = None):
    """Returns list of filtered TP algo spans (control)."""
    out = []
    for f in sorted(manifest_dir.glob("*.json")):
        d = json.load(open(f))
        vid = d["video_id"]
        corpus = d["corpus"]
        count = 0
        for ev in d.get("events", []):
            if ev.get("topology") != "TP":
                continue
            if ev.get("kinematically_excluded") or ev.get("outside_gt_segmentation"):
                continue
            det = ev.get("detector")
            if not det:
                continue
            out.append({
                "video": vid,
                "corpus": corpus,
                "algo_start": det["start"],
                "algo_end": det["end"],
            })
            count += 1
            if max_per_video and count >= max_per_video:
                break
    return out


# ---------- Dip metrics ----------

def interior_window(start: int, end: int, edge_protect: int = 3):
    """Return (lo, hi) inclusive frame range for the 'interior' of an
    algo span, protecting `edge_protect` frames from each end so we
    don't pick up the leading/trailing low-lk runs that the existing
    trims already handle."""
    lo = start + edge_protect
    hi = end - edge_protect
    if hi < lo:
        return None
    return lo, hi


def longest_sustained_low_run(values: np.ndarray, threshold: float) -> int:
    """Returns the length of the longest consecutive run of values < threshold."""
    if values.size == 0:
        return 0
    below = values < threshold
    if not np.any(below):
        return 0
    longest = cur = 0
    for b in below:
        if b:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0
    return longest


def dip_at_position(values: np.ndarray, threshold: float, sustain_n: int):
    """Returns (has_dip, start_idx_in_values, length) for the first run >= sustain_n
    of values < threshold, else (False, None, 0)."""
    if values.size == 0:
        return False, None, 0
    below = values < threshold
    i = 0
    n = len(below)
    while i < n:
        if below[i]:
            j = i
            while j < n and below[j]:
                j += 1
            if j - i >= sustain_n:
                return True, i, j - i
            i = j
        else:
            i += 1
    return False, None, 0


# ---------- Main ----------

def main():
    cal_lk_by_vid = load_cal_paw_lk_per_video()

    cal_merged = collect_filtered_merged(MANIFEST_ROOT / "calibration_loocv")
    hol_merged = collect_filtered_merged(MANIFEST_ROOT / "holdout_2026_05_11")
    all_merged = cal_merged + hol_merged
    print(f"\nFiltered MERGED components: cal={len(cal_merged)}, hol={len(hol_merged)}")

    cal_tp = collect_filtered_tps(MANIFEST_ROOT / "calibration_loocv")
    hol_tp = collect_filtered_tps(MANIFEST_ROOT / "holdout_2026_05_11")
    all_tp = cal_tp + hol_tp
    print(f"Filtered TPs (control):     cal={len(cal_tp)}, hol={len(hol_tp)}")

    # Cache holdout lk per video on demand
    hol_lk_cache = {}
    def get_lk(vid: str, corpus: str) -> Optional[np.ndarray]:
        if corpus == "calibration_loocv":
            return cal_lk_by_vid.get(vid)
        if vid not in hol_lk_cache:
            hol_lk_cache[vid] = load_hol_paw_lk(vid)
        return hol_lk_cache[vid]

    # =========================================================
    # Per-MERGED event detail
    # =========================================================
    print("\n=== Per-MERGED event paw_lk interior signature ===")
    print(f"{'video':<22} {'corpus':<3} {'algo':>15} {'span':>4} "
          f"{'min_lk_int':>10} {'p10_int':>8} {'longest@.40':>11} "
          f"{'longest@.60':>11} {'gap_min_lk':>10} {'gap_n':>5}")
    merged_records = []
    for m in all_merged:
        vid = m["video"]
        corp = m["corpus"]
        lk = get_lk(vid, corp)
        if lk is None:
            print(f"  [skip] no lk: {vid}")
            continue
        s, e = m["algo_start"], m["algo_end"]
        if e >= len(lk):
            e_use = len(lk) - 1
        else:
            e_use = e
        algo_vals = lk[s:e_use + 1]
        if algo_vals.size == 0 or np.all(np.isnan(algo_vals)):
            continue
        interior = interior_window(s, e_use, edge_protect=3)
        if interior is None:
            int_vals = np.array([], dtype=np.float32)
        else:
            lo, hi = interior
            int_vals = lk[lo:hi + 1]
            int_vals = int_vals[~np.isnan(int_vals)]
        min_int = float(np.nanmin(int_vals)) if int_vals.size else float("nan")
        p10_int = float(np.nanpercentile(int_vals, 10)) if int_vals.size else float("nan")
        longest_40 = longest_sustained_low_run(int_vals, 0.40)
        longest_60 = longest_sustained_low_run(int_vals, 0.60)
        # Inter-GT gap analysis: between consecutive GT spans inside algo span
        gt_spans = m["all_gt_spans"]
        gap_min = float("nan")
        gap_n = 0
        for i in range(len(gt_spans) - 1):
            gs1_end = gt_spans[i][1]
            gs2_start = gt_spans[i + 1][0]
            if gs2_start <= gs1_end + 1:
                continue
            gap_lo = max(gs1_end + 1, s)
            gap_hi = min(gs2_start - 1, e_use)
            if gap_hi < gap_lo:
                continue
            gap_vals = lk[gap_lo:gap_hi + 1]
            gap_vals = gap_vals[~np.isnan(gap_vals)]
            if gap_vals.size == 0:
                continue
            this_min = float(np.nanmin(gap_vals))
            if np.isnan(gap_min) or this_min < gap_min:
                gap_min = this_min
            gap_n = max(gap_n, gap_vals.size)
        print(f"{vid:<22} {corp[:3]:<3} {f'{s}-{e}':>15} {e_use-s+1:>4} "
              f"{min_int:>10.3f} {p10_int:>8.3f} {longest_40:>11} "
              f"{longest_60:>11} {gap_min:>10.3f} {gap_n:>5}")
        merged_records.append({
            "video": vid, "corpus": corp, "span": e_use - s + 1,
            "min_int": min_int, "longest_40": longest_40, "longest_60": longest_60,
            "gap_min": gap_min, "gap_n": gap_n,
        })

    # =========================================================
    # Threshold/sustain sweep: prevalence of "has interior dip"
    # in MERGED vs TP
    # =========================================================
    print("\n=== Dip-event prevalence by threshold x sustain_n ===")
    print("Cell = (MERGED hits / total) | (TP hits / total) | discrimination ratio")

    def survey(items, t, n, edge_protect=3):
        hits = 0
        total = 0
        for m in items:
            vid, corp = m["video"], m["corpus"]
            lk = get_lk(vid, corp)
            if lk is None:
                continue
            s, e = m["algo_start"], m["algo_end"]
            e_use = min(e, len(lk) - 1)
            interior = interior_window(s, e_use, edge_protect=edge_protect)
            if interior is None:
                continue
            lo, hi = interior
            vals = lk[lo:hi + 1]
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            total += 1
            has, _, _ = dip_at_position(vals, t, n)
            if has:
                hits += 1
        return hits, total

    thresholds = [0.30, 0.40, 0.50, 0.60]
    sustains = [2, 3, 4, 5]

    print(f"\n{'thresh':>7} {'sustain':>8} {'merged_hit/tot':>15} {'tp_hit/tot':>14} "
          f"{'merged_rate':>12} {'tp_rate':>9} {'M/TP_ratio':>11}")
    for t in thresholds:
        for n in sustains:
            mh, mt = survey(all_merged, t, n)
            th, tt = survey(all_tp, t, n)
            m_rate = mh / mt if mt else 0
            tp_rate = th / tt if tt else 0
            ratio = (m_rate / tp_rate) if tp_rate > 0 else float("inf")
            print(f"{t:>7.2f} {n:>8} {f'{mh}/{mt}':>15} {f'{th}/{tt}':>14} "
                  f"{m_rate:>12.3f} {tp_rate:>9.3f} {ratio:>11.2f}")

    # =========================================================
    # Inter-GT gap signature: do the GAP frames specifically dip?
    # =========================================================
    print("\n=== Inter-GT gap paw_lk distribution (MERGED only) ===")
    gap_mins = [r["gap_min"] for r in merged_records if not np.isnan(r["gap_min"])]
    if gap_mins:
        print(f"  N MERGEDs with inter-GT gap: {len(gap_mins)}")
        print(f"  gap_min_lk distribution:")
        print(f"    min:    {min(gap_mins):.3f}")
        print(f"    p25:    {np.percentile(gap_mins, 25):.3f}")
        print(f"    median: {np.percentile(gap_mins, 50):.3f}")
        print(f"    p75:    {np.percentile(gap_mins, 75):.3f}")
        print(f"    max:    {max(gap_mins):.3f}")
        for t in (0.40, 0.50, 0.60):
            n_below = sum(1 for v in gap_mins if v < t)
            print(f"    fraction with gap_min < {t}: {n_below}/{len(gap_mins)}")
    else:
        print("  No MERGEDs had a multi-GT inter-gap (all GTs touching or N=1?)")


if __name__ == "__main__":
    main()

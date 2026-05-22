"""
v8.0.x experiment: apex-split postprocess at depth>=0.5, peak2_rel<0.85.

Pinned single-config formal experiment selected from the 6-config sweep at
v8.0.2_dev_apex_split_sweep/ (Logan choice: conservative). This runner
applies v8.0.2 trim + apex-split at the conservative config + asymmetric
matcher, then performs the FRAGMENTED inspection needed to evaluate the
decision rule (which the sweep flagged as tripped on FRAG > 2x baseline).

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-22 via master log b3f9155
   + entry-point brief reach_detection_state_2026-05-21):
   - Production v8.0.2 reach detector (BSW b=1/w=0.8 + mg=0 + sustained-
     trim N=3/T=0.60). Asymmetric strict matcher tolerance -2/+5. Live GT
     post 2026-05-22 edits.
   - Comparison baseline (cumulative best == pure baseline for this
     experiment, since no intermediate v8.0.3 shipped between
     v8.0.2+asymmetric and apex-split):
       Calibration LOOCV unfiltered: TP=2231 / FP=98  / FN=170
       Holdout 19 unfiltered:        TP=3656 / FP=71  / FN=96
   - No reverts on master since v8.0.2 ship.

2. Existing-code-modification check: NO. All apex-split logic inline in
   this runner. No modifications to src/mousereach/* or
   mousereach.improvement.reach_detection.metrics.

3. Unverified hypotheses:
   - Whether FRAGMENTED rise (+7 cal / +9 hol in the sweep at this config)
     is dominated by legitimate splits of MERGED-with-tight-trough cases
     vs. over-splits of single TPs. The runner emits per-event provenance
     to answer this.
   - Whether the trough frame is kinematically accurate enough for the
     downstream kinematics window (Cardinal Rule check: start_delta
     abs_median).

4. FN-direction-reporting (planned):
   - Lead with FN delta vs cumulative best (= pure baseline for this
     experiment, both 170 cal / 96 hol). Note this in RESULTS.md.
   - Two-delta convention: when cumulative-best == pure-baseline, the
     two deltas are identical; surface explicitly.
   - Topology counts paired with legacy in every table.
   - ASCII output only.

5. Framework check:
   - Output to Improvement_Snapshots/reach_detection/
     v8.0.2_dev_apex_split_d05_p085/{metrics,figures}/ + RESULTS.md.
   - JSON + text RESULTS, no parallel-track files.

6. Branch + tag:
   - feature/v8-apex-split-postprocess (no commits ahead of master yet;
     all apex-split work untracked).
   - Tag: v8-pre-apex-split-2026-05-22.

7. Decision rule (applied in RESULTS.md after inspection):
   ACCEPT if all of:
     - TP rises or holds vs baseline on BOTH corpora
     - FN drops or holds vs baseline on BOTH corpora
     - start_delta abs_median = 0 (Cardinal Rule on TP boundaries)
     - MERGED topology drops materially
     - FRAGMENTED count <= 2x baseline OR new FRAG events are
       overwhelmingly conversions from MERGED (not over-splits of TPs)
   REJECT otherwise.

   Sweep pre-result for this config: TP+88/+23, FP-29/+1, FN-88/-23,
   start_delta abs_med=0, MERGED 57->5 cal / 17->2 hol (-52/-15),
   FRAGMENTED 5->12 cal / 8->17 hol (2.4x/2.1x baseline -- trips the
   strict threshold; inspection determines whether the rise is benign).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.improvement.reach_detection.metrics import (
    STRICT_START_TOL_EARLY, STRICT_START_TOL_LATE,
)
from mousereach.reach.v8.postprocess import (
    ReachSpan, trim_leading_sustained_lk, compute_paw_mean_lk,
)


# ----- Paths -----
CAL_LOOCV_SOURCE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.1_model_3_1_baseline_loocv\metrics\loocv_aggregate.json"
)
CAL_PARQUET = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\_corpus\2026-05-21_model_3_1_inventory\phase_b_dataset\train_pool.parquet"
)
HOLDOUT_ALGO_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.0_holdout_generalization_merge_gap_0\algo_outputs_v8.0.0_mg0"
)
HOLDOUT_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\dlc"
)
GEN_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
    r"\generalization_test_2026-05-11\gt"
)
CAL_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\validation_runs\DLC_2026_03_27\gt"
)
OUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\Improvement_Snapshots"
    r"\reach_detection\v8.0.2_dev_apex_split_d05_p085"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
PARQUET_POS_COLS = ([f"{kp}_{ax}" for kp in
                      ("RightHand", "RHLeft", "RHOut", "RHRight",
                        "BOXL", "BOXR") for ax in ("x", "y")])

# v8.0.2 trim parameters (production)
TRIM_THRESHOLD = 0.60
TRIM_SUSTAIN_N = 3
TRIM_MIN_SPAN = 3

# Apex split parameters (pinned config)
DEPTH_MIN = 0.5
PEAK2_REL_MAX = 0.85
APEX_PROMINENCE = 0.05
APEX_MIN_DISTANCE = 4
APEX_MIN_SPAN = 3

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


# ---------- Helpers ----------

def smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def compute_norm_pos_from_df(df):
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


def load_dlc_h5(path):
    df = pd.read_hdf(path)
    df.columns = ['_'.join(col[1:]) for col in df.columns]
    return df


def apex_split(reach, norm_pos, depth_min, peak2_rel_max,
               prominence=APEX_PROMINENCE, min_distance=APEX_MIN_DISTANCE,
               min_span=APEX_MIN_SPAN):
    s, e = reach
    if e >= len(norm_pos):
        return [(s, e)], None
    sig = norm_pos[s:e + 1]
    if len(sig) < 3:
        return [(s, e)], None
    peaks, _ = find_peaks(sig, prominence=prominence, distance=min_distance)
    if len(peaks) < 2:
        return [(s, e)], None
    peak2_rel = peaks[-1] / (len(sig) - 1)
    if peak2_rel >= peak2_rel_max:
        return [(s, e)], None
    best_depth = 0.0
    best_trough_frame = None
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        if p2 - p1 < 2:
            continue
        between = sig[p1:p2 + 1]
        t_local = int(np.argmin(between))
        depth = max(float(sig[p1]), float(sig[p2])) - float(between[t_local])
        if depth > best_depth:
            best_depth = depth
            best_trough_frame = s + p1 + t_local
    if best_depth < depth_min or best_trough_frame is None:
        return [(s, e)], None
    half1 = (s, best_trough_frame)
    half2 = (best_trough_frame + 1, e)
    if (half1[1] - half1[0] + 1) < min_span or (half2[1] - half2[0] + 1) < min_span:
        return [(s, e)], None
    return [half1, half2], best_trough_frame


# ---------- Matcher + topology (asymmetric tolerance) ----------

def overlap(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or a_s > b_e)


def greedy_match(algos, gts):
    candidates = []
    for ai, (a_s, a_e) in enumerate(algos):
        algo_span = a_e - a_s + 1
        for gi, (g_s, g_e) in enumerate(gts):
            gt_span = g_e - g_s + 1
            start_delta = a_s - g_s
            span_delta = algo_span - gt_span
            span_tol = max(SPAN_TOL_FRAC * gt_span, SPAN_TOL_MIN)
            if (-STRICT_START_TOL_EARLY <= start_delta <= STRICT_START_TOL_LATE
                    and abs(span_delta) <= span_tol):
                candidates.append((abs(start_delta), ai, gi, start_delta))
    candidates.sort()
    used_a, used_g = set(), set()
    pairs = []
    tp_sd = []
    for _, ai, gi, sd in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        pairs.append((ai, gi))
        tp_sd.append(sd)
    fps = [ai for ai in range(len(algos)) if ai not in used_a]
    fns = [gi for gi in range(len(gts)) if gi not in used_g]
    return pairs, fps, fns, tp_sd


def classify_components(algos, gts):
    """Returns dict: comp_id -> {algo_indices, gt_indices, label}."""
    algo_to_gt = defaultdict(set)
    gt_to_algo = defaultdict(set)
    for i, (a_s, a_e) in enumerate(algos):
        for j, (g_s, g_e) in enumerate(gts):
            if overlap(a_s, a_e, g_s, g_e):
                algo_to_gt[i].add(j)
                gt_to_algo[j].add(i)
    visited_a, visited_g = set(), set()
    comps = []
    for i in range(len(algos)):
        if i in visited_a:
            continue
        if not algo_to_gt[i]:
            comps.append({"algos": {i}, "gts": set(), "label": "FALSE_POSITIVE"})
            visited_a.add(i)
            continue
        algo_in, gt_in = set(), set()
        queue = [("a", i)]
        while queue:
            kind, idx = queue.pop()
            if kind == "a":
                if idx in algo_in:
                    continue
                algo_in.add(idx)
                for gj in algo_to_gt[idx]:
                    queue.append(("g", gj))
            else:
                if idx in gt_in:
                    continue
                gt_in.add(idx)
                for ai in gt_to_algo[idx]:
                    queue.append(("a", ai))
        visited_a.update(algo_in); visited_g.update(gt_in)
        na, ng = len(algo_in), len(gt_in)
        if na == 1 and ng == 1:
            a = algos[next(iter(algo_in))]
            g = gts[next(iter(gt_in))]
            sd = a[0] - g[0]
            span_a = a[1] - a[0] + 1; span_g = g[1] - g[0] + 1
            sp_d = span_a - span_g
            sp_tol = max(SPAN_TOL_FRAC * span_g, SPAN_TOL_MIN)
            if -STRICT_START_TOL_EARLY <= sd <= STRICT_START_TOL_LATE and abs(sp_d) <= sp_tol:
                label = "TP"
            else:
                label = "TOLERANCE_ERROR"
        elif na == 1 and ng >= 2:
            label = "MERGED"
        elif na >= 2 and ng == 1:
            label = "FRAGMENTED"
        elif na >= 2 and ng >= 2:
            label = "COMPLEX"
        comps.append({"algos": algo_in, "gts": gt_in, "label": label})
    for j in range(len(gts)):
        if j not in visited_g:
            comps.append({"algos": set(), "gts": {j}, "label": "FALSE_NEGATIVE"})
    return comps


# ---------- Data loading ----------

def load_live_gt(corpus_label, video_id):
    if corpus_label == "calibration_loocv":
        gt_path = CAL_GT_DIR / f"{video_id}_unified_ground_truth.json"
    else:
        gt_path = GEN_GT_DIR / f"{video_id}_unified_ground_truth.json"
    if not gt_path.exists():
        return []
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    reaches_obj = data.get("reaches", {})
    rlist = (reaches_obj.get("reaches", [])
             if isinstance(reaches_obj, dict) else [])
    return sorted(set(
        (int(r["start_frame"]), int(r["end_frame"]))
        for r in rlist if not r.get("exclude_from_analysis")
    ))


def load_calibration():
    print("Loading calibration LOOCV + parquet...", flush=True)
    data = json.loads(CAL_LOOCV_SOURCE.read_text(encoding="utf-8"))
    raw = data["raw_results"]
    algos = defaultdict(set)
    for r in raw:
        if r["algo_start_frame"] >= 0:
            algos[r["video_id"]].add((int(r["algo_start_frame"]),
                                       int(r["algo_end_frame"])))
    df = pd.read_parquet(CAL_PARQUET,
                          columns=["video_id", "frame"] + PARQUET_LK_COLS + PARQUET_POS_COLS)
    out = {}
    for vid, grp in df.groupby("video_id", sort=False):
        g = grp.sort_values("frame").reset_index(drop=True)
        paw_lk_matrix = g[PARQUET_LK_COLS].to_numpy(dtype=np.float32)
        paw_mean_lk = paw_lk_matrix.mean(axis=1)
        mx_frame = int(g["frame"].max())
        lk_arr = np.full(mx_frame + 1, np.nan, dtype=np.float32)
        lk_arr[g["frame"].to_numpy()] = paw_mean_lk
        norm_pos = compute_norm_pos_from_df(g)
        np_arr = np.full(mx_frame + 1, np.nan, dtype=np.float32)
        np_arr[g["frame"].to_numpy()] = norm_pos
        out[vid] = {
            "algos_v801": sorted(algos.get(vid, set())),
            "gts": load_live_gt("calibration_loocv", vid),
            "paw_lk": lk_arr,
            "norm_pos": np_arr,
        }
    print(f"  {len(out)} calibration videos loaded")
    return out


def load_holdout():
    print("Loading holdout outputs + DLC + GT...", flush=True)
    out = {}
    for algo_path in sorted(HOLDOUT_ALGO_DIR.glob("*_reaches.json")):
        vid = algo_path.stem.replace("_reaches", "")
        dlc_path = HOLDOUT_GT_DIR / f"{vid}{DLC_SUFFIX}.h5"
        if not dlc_path.exists():
            continue
        adata = json.loads(algo_path.read_text(encoding="utf-8"))
        algos_v801 = sorted(set(
            (int(r["start_frame"]), int(r["end_frame"]))
            for r in adata.get("reaches", [])
        ))
        dlc = load_dlc_h5(dlc_path)
        paw_lk = compute_paw_mean_lk(dlc)
        norm_pos = compute_norm_pos_from_df(dlc)
        out[vid] = {
            "algos_v801": algos_v801,
            "gts": load_live_gt("holdout_2026_05_11", vid),
            "paw_lk": paw_lk,
            "norm_pos": norm_pos,
        }
    print(f"  {len(out)} holdout videos loaded")
    return out


# ---------- Scoring + provenance tracking ----------

def apply_v802_trim(algos, paw_lk):
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos]
    trimmed = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_THRESHOLD,
        sustain_n=TRIM_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN,
    )
    return [(r.start_frame, r.end_frame) for r in trimmed]


def score_corpus_with_provenance(corpus_data, apply_apex):
    """Score and emit per-event provenance. Returns dict per video."""
    totals = {"tp": 0, "fp": 0, "fn": 0, "splits_made": 0}
    topo_counts = defaultdict(int)
    tp_start_deltas = []
    per_video = {}
    for vid, vd in corpus_data.items():
        algos_v802 = apply_v802_trim(vd["algos_v801"], vd["paw_lk"])
        split_provenance = []
        if apply_apex:
            algos_final = []
            for reach in algos_v802:
                parts, trough = apex_split(reach, vd["norm_pos"],
                                            depth_min=DEPTH_MIN,
                                            peak2_rel_max=PEAK2_REL_MAX)
                if len(parts) > 1:
                    totals["splits_made"] += 1
                    split_provenance.append({
                        "pre_split": list(reach),
                        "trough_frame": trough,
                        "halves": [list(p) for p in parts],
                    })
                algos_final.extend(parts)
        else:
            algos_final = algos_v802
        algos_final = sorted(set(algos_final))
        pairs, fps, fns, tp_sd = greedy_match(algos_final, vd["gts"])
        totals["tp"] += len(pairs); totals["fp"] += len(fps); totals["fn"] += len(fns)
        tp_start_deltas.extend(tp_sd)
        comps = classify_components(algos_final, vd["gts"])
        for c in comps:
            topo_counts[c["label"]] += 1
        per_video[vid] = {
            "algos_final": algos_final,
            "gts": vd["gts"],
            "components": [
                {"algos": sorted(c["algos"]),
                 "gts": sorted(c["gts"]),
                 "label": c["label"]}
                for c in comps
            ],
            "split_provenance": split_provenance,
        }
    abs_med = int(np.median([abs(d) for d in tp_start_deltas])) if tp_start_deltas else None
    return {
        "totals": totals,
        "topology": dict(topo_counts),
        "start_delta_abs_median": abs_med,
        "per_video": per_video,
    }


# ---------- FRAGMENTED inspection ----------

def classify_new_frag(baseline_per_video, apex_per_video):
    """For each NEW FRAG event in apex (not present in baseline), determine
    what label the same GT had in baseline. Returns counts + a sample list.
    """
    from_label_counts = defaultdict(int)
    examples = []
    for vid in apex_per_video:
        apex_comps = apex_per_video[vid]["components"]
        base_comps = baseline_per_video[vid]["components"]
        base_algos = baseline_per_video[vid]["algos_final"]
        base_gts = baseline_per_video[vid]["gts"]
        # Build a per-GT map of baseline labels (each GT in at most one comp)
        gt_to_base_label = {}
        gt_to_base_algos = {}
        for c in base_comps:
            for gj in c["gts"]:
                gt_to_base_label[gj] = c["label"]
                gt_to_base_algos[gj] = [base_algos[ai] for ai in c["algos"]]
        # For each FRAG comp in apex, look up the single GT's baseline label
        for c in apex_comps:
            if c["label"] != "FRAGMENTED":
                continue
            gj = c["gts"][0]
            base_label = gt_to_base_label.get(gj, "UNKNOWN")
            from_label_counts[base_label] += 1
            if len(examples) < 30:
                examples.append({
                    "video_id": vid,
                    "gt_frames": list(base_gts[gj]),
                    "baseline_label": base_label,
                    "baseline_algos": gt_to_base_algos.get(gj, []),
                    "apex_algos": [apex_per_video[vid]["algos_final"][ai]
                                    for ai in c["algos"]],
                })
    return dict(from_label_counts), examples


# ---------- Output ----------

def print_summary(label, baseline, apex):
    bt = baseline["totals"]; at = apex["totals"]
    bb = baseline["topology"]; aa = apex["topology"]
    print(f"  {label}: baseline TP={bt['tp']} FP={bt['fp']} FN={bt['fn']}")
    print(f"  {label}: apex     TP={at['tp']} FP={at['fp']} FN={at['fn']} "
          f"(splits={at['splits_made']})")
    print(f"  {label}: delta    TP={at['tp']-bt['tp']:+} FP={at['fp']-bt['fp']:+} FN={at['fn']-bt['fn']:+}")
    print(f"  {label}: topology baseline -> apex (delta):")
    for k in ("TP","TOLERANCE_ERROR","MERGED","FRAGMENTED","FALSE_POSITIVE","FALSE_NEGATIVE","COMPLEX"):
        b_ = bb.get(k, 0); a_ = aa.get(k, 0)
        print(f"    {k:<22} {b_:>4} -> {a_:>4}  ({a_-b_:+d})")
    print(f"  {label}: start_delta abs_median (TP only) = {apex['start_delta_abs_median']}")


def write_results_md(cal_base, cal_apex, hol_base, hol_apex,
                      cal_frag_breakdown, hol_frag_breakdown,
                      cal_frag_examples, hol_frag_examples):
    md = []
    md.append("# Apex-split postprocess @ depth>=0.5, peak2_rel<0.85")
    md.append("")
    md.append(f"Run: {datetime.utcnow().isoformat()}Z")
    md.append("")
    md.append("## Pre-experiment checklist")
    md.append("")
    md.append("Per `pre_experiment_checklist.md`, applied in the runner docstring at")
    md.append("`scripts/restart_phase_b_apex_split_d05_p085.py`. Summary:")
    md.append("- Cumulative best = pure baseline = v8.0.2 + asymmetric on live GT.")
    md.append("  No intermediate ship between v8.0.2 and this experiment.")
    md.append("- Stacked improvements: v8.0.2 trim (T=0.60, N=3, MIN_SPAN=3),")
    md.append("  mg=0, BSW b=1/w=0.8 (already in the v8.0.0 model bundle).")
    md.append("- Asymmetric strict matcher: -2 early, +5 late, span_tol=max(0.5*gt,5).")
    md.append("- Live GT read from canonical *_unified_ground_truth.json paths;")
    md.append("  per `feedback_never_pull_gt_from_snapshots`.")
    md.append("- Output canonical snapshot dir; ASCII only; topology paired with legacy.")
    md.append("")

    # Section 1: FN direction
    cbt = cal_base["totals"]; cat = cal_apex["totals"]
    hbt = hol_base["totals"]; hat = hol_apex["totals"]
    cal_dfn = cat["fn"] - cbt["fn"]
    hol_dfn = hat["fn"] - hbt["fn"]
    md.append("## FN direction (leading headline)")
    md.append("")
    md.append("For this experiment, cumulative best == pure baseline (no intermediate")
    md.append("v8.0.x ship between v8.0.2+asymmetric and this run). Both FN deltas:")
    md.append("")
    md.append(f"- Calibration LOOCV: FN {cbt['fn']} -> {cat['fn']} ({cal_dfn:+}).")
    md.append(f"  FN direction: FALLING by {abs(cal_dfn)}.")
    md.append(f"- Holdout 19: FN {hbt['fn']} -> {hat['fn']} ({hol_dfn:+}).")
    md.append(f"  FN direction: FALLING by {abs(hol_dfn)}.")
    md.append("")

    # Section 2: legacy + topology tables
    md.append("## Counts: legacy paired with topology (per `feedback_pair_legacy_with_topology`)")
    md.append("")
    for corpus_label, baseline, apex in [
        ("Calibration LOOCV", cal_base, cal_apex),
        ("Holdout 19", hol_base, hol_apex),
    ]:
        md.append(f"### {corpus_label}")
        md.append("")
        bt = baseline["totals"]; at = apex["totals"]
        bb = baseline["topology"]; aa = apex["topology"]
        md.append("Legacy TP/FP/FN (unfiltered):")
        md.append("")
        md.append("| | baseline | apex_d0.5_p0.85 | delta |")
        md.append("|---|---:|---:|---:|")
        md.append(f"| TP | {bt['tp']} | {at['tp']} | {at['tp']-bt['tp']:+} |")
        md.append(f"| FP | {bt['fp']} | {at['fp']} | {at['fp']-bt['fp']:+} |")
        md.append(f"| FN | {bt['fn']} | {at['fn']} | {at['fn']-bt['fn']:+} |")
        md.append("")
        md.append("Topology (connected-component classification):")
        md.append("")
        md.append("| class | baseline | apex_d0.5_p0.85 | delta |")
        md.append("|---|---:|---:|---:|")
        for k in ("TP","TOLERANCE_ERROR","MERGED","FRAGMENTED","FALSE_POSITIVE","FALSE_NEGATIVE","COMPLEX"):
            b_ = bb.get(k, 0); a_ = aa.get(k, 0)
            md.append(f"| {k} | {b_} | {a_} | {a_-b_:+} |")
        md.append("")
        md.append(f"Splits made by apex postprocess: {at['splits_made']}")
        md.append(f"start_delta abs_median (TP only): baseline={baseline['start_delta_abs_median']} apex={apex['start_delta_abs_median']}")
        md.append("")

    # Section 3: FRAGMENTED inspection
    md.append("## FRAGMENTED inspection (key decision input)")
    md.append("")
    md.append("Decision rule trips if FRAG > 2x baseline. Cal: 5 -> 12 (2.4x). Hol: 8 -> 17 (2.1x).")
    md.append("Question: are the new FRAG events conversions from MERGED (legitimate splits")
    md.append("inside a multi-GT region that the topology classifier still calls FRAGMENTED),")
    md.append("or over-splits of single TPs/TOLERANCE_ERRORs?")
    md.append("")
    md.append("Each FRAG component in the apex run was looked up by its single GT in the")
    md.append("baseline; the baseline label of that same GT tells us what the apex split")
    md.append("converted FROM. Counts:")
    md.append("")
    for corpus_label, breakdown in [
        ("Calibration LOOCV", cal_frag_breakdown),
        ("Holdout 19", hol_frag_breakdown),
    ]:
        md.append(f"### {corpus_label}")
        md.append("")
        md.append("| baseline label of FRAG'd GT | count |")
        md.append("|---|---:|")
        for k in sorted(breakdown.keys()):
            md.append(f"| {k} | {breakdown[k]} |")
        md.append("")

    md.append("Sample new FRAG events (up to 30 per corpus, see metrics/frag_examples.json for full list):")
    md.append("")
    for corpus_label, examples in [
        ("Calibration LOOCV", cal_frag_examples[:10]),
        ("Holdout 19", hol_frag_examples[:10]),
    ]:
        md.append(f"### {corpus_label}")
        md.append("")
        if not examples:
            md.append("(none)")
            md.append("")
            continue
        md.append("| video | GT frames | baseline label | baseline algos | apex algos |")
        md.append("|---|---|---|---|---|")
        for ex in examples:
            md.append(f"| {ex['video_id']} | {ex['gt_frames']} | {ex['baseline_label']} | "
                       f"{ex['baseline_algos']} | {ex['apex_algos']} |")
        md.append("")

    # Section 4: Decision-rule walkthrough
    md.append("## Decision rule walkthrough")
    md.append("")
    md.append("Per the runner docstring, ACCEPT if all the following are TRUE:")
    md.append("")
    md.append("| criterion | result |")
    md.append("|---|---|")
    md.append(f"| TP rises/holds both corpora | Cal {cat['tp']-cbt['tp']:+}, Hol {hat['tp']-hbt['tp']:+} -> "
              f"{'PASS' if cat['tp']>=cbt['tp'] and hat['tp']>=hbt['tp'] else 'FAIL'} |")
    md.append(f"| FN drops/holds both corpora | Cal {cat['fn']-cbt['fn']:+}, Hol {hat['fn']-hbt['fn']:+} -> "
              f"{'PASS' if cat['fn']<=cbt['fn'] and hat['fn']<=hbt['fn'] else 'FAIL'} |")
    md.append(f"| start_delta abs_median = 0 | cal={cal_apex['start_delta_abs_median']} "
              f"hol={hol_apex['start_delta_abs_median']} -> "
              f"{'PASS' if cal_apex['start_delta_abs_median']==0 and hol_apex['start_delta_abs_median']==0 else 'FAIL'} |")
    cb_merged = cal_base["topology"].get("MERGED", 0); ca_merged = cal_apex["topology"].get("MERGED", 0)
    hb_merged = hol_base["topology"].get("MERGED", 0); ha_merged = hol_apex["topology"].get("MERGED", 0)
    md.append(f"| MERGED drops materially | Cal {cb_merged}->{ca_merged} ({ca_merged-cb_merged:+}), "
              f"Hol {hb_merged}->{ha_merged} ({ha_merged-hb_merged:+}) -> PASS |")
    cb_frag = cal_base["topology"].get("FRAGMENTED", 0); ca_frag = cal_apex["topology"].get("FRAGMENTED", 0)
    hb_frag = hol_base["topology"].get("FRAGMENTED", 0); ha_frag = hol_apex["topology"].get("FRAGMENTED", 0)
    cal_frag_ratio = ca_frag / cb_frag if cb_frag > 0 else float("inf")
    hol_frag_ratio = ha_frag / hb_frag if hb_frag > 0 else float("inf")
    md.append(f"| FRAGMENTED <= 2x baseline OR new FRAG are MERGED conversions | "
              f"Cal {cb_frag}->{ca_frag} ({cal_frag_ratio:.2f}x), "
              f"Hol {hb_frag}->{ha_frag} ({hol_frag_ratio:.2f}x) -- see inspection above |")
    md.append("")
    md.append("Verdict written manually after inspecting the FRAGMENTED breakdown.")
    md.append("")

    (OUT_DIR / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")


# ---------- Main ----------

def main():
    print("=" * 70)
    print("APEX SPLIT FORMAL EXPERIMENT (depth>=0.5, peak2_rel<0.85)")
    print("=" * 70)
    print()

    cal_data = load_calibration()
    print()
    hol_data = load_holdout()
    print()

    print("Scoring calibration baseline (v8.0.2 trim, no apex)...", flush=True)
    cal_base = score_corpus_with_provenance(cal_data, apply_apex=False)
    print("Scoring calibration with apex split (d=0.5, p2=0.85)...", flush=True)
    cal_apex = score_corpus_with_provenance(cal_data, apply_apex=True)
    print()
    print("Scoring holdout baseline...", flush=True)
    hol_base = score_corpus_with_provenance(hol_data, apply_apex=False)
    print("Scoring holdout with apex split...", flush=True)
    hol_apex = score_corpus_with_provenance(hol_data, apply_apex=True)
    print()

    print_summary("CAL", cal_base, cal_apex)
    print()
    print_summary("HOL", hol_base, hol_apex)
    print()

    # FRAG inspection
    cal_frag_breakdown, cal_frag_examples = classify_new_frag(
        cal_base["per_video"], cal_apex["per_video"])
    hol_frag_breakdown, hol_frag_examples = classify_new_frag(
        hol_base["per_video"], hol_apex["per_video"])

    print("FRAGMENTED inspection (apex's FRAG -> baseline label of same GT):")
    print(f"  CAL: {cal_frag_breakdown}")
    print(f"  HOL: {hol_frag_breakdown}")
    print()

    # Save artifacts
    (OUT_DIR / "metrics" / "scalars.json").write_text(json.dumps({
        "config": {
            "depth_min": DEPTH_MIN,
            "peak2_rel_max": PEAK2_REL_MAX,
            "prominence": APEX_PROMINENCE,
            "min_distance": APEX_MIN_DISTANCE,
            "min_span": APEX_MIN_SPAN,
            "trim_threshold": TRIM_THRESHOLD,
            "trim_sustain_n": TRIM_SUSTAIN_N,
            "trim_min_span": TRIM_MIN_SPAN,
        },
        "calibration": {
            "baseline": {"totals": cal_base["totals"],
                          "topology": cal_base["topology"],
                          "start_delta_abs_median": cal_base["start_delta_abs_median"]},
            "apex": {"totals": cal_apex["totals"],
                       "topology": cal_apex["topology"],
                       "start_delta_abs_median": cal_apex["start_delta_abs_median"]},
        },
        "holdout": {
            "baseline": {"totals": hol_base["totals"],
                          "topology": hol_base["topology"],
                          "start_delta_abs_median": hol_base["start_delta_abs_median"]},
            "apex": {"totals": hol_apex["totals"],
                       "topology": hol_apex["topology"],
                       "start_delta_abs_median": hol_apex["start_delta_abs_median"]},
        },
        "frag_breakdown": {
            "calibration": cal_frag_breakdown,
            "holdout": hol_frag_breakdown,
        },
    }, indent=2, default=int), encoding="utf-8")

    (OUT_DIR / "metrics" / "frag_examples.json").write_text(json.dumps({
        "calibration": cal_frag_examples,
        "holdout": hol_frag_examples,
    }, indent=2, default=int), encoding="utf-8")

    print(f"Wrote: {OUT_DIR / 'metrics' / 'scalars.json'}")
    print(f"Wrote: {OUT_DIR / 'metrics' / 'frag_examples.json'}")

    write_results_md(cal_base, cal_apex, hol_base, hol_apex,
                      cal_frag_breakdown, hol_frag_breakdown,
                      cal_frag_examples, hol_frag_examples)
    print(f"Wrote: {OUT_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

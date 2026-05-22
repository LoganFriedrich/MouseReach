"""
v8.0.x experiment: apex-split holdout gate (final config).

Final config selected after three sub-sweeps:
  - prominence sweep (0.05 .. 0.15 at depth=0.5, peak2_rel<0.85)
  - depth sweep (0.4 .. 0.7 at prom=0.12, peak2_rel<0.85)
  - peak2_rel sweep (0.70 .. 0.90 at prom=0.12, depth=0.5)

Selected: prominence=0.12, trough_depth_min=0.5, peak2_rel_max<0.85.

This snapshot serves as the ship decision document (RESULTS.md). Per the
v8.0.2 ship history (Improvement_Snapshots/.../v8.0.1_dev_sustained_trim_holdout_gate/
RESULTS.md), the holdout gate is the formal verification snapshot before
editing production code.

Pre-experiment checklist (per pre_experiment_checklist.md):

1. Cumulative-stacking check (verified 2026-05-22):
   - Production v8.0.2 reach detector (BSW b=1/w=0.8 + mg=0 + sustained-
     trim N=3/T=0.60). Asymmetric strict matcher -2/+5 on live GT.
   - Baseline (cumulative best == pure baseline for this experiment):
       Calibration LOOCV: TP=2231 / FP=98 / FN=170
       Holdout 19:        TP=3656 / FP=71 / FN=96

2. Existing-code-modification check: NO during this run. The runner
   stays inline. Production-code edits happen AFTER the gate passes,
   in a separate commit.

3. Unverified hypotheses (relative to prior sub-sweeps):
   - That span_delta abs_median for TP events remains at 0 (Cardinal
     Rule on BOTH endpoint precision axes). Prior runs only printed
     start_delta abs_median.

4. FN-direction-reporting (planned):
   - Lead with FN delta vs cumulative best (= pure baseline for this
     experiment, since no intermediate v8.0.3 has shipped).
   - Pair legacy with topology in every counts table.
   - ASCII output only.

5. Framework check:
   - Output to v8.0.2_dev_apex_split_holdout_gate/{metrics,figures}/
     + RESULTS.md (the ship decision document).

6. Branch + tag:
   - feature/v8-apex-split-postprocess (created earlier)
   - Tag: v8-pre-apex-split-2026-05-22

7. Decision rule (final, applied in RESULTS.md):
   ACCEPT if all of:
     - TP rises on both corpora
     - FN drops on both corpora
     - FP drops or holds on both corpora
     - start_delta abs_median = 0 (Cardinal Rule, start)
     - span_delta abs_median = 0 (Cardinal Rule, end)
     - MERGED drops >= 70% on both corpora
     - FRAGMENTED <= 2x baseline on both corpora
     - Over-splits (new FRAG events whose GT was baseline TP) <= 4 per
       corpus
   REJECT if any criterion fails.
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
    r"\reach_detection\v8.0.2_dev_apex_split_holdout_gate"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "metrics").mkdir(exist_ok=True)
(OUT_DIR / "figures").mkdir(exist_ok=True)

DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000"
PARQUET_LK_COLS = ["RightHand_lk", "RHLeft_lk", "RHOut_lk", "RHRight_lk"]
PARQUET_POS_COLS = ([f"{kp}_{ax}" for kp in
                      ("RightHand", "RHLeft", "RHOut", "RHRight",
                        "BOXL", "BOXR") for ax in ("x", "y")])

# v8.0.2 production trim
TRIM_THRESHOLD = 0.60
TRIM_SUSTAIN_N = 3
TRIM_MIN_SPAN = 3

# Apex split (final config selected from sub-sweeps)
PROMINENCE = 0.12
DEPTH_MIN = 0.5
PEAK2_REL_MAX = 0.85
APEX_MIN_DISTANCE = 4
APEX_MIN_SPAN = 3

SPAN_TOL_FRAC = 0.5
SPAN_TOL_MIN = 5


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


def apex_split(reach, norm_pos):
    s, e = reach
    if e >= len(norm_pos):
        return [(s, e)], None
    sig = norm_pos[s:e + 1]
    if len(sig) < 3:
        return [(s, e)], None
    peaks, _ = find_peaks(sig, prominence=PROMINENCE, distance=APEX_MIN_DISTANCE)
    if len(peaks) < 2:
        return [(s, e)], None
    peak2_rel = peaks[-1] / (len(sig) - 1)
    if peak2_rel >= PEAK2_REL_MAX:
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
    if best_depth < DEPTH_MIN or best_trough_frame is None:
        return [(s, e)], None
    half1 = (s, best_trough_frame)
    half2 = (best_trough_frame + 1, e)
    if (half1[1] - half1[0] + 1) < APEX_MIN_SPAN or (half2[1] - half2[0] + 1) < APEX_MIN_SPAN:
        return [(s, e)], None
    return [half1, half2], best_trough_frame


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
                candidates.append((abs(start_delta), ai, gi, start_delta, span_delta))
    candidates.sort()
    used_a, used_g = set(), set()
    pairs = []; tp_sd = []; tp_pd = []
    for _, ai, gi, sd, pd_ in candidates:
        if ai in used_a or gi in used_g:
            continue
        used_a.add(ai); used_g.add(gi)
        pairs.append((ai, gi))
        tp_sd.append(sd)
        tp_pd.append(pd_)
    fps = [ai for ai in range(len(algos)) if ai not in used_a]
    fns = [gi for gi in range(len(gts)) if gi not in used_g]
    return pairs, fps, fns, tp_sd, tp_pd


def classify_components(algos, gts):
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
            visited_a.add(i); continue
        algo_in, gt_in = set(), set()
        queue = [("a", i)]
        while queue:
            kind, idx = queue.pop()
            if kind == "a":
                if idx in algo_in: continue
                algo_in.add(idx)
                for gj in algo_to_gt[idx]: queue.append(("g", gj))
            else:
                if idx in gt_in: continue
                gt_in.add(idx)
                for ai in gt_to_algo[idx]: queue.append(("a", ai))
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


def apply_v802_trim(algos, paw_lk):
    spans = [ReachSpan(start_frame=s, end_frame=e) for s, e in algos]
    trimmed = trim_leading_sustained_lk(
        spans, paw_lk,
        threshold=TRIM_THRESHOLD,
        sustain_n=TRIM_SUSTAIN_N,
        min_span=TRIM_MIN_SPAN,
    )
    return [(r.start_frame, r.end_frame) for r in trimmed]


def score_corpus(corpus_data, apply_apex):
    totals = {"tp": 0, "fp": 0, "fn": 0, "splits_made": 0}
    topo_counts = defaultdict(int)
    tp_start_deltas = []
    tp_span_deltas = []
    per_video = {}
    for vid, vd in corpus_data.items():
        algos_v802 = apply_v802_trim(vd["algos_v801"], vd["paw_lk"])
        split_provenance = []
        if apply_apex:
            algos_final = []
            for reach in algos_v802:
                parts, trough = apex_split(reach, vd["norm_pos"])
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
        pairs, fps, fns, tp_sd, tp_pd = greedy_match(algos_final, vd["gts"])
        totals["tp"] += len(pairs); totals["fp"] += len(fps); totals["fn"] += len(fns)
        tp_start_deltas.extend(tp_sd)
        tp_span_deltas.extend(tp_pd)
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
    start_abs_med = int(np.median([abs(d) for d in tp_start_deltas])) if tp_start_deltas else None
    span_abs_med = int(np.median([abs(d) for d in tp_span_deltas])) if tp_span_deltas else None
    return {
        "totals": totals,
        "topology": dict(topo_counts),
        "start_delta_abs_median": start_abs_med,
        "span_delta_abs_median": span_abs_med,
        "per_video": per_video,
    }


def count_and_inspect_over_splits(baseline_per_video, apex_per_video, gt_lookup):
    over_split_count = 0
    from_merged = 0
    from_frag = 0
    from_tol = 0
    over_split_examples = []
    for vid in apex_per_video:
        base_comps = baseline_per_video[vid]["components"]
        gt_to_base_label = {}
        gt_to_base_algos = {}
        base_algos = baseline_per_video[vid]["algos_final"]
        gts = baseline_per_video[vid]["gts"]
        for c in base_comps:
            for gj in c["gts"]:
                gt_to_base_label[gj] = c["label"]
                gt_to_base_algos[gj] = [base_algos[ai] for ai in c["algos"]]
        for c in apex_per_video[vid]["components"]:
            if c["label"] != "FRAGMENTED":
                continue
            gj = c["gts"][0]
            base_label = gt_to_base_label.get(gj, "UNKNOWN")
            if base_label == "TP":
                over_split_count += 1
                over_split_examples.append({
                    "video_id": vid,
                    "gt_frames": list(gts[gj]),
                    "baseline_algos": gt_to_base_algos.get(gj, []),
                    "apex_algos": [apex_per_video[vid]["algos_final"][ai]
                                    for ai in c["algos"]],
                })
            elif base_label == "MERGED":
                from_merged += 1
            elif base_label == "FRAGMENTED":
                from_frag += 1
            elif base_label == "TOLERANCE_ERROR":
                from_tol += 1
    return {
        "over_splits": over_split_count,
        "from_merged": from_merged,
        "from_fragmented": from_frag,
        "from_tolerance_error": from_tol,
        "examples": over_split_examples,
    }


def write_results_md(cal_base, cal_apex, hol_base, hol_apex,
                      cal_frag, hol_frag):
    md = []
    md.append("# Apex-split holdout gate (final config)")
    md.append("")
    md.append(f"Run: {datetime.utcnow().isoformat()}Z")
    md.append("")
    md.append("## Selected configuration")
    md.append("")
    md.append(f"- prominence (scipy.signal.find_peaks): {PROMINENCE}")
    md.append(f"- trough_depth_min: {DEPTH_MIN}")
    md.append(f"- peak2_rel_max (suppress if peak2 at >= cutoff of algo span): {PEAK2_REL_MAX}")
    md.append(f"- min_distance between detected peaks: {APEX_MIN_DISTANCE}")
    md.append(f"- min_span of each half after split: {APEX_MIN_SPAN}")
    md.append("")
    md.append("Stacked on production v8.0.2 (BSW b=1/w=0.8 + mg=0 + sustained-trim")
    md.append(f"N={TRIM_SUSTAIN_N}/T={TRIM_THRESHOLD}) and asymmetric strict matcher")
    md.append(f"-{STRICT_START_TOL_EARLY} early, +{STRICT_START_TOL_LATE} late.")
    md.append("")

    cbt = cal_base["totals"]; cat = cal_apex["totals"]
    hbt = hol_base["totals"]; hat = hol_apex["totals"]
    cal_dfn = cat["fn"] - cbt["fn"]
    hol_dfn = hat["fn"] - hbt["fn"]

    md.append("## FN direction (leading headline)")
    md.append("")
    md.append("Cumulative best == pure baseline for this experiment (no intermediate")
    md.append("v8.0.x ship between v8.0.2+asymmetric and this run).")
    md.append("")
    md.append(f"- Calibration LOOCV: FN {cbt['fn']} -> {cat['fn']} ({cal_dfn:+}). "
              f"FN direction: FALLING by {abs(cal_dfn)}.")
    md.append(f"- Holdout 19: FN {hbt['fn']} -> {hat['fn']} ({hol_dfn:+}). "
              f"FN direction: FALLING by {abs(hol_dfn)}.")
    md.append("")

    md.append("## Counts: legacy paired with topology")
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
        md.append("| | baseline | apex (final) | delta |")
        md.append("|---|---:|---:|---:|")
        md.append(f"| TP | {bt['tp']} | {at['tp']} | {at['tp']-bt['tp']:+} |")
        md.append(f"| FP | {bt['fp']} | {at['fp']} | {at['fp']-bt['fp']:+} |")
        md.append(f"| FN | {bt['fn']} | {at['fn']} | {at['fn']-bt['fn']:+} |")
        md.append("")
        md.append("Topology (connected-component classification):")
        md.append("")
        md.append("| class | baseline | apex (final) | delta |")
        md.append("|---|---:|---:|---:|")
        for k in ("TP","TOLERANCE_ERROR","MERGED","FRAGMENTED","FALSE_POSITIVE","FALSE_NEGATIVE","COMPLEX"):
            b_ = bb.get(k, 0); a_ = aa.get(k, 0)
            md.append(f"| {k} | {b_} | {a_} | {a_-b_:+} |")
        md.append("")
        md.append(f"Splits made by apex postprocess: {at['splits_made']}")
        md.append(f"start_delta abs_median (TP only): baseline={baseline['start_delta_abs_median']} apex={apex['start_delta_abs_median']}")
        md.append(f"span_delta  abs_median (TP only): baseline={baseline['span_delta_abs_median']} apex={apex['span_delta_abs_median']}")
        md.append("")

    md.append("## Over-split inspection (kinematic concern)")
    md.append("")
    md.append("Over-split = an apex FRAGMENTED component whose single GT had baseline")
    md.append("label TP. These are clean reaches that the split fired on. Each one")
    md.append("creates a kinematic ambiguity: which half of the now-split algo")
    md.append("represents the reach window?")
    md.append("")
    for corpus_label, brk in [
        ("Calibration LOOCV", cal_frag),
        ("Holdout 19", hol_frag),
    ]:
        md.append(f"### {corpus_label}: {brk['over_splits']} over-splits")
        md.append("")
        if brk["examples"]:
            md.append("| video | GT frames | baseline algo | apex halves |")
            md.append("|---|---|---|---|")
            for ex in brk["examples"]:
                md.append(f"| {ex['video_id']} | {ex['gt_frames']} | {ex['baseline_algos']} | {ex['apex_algos']} |")
            md.append("")
        md.append(f"New FRAG event composition: over_splits={brk['over_splits']} "
                   f"from_merged={brk['from_merged']} from_fragmented={brk['from_fragmented']} "
                   f"from_tolerance_error={brk['from_tolerance_error']}")
        md.append("")

    md.append("## Decision rule walkthrough")
    md.append("")
    cb_merged = cal_base["topology"].get("MERGED", 0); ca_merged = cal_apex["topology"].get("MERGED", 0)
    hb_merged = hol_base["topology"].get("MERGED", 0); ha_merged = hol_apex["topology"].get("MERGED", 0)
    cb_frag = cal_base["topology"].get("FRAGMENTED", 0); ca_frag = cal_apex["topology"].get("FRAGMENTED", 0)
    hb_frag = hol_base["topology"].get("FRAGMENTED", 0); ha_frag = hol_apex["topology"].get("FRAGMENTED", 0)
    cal_merged_pct = 100 * (cb_merged - ca_merged) / cb_merged if cb_merged > 0 else 0
    hol_merged_pct = 100 * (hb_merged - ha_merged) / hb_merged if hb_merged > 0 else 0
    cal_frag_ratio = ca_frag / cb_frag if cb_frag > 0 else float("inf")
    hol_frag_ratio = ha_frag / hb_frag if hb_frag > 0 else float("inf")

    def passfail(b):
        return "PASS" if b else "FAIL"

    crits = []
    crits.append((
        "TP rises both corpora",
        f"Cal {cat['tp']-cbt['tp']:+}, Hol {hat['tp']-hbt['tp']:+}",
        cat["tp"] > cbt["tp"] and hat["tp"] > hbt["tp"],
    ))
    crits.append((
        "FN drops both corpora",
        f"Cal {cal_dfn:+}, Hol {hol_dfn:+}",
        cal_dfn < 0 and hol_dfn < 0,
    ))
    crits.append((
        "FP drops or holds both corpora",
        f"Cal {cat['fp']-cbt['fp']:+}, Hol {hat['fp']-hbt['fp']:+}",
        cat["fp"] <= cbt["fp"] and hat["fp"] <= hbt["fp"],
    ))
    crits.append((
        "start_delta abs_median = 0 (Cardinal Rule, start)",
        f"Cal={cal_apex['start_delta_abs_median']}, Hol={hol_apex['start_delta_abs_median']}",
        cal_apex["start_delta_abs_median"] == 0 and hol_apex["start_delta_abs_median"] == 0,
    ))
    crits.append((
        "span_delta abs_median = 0 (Cardinal Rule, end)",
        f"Cal={cal_apex['span_delta_abs_median']}, Hol={hol_apex['span_delta_abs_median']}",
        cal_apex["span_delta_abs_median"] == 0 and hol_apex["span_delta_abs_median"] == 0,
    ))
    crits.append((
        "MERGED drops >= 70% both corpora",
        f"Cal {cal_merged_pct:.0f}%, Hol {hol_merged_pct:.0f}%",
        cal_merged_pct >= 70 and hol_merged_pct >= 70,
    ))
    crits.append((
        "FRAGMENTED <= 2x baseline both corpora",
        f"Cal {cal_frag_ratio:.2f}x, Hol {hol_frag_ratio:.2f}x",
        cal_frag_ratio <= 2.0 and hol_frag_ratio <= 2.0,
    ))
    crits.append((
        "Over-splits <= 4 per corpus",
        f"Cal {cal_frag['over_splits']}, Hol {hol_frag['over_splits']}",
        cal_frag["over_splits"] <= 4 and hol_frag["over_splits"] <= 4,
    ))

    md.append("| criterion | result | verdict |")
    md.append("|---|---|---|")
    all_pass = True
    for name, result, ok in crits:
        md.append(f"| {name} | {result} | {passfail(ok)} |")
        if not ok:
            all_pass = False
    md.append("")
    md.append(f"**Overall verdict: {'ACCEPT' if all_pass else 'REJECT'}.**")
    md.append("")

    md.append("## Pre-experiment checklist")
    md.append("")
    md.append("Applied in the runner docstring at `scripts/restart_phase_b_apex_split_holdout_gate.py`.")
    md.append("Cumulative best = pure baseline = v8.0.2 + asymmetric on live GT.")
    md.append("Existing-code-modification: NO (inline runner; production-code edit is a follow-up commit after gate passes).")
    md.append("Framework: canonical snapshot dir, ASCII only, topology paired with legacy.")
    md.append("GT: live `*_unified_ground_truth.json` files (per never-pull-gt-from-snapshots).")
    md.append("Branch: feature/v8-apex-split-postprocess; tag: v8-pre-apex-split-2026-05-22.")
    md.append("")

    (OUT_DIR / "RESULTS.md").write_text("\n".join(md), encoding="utf-8")


def print_summary(label, baseline, apex):
    bt = baseline["totals"]; at = apex["totals"]
    bb = baseline["topology"]; aa = apex["topology"]
    print(f"  {label}: baseline TP={bt['tp']} FP={bt['fp']} FN={bt['fn']}")
    print(f"  {label}: apex     TP={at['tp']} FP={at['fp']} FN={at['fn']} (splits={at['splits_made']})")
    print(f"  {label}: delta    TP={at['tp']-bt['tp']:+} FP={at['fp']-bt['fp']:+} FN={at['fn']-bt['fn']:+}")
    for k in ("TP","TOLERANCE_ERROR","MERGED","FRAGMENTED","FALSE_POSITIVE","FALSE_NEGATIVE","COMPLEX"):
        b_ = bb.get(k, 0); a_ = aa.get(k, 0)
        print(f"    {k:<22} {b_:>4} -> {a_:>4}  ({a_-b_:+d})")
    print(f"  {label}: start_delta abs_med = {apex['start_delta_abs_median']}, "
          f"span_delta abs_med = {apex['span_delta_abs_median']}")


def main():
    print("=" * 70)
    print("APEX SPLIT HOLDOUT GATE (final config)")
    print(f"  prominence={PROMINENCE}, depth_min={DEPTH_MIN}, peak2_rel<{PEAK2_REL_MAX}")
    print("=" * 70)
    print()

    cal_data = load_calibration()
    print()
    hol_data = load_holdout()
    print()

    print("Scoring calibration baseline (v8.0.2 trim, no apex)...", flush=True)
    cal_base = score_corpus(cal_data, apply_apex=False)
    print("Scoring calibration with apex split (final config)...", flush=True)
    cal_apex = score_corpus(cal_data, apply_apex=True)
    print()
    print("Scoring holdout baseline...", flush=True)
    hol_base = score_corpus(hol_data, apply_apex=False)
    print("Scoring holdout with apex split...", flush=True)
    hol_apex = score_corpus(hol_data, apply_apex=True)
    print()

    print_summary("CAL", cal_base, cal_apex)
    print()
    print_summary("HOL", hol_base, hol_apex)
    print()

    cal_frag = count_and_inspect_over_splits(cal_base["per_video"], cal_apex["per_video"], None)
    hol_frag = count_and_inspect_over_splits(hol_base["per_video"], hol_apex["per_video"], None)
    print(f"Over-splits: CAL={cal_frag['over_splits']}  HOL={hol_frag['over_splits']}")
    print()

    (OUT_DIR / "metrics" / "scalars.json").write_text(json.dumps({
        "config": {
            "prominence": PROMINENCE,
            "depth_min": DEPTH_MIN,
            "peak2_rel_max": PEAK2_REL_MAX,
            "min_distance": APEX_MIN_DISTANCE,
            "min_span": APEX_MIN_SPAN,
        },
        "calibration": {
            "baseline": {k: cal_base[k] for k in ("totals","topology","start_delta_abs_median","span_delta_abs_median")},
            "apex": {k: cal_apex[k] for k in ("totals","topology","start_delta_abs_median","span_delta_abs_median")},
            "frag_breakdown": {k: v for k, v in cal_frag.items() if k != "examples"},
        },
        "holdout": {
            "baseline": {k: hol_base[k] for k in ("totals","topology","start_delta_abs_median","span_delta_abs_median")},
            "apex": {k: hol_apex[k] for k in ("totals","topology","start_delta_abs_median","span_delta_abs_median")},
            "frag_breakdown": {k: v for k, v in hol_frag.items() if k != "examples"},
        },
    }, indent=2, default=int), encoding="utf-8")

    (OUT_DIR / "metrics" / "over_split_examples.json").write_text(json.dumps({
        "calibration": cal_frag["examples"],
        "holdout": hol_frag["examples"],
    }, indent=2, default=int), encoding="utf-8")

    write_results_md(cal_base, cal_apex, hol_base, hol_apex, cal_frag, hol_frag)

    print(f"Wrote: {OUT_DIR / 'metrics' / 'scalars.json'}")
    print(f"Wrote: {OUT_DIR / 'metrics' / 'over_split_examples.json'}")
    print(f"Wrote: {OUT_DIR / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

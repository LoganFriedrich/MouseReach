"""For every segment in the corpus, compute the candidate
'predominantly on pillar' metrics and check the distribution by GT
class. The rule we want to add commits untouched if:

  frac_inside_pillar_circle >= FRAC_THR
  AND off_pillar_count <= MAX_OFF_PILLAR

Test passes only if NO GT-displaced_sa, GT-retrieved, or
GT-abnormal_exception segment passes. Otherwise the rule is too loose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_7_pellet_displaced_to_sa import PAW_BODYPARTS
from mousereach.reach.v8.features import load_dlc_h5

QUARANTINE = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough")
DLC_DIR = QUARANTINE / "dlc"
GT_DIR = QUARANTINE / "gt"
CORPUS_DIR = Path(r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots\_corpus\2026-04-30_restart_inventory")

PELLET_LK_THR = 0.7
PAW_LK_THR = 0.5
ON_PILLAR_RADII = 1.0  # pellet inside pillar circle = within 1 radius
TRANSITION_ZONE_HALF = 5


def find_dlc(vid):
    return next(DLC_DIR.glob(f"{vid}DLC_*.h5"))


def gt_payload(vid):
    return json.loads((GT_DIR / f"{vid}_unified_ground_truth.json").read_text(encoding="utf-8"))


def gt_segment_bounds(gt):
    gt_b = [int(b["frame"]) for b in gt.get("segmentation", {}).get("boundaries", [])]
    return {i + 1: (gt_b[i], gt_b[i + 1] - 1) for i in range(len(gt_b) - 1)}


def reaches_for_segment(gt, sn):
    out = []
    for r in gt.get("reaches", {}).get("reaches", []) or []:
        if r.get("segment_num") == sn:
            s = r.get("start_frame"); e = r.get("end_frame")
            if s is not None and e is not None:
                out.append((int(s), int(e)))
    return out


def during_reach_mask(seg_start, seg_end, reaches):
    n = seg_end - seg_start + 1
    m = np.zeros(n, dtype=bool)
    for rs, re in reaches:
        s = max(seg_start, int(rs))
        e = min(seg_end, int(re))
        if e < s:
            continue
        m[s - seg_start:e - seg_start + 1] = True
    return m


def compute(vid, sn, seg_start, seg_end, dlc, reaches):
    clean_end = seg_end - TRANSITION_ZONE_HALF
    if clean_end <= seg_start:
        return None
    sub_raw = dlc.iloc[seg_start:clean_end + 1]
    n = len(sub_raw)
    if n == 0:
        return None
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pillar_cx = geom["pillar_cx"].to_numpy(dtype=float)
    pillar_cy = geom["pillar_cy"].to_numpy(dtype=float)
    pillar_r = geom["pillar_r"].to_numpy(dtype=float)

    pellet_x = sub["Pellet_x"].to_numpy(dtype=float)
    pellet_y = sub["Pellet_y"].to_numpy(dtype=float)
    pellet_lk = sub_raw["Pellet_likelihood"].to_numpy(dtype=float)
    dist_radii = (np.sqrt((pellet_x - pillar_cx)**2 + (pellet_y - pillar_cy)**2)
                  / np.maximum(pillar_r, 1e-6))

    during = during_reach_mask(seg_start, clean_end, reaches)
    not_during = ~during
    n_not_during = int(not_during.sum())
    confident = pellet_lk >= PELLET_LK_THR
    eligible = not_during & confident
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        return None

    inside = (dist_radii <= ON_PILLAR_RADII) & eligible
    outside = (dist_radii > ON_PILLAR_RADII) & eligible
    frac_inside = float(inside.sum() / n_eligible)
    n_off_pillar = int(outside.sum())
    # Visibility fraction: pellet confidently visible across non-during-
    # reach frames. Untouched ~ 1.0; retrieved drops because the
    # pellet disappears (mouse eats it).
    visibility_frac = float(n_eligible / max(n_not_during, 1))

    # Late-segment frac_inside: in last 25% of the clean zone, was the
    # pellet still on the pillar? For untouched: yes (pellet never moved).
    # For displaced: no (pellet must be off-pillar at end). For
    # retrieved: pellet may be missing (low visibility) so this is
    # capped by visibility too.
    n = len(sub_raw)
    late_start = int(n * 0.75)
    late_eligible = eligible.copy()
    late_eligible[:late_start] = False
    late_eligible_count = int(late_eligible.sum())
    late_inside_count = int(((dist_radii <= ON_PILLAR_RADII) & late_eligible).sum())
    late_frac_inside = float(late_inside_count / max(late_eligible_count, 1)) if late_eligible_count else 0.0
    late_off_pillar_count = int(((dist_radii > ON_PILLAR_RADII) & late_eligible).sum())

    return {
        "n_eligible": n_eligible,
        "n_not_during": n_not_during,
        "frac_inside": frac_inside,
        "n_off_pillar": n_off_pillar,
        "visibility_frac": visibility_frac,
        "late_eligible_count": late_eligible_count,
        "late_inside_count": late_inside_count,
        "late_frac_inside": late_frac_inside,
        "late_off_pillar_count": late_off_pillar_count,
    }


def main():
    folds = json.loads((CORPUS_DIR / "cv_folds.json").read_text())
    train_pool_ids = folds["train_pool"]["video_ids"]
    rows = []
    for vid in train_pool_ids:
        dlc = load_dlc_h5(find_dlc(vid))
        gt = gt_payload(vid)
        bounds = gt_segment_bounds(gt)
        gt_outs = {s["segment_num"]: s for s in gt.get("outcomes", {}).get("segments", []) or []}
        for sn, (seg_start, seg_end) in bounds.items():
            entry = gt_outs.get(sn, {})
            outcome = entry.get("outcome")
            if outcome == "displaced_outside":
                outcome = "displaced_sa"
            reaches = reaches_for_segment(gt, sn)
            m = compute(vid, sn, seg_start, seg_end, dlc, reaches)
            if m is None:
                continue
            rows.append({
                "vid": vid, "sn": sn, "outcome": outcome,
                "n_eligible": m["n_eligible"],
                "n_not_during": m["n_not_during"],
                "frac_inside": m["frac_inside"],
                "n_off_pillar": m["n_off_pillar"],
                "visibility_frac": m["visibility_frac"],
                "late_frac_inside": m["late_frac_inside"],
                "late_off_pillar_count": m["late_off_pillar_count"],
                "late_eligible_count": m["late_eligible_count"],
            })

    by_oc = defaultdict(list)
    for r in rows:
        by_oc[r["outcome"]].append(r)

    print(f"Distributions of (frac_inside, n_off_pillar, visibility_frac) by GT class:")
    for oc, rs in sorted(by_oc.items()):
        fr = sorted(r["frac_inside"] for r in rs)
        op = sorted(r["n_off_pillar"] for r in rs)
        vf = sorted(r["visibility_frac"] for r in rs)
        n = len(fr)
        print(f"\n{oc}: n={n}")
        if not fr: continue
        for label, arr in [("frac_inside", fr), ("n_off_pillar", op), ("visibility_frac", vf)]:
            line = f"  {label}: "
            for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
                i = max(0, min(n - 1, int(p / 100 * (n - 1))))
                v = arr[i]
                line += f"p{p}={v:.3f}  " if isinstance(v, float) else f"p{p}={v}  "
            print(line)

    # Test rule with late_frac_inside discriminator
    print()
    print("Candidate rule with LATE-SEGMENT frac_inside discriminator:")
    KNOWN_RESIDUAL = {("20250624_CNT0115_P2", 17), ("20250630_CNT0104_P3", 2),
                      ("20250630_CNT0104_P3", 6), ("20250806_CNT0311_P2", 17),
                      ("20250806_CNT0311_P2", 19), ("20250806_CNT0311_P2", 20),
                      ("20250806_CNT0312_P2", 18), ("20250811_CNT0305_P2", 14),
                      ("20250909_CNT0209_P4", 4), ("20250912_CNT0209_P3", 9),
                      ("20251008_CNT0301_P4", 8)}
    for frac_thr in [0.99, 0.97, 0.95, 0.94, 0.90, 0.85]:
        for late_thr in [0.99, 0.95, 0.90]:
            for vis_thr in [0.99, 0.95, 0.85]:
                counts = defaultdict(int)
                caught_resid = 0
                for r in rows:
                    if (r["frac_inside"] >= frac_thr
                            and r["visibility_frac"] >= vis_thr
                            and r["late_frac_inside"] >= late_thr):
                        counts[r["outcome"]] += 1
                        if (r["vid"], r["sn"]) in KNOWN_RESIDUAL:
                            caught_resid += 1
                disp_or_ret = (counts.get("displaced_sa", 0)
                               + counts.get("retrieved", 0)
                               + counts.get("abnormal_exception", 0))
                print(f"  frac>={frac_thr:.2f} AND vis>={vis_thr:.2f} AND late>={late_thr:.2f}: "
                      f"untouched={counts.get('untouched', 0):3d}/{len(by_oc.get('untouched', [])):3d}  "
                      f"contam={disp_or_ret} (d={counts.get('displaced_sa', 0)}, "
                      f"r={counts.get('retrieved', 0)}, a={counts.get('abnormal_exception', 0)})  "
                      f"resid_caught: {caught_resid}/11")
    print()
    # Show late_frac_inside distribution by class
    print("late_frac_inside distribution by GT class:")
    for oc, rs in sorted(by_oc.items()):
        arr = sorted(r["late_frac_inside"] for r in rs)
        n = len(arr)
        if not arr: continue
        line = f"  {oc}: "
        for p in [0, 1, 5, 25, 50, 75, 95, 99, 100]:
            i = max(0, min(n - 1, int(p / 100 * (n - 1))))
            line += f"p{p}={arr[i]:.3f}  "
        print(line)
    print()
    print("4 short-segment residuals -- their late_frac_inside:")
    SHORT_RESIDUALS = [("20250806_CNT0311_P2", 17), ("20250806_CNT0311_P2", 19),
                      ("20250806_CNT0311_P2", 20), ("20250806_CNT0312_P2", 18)]
    for vid, sn in SHORT_RESIDUALS:
        match = next((r for r in rows if r["vid"] == vid and r["sn"] == sn), None)
        if match:
            print(f"  {vid} seg {sn:2d}: frac={match['frac_inside']:.3f}  "
                  f"late_frac={match['late_frac_inside']:.3f}  "
                  f"vis={match['visibility_frac']:.3f}  "
                  f"n_eligible={match['n_eligible']}  "
                  f"late_off={match['late_off_pillar_count']}/{match['late_eligible_count']}")

    # Test the candidate rule WITHOUT n_off_pillar (since frac_inside
    # already encodes 1 - frac_off / frac_eligible)
    print()
    print("Candidate rule (frac_inside + visibility only):")
    for frac_thr in [0.99, 0.97, 0.96, 0.95, 0.94]:
        for vis_thr in [0.85, 0.90, 0.95, 0.97, 0.99]:
            counts = defaultdict(int)
            for r in rows:
                if (r["frac_inside"] >= frac_thr
                        and r["visibility_frac"] >= vis_thr):
                    counts[r["outcome"]] += 1
            disp_or_ret = counts.get("displaced_sa", 0) + counts.get("retrieved", 0) + counts.get("abnormal_exception", 0)
            n_residuals_caught = sum(1 for r in rows
                                     if (r["vid"], r["sn"]) in {("20250624_CNT0115_P2", 17),
                                                                ("20250630_CNT0104_P3", 2),
                                                                ("20250630_CNT0104_P3", 6),
                                                                ("20250806_CNT0311_P2", 17),
                                                                ("20250806_CNT0311_P2", 19),
                                                                ("20250806_CNT0311_P2", 20),
                                                                ("20250806_CNT0312_P2", 18),
                                                                ("20250811_CNT0305_P2", 14),
                                                                ("20250909_CNT0209_P4", 4),
                                                                ("20250912_CNT0209_P3", 9),
                                                                ("20251008_CNT0301_P4", 8)}
                                     and r["frac_inside"] >= frac_thr
                                     and r["visibility_frac"] >= vis_thr)
            print(f"  frac>={frac_thr:.2f} AND vis>={vis_thr:.2f}: "
                  f"untouched={counts.get('untouched', 0):3d}/{len(by_oc.get('untouched', [])):3d}  "
                  f"displaced_sa={counts.get('displaced_sa', 0):3d}  "
                  f"retrieved={counts.get('retrieved', 0):3d}  "
                  f"abnormal={counts.get('abnormal_exception', 0):3d}  "
                  f"==> contamination: {disp_or_ret}  residuals_caught: {n_residuals_caught}/11")

    # Per-case status of the 11 known residuals -- did the proposed
    # rule capture them?
    KNOWN_RESIDUAL = [
        ("20250624_CNT0115_P2", 17), ("20250630_CNT0104_P3", 2),
        ("20250630_CNT0104_P3", 6), ("20250806_CNT0311_P2", 17),
        ("20250806_CNT0311_P2", 19), ("20250806_CNT0311_P2", 20),
        ("20250806_CNT0312_P2", 18), ("20250811_CNT0305_P2", 14),
        ("20250909_CNT0209_P4", 4), ("20250912_CNT0209_P3", 9),
        ("20251008_CNT0301_P4", 8),
    ]
    print()
    print("Known 11 residual untouched cases -- their (frac_inside, n_off_pillar):")
    for vid, sn in KNOWN_RESIDUAL:
        match = next((r for r in rows if r["vid"] == vid and r["sn"] == sn), None)
        if match:
            print(f"  {vid} seg {sn:2d}  frac_inside={match['frac_inside']:.4f}  "
                  f"n_off_pillar={match['n_off_pillar']:3d}  "
                  f"n_eligible={match['n_eligible']:4d}  "
                  f"visibility_frac={match['visibility_frac']:.3f}")


if __name__ == "__main__":
    main()

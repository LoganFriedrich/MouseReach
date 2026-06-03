"""
DIAGNOSTIC (not an experiment): deep-dive on the 11 remaining model-3.1
outcome errors after v6.0.4 (Lever A). For each segment, dump the DLC
reality + reaches + the full gate trace so we can judge whether there's a
signal the gates are missing vs genuinely-untrackable DLC.

Per segment prints:
  - identity, GT, reaches (local windows)
  - pellet tracking: % confident frames; low-lk runs; post-last-reach vanish
  - pellet position classification (confident frames): on-pillar / in-SA /
    outside-SA; plus the last-30-confident-frame dominant state
  - pillar-reveal: mean Pillar lk pre-first-reach vs post-last-reach
  - per-reach displacement (Stage 27 signal) + per-reach vanish
  - full gate trace: every stage's decision + reason

Read-only; nothing written.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_leverA", SCRIPTS_DIR / "outcome_leverA_net_displaced_sa_2026-06-03.py")
lva = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lva)

detect_reaches_v8 = lva.detect_reaches_v8
load_dlc_h5 = lva.load_dlc_h5
SegmentInput = lva.SegmentInput
build_stages_with_leverA = lva.build_stages_with_leverA
load_gt_segments = lva.load_gt_segments

from mousereach.lib.dlc_cleaning import clean_dlc_bodyparts
from mousereach.lib.pillar_geometry import compute_pillar_geometry_series
from mousereach.outcomes.v6_cascade.stage_27_displaced_sa_via_unique_high_displacement_reach import (
    Stage27DisplacedSaViaUniqueHighDisplacement)

M31_DLC = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
               r"\DLC_2026_03_27\Processing\updated dlc model 3.1")
M31_GT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs"
              r"\DLC_2026_03_27\iterations\2026-04-28_outcome_v4.0.0_dev_walkthrough\gt")
GEN_DLC = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
               r"\generalization_test_2026-05-11\algo_outputs_current")
GEN_GT = Path(r"Y:\2_Connectome\Behavior\MouseReach_Improvement\iterations"
              r"\generalization_test_2026-05-11\gt")
CORPUS = {"m31": (M31_DLC, M31_GT), "gen": (GEN_DLC, GEN_GT)}

TARGETS = [
    # corpus, video, seg, GT, note
    ("m31", "20250626_CNT0102_P4", 16, "displaced_sa", "DLC corner"),
    ("m31", "20250627_CNT0105_P1", 8,  "displaced_sa", "DLC loses"),
    ("m31", "20250627_CNT0105_P1", 14, "displaced_sa", "DLC loses, dark"),
    ("m31", "20250630_CNT0104_P3", 17, "displaced_sa", "near end"),
    ("m31", "20250710_CNT0215_P4", 8,  "displaced_sa", "displaced_outside (expected_triage)"),
    ("m31", "20250820_CNT0103_P3", 13, "retrieved",    "phantom pellet"),
    ("m31", "20250821_CNT0110_P4", 8,  "displaced_sa", "displaced_outside"),
    ("m31", "20251009_CNT0310_P2", 14, "abnormal",     "tail knockover"),
    ("m31", "20251022_CNT0413_P4", 12, "retrieved",    "phantom pellet"),
    ("m31", "20251024_CNT0402_P4", 5,  "displaced_sa", "DLC under boxl"),
    ("m31", "20251031_CNT0407_P1", 19, "displaced_sa", "near end (Lever B)"),
    # generalization
    ("gen", "20250625_CNT0102_P4", 3,  "displaced_sa", "triaged"),
    ("gen", "20250625_CNT0102_P4", 10, "displaced_sa", "->retrieved; displaced_outside"),
    ("gen", "20250625_CNT0102_P4", 12, "untouched",    "->displaced_sa; phantom (Lever A overfire)"),
    ("gen", "20250711_CNT0216_P1", 1,  "displaced_sa", "->retrieved; pellet under BOXL"),
    ("gen", "20250715_CNT0209_P2", 17, "displaced_sa", "->retrieved; displaced_outside"),
    ("gen", "20250718_CNT0214_P1", 2,  "displaced_sa", "->retrieved; corner"),
    ("gen", "20250718_CNT0214_P1", 19, "retrieved",    "->displaced_sa; phantom pellet"),
    ("gen", "20250822_CNT0110_P2", 6,  "retrieved",    "triaged; GT uncertain"),
    ("gen", "20251008_CNT0303_P2", 6,  "displaced_sa", "->retrieved; displaced_outside"),
    ("gen", "20251008_CNT0303_P2", 7,  "retrieved",    "->displaced_sa; two pellets"),
    ("gen", "20251022_CNT0402_P4", 17, "retrieved",    "triaged; fringe displaced_outside->fell back"),
    ("gen", "20251027_CNT0404_P4", 3,  "retrieved",    "->displaced_sa; phantom"),
    ("gen", "20251027_CNT0404_P4", 4,  "retrieved",    "->displaced_sa; phantom"),
    ("gen", "20251027_CNT0404_P4", 11, "retrieved",    "triaged; phantom"),
    ("gen", "20251027_CNT0404_P4", 18, "retrieved",    "triaged"),
]

LK = 0.7


def _runs_below(mask, min_len):
    runs = []; start = -1
    for i, b in enumerate(mask):
        if b and start < 0:
            start = i
        elif not b and start >= 0:
            if i - start >= min_len:
                runs.append((start, i - 1))
            start = -1
    if start >= 0 and len(mask) - start >= min_len:
        runs.append((start, len(mask) - 1))
    return runs


def characterize(seg, reaches_local):
    s, e = seg.seg_start, seg.seg_end
    sub_raw = seg.dlc_df.iloc[s:e + 1]
    n = len(sub_raw)
    sub = clean_dlc_bodyparts(sub_raw, other_bodyparts_to_clean=("Pellet",))
    geom = compute_pillar_geometry_series(sub)
    pcx = geom["pillar_cx"].to_numpy(float); pcy = geom["pillar_cy"].to_numpy(float)
    pr = geom["pillar_r"].to_numpy(float)
    px = sub["Pellet_x"].to_numpy(float); py = sub["Pellet_y"].to_numpy(float)
    plk = sub_raw["Pellet_likelihood"].to_numpy(float)
    pil_lk = sub_raw["Pillar_likelihood"].to_numpy(float)
    dist_r = np.sqrt((px - pcx) ** 2 + (py - pcy) ** 2) / np.maximum(pr, 1e-6)
    conf = plk >= LK
    # SA box (median over confident off-pillar frames; fallback all conf)
    off = conf & (dist_r > 1.0)
    base = off if off.sum() >= 5 else conf
    def med(col, m):
        a = sub[col].to_numpy(float)[m]; return float(np.median(a)) if len(a) else float("nan")
    sa_top = (med("SATL_y", base) + med("SATR_y", base)) / 2
    sa_bot = (med("SABL_y", base) + med("SABR_y", base)) / 2
    sa_left = min(med("SABL_x", base), med("SATL_x", base))
    sa_right = max(med("SABR_x", base), med("SATR_x", base))
    in_sa = (py >= sa_top) & (py <= sa_bot) & (px >= sa_left) & (px <= sa_right)
    on_pillar = conf & (dist_r <= 1.0)
    in_sa_off = conf & (dist_r > 1.0) & in_sa
    outside = conf & (dist_r > 1.0) & (~in_sa)
    print(f"  pellet tracked (lk>={LK}): {100*conf.mean():.0f}% of {n} frames")
    lowruns = _runs_below(~conf, 15)
    if lowruns:
        print(f"  low-lk runs (>=15f), local: {lowruns[:6]}{' ...' if len(lowruns)>6 else ''}")
    print(f"  confident-frame position: on_pillar={int(on_pillar.sum())}  "
          f"in_SA_off={int(in_sa_off.sum())}  outside_SA={int(outside.sum())}")
    # last 30 confident frames dominant state
    last_idx = np.where(conf)[0]
    if len(last_idx):
        tail = last_idx[-30:]
        states = []
        for i in tail:
            states.append("onP" if on_pillar[i] else ("SA" if in_sa_off[i] else "OUT"))
        from collections import Counter
        print(f"  last-30-confident-frame states: {dict(Counter(states))}")
        lp = last_idx[-1]
        print(f"  LAST confident pellet @ local {lp}: dist={dist_r[lp]:.2f}r "
              f"{'on_pillar' if on_pillar[lp] else ('in_SA' if in_sa_off[lp] else 'OUTSIDE_SA')}")
    # pillar reveal
    if reaches_local:
        fr = reaches_local[0][0]; lr = reaches_local[-1][1]
        pre = pil_lk[:fr]; post = pil_lk[lr + 1:]
        print(f"  pillar lk: pre-first-reach mean={np.mean(pre) if len(pre) else float('nan'):.2f}  "
              f"post-last-reach mean={np.mean(post) if len(post) else float('nan'):.2f}")
        # post-last-reach vanish
        post_conf = conf[lr + 1:]
        vr = _runs_below(~post_conf, 1)
        maxv = max((b - a + 1 for a, b in vr), default=0)
        print(f"  post-last-reach: {len(post_conf)}f window, longest vanish run={maxv}f")


def main():
    st27 = Stage27DisplacedSaViaUniqueHighDisplacement()
    stages = build_stages_with_leverA(video_dir=None)
    cache = {}
    for corpus, vid, seg_num, gt, note in TARGETS:
        dlc_dir, gt_dir = CORPUS[corpus]
        key = (corpus, vid)
        if key not in cache:
            cache[key] = (load_dlc_h5(sorted(dlc_dir.glob(f"{vid}DLC_*.h5"))[0]),
                          load_gt_segments(gt_dir, vid))
        dlc, segments = cache[key]
        reaches = detect_reaches_v8(dlc)
        s, e = segments[seg_num - 1]
        seg_r = [(r0, r1) for r0, r1 in reaches if s <= r0 <= e]
        seg = SegmentInput(video_id=vid, segment_num=seg_num, seg_start=s, seg_end=e,
                           dlc_df=dlc, reach_windows=seg_r)
        rl = [(a - s, b - s) for a, b in seg_r]
        print("=" * 80)
        print(f"[{corpus}] {vid} s{seg_num}  GT={gt}  [{note}]")
        print(f"  frames [{s},{e}] len={e-s+1}  n_reaches={len(seg_r)}  reaches(local)={rl[:10]}{' ...' if len(rl)>10 else ''}")
        characterize(seg, rl)
        # per-reach displacement + vanish (Stage 27 signal)
        sig = []
        sr = sorted(seg_r)
        for i, (rs, re) in enumerate(sr):
            prev = sr[i-1][1] if i > 0 else None
            nxt = sr[i+1][0] if i+1 < len(sr) else None
            v, d = st27._per_reach_signal(dlc, rs, re, prev, nxt, s, e)
            sig.append((bool(v), None if d is None else round(d, 1)))
        print(f"  per-reach (vanish,disp_px): {sig[:12]}{' ...' if len(sig)>12 else ''}")
        # gate trace (all stages)
        print("  -- gate trace --")
        for label, stage in stages:
            d = stage.decide(seg)
            mark = " <<< " + d.decision.upper() if d.decision != "continue" else ""
            print(f"    {label:46s} {d.decision:8s}{mark}")
            if d.decision != "continue" or "displaced" in label or "stage_2" in label or label.endswith(("_26","27","29")):
                print(f"        {(d.reason or '')[:110]}")
            if d.decision in ("commit", "triage"):
                break
        print()


if __name__ == "__main__":
    main()

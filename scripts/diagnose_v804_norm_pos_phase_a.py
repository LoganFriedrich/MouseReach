"""Phase A diagnostic for the norm_pos-based postprocess proposals.

Two design proposals dropped by A-side Claude 2026-06-02 (no algo runs):
- leading-trim rescue via norm_pos rising edge -> targets 7 hol LEADING_TRIM_STRANDED FNs
- trailing-trim refinement via norm_pos return-to-baseline -> targets 5 hol end-extended TOL pairs

Phase A is read-only verification that the norm_pos signal contains the structure
the proposals assume. No postprocess changes, no algo re-runs. Just compute
`compute_hand_to_boxl_norm_pos` over the targeted cases and report whether the
expected pattern exists.

Pre-experiment checklist (in writing per pre_experiment_checklist.md)
---------------------------------------------------------------------
1. Cumulative-stacking: Production is v8.0.4 (BSW b=1/w=0.8 + leading-trim
   T=0.60/N=3 + apex-split + trailing-trim T=0.60/N=3) + asymmetric -2/+5
   matcher + matcher-aware topology classifier. Verified 2026-06-02 against
   `pipeline_versions.json` and `outcome_cascade_state_2026-06-02.md`. No
   accepted-but-pending reach-detection improvements; cumulative best is
   simply current production. Comparison baseline for any downstream Phase B
   would be v8.0.4.
2. Existing-module modification: NONE. Pure-read diagnostic; consumes
   `mousereach.reach.v8.postprocess.compute_hand_to_boxl_norm_pos` as-is.
3. Unverified hypotheses (from proposals):
   - "norm_pos rises near GT_start for LEADING_TRIM_STRANDED cases" -- this
     diagnostic tests it.
   - "norm_pos drops below 0.5 * peak after apex for end-extended TOL pairs"
     -- this diagnostic tests it.
   - "norm_pos does NOT drop similarly in clean TPs or in CNT0303_P2
     under-extension TOLs" -- this diagnostic tests both controls.
   - Smoothing-kernel washout (CNT0314_P4-style fast-reach pattern) might
     wash out the targeted signals -- diagnostic surfaces this if present.
4. FN-direction: diagnostic only, no FN delta produced. Phase B (if pursued)
   would have to lead-report FN delta both corpora vs cumulative best.
5. Framework: canonical scoring not invoked. Output is a per-case probe
   plus aggregate yes/no table written to snapshot dir.
6. Branch + tag: created `feature/v8-norm-pos-phase-a-diagnostic` from
   master at e72926d; pre-tag `pre-norm-pos-phase-a-diagnostic-2026-06-02`.
7. Decision-rule: this is a diagnostic, no accept/reject. The outcome
   decides whether Phase B implementation is worth running. Specifically:
   - Leading-trim rescue: pursue Phase B if >=5 of 7 stranded FNs have a
     norm_pos rising edge landing within +/-5 of GT_start under at least
     one (threshold, window) cell, AND control FPs don't show the same
     pattern.
   - Trailing-trim refinement: pursue Phase B if >=4 of 5 end-extended
     TOL pairs have norm_pos crossing 0.5*peak after apex landing within
     +/-5 of GT_end, AND CNT0303_P2 under-extension cases do NOT cross.

Inputs
------
- Live GT JSONs: `Y:\\2_Connectome\\Behavior\\MouseReach_Improvement\\iterations\\generalization_test_2026-05-11\\gt\\*_unified_ground_truth.json`
  (per `feedback_never_pull_gt_from_snapshots`)
- DLC h5s holdout: `iterations\\generalization_test_2026-05-11\\dlc\\`
- v8.0.3 manifests for case discovery (validated upstream by probe script)
- `mousereach.reach.v8.postprocess.compute_hand_to_boxl_norm_pos`

Outputs
-------
Written to `Improvement_Snapshots\\reach_detection\\
v8.0.4_dev_norm_pos_phase_a_diagnostic\\metrics\\diagnostic.json`,
plus RESULTS.md prose.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# C: runtime copy of mousereach is stale (May 6, pre-v8.0.4). Pin imports to
# the Y: master copy for this diagnostic. Removed when C: sync happens.
_Y_SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
if _Y_SRC not in sys.path:
    sys.path.insert(0, _Y_SRC)

# Drop any pre-imported stale C: copies so the Y: ones win.
for _mod in [m for m in list(sys.modules) if m.startswith("mousereach")]:
    del sys.modules[_mod]

import numpy as np

from mousereach.reach.core.geometry import load_dlc
from mousereach.reach.v8.postprocess import (
    compute_hand_to_boxl_norm_pos,
    compute_paw_mean_lk,
)

print(f"[sys.path hack] using mousereach from: {Path(__import__('mousereach').__file__).parent}")

# Paths ----------------------------------------------------------------------
HOL_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\dlc"
)
HOL_GT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\gt"
)
HOL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\holdout_2026_05_11"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\Improvement_Snapshots\reach_detection"
    r"\v8.0.4_dev_norm_pos_phase_a_diagnostic"
)
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000.h5"

# Cases ----------------------------------------------------------------------
# 7 LEADING_TRIM_STRANDED holdout FNs (from
# v8.0.4_dev_stranded_fn_failure_modes/metrics/diagnostic.json)
STRANDED_FNS: List[Dict[str, Any]] = [
    {"video": "20250715_CNT0209_P2", "gt_start": 26307, "gt_end": 26312, "peak_proba": 0.968},
    {"video": "20250718_CNT0206_P1", "gt_start":  6975, "gt_end":  6978, "peak_proba": 0.891},
    {"video": "20250806_CNT0316_P3", "gt_start": 11161, "gt_end": 11163, "peak_proba": 0.871},
    {"video": "20250822_CNT0110_P2", "gt_start":   439, "gt_end":   443, "peak_proba": 0.951},
    {"video": "20250822_CNT0110_P2", "gt_start": 16896, "gt_end": 16899, "peak_proba": 0.962},
    {"video": "20251023_CNT0407_P3", "gt_start": 12156, "gt_end": 12159, "peak_proba": 0.959},
    {"video": "20251027_CNT0404_P4", "gt_start": 26654, "gt_end": 26656, "peak_proba": 0.986},
]

# 5 holdout end-extended TOL pairs (all on CNT0316_P3, end_delta > +5
# from probe script)
END_EXTENDED_TOLS: List[Dict[str, Any]] = [
    {"video": "20250806_CNT0316_P3", "algo": (5854,  5873),  "gt": (5856,  5862),
     "start_delta": -2, "end_delta": 11},
    {"video": "20250806_CNT0316_P3", "algo": (33388, 33417), "gt": (33390, 33401),
     "start_delta": -2, "end_delta": 16},
    {"video": "20250806_CNT0316_P3", "algo": (33475, 33501), "gt": (33478, 33493),
     "start_delta": -3, "end_delta": 8},
    {"video": "20250806_CNT0316_P3", "algo": (33530, 33557), "gt": (33532, 33546),
     "start_delta": -2, "end_delta": 11},
    {"video": "20250806_CNT0316_P3", "algo": (33558, 33584), "gt": (33560, 33575),
     "start_delta": -2, "end_delta": 9},
]

# 3 CNT0303_P2 under-extension TOL pairs (control: trailing trim must NOT
# fire here) + 1 CNT0310_P2 short-algo TOL (additional control)
UNDER_EXTENSION_CONTROLS: List[Dict[str, Any]] = [
    {"video": "20251008_CNT0303_P2", "algo": (1884,  1894),  "gt": (1884,  1915),
     "end_delta": -21},
    {"video": "20251008_CNT0303_P2", "algo": (6675,  6695),  "gt": (6675,  6739),
     "end_delta": -44},
    {"video": "20251008_CNT0303_P2", "algo": (13473, 13499), "gt": (13473, 13535),
     "end_delta": -36},
    {"video": "20250811_CNT0310_P2", "algo": (22417, 22419), "gt": (22417, 22427),
     "end_delta": -8},
]

# Thresholds for the sweeps. The proposals nominate these specifically.
RISING_THRESHOLDS_ABS = [0.25, 0.30, 0.35, 0.40]  # absolute norm_pos units
RISING_THRESHOLDS_REL = [0.30, 0.40, 0.50]  # fraction of in-window peak
RISING_WINDOWS_K = [2, 3, 4]  # frames of sustained rising required
RETURN_BASELINE_FRACS = [0.4, 0.5, 0.6, 0.7]  # fraction of peak

ASYM_TOL_START_EARLY = 2
ASYM_TOL_START_LATE = 5
ASYM_TOL_END_LATE_FOR_TRAILING = 5  # span_delta tolerance is more complex;
                                    # for diagnostic we report end_delta directly


# Helpers --------------------------------------------------------------------
def _video_dlc_path(video_id: str) -> Path:
    return HOL_DLC_DIR / f"{video_id}{DLC_SUFFIX}"


def _video_gt_path(video_id: str) -> Path:
    return HOL_GT_DIR / f"{video_id}_unified_ground_truth.json"


def _video_manifest_path(video_id: str) -> Path:
    return HOL_MANIFEST_DIR / f"{video_id}.json"


def _load_norm_pos(video_id: str) -> np.ndarray:
    """Compute the v8.0.4 production norm_pos signal across the full video."""
    dlc = load_dlc(_video_dlc_path(video_id))
    return compute_hand_to_boxl_norm_pos(dlc)


def _load_paw_mean_lk(video_id: str) -> np.ndarray:
    """Compute the v8.0.4 production paw_mean_lk signal across the full video."""
    dlc = load_dlc(_video_dlc_path(video_id))
    return compute_paw_mean_lk(dlc)


TRAILING_TRIM_LK_THRESHOLD = 0.60
TRAILING_TRIM_SUSTAIN_N = 3


def _option_c_trim_active(
    paw_mean_lk: np.ndarray, algo_end: int
) -> Tuple[bool, list, str]:
    """Decide whether v8.0.4 paw_lk trailing-trim was active on a reach
    whose post-trim end is `algo_end`.

    The trim walks inward from the raw GBM end; it STOPS at the first
    frame whose [end-2 ... end] window contains any frame >= threshold.
    So:
      - If paw_mean_lk[algo_end + 1 .. algo_end + 3] are ALL <
        threshold, those frames would have been eaten by trim (sustain
        window all low-lk -> trim advances inward past them). The trim
        was ACTIVE.
      - If any of those post-end frames is >= threshold, the trim could
        not have eaten them (it would have stopped earlier). So the
        algo_end is just the raw GBM end and trim was INACTIVE.

    Edge: if algo_end + 3 is past video length, fall back to whatever
    frames exist; if no frames exist past algo_end, the trim cannot have
    been active there.

    Returns (was_active, post_end_lk_values, reason).
    """
    end_plus = []
    for j in range(1, TRAILING_TRIM_SUSTAIN_N + 1):
        idx = algo_end + j
        if idx >= len(paw_mean_lk):
            return False, end_plus, "post_end_past_video"
        v = float(paw_mean_lk[idx])
        end_plus.append(v)
    if all(v < TRAILING_TRIM_LK_THRESHOLD for v in end_plus):
        return True, end_plus, "all_3_post_end_low_lk"
    return False, end_plus, "post_end_has_confident_frame"


def _find_rising_edge_abs(
    np_arr: np.ndarray, start: int, end: int, threshold: float, k: int
) -> Optional[int]:
    """Find first frame in [start, end] where norm_pos >= threshold AND
    norm_pos is rising over k frames (last value > first value)."""
    for t in range(start, end + 1):
        if t < k or t + k >= len(np_arr):
            continue
        if np_arr[t] < threshold:
            continue
        if np_arr[t + k] > np_arr[t - k]:
            return t
    return None


def _find_rising_edge_rel(
    np_arr: np.ndarray, start: int, end: int, frac_of_peak: float, k: int
) -> Optional[int]:
    """Same as _find_rising_edge_abs but threshold = frac_of_peak * peak(np in [start, end])."""
    if start >= end:
        return None
    window = np_arr[start : end + 1]
    if window.size == 0:
        return None
    peak = float(np.nanmax(window))
    threshold = frac_of_peak * peak
    return _find_rising_edge_abs(np_arr, start, end, threshold, k)


def _find_return_to_baseline(
    np_arr: np.ndarray, start: int, end: int, frac_of_peak: float
) -> Tuple[Optional[int], Optional[int], float]:
    """For the trailing-trim refinement test: walk the algo span [start, end],
    find peak frame and peak value, then find first frame AFTER peak where
    norm_pos[t] < frac_of_peak * peak.

    Returns (peak_frame, crossing_frame_or_None, peak_value).
    """
    if start >= end:
        return None, None, float("nan")
    window = np_arr[start : end + 1]
    if window.size == 0:
        return None, None, float("nan")
    if np.all(np.isnan(window)):
        return None, None, float("nan")
    peak_offset = int(np.nanargmax(window))
    peak_frame = start + peak_offset
    peak_value = float(window[peak_offset])
    threshold = frac_of_peak * peak_value
    crossing = None
    for t in range(peak_frame + 1, end + 1):
        if np.isnan(np_arr[t]):
            continue
        if np_arr[t] < threshold:
            crossing = t
            break
    return peak_frame, crossing, peak_value


# Test 1: Leading-trim rescue --------------------------------------------------
@dataclass
class StrandedFNResult:
    video: str
    gt_start: int
    gt_end: int
    gt_span: int
    peak_proba: float
    norm_pos_at_gt_start: Optional[float]
    norm_pos_peak_in_gt: Optional[float]
    norm_pos_min_pre_gt: Optional[float]  # min in [gt_start-15, gt_start]
    # For each (threshold_kind, threshold_value, k): rising-edge frame and delta
    rising_edge_probes: Dict[str, Optional[int]]
    rising_edge_deltas: Dict[str, Optional[int]]
    rising_edge_within_tol: Dict[str, bool]
    # Did at least one (threshold, k) cell land within tolerance?
    any_cell_within_tol: bool


def probe_stranded_fn(case: Dict[str, Any], norm_pos: np.ndarray) -> StrandedFNResult:
    """Probe whether a rising norm_pos edge exists near the GT_start for one
    stranded FN case. Search window is [gt_start - 20, gt_end + 5] -- captures
    where the rescue would look in the pre-trim algo span.
    """
    gt_s = case["gt_start"]
    gt_e = case["gt_end"]
    span = gt_e - gt_s + 1

    # Context windows
    s_search = max(0, gt_s - 20)
    e_search = min(len(norm_pos) - 1, gt_e + 5)
    s_pre = max(0, gt_s - 15)

    np_at_gt = float(norm_pos[gt_s]) if 0 <= gt_s < len(norm_pos) else None
    np_peak_in_gt = (
        float(np.nanmax(norm_pos[gt_s : gt_e + 1]))
        if gt_s < len(norm_pos)
        else None
    )
    np_min_pre = (
        float(np.nanmin(norm_pos[s_pre : gt_s + 1]))
        if gt_s < len(norm_pos)
        else None
    )

    probes: Dict[str, Optional[int]] = {}
    deltas: Dict[str, Optional[int]] = {}
    within_tol: Dict[str, bool] = {}
    any_in_tol = False

    for thr in RISING_THRESHOLDS_ABS:
        for k in RISING_WINDOWS_K:
            key = f"abs_{thr:.2f}_k{k}"
            t = _find_rising_edge_abs(norm_pos, s_search, e_search, thr, k)
            probes[key] = t
            delta = (t - gt_s) if t is not None else None
            deltas[key] = delta
            in_tol = (
                t is not None
                and -ASYM_TOL_START_EARLY <= delta <= ASYM_TOL_START_LATE
            )
            within_tol[key] = in_tol
            if in_tol:
                any_in_tol = True

    for frac in RISING_THRESHOLDS_REL:
        for k in RISING_WINDOWS_K:
            key = f"rel_{frac:.2f}_k{k}"
            t = _find_rising_edge_rel(norm_pos, s_search, e_search, frac, k)
            probes[key] = t
            delta = (t - gt_s) if t is not None else None
            deltas[key] = delta
            in_tol = (
                t is not None
                and -ASYM_TOL_START_EARLY <= delta <= ASYM_TOL_START_LATE
            )
            within_tol[key] = in_tol
            if in_tol:
                any_in_tol = True

    return StrandedFNResult(
        video=case["video"],
        gt_start=gt_s,
        gt_end=gt_e,
        gt_span=span,
        peak_proba=case["peak_proba"],
        norm_pos_at_gt_start=np_at_gt,
        norm_pos_peak_in_gt=np_peak_in_gt,
        norm_pos_min_pre_gt=np_min_pre,
        rising_edge_probes=probes,
        rising_edge_deltas=deltas,
        rising_edge_within_tol=within_tol,
        any_cell_within_tol=any_in_tol,
    )


# Test 2: Trailing-trim refinement --------------------------------------------
@dataclass
class EndExtendedResult:
    video: str
    algo_start: int
    algo_end: int
    gt_start: int
    gt_end: int
    start_delta: int
    end_delta: int  # algo_end - gt_end
    norm_pos_peak: float
    norm_pos_peak_frame: int
    norm_pos_peak_rel_pos: float  # peak_frame's position as fraction of [algo_start, algo_end]
    paw_lk_trim_was_active: bool
    paw_lk_post_end_3f: list
    option_c_reason: str
    crossings: Dict[str, Optional[int]]
    cross_to_gt_end_delta: Dict[str, Optional[int]]
    cross_within_tol: Dict[str, bool]
    # New end_delta if we adopted the crossing as new algo_end (per cell)
    new_end_delta: Dict[str, Optional[int]]
    any_cell_resolves: bool


def probe_end_extended(
    case: Dict[str, Any], norm_pos: np.ndarray, paw_mean_lk: np.ndarray
) -> EndExtendedResult:
    algo_s, algo_e = case["algo"]
    gt_s, gt_e = case["gt"]
    trim_active, post_end_lk, reason = _option_c_trim_active(paw_mean_lk, algo_e)
    peak_frame, _, peak_val = _find_return_to_baseline(norm_pos, algo_s, algo_e, 0.5)
    if peak_frame is None:
        return EndExtendedResult(
            video=case["video"],
            algo_start=algo_s,
            algo_end=algo_e,
            gt_start=gt_s,
            gt_end=gt_e,
            start_delta=case["start_delta"],
            end_delta=case["end_delta"],
            norm_pos_peak=float("nan"),
            norm_pos_peak_frame=-1,
            norm_pos_peak_rel_pos=float("nan"),
            paw_lk_trim_was_active=trim_active,
            paw_lk_post_end_3f=post_end_lk,
            option_c_reason=reason,
            crossings={}, cross_to_gt_end_delta={}, cross_within_tol={},
            new_end_delta={}, any_cell_resolves=False,
        )
    rel_pos = (peak_frame - algo_s) / max(algo_e - algo_s, 1)

    crossings: Dict[str, Optional[int]] = {}
    deltas: Dict[str, Optional[int]] = {}
    within_tol: Dict[str, bool] = {}
    new_end_delta: Dict[str, Optional[int]] = {}
    any_resolves = False

    for frac in RETURN_BASELINE_FRACS:
        _, cross, _ = _find_return_to_baseline(norm_pos, algo_s, algo_e, frac)
        key = f"frac_{frac:.2f}"
        crossings[key] = cross
        if cross is None:
            deltas[key] = None
            within_tol[key] = False
            new_end_delta[key] = None
            continue
        delta = cross - gt_e
        deltas[key] = delta
        # New end_delta would be cross - gt_e (the new algo_end - gt_end)
        new_end_delta[key] = delta
        in_tol = -ASYM_TOL_END_LATE_FOR_TRAILING <= delta <= ASYM_TOL_END_LATE_FOR_TRAILING
        within_tol[key] = in_tol
        if in_tol:
            any_resolves = True

    return EndExtendedResult(
        video=case["video"],
        algo_start=algo_s,
        algo_end=algo_e,
        gt_start=gt_s,
        gt_end=gt_e,
        start_delta=case["start_delta"],
        end_delta=case["end_delta"],
        norm_pos_peak=peak_val,
        norm_pos_peak_frame=peak_frame,
        norm_pos_peak_rel_pos=rel_pos,
        paw_lk_trim_was_active=trim_active,
        paw_lk_post_end_3f=post_end_lk,
        option_c_reason=reason,
        crossings=crossings,
        cross_to_gt_end_delta=deltas,
        cross_within_tol=within_tol,
        new_end_delta=new_end_delta,
        any_cell_resolves=any_resolves,
    )


# Test 2 controls -------------------------------------------------------------
@dataclass
class UnderExtensionResult:
    video: str
    algo_start: int
    algo_end: int
    gt_start: int
    gt_end: int
    end_delta: int
    paw_lk_trim_was_active: bool
    paw_lk_post_end_3f: list
    option_c_reason: str
    # Would the secondary trim fire? Yes only if a return-crossing exists
    # within the algo span AT each frac_of_peak cell AND option-c allows.
    crossings: Dict[str, Optional[int]]
    fires_at_no_gate: Dict[str, bool]  # crossing exists (no option-c gate)
    fires_at_option_c: Dict[str, bool]  # crossing exists AND trim was inactive
    any_cell_fires_no_gate: bool
    any_cell_fires_option_c: bool


def probe_under_extension(
    case: Dict[str, Any], norm_pos: np.ndarray, paw_mean_lk: np.ndarray
) -> UnderExtensionResult:
    algo_s, algo_e = case["algo"]
    trim_active, post_end_lk, reason = _option_c_trim_active(paw_mean_lk, algo_e)
    crossings: Dict[str, Optional[int]] = {}
    fires_no_gate: Dict[str, bool] = {}
    fires_oc: Dict[str, bool] = {}
    any_no_gate = False
    any_oc = False
    for frac in RETURN_BASELINE_FRACS:
        _, cross, _ = _find_return_to_baseline(norm_pos, algo_s, algo_e, frac)
        key = f"frac_{frac:.2f}"
        crossings[key] = cross
        fng = cross is not None
        foc = fng and not trim_active
        fires_no_gate[key] = fng
        fires_oc[key] = foc
        if fng:
            any_no_gate = True
        if foc:
            any_oc = True
    return UnderExtensionResult(
        video=case["video"],
        algo_start=algo_s,
        algo_end=algo_e,
        gt_start=case["gt"][0],
        gt_end=case["gt"][1],
        end_delta=case["end_delta"],
        paw_lk_trim_was_active=trim_active,
        paw_lk_post_end_3f=post_end_lk,
        option_c_reason=reason,
        crossings=crossings,
        fires_at_no_gate=fires_no_gate,
        fires_at_option_c=fires_oc,
        any_cell_fires_no_gate=any_no_gate,
        any_cell_fires_option_c=any_oc,
    )


# Test 2 clean-TP control on CNT0316_P3 ---------------------------------------
def load_clean_tps_for_video(video_id: str, n_max: int = 10) -> List[Dict[str, Any]]:
    """Pull clean TPs (start_delta=0 AND span_delta=0) from the manifest for
    a video. Used as a control: trailing-trim refinement must NOT inappropriately
    cut these.
    """
    path = _video_manifest_path(video_id)
    if not path.exists():
        return []
    with open(path) as f:
        d = json.load(f)
    tps = []
    for e in d.get("events", []):
        if e.get("kind") != "TP":
            continue
        if (e.get("start_delta") == 0) and (e.get("span_delta") == 0):
            algo = e.get("detector") or {}
            gt = e.get("gt") or {}
            tps.append(
                {
                    "video": video_id,
                    "algo": (algo.get("start"), algo.get("end")),
                    "gt": (gt.get("start"), gt.get("end")),
                }
            )
        if len(tps) >= n_max:
            break
    return tps


@dataclass
class CleanTPResult:
    video: str
    algo_start: int
    algo_end: int
    gt_end: int
    paw_lk_trim_was_active: bool
    paw_lk_post_end_3f: list
    option_c_reason: str
    crossings: Dict[str, Optional[int]]
    fires_at_no_gate: Dict[str, bool]
    fires_at_option_c: Dict[str, bool]
    new_end_delta_if_fires: Dict[str, Optional[int]]
    any_cell_fires_no_gate: bool
    any_cell_fires_option_c: bool
    worst_new_end_delta_no_gate: Optional[int]
    worst_new_end_delta_option_c: Optional[int]


def probe_clean_tp(
    case: Dict[str, Any], norm_pos: np.ndarray, paw_mean_lk: np.ndarray
) -> CleanTPResult:
    algo_s, algo_e = case["algo"]
    gt_e = case["gt"][1]
    trim_active, post_end_lk, reason = _option_c_trim_active(paw_mean_lk, algo_e)
    crossings: Dict[str, Optional[int]] = {}
    fires_no_gate: Dict[str, bool] = {}
    fires_oc: Dict[str, bool] = {}
    new_end_delta: Dict[str, Optional[int]] = {}
    any_no_gate = False
    any_oc = False
    worst_no_gate = None
    worst_oc = None
    for frac in RETURN_BASELINE_FRACS:
        _, cross, _ = _find_return_to_baseline(norm_pos, algo_s, algo_e, frac)
        key = f"frac_{frac:.2f}"
        crossings[key] = cross
        fng = cross is not None
        foc = fng and not trim_active
        fires_no_gate[key] = fng
        fires_oc[key] = foc
        if fng:
            any_no_gate = True
            d = cross - gt_e
            new_end_delta[key] = d
            if worst_no_gate is None or abs(d) > abs(worst_no_gate):
                worst_no_gate = d
            if foc:
                any_oc = True
                if worst_oc is None or abs(d) > abs(worst_oc):
                    worst_oc = d
        else:
            new_end_delta[key] = None
    return CleanTPResult(
        video=case["video"],
        algo_start=algo_s,
        algo_end=algo_e,
        gt_end=gt_e,
        paw_lk_trim_was_active=trim_active,
        paw_lk_post_end_3f=post_end_lk,
        option_c_reason=reason,
        crossings=crossings,
        fires_at_no_gate=fires_no_gate,
        fires_at_option_c=fires_oc,
        new_end_delta_if_fires=new_end_delta,
        any_cell_fires_no_gate=any_no_gate,
        any_cell_fires_option_c=any_oc,
        worst_new_end_delta_no_gate=worst_no_gate,
        worst_new_end_delta_option_c=worst_oc,
    )


# Main -----------------------------------------------------------------------
def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Cache norm_pos + paw_mean_lk signals per video (avoid recomputing)
    norm_pos_cache: Dict[str, np.ndarray] = {}
    paw_lk_cache: Dict[str, np.ndarray] = {}

    def get_norm_pos(video_id: str) -> np.ndarray:
        if video_id not in norm_pos_cache:
            print(f"  loading norm_pos for {video_id}...")
            norm_pos_cache[video_id] = _load_norm_pos(video_id)
        return norm_pos_cache[video_id]

    def get_paw_lk(video_id: str) -> np.ndarray:
        if video_id not in paw_lk_cache:
            print(f"  loading paw_mean_lk for {video_id}...")
            paw_lk_cache[video_id] = _load_paw_mean_lk(video_id)
        return paw_lk_cache[video_id]

    # Test 1: leading-trim rescue (7 cases)
    print("=" * 76)
    print("Test 1: Leading-trim rescue -- norm_pos rising edge near GT_start")
    print("=" * 76)
    test1_results: List[StrandedFNResult] = []
    for case in STRANDED_FNS:
        np_arr = get_norm_pos(case["video"])
        res = probe_stranded_fn(case, np_arr)
        test1_results.append(res)
        n_in_tol = sum(1 for v in res.rising_edge_within_tol.values() if v)
        total = len(res.rising_edge_within_tol)
        print(
            f"  {case['video']:30s} GT=[{case['gt_start']:>6},{case['gt_end']:>6}] "
            f"span={res.gt_span:>2}f peak_proba={res.peak_proba:.3f} "
            f"np_at_gt={res.norm_pos_at_gt_start:.3f} np_peak={res.norm_pos_peak_in_gt:.3f}"
            f" -> {n_in_tol}/{total} cells land in tolerance"
        )

    # Test 2: trailing-trim refinement (5 end-extended TOL pairs)
    print()
    print("=" * 76)
    print("Test 2: Trailing-trim refinement -- norm_pos return below frac*peak")
    print("=" * 76)
    test2_results: List[EndExtendedResult] = []
    for case in END_EXTENDED_TOLS:
        np_arr = get_norm_pos(case["video"])
        lk_arr = get_paw_lk(case["video"])
        res = probe_end_extended(case, np_arr, lk_arr)
        test2_results.append(res)
        n_in_tol = sum(1 for v in res.cross_within_tol.values() if v)
        total = len(res.cross_within_tol)
        cross_str = ", ".join(
            f"{k}=>{('-' if v is None else str(v))}" for k, v in res.cross_to_gt_end_delta.items()
        )
        print(
            f"  {case['video']:30s} algo=[{case['algo'][0]:>6},{case['algo'][1]:>6}] "
            f"gt_end={case['gt'][1]:>6} ed={case['end_delta']:>+3} "
            f"peak={res.norm_pos_peak:.3f}@{res.norm_pos_peak_frame:>6} "
            f"rel_pos={res.norm_pos_peak_rel_pos:.2f} trim_act={res.paw_lk_trim_was_active} "
            f"post_lk={res.paw_lk_post_end_3f} "
            f"cross_to_gt_end: [{cross_str}] {n_in_tol}/{total} cells in tol"
        )

    # Test 2 control A: under-extension (must NOT fire)
    print()
    print("=" * 76)
    print("Test 2-A control: Under-extension TOLs -- must NOT trigger secondary trim")
    print("=" * 76)
    test2_a_results: List[UnderExtensionResult] = []
    for case in UNDER_EXTENSION_CONTROLS:
        np_arr = get_norm_pos(case["video"])
        lk_arr = get_paw_lk(case["video"])
        res = probe_under_extension(case, np_arr, lk_arr)
        test2_a_results.append(res)
        n_ng = sum(1 for v in res.fires_at_no_gate.values() if v)
        n_oc = sum(1 for v in res.fires_at_option_c.values() if v)
        total = len(res.fires_at_no_gate)
        print(
            f"  {case['video']:30s} algo=[{case['algo'][0]:>6},{case['algo'][1]:>6}] "
            f"gt=[{case['gt'][0]:>6},{case['gt'][1]:>6}] ed={case['end_delta']:>+3} "
            f"trim_act={res.paw_lk_trim_was_active} "
            f"FIRES no_gate={n_ng}/{total} option_c={n_oc}/{total}"
        )

    # Test 2 control B: clean TPs on CNT0316_P3 (must NOT fire)
    print()
    print("=" * 76)
    print("Test 2-B control: Clean TPs on CNT0316_P3 -- secondary must not shorten")
    print("=" * 76)
    clean_tps = load_clean_tps_for_video("20250806_CNT0316_P3", n_max=20)
    test2_b_results: List[CleanTPResult] = []
    for case in clean_tps:
        np_arr = get_norm_pos(case["video"])
        lk_arr = get_paw_lk(case["video"])
        res = probe_clean_tp(case, np_arr, lk_arr)
        test2_b_results.append(res)
        n_ng = sum(1 for v in res.fires_at_no_gate.values() if v)
        n_oc = sum(1 for v in res.fires_at_option_c.values() if v)
        total = len(res.fires_at_no_gate)
        wng = res.worst_new_end_delta_no_gate
        woc = res.worst_new_end_delta_option_c
        wng_str = f"{wng:+d}" if wng is not None else "n/a"
        woc_str = f"{woc:+d}" if woc is not None else "n/a"
        print(
            f"  algo=[{case['algo'][0]:>6},{case['algo'][1]:>6}] "
            f"gt_end={case['gt'][1]:>6} trim_act={res.paw_lk_trim_was_active} "
            f"FIRES no_gate={n_ng}/{total} (worst={wng_str}) "
            f"option_c={n_oc}/{total} (worst={woc_str})"
        )

    # Aggregate
    summary = {
        "test1_leading_trim_rescue": {
            "n_cases": len(test1_results),
            "n_with_at_least_one_cell_in_tol": sum(1 for r in test1_results if r.any_cell_within_tol),
            "per_case": [asdict(r) for r in test1_results],
        },
        "test2_trailing_trim_refinement": {
            "n_cases": len(test2_results),
            "n_resolved_by_any_cell": sum(1 for r in test2_results if r.any_cell_resolves),
            "per_case": [asdict(r) for r in test2_results],
        },
        "test2_control_a_under_extension": {
            "n_cases": len(test2_a_results),
            "n_fires_no_gate": sum(1 for r in test2_a_results if r.any_cell_fires_no_gate),
            "n_fires_option_c": sum(1 for r in test2_a_results if r.any_cell_fires_option_c),
            "per_case": [asdict(r) for r in test2_a_results],
        },
        "test2_control_b_clean_tps_cnt0316_p3": {
            "n_cases": len(test2_b_results),
            "n_fires_no_gate": sum(1 for r in test2_b_results if r.any_cell_fires_no_gate),
            "n_fires_option_c": sum(1 for r in test2_b_results if r.any_cell_fires_option_c),
            "max_new_end_delta_abs_no_gate": max(
                (abs(r.worst_new_end_delta_no_gate) for r in test2_b_results if r.worst_new_end_delta_no_gate is not None),
                default=0,
            ),
            "max_new_end_delta_abs_option_c": max(
                (abs(r.worst_new_end_delta_option_c) for r in test2_b_results if r.worst_new_end_delta_option_c is not None),
                default=0,
            ),
            "per_case": [asdict(r) for r in test2_b_results],
        },
    }

    out_path = metrics_dir / "diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print()
    print(f"Wrote diagnostic JSON: {out_path}")

    # Headline
    print()
    print("=" * 76)
    print("HEADLINE")
    print("=" * 76)
    t1 = summary["test1_leading_trim_rescue"]
    t2 = summary["test2_trailing_trim_refinement"]
    t2a = summary["test2_control_a_under_extension"]
    t2b = summary["test2_control_b_clean_tps_cnt0316_p3"]
    print(
        f"Test 1 leading-trim rescue: {t1['n_with_at_least_one_cell_in_tol']} / "
        f"{t1['n_cases']} stranded FNs have a norm_pos rising edge within "
        f"start tolerance under at least one (threshold, k) cell."
    )
    print(
        f"Test 2 trailing-trim refinement: {t2['n_resolved_by_any_cell']} / "
        f"{t2['n_cases']} end-extended TOLs resolved by a norm_pos return-"
        f"crossing within end tolerance under at least one frac_of_peak cell."
    )
    print(
        f"Test 2-A control: under-extension TOLs that would fire -- "
        f"no_gate {t2a['n_fires_no_gate']}/{t2a['n_cases']}, "
        f"option_c {t2a['n_fires_option_c']}/{t2a['n_cases']}."
    )
    print(
        f"Test 2-B control: clean TPs that would fire -- "
        f"no_gate {t2b['n_fires_no_gate']}/{t2b['n_cases']} "
        f"(worst abs delta {t2b['max_new_end_delta_abs_no_gate']}), "
        f"option_c {t2b['n_fires_option_c']}/{t2b['n_cases']} "
        f"(worst abs delta {t2b['max_new_end_delta_abs_option_c']})."
    )


if __name__ == "__main__":
    main()

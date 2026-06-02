"""Phase A step (a): characterize all FRAGMENTED components on cal + hol.

Goal: split the FRAG residual (cal 5 + hol 7 filtered) into actionable
sub-populations:
  - hold-during-extension (the target mechanism): algo emits the extension
    piece, paw_lk drops during a stationary apex hold, GBM re-emits during
    retraction. Net FRAGMENTED. 2026-05-22 standing finding identified 2
    hol cases (CNT0303_P2 cid=135 gap=12f, CNT0407_P3 cid=259 gap=19f).
  - apex-split over-split (different mechanism): v8.0.3 apex-split fired
    inappropriately on a clean single reach. 2026-05-22 ship doc says 2
    cal + 1 hol such cases.
  - other (residual): different mechanisms; would need separate analysis.

Pre-experiment checklist (in writing per pre_experiment_checklist.md)
---------------------------------------------------------------------
1. Cumulative-stacking: production v8.0.4 (BSW b=1/w=0.8 + leading-trim
   + apex-split + trailing-trim) + asymmetric -2/+5 matcher +
   matcher-aware topology classifier. No pending stacked changes.
   Comparison baseline for any downstream Phase B would be v8.0.4.
2. Existing-module modification: NONE. Pure-read probe.
3. Unverified hypotheses (this probe tests):
   - "FRAG events split into 3 subcategories"
   - "hold-during-extension subset is a clean target population"
   - Apex-split over-splits have small inter-piece gaps; hold cases
     have larger gaps (5+ frames) with the second piece at the GT tail.
4. FN direction: not applicable; characterization only.
5. Framework: snapshot at
   Improvement_Snapshots/reach_detection/v8.0.4_dev_frag_characterization/
6. Branch + tag: `feature/v8-frag-characterization` from master @ 34cb123;
   tag `pre-frag-characterization-2026-06-02`.
7. Decision-rule: characterization probe, no accept/reject. Output
   decides whether step (b) Phase A discriminator test on the
   hold-during-extension subset is worth running.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Y: src pin (C: runtime is pre-v8.0.4 stale; per feedback_no_agentic_behavior
# avoid touching C: until a deliberate sync.)
_Y_SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
if _Y_SRC not in sys.path:
    sys.path.insert(0, _Y_SRC)
for _mod in [m for m in list(sys.modules) if m.startswith("mousereach")]:
    del sys.modules[_mod]

import numpy as np

from mousereach.reach.core.geometry import load_dlc
from mousereach.reach.v8.postprocess import (
    compute_hand_to_boxl_norm_pos,
    compute_paw_mean_lk,
)

# Paths ----------------------------------------------------------------------
HOL_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\iterations\generalization_test_2026-05-11\dlc"
)
HOL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\holdout_2026_05_11"
)
CAL_DLC_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\validation_runs\DLC_2026_03_27\Processing\updated dlc model 3.1"
)
CAL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\calibration_loocv"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\Improvement_Snapshots\reach_detection"
    r"\v8.0.4_dev_frag_characterization"
)
DLC_SUFFIX = "DLC_resnet50_MPSAOct27shuffle1_100000.h5"

# Filters that apply to "filtered" topology counts (see pipeline_versions.json
# notes).
MIN_REPORTED_SPAN = 4


# Helpers --------------------------------------------------------------------
def _algo_span_len(s: int, e: int) -> int:
    return e - s + 1


def _algo_outside_gt_seg(event: Dict[str, Any]) -> bool:
    return bool(event.get("outside_gt_segmentation", False))


def _span_passes_filter(event: Dict[str, Any]) -> bool:
    """A FP / FN passes the headline-metric filters if the algo span is
    >= MIN_REPORTED_SPAN AND not flagged outside_gt_segmentation."""
    if _algo_outside_gt_seg(event):
        return False
    algo = event.get("detector")
    if algo is None:
        return True  # GT-side event (no algo span to filter on)
    s, e = algo["start"], algo["end"]
    return _algo_span_len(s, e) >= MIN_REPORTED_SPAN


@dataclass
class FragComponent:
    corpus: str
    video: str
    component_id: int
    algo_pieces: List[Tuple[int, int]]  # all algo intervals in component, sorted
    gt_span: Optional[Tuple[int, int]]  # the GT this component represents
    span_passes_filter: bool  # all algo pieces pass filter?
    # Matcher-pair info (drawn from manifest event categories/start_delta/span_delta)
    has_matcher_pair: bool  # one piece is the TP-paired algo
    matched_piece: Optional[Tuple[int, int]]  # the algo span the matcher matched
    matched_start_delta: Optional[int]
    matched_span_delta: Optional[int]
    matched_kind: Optional[str]  # "TP" or "TOL" if matcher matched within tolerance
    unmatched_pieces: List[Tuple[int, int]]  # algo pieces with no GT match (FP-within-gt etc.)
    # Derived characterization
    n_pieces: int
    inter_piece_gaps: List[int]  # gap (frames) between consecutive pieces
    largest_gap: int
    second_piece_at_gt_tail: bool  # second piece starts AT or AFTER 50% of GT span
    classification: str  # "hold_during_extension" | "apex_split_over_split"
                         # | "other"
    classification_reason: str


def _classify(comp: FragComponent) -> Tuple[str, str]:
    """Classify a FRAG component into hold/apex-over/other.

    Decision rules (heuristic, to be revised after looking at the data):
      - "apex_split_over_split": all pieces are within the GT span, max gap
        between pieces <= 3 frames. Apex-split's prom=0.12 trough splits
        produce gaps of 1-3 frames typically.
      - "hold_during_extension": >= 2 pieces, max gap >= 5 frames, second
        piece starts at or after the midpoint of the GT span (suggests
        retract-after-hold).
      - "other": everything else.
    """
    if not comp.algo_pieces or comp.gt_span is None:
        return "other", "no_gt_or_pieces"

    gt_s, gt_e = comp.gt_span
    gt_span_len = gt_e - gt_s + 1

    pieces = sorted(comp.algo_pieces)

    # Outside-GT extent? Pieces that extend much past GT_end suggest something
    # different (over-extension, not hold-during-extension).
    last_piece_end = pieces[-1][1]
    pre_gt_start = pieces[0][0]
    pre_gt_overshoot = max(0, gt_s - pre_gt_start)
    post_gt_overshoot = max(0, last_piece_end - gt_e)

    if comp.largest_gap <= 3 and post_gt_overshoot <= 2 and pre_gt_overshoot <= 2:
        return "apex_split_over_split", f"max_gap={comp.largest_gap}f pieces_inside_gt"

    if (
        comp.largest_gap >= 5
        and comp.second_piece_at_gt_tail
        and gt_span_len >= 8  # hold-during-extension typically on longer reaches
    ):
        return (
            "hold_during_extension",
            f"max_gap={comp.largest_gap}f second_piece_at_tail "
            f"gt_span={gt_span_len}f",
        )

    return (
        "other",
        f"max_gap={comp.largest_gap}f second_at_tail={comp.second_piece_at_gt_tail} "
        f"gt_span={gt_span_len}f pre_over={pre_gt_overshoot} post_over={post_gt_overshoot}",
    )


def _extract_frag_components(manifest_path: Path, corpus: str) -> List[FragComponent]:
    with open(manifest_path) as f:
        d = json.load(f)
    video = d["video_id"]
    comps_by_id: Dict[int, List[Dict[str, Any]]] = {}
    for e in d.get("events", []):
        if e.get("topology") != "FRAGMENTED":
            continue
        cid = e.get("component_id")
        if cid is None:
            continue
        comps_by_id.setdefault(cid, []).append(e)

    results: List[FragComponent] = []
    for cid, evs in comps_by_id.items():
        # Collect algo pieces + identify matcher state
        algo_pieces: List[Tuple[int, int]] = []
        unmatched_pieces: List[Tuple[int, int]] = []
        gt_span: Optional[Tuple[int, int]] = None
        all_passes = True
        matched_piece: Optional[Tuple[int, int]] = None
        matched_start_d: Optional[int] = None
        matched_span_d: Optional[int] = None
        matched_kind: Optional[str] = None
        for e in evs:
            algo = e.get("detector")
            if algo is not None:
                p = (algo["start"], algo["end"])
                algo_pieces.append(p)
                if not _span_passes_filter(e):
                    all_passes = False
                # Identify the matcher-paired algo: in manifest, the TP entry
                # has kind="TP", or a TOL entry (FP+FN paired in same component).
                # We look at this event's "kind" + "category" to classify.
                if e.get("kind") == "TP":
                    matched_piece = p
                    matched_start_d = e.get("start_delta")
                    matched_span_d = e.get("span_delta")
                    matched_kind = "TP"
                else:
                    unmatched_pieces.append(p)
            gt = e.get("gt")
            if gt is not None and gt_span is None:
                gt_span = (gt["start"], gt["end"])
        if not algo_pieces:
            continue
        algo_pieces.sort()
        unmatched_pieces.sort()
        # Inter-piece gaps
        gaps = [
            algo_pieces[i + 1][0] - algo_pieces[i][1] - 1
            for i in range(len(algo_pieces) - 1)
        ]
        largest_gap = max(gaps) if gaps else 0
        # Second piece at GT tail?
        second_at_tail = False
        if gt_span and len(algo_pieces) >= 2:
            gt_s, gt_e = gt_span
            second_start = algo_pieces[1][0]
            mid = gt_s + (gt_e - gt_s) // 2
            second_at_tail = second_start >= mid

        comp = FragComponent(
            corpus=corpus,
            video=video,
            component_id=cid,
            algo_pieces=algo_pieces,
            gt_span=gt_span,
            span_passes_filter=all_passes,
            has_matcher_pair=matched_piece is not None,
            matched_piece=matched_piece,
            matched_start_delta=matched_start_d,
            matched_span_delta=matched_span_d,
            matched_kind=matched_kind,
            unmatched_pieces=unmatched_pieces,
            n_pieces=len(algo_pieces),
            inter_piece_gaps=gaps,
            largest_gap=largest_gap,
            second_piece_at_gt_tail=second_at_tail,
            classification="(pending)",
            classification_reason="",
        )
        cls, reason = _classify(comp)
        comp.classification = cls
        comp.classification_reason = reason
        results.append(comp)
    return results


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    all_comps: List[FragComponent] = []
    for label, dirpath in (("calibration", CAL_MANIFEST_DIR), ("holdout", HOL_MANIFEST_DIR)):
        files = sorted(dirpath.glob("*.json"))
        for f in files:
            all_comps.extend(_extract_frag_components(f, label))

    # Summaries
    from collections import Counter
    cls_count_all: Counter = Counter()
    cls_count_filtered: Counter = Counter()
    by_corpus_filtered: Dict[str, Counter] = {"calibration": Counter(), "holdout": Counter()}
    for c in all_comps:
        cls_count_all[c.classification] += 1
        if c.span_passes_filter:
            cls_count_filtered[c.classification] += 1
            by_corpus_filtered[c.corpus][c.classification] += 1

    # Report
    print("=" * 78)
    print("FRAG component characterization (v8.0.4 manifests, v8.0.3 manifest dir)")
    print("=" * 78)
    print(f"Total components: {len(all_comps)}")
    print(f"Components passing MIN_SPAN+outside_gt_seg filter: "
          f"{sum(cls_count_filtered.values())}")
    print()
    print("Classification (filtered):")
    for cls in ("hold_during_extension", "apex_split_over_split", "other"):
        print(
            f"  {cls:30s} cal={by_corpus_filtered['calibration'][cls]:>2}  "
            f"hol={by_corpus_filtered['holdout'][cls]:>2}  "
            f"total={cls_count_filtered[cls]:>2}"
        )
    print()
    print("Classification (unfiltered, for reference):")
    for cls, n in cls_count_all.most_common():
        print(f"  {cls:30s} unfiltered={n:>2}")
    print()
    n_filt_with_match = sum(
        1 for c in all_comps if c.span_passes_filter and c.has_matcher_pair
    )
    n_filt_no_match = sum(
        1 for c in all_comps if c.span_passes_filter and not c.has_matcher_pair
    )
    print(
        f"Matcher-pair distribution (filtered): with_match={n_filt_with_match}, "
        f"no_match={n_filt_no_match}"
    )
    print()
    print("Per-component detail (filtered only):")
    for c in sorted(all_comps, key=lambda x: (x.corpus, x.video, x.component_id)):
        if not c.span_passes_filter:
            continue
        pieces_str = ", ".join(f"[{s},{e}]" for s, e in c.algo_pieces)
        gt_str = f"[{c.gt_span[0]},{c.gt_span[1]}]" if c.gt_span else "None"
        match_str = "no_match"
        if c.has_matcher_pair and c.matched_piece is not None:
            match_str = (
                f"matched=[{c.matched_piece[0]},{c.matched_piece[1]}] "
                f"sd={c.matched_start_delta:+d} span_d={c.matched_span_delta:+d}"
            )
        print(
            f"  {c.corpus:11s} {c.video:30s} cid={c.component_id:>4} "
            f"algo={pieces_str} gt={gt_str} max_gap={c.largest_gap:>3} "
            f"{match_str} "
            f"cls={c.classification}"
        )
    print()
    print("Per-component detail (all -- unfiltered for full audit):")
    n_unfiltered_extra = 0
    for c in sorted(all_comps, key=lambda x: (x.corpus, x.video, x.component_id)):
        if c.span_passes_filter:
            continue
        n_unfiltered_extra += 1
        if n_unfiltered_extra > 12:
            print("  ... (truncated)")
            break
        pieces_str = ", ".join(f"[{s},{e}]" for s, e in c.algo_pieces)
        gt_str = f"[{c.gt_span[0]},{c.gt_span[1]}]" if c.gt_span else "None"
        print(
            f"  [unfilt] {c.corpus:9s} {c.video:30s} cid={c.component_id:>4} "
            f"algo={pieces_str} gt={gt_str} cls={c.classification}"
        )

    # Write JSON
    out = {
        "summary_all": dict(cls_count_all),
        "summary_filtered": dict(cls_count_filtered),
        "summary_filtered_by_corpus": {k: dict(v) for k, v in by_corpus_filtered.items()},
        "components": [asdict(c) for c in all_comps],
    }
    out_path = metrics_dir / "characterization.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

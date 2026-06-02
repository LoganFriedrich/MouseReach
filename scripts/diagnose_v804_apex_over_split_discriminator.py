"""Phase A discriminator probe for apex-split over-split repair.

5 of 8 filtered FRAG components on cal+hol are "apex over-splits" --
v8.0.3 apex-split fired on what should have been one reach. They merge
cleanly to GT (4 of 5 to within span_delta=0; the 5th to span_delta=-3).

Goal of this probe: design a structural discriminator that identifies the
5 over-split positives WITHOUT false-firing on real adjacent reach pairs
across the corpus.

Candidate criteria swept:
  - gap_le_2: gap <= 2 frames between consecutive pieces
  - gap_le_1: gap <= 1 frame (tighter)
  - gap_le_2_AND_min_span_le_8: structural pair signature
  - gap_le_2_AND_min_span_le_10: looser min_span
  - gap_le_2_AND_ratio_le_0.6: relative-size variant
  - gap_le_2_AND_combined_span_le_60: combined-span guard

Read-only diagnostic. No algo changes. No accept/reject.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_Y_SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
if _Y_SRC not in sys.path:
    sys.path.insert(0, _Y_SRC)
for _mod in [m for m in list(sys.modules) if m.startswith("mousereach")]:
    del sys.modules[_mod]

HOL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\holdout_2026_05_11"
)
CAL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\calibration_loocv"
)
SNAPSHOT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\Improvement_Snapshots\reach_detection"
    r"\v8.0.4_dev_apex_over_split_discriminator_phase_a"
)
MIN_REPORTED_SPAN = 4


# Known 5 apex over-split positives (from frag characterization probe)
KNOWN_POSITIVES = {
    ("calibration", "20250820_CNT0103_P3", 37),
    ("calibration", "20250821_CNT0110_P4", 53),
    ("holdout", "20250625_CNT0102_P4", 50),
    ("holdout", "20250718_CNT0206_P1", 155),
    ("holdout", "20250718_CNT0214_P1", 245),
}


@dataclass
class AlgoPair:
    corpus: str
    video: str
    piece_a: Tuple[int, int]
    piece_b: Tuple[int, int]
    span_a: int
    span_b: int
    min_span: int
    max_span: int
    combined_span: int
    gap: int
    ratio_min_over_max: float
    # Topology info
    same_component: bool  # both pieces in same component_id
    component_a: Optional[int]
    component_b: Optional[int]
    kind_a: str  # TP/FP
    kind_b: str
    cat_a: str
    cat_b: str
    topo_a: Optional[str]
    topo_b: Optional[str]
    # GT info (if available)
    gt_a: Optional[Tuple[int, int]]  # GT linked to piece_a if any
    gt_b: Optional[Tuple[int, int]]
    # Both pieces pass MIN_REPORTED_SPAN filter
    both_pass_filter: bool


def _algo_event_records(manifest_path: Path):
    """Yield (algo_span, kind, cat, topo, component_id, gt_span) for each algo
    event in a manifest."""
    with open(manifest_path) as f:
        d = json.load(f)
    video = d["video_id"]
    out = []
    for e in d.get("events", []):
        algo = e.get("detector")
        if algo is None:
            continue
        gt = e.get("gt")
        gt_span = (gt["start"], gt["end"]) if gt else None
        out.append({
            "video": video,
            "algo": (algo["start"], algo["end"]),
            "kind": e.get("kind"),
            "category": e.get("category", ""),
            "topology": e.get("topology"),
            "topology_sub": e.get("topology_sub"),
            "component_id": e.get("component_id"),
            "gt_span": gt_span,
            "outside_gt_seg": bool(e.get("outside_gt_segmentation", False)),
        })
    return video, out


def _build_pairs(corpus: str, manifest_dir: Path) -> List[AlgoPair]:
    pairs = []
    for f in sorted(manifest_dir.glob("*.json")):
        video, recs = _algo_event_records(f)
        recs_sorted = sorted(recs, key=lambda r: r["algo"][0])
        for i in range(len(recs_sorted) - 1):
            a = recs_sorted[i]
            b = recs_sorted[i + 1]
            sa, ea = a["algo"]
            sb, eb = b["algo"]
            if sb < sa:
                continue
            gap = sb - ea - 1
            span_a = ea - sa + 1
            span_b = eb - sb + 1
            min_span = min(span_a, span_b)
            max_span = max(span_a, span_b)
            ratio = min_span / max_span if max_span > 0 else 1.0
            combined = eb - sa + 1
            both_pass = (
                span_a >= MIN_REPORTED_SPAN
                and span_b >= MIN_REPORTED_SPAN
                and not a["outside_gt_seg"]
                and not b["outside_gt_seg"]
            )
            pairs.append(AlgoPair(
                corpus=corpus,
                video=video,
                piece_a=(sa, ea),
                piece_b=(sb, eb),
                span_a=span_a,
                span_b=span_b,
                min_span=min_span,
                max_span=max_span,
                combined_span=combined,
                gap=gap,
                ratio_min_over_max=ratio,
                same_component=a["component_id"] == b["component_id"],
                component_a=a["component_id"],
                component_b=b["component_id"],
                kind_a=a["kind"],
                kind_b=b["kind"],
                cat_a=a["category"] or "",
                cat_b=b["category"] or "",
                topo_a=a["topology"],
                topo_b=b["topology"],
                gt_a=a["gt_span"],
                gt_b=b["gt_span"],
                both_pass_filter=both_pass,
            ))
    return pairs


# Candidate criteria
def crit_gap_le_2(p: AlgoPair) -> bool:
    return p.gap <= 2


def crit_gap_le_1(p: AlgoPair) -> bool:
    return p.gap <= 1


def crit_gap_le_2_min8(p: AlgoPair) -> bool:
    return p.gap <= 2 and p.min_span <= 8


def crit_gap_le_2_min10(p: AlgoPair) -> bool:
    return p.gap <= 2 and p.min_span <= 10


def crit_gap_le_2_ratio06(p: AlgoPair) -> bool:
    return p.gap <= 2 and p.ratio_min_over_max <= 0.6


def crit_gap_le_2_combined60(p: AlgoPair) -> bool:
    return p.gap <= 2 and p.combined_span <= 60


def crit_gap_le_2_min10_combined60(p: AlgoPair) -> bool:
    return p.gap <= 2 and p.min_span <= 10 and p.combined_span <= 60


CRITERIA = [
    ("gap_le_2", crit_gap_le_2),
    ("gap_le_1", crit_gap_le_1),
    ("gap_le_2_AND_min_span_le_8", crit_gap_le_2_min8),
    ("gap_le_2_AND_min_span_le_10", crit_gap_le_2_min10),
    ("gap_le_2_AND_ratio_le_0.6", crit_gap_le_2_ratio06),
    ("gap_le_2_AND_combined_span_le_60", crit_gap_le_2_combined60),
    ("gap_le_2_AND_min_span_le_10_AND_combined_le_60", crit_gap_le_2_min10_combined60),
]


def _pair_classification(p: AlgoPair) -> str:
    """Classify a pair into target / damage-risk / phantom-merge.

    - target: same component_id with FRAG topology (the known over-split target population)
    - real_adjacent: different component, both TP (would damage if merged)
    - tp_plus_fp: different component, one TP one FP (merging into TP keeps the TP,
      may help filter the FP)
    - phantom_pair: different component, both FP
    - other: anything else (TOL pairs, FN involvement, etc.)
    """
    if p.same_component and (p.topo_a == "FRAGMENTED" or p.topo_b == "FRAGMENTED"):
        return "target_apex_over_split"
    if (not p.same_component) and p.kind_a == "TP" and p.kind_b == "TP":
        return "real_adjacent_both_tp"
    if (not p.same_component) and {p.kind_a, p.kind_b} == {"TP", "FP"}:
        return "tp_plus_fp"
    if (not p.same_component) and p.kind_a == "FP" and p.kind_b == "FP":
        return "phantom_pair"
    return "other"


def main():
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = SNAPSHOT_DIR / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    pairs: List[AlgoPair] = []
    pairs.extend(_build_pairs("calibration", CAL_MANIFEST_DIR))
    pairs.extend(_build_pairs("holdout", HOL_MANIFEST_DIR))

    print(f"Total consecutive algo pairs across cal+hol: {len(pairs)}")
    print()

    # Gap distribution
    from collections import Counter
    gap_hist: Counter = Counter()
    for p in pairs:
        if p.gap < 0:
            continue
        bucket = p.gap if p.gap <= 20 else 999
        gap_hist[bucket] += 1
    print("Gap distribution (gap -> count):")
    for g in sorted(gap_hist.keys()):
        if g == 999:
            print(f"  >20: {gap_hist[g]}")
        else:
            print(f"  {g:>3}: {gap_hist[g]}")
    print()

    # Per-criterion evaluation
    print("Per-criterion evaluation (filtered to both_pass_filter only):")
    print()
    for crit_name, crit_fn in CRITERIA:
        firings = [p for p in pairs if p.both_pass_filter and crit_fn(p)]
        cls_count: Counter = Counter()
        for p in firings:
            cls_count[_pair_classification(p)] += 1
        n_known_pos_caught = sum(
            1
            for p in firings
            if p.same_component
            and (p.corpus, p.video, p.component_a) in KNOWN_POSITIVES
        )
        print(f"== {crit_name} ==")
        print(f"   total firings: {len(firings)}")
        print(f"   known positives caught: {n_known_pos_caught} / 5")
        for cls, n in cls_count.most_common():
            print(f"     {cls:30s}: {n}")
        # Show real_adjacent_both_tp details (the dangerous false-fires)
        damage_cases = [
            p for p in firings if _pair_classification(p) == "real_adjacent_both_tp"
        ]
        if damage_cases:
            print(f"   DAMAGE-RISK cases ({len(damage_cases)}):")
            for p in damage_cases[:20]:
                print(
                    f"     {p.corpus:11s} {p.video:30s} "
                    f"a={p.piece_a} b={p.piece_b} gap={p.gap} "
                    f"spans={p.span_a}+{p.span_b}"
                )
            if len(damage_cases) > 20:
                print(f"     ... ({len(damage_cases) - 20} more)")
        print()

    # Also: list all "target_apex_over_split" pairs the most-restrictive
    # criterion catches, sanity-check we got all 5
    print("Sanity: list known-positive components and what gap each has:")
    for p in pairs:
        if p.same_component and (p.corpus, p.video, p.component_a) in KNOWN_POSITIVES:
            print(
                f"  {p.corpus:11s} {p.video:30s} cid={p.component_a:>4} "
                f"a={p.piece_a} b={p.piece_b} gap={p.gap} "
                f"spans={p.span_a}+{p.span_b} min_span={p.min_span} "
                f"ratio={p.ratio_min_over_max:.2f} combined={p.combined_span}"
            )
    print()

    # Write JSON
    out = {
        "n_pairs_total": len(pairs),
        "gap_histogram": {str(k): v for k, v in gap_hist.items()},
        "per_criterion": {
            crit_name: {
                "firings": [asdict(p) for p in pairs if p.both_pass_filter and crit_fn(p)],
            }
            for crit_name, crit_fn in CRITERIA
        },
    }
    out_path = metrics_dir / "discriminator_probe.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

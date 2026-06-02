"""Re-tally apex-split outcomes using the correct identification.

Apex-split produces algo pairs with gap == 0 EXACTLY. Per-video apex_fires
count matches per-video count of gap==0 pairs across all 17 videos with fires.

For each gap=0 pair (= one apex-split fire):
  - same component_id, both TP: would be impossible (same GT, both TP)
  - different component_id, both TP: correct MERGED catch (intended target)
  - same component_id, FRAGMENTED topology: over-split (algo split a single GT)
  - one TP + one FP (any cat): partial (matched piece is TP, other is leftover)
  - both FP: phantom split (both pieces are unmatched FPs)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_Y_SRC = r"Y:\2_Connectome\Behavior\MouseReach\src"
if _Y_SRC not in sys.path:
    sys.path.insert(0, _Y_SRC)

HOL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\holdout_2026_05_11"
)
CAL_MANIFEST_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement"
    r"\fpfn_review_manifests\v8.0.3\calibration_loocv"
)


def main():
    by_corpus_outcomes: dict = {"calibration": Counter(), "holdout": Counter()}
    per_pair_records = []
    for label, dirpath in (("calibration", CAL_MANIFEST_DIR), ("holdout", HOL_MANIFEST_DIR)):
        for f in sorted(dirpath.glob("*.json")):
            with open(f) as fh:
                d = json.load(fh)
            video = d["video_id"]
            apex_fires = d.get("v803_apex_splits_added", 0)
            if apex_fires == 0:
                continue

            # Build event lookup by algo (start, end) -> event
            event_by_algo = {}
            for e in d.get("events", []):
                algo = e.get("detector")
                if algo is None:
                    continue
                key = (algo["start"], algo["end"])
                event_by_algo[key] = e

            # Collect algo events sorted by start
            algos_sorted = sorted(event_by_algo.keys())

            # Find gap == 0 pairs
            gap0_pairs = []
            for i in range(len(algos_sorted) - 1):
                a = algos_sorted[i]
                b = algos_sorted[i + 1]
                if b[0] - a[1] - 1 == 0:
                    gap0_pairs.append((a, b))

            if len(gap0_pairs) != apex_fires:
                print(
                    f"WARN {label} {video}: apex_fires={apex_fires} but found "
                    f"{len(gap0_pairs)} gap=0 pairs"
                )

            # Classify each gap=0 pair
            for a, b in gap0_pairs:
                ea = event_by_algo[a]
                eb = event_by_algo[b]
                kind_a = ea.get("kind")
                kind_b = eb.get("kind")
                cat_a = ea.get("category") or ""
                cat_b = eb.get("category") or ""
                cid_a = ea.get("component_id")
                cid_b = eb.get("component_id")
                topo_a = ea.get("topology")
                topo_b = eb.get("topology")
                same_cid = cid_a == cid_b
                span_a = a[1] - a[0] + 1
                span_b = b[1] - b[0] + 1

                # Decision tree per the docstring
                if kind_a == "TP" and kind_b == "TP" and not same_cid:
                    outcome = "correct_merged_catch"
                elif same_cid and (topo_a == "FRAGMENTED" or topo_b == "FRAGMENTED"):
                    outcome = "over_split_fragmented"
                elif {kind_a, kind_b} == {"TP", "FP"}:
                    outcome = "partial_one_tp_one_fp"
                elif kind_a == "FP" and kind_b == "FP" and not same_cid:
                    outcome = "phantom_split_both_fp_separate"
                elif kind_a == "FP" and kind_b == "FP" and same_cid:
                    outcome = "phantom_split_both_fp_same_cid"
                else:
                    outcome = f"other_kind_{kind_a}_{kind_b}_samecid_{same_cid}"

                by_corpus_outcomes[label][outcome] += 1
                per_pair_records.append({
                    "corpus": label,
                    "video": video,
                    "a": a,
                    "b": b,
                    "spans": (span_a, span_b),
                    "min_span": min(span_a, span_b),
                    "kinds": (kind_a, kind_b),
                    "cats": (cat_a, cat_b),
                    "cids": (cid_a, cid_b),
                    "topo": (topo_a, topo_b),
                    "outcome": outcome,
                })

    print("=" * 76)
    print("Apex-split actual outcomes (gap=0 pairs = apex-split fires)")
    print("=" * 76)
    print()
    for label in ("calibration", "holdout"):
        print(f"{label}:")
        for outcome, n in by_corpus_outcomes[label].most_common():
            print(f"  {outcome:35s}: {n}")
        print(f"  TOTAL: {sum(by_corpus_outcomes[label].values())}")
        print()

    print("Per-pair detail by outcome:")
    for outcome in sorted({r["outcome"] for r in per_pair_records}):
        recs = [r for r in per_pair_records if r["outcome"] == outcome]
        print()
        print(f"=== {outcome} ({len(recs)}) ===")
        for r in sorted(recs, key=lambda x: (x["corpus"], x["video"])):
            print(
                f"  {r['corpus']:11s} {r['video']:30s} "
                f"a={r['a']} b={r['b']} spans={r['spans']} "
                f"min={r['min_span']} kinds={r['kinds']} cats={r['cats']}"
            )


if __name__ == "__main__":
    main()

"""
Phase A of the v8/v5/v1 restart: corpus inventory + CV fold definition.

For each of the 47 GT videos in the v4.0.0_dev_walkthrough quarantine:
  - Pull exhaustive flag from GT JSON (gold standard vs supplementary).
  - Extract cohort from video name.
  - Count segments, reaches, and per-class outcome distribution.
  - Verify every segment has a non-null outcome (sanity check on the
    "outcome labels are per-segment-complete in both kinds" assumption).

Then build deterministic CV fold assignments:
  - Hold out a final test split (10 videos) -- never touched until
    Phase E.
  - Remaining 37 videos: leave-one-video-out (LOOCV) for training/val
    during Phases B-D.
  - Both splits cohort-stratified and exhaustive-flag-stratified so
    every fold contains the full population.

Outputs go under
  MouseReach_Pipeline/Improvement_Snapshots/_corpus/2026-04-30_restart_inventory/

Files:
  inventory.json   -- per-video metadata
  cv_folds.json    -- locked-in fold assignments
  summary.txt      -- human-readable summary printed to console too
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

# Make mousereach importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


QUARANTINE = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Improvement\validation_runs\DLC_2026_03_27\iterations"
    r"\2026-04-28_outcome_v4.0.0_dev_walkthrough"
)
GT_DIR = QUARANTINE / "gt"
ALGO_DIR = QUARANTINE / "algo_outputs"

OUTPUT_DIR = Path(
    r"Y:\2_Connectome\Behavior\MouseReach_Pipeline\Improvement_Snapshots"
    r"\_corpus\2026-04-30_restart_inventory"
)


def cohort_from_video(video_id: str) -> str:
    """20250624_CNT0107_P3 -> CNT_01 (cohort 01, subject 07)."""
    parts = video_id.split("_")
    if len(parts) >= 2 and parts[1].startswith("CNT"):
        cnt = parts[1]  # e.g. CNT0107
        if len(cnt) == 7:
            return f"CNT_{cnt[3:5]}"
    return "UNKNOWN"


def load_inventory() -> list:
    """Walk every GT JSON; collect per-video metadata."""
    rows = []
    gt_files = sorted(GT_DIR.glob("*_unified_ground_truth.json"))
    for gt_path in gt_files:
        video_id = gt_path.stem.replace("_unified_ground_truth", "")
        gt = json.loads(gt_path.read_text(encoding="utf-8"))

        outcomes_block = gt.get("outcomes", {})
        exhaustive = bool(outcomes_block.get("exhaustive", False))
        gt_segments = outcomes_block.get("segments", []) or []
        gt_reaches_block = gt.get("reaches", {})
        gt_reaches = gt_reaches_block.get("reaches", []) or []

        # Outcome distribution
        outcome_dist = defaultdict(int)
        n_outcome_null = 0
        for seg in gt_segments:
            o = seg.get("outcome")
            if o is None:
                n_outcome_null += 1
            else:
                outcome_dist[o] += 1

        # Reach span stats
        reach_spans = []
        for r in gt_reaches:
            s, e = r.get("start_frame"), r.get("end_frame")
            if s is not None and e is not None:
                reach_spans.append(e - s + 1)

        rows.append({
            "video_id": video_id,
            "cohort": cohort_from_video(video_id),
            "exhaustive": exhaustive,
            "n_segments": len(gt_segments),
            "n_reaches": len(gt_reaches),
            "n_outcome_null": n_outcome_null,
            "outcome_distribution": dict(outcome_dist),
            "reach_span_min": min(reach_spans) if reach_spans else None,
            "reach_span_max": max(reach_spans) if reach_spans else None,
            "reach_span_median": (
                sorted(reach_spans)[len(reach_spans) // 2]
                if reach_spans else None
            ),
        })
    return rows


def stratified_test_holdout(rows: list, n_test: int = 10, seed: int = 42) -> tuple:
    """Pick n_test videos as the locked test set, stratified by
    (cohort, exhaustive). Deterministic via SHA-256 sort of video_id +
    seed.

    Returns (test_video_ids, train_video_ids).
    """
    # Bucket by (cohort, exhaustive). Within a bucket, deterministic
    # ordering via hash.
    buckets: dict = defaultdict(list)
    for r in rows:
        key = (r["cohort"], r["exhaustive"])
        h = hashlib.sha256(f"{r['video_id']}_{seed}".encode()).hexdigest()
        buckets[key].append((h, r["video_id"]))

    # Sort within each bucket
    for k in buckets:
        buckets[k].sort()

    # Round-robin assign test slots across buckets in proportion to size.
    test_ids = set()
    bucket_total = sum(len(v) for v in buckets.values())
    quotas = {k: max(1, round(n_test * len(v) / bucket_total)) for k, v in buckets.items()}
    # Trim to n_test exactly (largest buckets give up first if over,
    # smallest grow first if under)
    while sum(quotas.values()) > n_test:
        k = max(quotas, key=lambda k: quotas[k])
        quotas[k] -= 1
    while sum(quotas.values()) < n_test:
        k = min(quotas, key=lambda k: quotas[k])
        quotas[k] += 1

    for k, vids in buckets.items():
        for _h, vid in vids[:quotas[k]]:
            test_ids.add(vid)

    train_ids = [r["video_id"] for r in rows if r["video_id"] not in test_ids]
    return sorted(test_ids), sorted(train_ids)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_inventory()

    # Sanity check on outcome completeness assumption
    n_with_nulls = sum(1 for r in rows if r["n_outcome_null"] > 0)

    # CV split
    test_ids, train_ids = stratified_test_holdout(rows, n_test=10, seed=42)

    # LOOCV folds within train_ids
    loocv_folds = [
        {"fold_id": i, "val_video": vid,
         "train_videos": [v for v in train_ids if v != vid]}
        for i, vid in enumerate(train_ids)
    ]

    inventory = {
        "corpus_root": str(QUARANTINE),
        "n_videos": len(rows),
        "n_exhaustive": sum(1 for r in rows if r["exhaustive"]),
        "n_supplementary": sum(1 for r in rows if not r["exhaustive"]),
        "videos": rows,
    }

    cv_folds = {
        "test_holdout": {
            "n_videos": len(test_ids),
            "video_ids": test_ids,
        },
        "train_pool": {
            "n_videos": len(train_ids),
            "video_ids": train_ids,
        },
        "loocv_folds": loocv_folds,
        "policy": (
            "Stratified by (cohort, exhaustive). Test set frozen and "
            "untouched until Phase E. LOOCV is for training/validation "
            "across Phases B-D. Reach detection metrics are reported "
            "headline on EXHAUSTIVE subset only; non-exhaustive is "
            "supplementary."
        ),
    }

    (OUTPUT_DIR / "inventory.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "cv_folds.json").write_text(
        json.dumps(cv_folds, indent=2), encoding="utf-8")

    # Human summary
    summary_lines = []

    def p(s):
        print(s)
        summary_lines.append(s)

    p("=" * 70)
    p("PHASE A CORPUS INVENTORY")
    p("=" * 70)
    p(f"Corpus root: {QUARANTINE}")
    p(f"Total videos: {len(rows)}")
    p(f"  Exhaustive (gold standard): {inventory['n_exhaustive']}")
    p(f"  Supplementary:               {inventory['n_supplementary']}")
    p("")

    # Cohort breakdown
    by_cohort = defaultdict(lambda: {"exhaustive": 0, "supplementary": 0})
    for r in rows:
        by_cohort[r["cohort"]]["exhaustive" if r["exhaustive"] else "supplementary"] += 1
    p("Per-cohort breakdown:")
    for cohort in sorted(by_cohort):
        b = by_cohort[cohort]
        p(f"  {cohort}: exhaustive={b['exhaustive']:>2}  supplementary={b['supplementary']:>2}")
    p("")

    # Outcome distribution
    p("Outcome distribution across all segments (both kinds):")
    total_outcome_dist = defaultdict(int)
    for r in rows:
        for o, c in r["outcome_distribution"].items():
            total_outcome_dist[o] += c
    for o, c in sorted(total_outcome_dist.items(), key=lambda x: -x[1]):
        p(f"  {o:>22s}: {c}")
    p("")

    # Outcome distribution -- exhaustive only (training-relevant)
    p("Outcome distribution -- EXHAUSTIVE subset only:")
    exh_outcome_dist = defaultdict(int)
    for r in rows:
        if r["exhaustive"]:
            for o, c in r["outcome_distribution"].items():
                exh_outcome_dist[o] += c
    for o, c in sorted(exh_outcome_dist.items(), key=lambda x: -x[1]):
        p(f"  {o:>22s}: {c}")
    p("")

    p(f"Sanity check: videos with any null-outcome segments: {n_with_nulls}/{len(rows)}")
    if n_with_nulls > 0:
        p("  WARNING: some segments lack outcome labels -- the 'outcome")
        p("  labels are per-segment-complete' assumption is violated.")
        p("  Affected videos:")
        for r in rows:
            if r["n_outcome_null"] > 0:
                p(f"    {r['video_id']}: {r['n_outcome_null']} null outcomes")
    p("")

    # Reach span summary
    all_spans = []
    exh_spans = []
    for r in rows:
        if r["reach_span_median"] is not None:
            all_spans.append(r["reach_span_median"])
            if r["exhaustive"]:
                exh_spans.append(r["reach_span_median"])
    if exh_spans:
        p(f"Reach span (median per video, exhaustive subset): "
          f"min={min(exh_spans)}, max={max(exh_spans)}, "
          f"mid={sorted(exh_spans)[len(exh_spans)//2]}")
    p("")

    p("CV fold assignments:")
    p(f"  Test holdout (frozen until Phase E): {len(test_ids)} videos")
    for vid in test_ids:
        r = next(r for r in rows if r["video_id"] == vid)
        p(f"    {vid:35s} cohort={r['cohort']} exhaustive={r['exhaustive']}")
    p(f"  Train pool (LOOCV across {len(train_ids)} folds for Phases B-D): "
      f"{len(train_ids)} videos")
    p("")

    p(f"Artifacts written to:")
    p(f"  {OUTPUT_DIR / 'inventory.json'}")
    p(f"  {OUTPUT_DIR / 'cv_folds.json'}")

    (OUTPUT_DIR / "summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

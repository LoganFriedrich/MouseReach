"""
mousereach.aspa.compare - Compare old ASPA vs new mousereach results.

Joins aspa_reaches and mousereach_reaches by video_id, aligns individual
reaches by frame overlap, and reports:
    - Reach count per video (old vs new)
    - Boundary agreement (frame overlap ratio)
    - Outcome agreement rate

CLI:
    mousereach-aspa-compare [--cohort H] [--output comparison.csv]
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from mousereach.aspa.database import get_connection, get_db_path


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------

def frame_overlap(s1: int, e1: int, s2: int, e2: int) -> int:
    """Number of frames shared by [s1, e1] and [s2, e2] (inclusive)."""
    start = max(s1, s2)
    end   = min(e1, e2)
    return max(0, end - start + 1)


def overlap_ratio(s1: int, e1: int, s2: int, e2: int) -> float:
    """Jaccard-style overlap: intersection / union."""
    intersection = frame_overlap(s1, e1, s2, e2)
    if intersection == 0:
        return 0.0
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - intersection
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_reaches(
    aspa_rows: List[dict],
    mr_rows: List[dict],
    min_overlap_ratio: float = 0.3,
) -> List[dict]:
    """Match each ASPA reach to the best-overlapping mousereach reach.

    Each ASPA reach is matched to at most one mousereach reach (greedy by
    overlap ratio, highest first).

    Args:
        aspa_rows:         Rows from aspa_reaches for one video.
        mr_rows:           Rows from mousereach_reaches for one video.
        min_overlap_ratio: Minimum Jaccard overlap to consider a match.

    Returns:
        List of alignment dicts, one per ASPA reach.
    """
    # Filter rows with valid frame boundaries
    def valid(r):
        return (
            r.get("start_frame") is not None
            and r.get("end_frame") is not None
        )

    valid_aspa = [r for r in aspa_rows if valid(r)]
    valid_mr   = [r for r in mr_rows   if valid(r)]

    used_mr = set()
    results = []

    # Build candidate matches for each aspa reach
    for ar in valid_aspa:
        best_ratio = 0.0
        best_mr    = None

        for i, mr in enumerate(valid_mr):
            if i in used_mr:
                continue
            ratio = overlap_ratio(
                ar["start_frame"], ar["end_frame"],
                mr["start_frame"], mr["end_frame"],
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_mr    = (i, mr)

        if best_mr and best_ratio >= min_overlap_ratio:
            idx, mr = best_mr
            used_mr.add(idx)
            results.append({
                "aspa_reach_num":  ar.get("reach_num"),
                "aspa_start":      ar["start_frame"],
                "aspa_end":        ar["end_frame"],
                "aspa_outcome":    ar.get("outcome"),
                "mr_reach_num":    mr.get("reach_num"),
                "mr_start":        mr["start_frame"],
                "mr_end":          mr["end_frame"],
                "mr_outcome":      mr.get("outcome"),
                "overlap_ratio":   round(best_ratio, 4),
                "outcomes_match":  ar.get("outcome") == mr.get("outcome"),
                "matched":         True,
            })
        else:
            results.append({
                "aspa_reach_num":  ar.get("reach_num"),
                "aspa_start":      ar["start_frame"],
                "aspa_end":        ar["end_frame"],
                "aspa_outcome":    ar.get("outcome"),
                "mr_reach_num":    None,
                "mr_start":        None,
                "mr_end":          None,
                "mr_outcome":      None,
                "overlap_ratio":   0.0,
                "outcomes_match":  False,
                "matched":         False,
            })

    return results


# ---------------------------------------------------------------------------
# Per-video metrics
# ---------------------------------------------------------------------------

def compute_video_metrics(
    video_id: str,
    aspa_rows: List[dict],
    mr_rows: List[dict],
    min_overlap_ratio: float = 0.3,
) -> dict:
    """Compute comparison metrics for a single video.

    Returns a dict suitable for CSV output.
    """
    alignments = align_reaches(aspa_rows, mr_rows, min_overlap_ratio)

    n_aspa = len(aspa_rows)
    n_mr   = len(mr_rows)
    n_matched = sum(1 for a in alignments if a["matched"])

    # Boundary agreement: mean overlap ratio for matched pairs
    matched = [a for a in alignments if a["matched"]]
    mean_overlap = (
        sum(a["overlap_ratio"] for a in matched) / len(matched)
        if matched else None
    )

    # Outcome agreement rate (among matched pairs)
    outcome_agree = (
        sum(1 for a in matched if a["outcomes_match"]) / len(matched)
        if matched else None
    )

    return {
        "video_id":              video_id,
        "aspa_reach_count":      n_aspa,
        "mr_reach_count":        n_mr,
        "matched_count":         n_matched,
        "unmatched_aspa_count":  n_aspa - n_matched,
        "mr_only_count":         n_mr - n_matched,
        "mean_overlap_ratio":    round(mean_overlap, 4) if mean_overlap is not None else "",
        "outcome_agreement_rate": round(outcome_agree, 4) if outcome_agree is not None else "",
    }


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def _fetch_aspa(conn, cohort: Optional[str]) -> Dict[str, List[dict]]:
    """Fetch aspa_reaches grouped by video_id."""
    if cohort:
        rows = conn.execute(
            "SELECT * FROM aspa_reaches WHERE cohort = ? ORDER BY video_id, reach_num",
            (cohort,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM aspa_reaches ORDER BY video_id, reach_num"
        ).fetchall()
    result: Dict[str, List[dict]] = {}
    for r in rows:
        d = dict(r)
        result.setdefault(d["video_id"], []).append(d)
    return result


def _fetch_mr(conn, cohort: Optional[str]) -> Dict[str, List[dict]]:
    """Fetch mousereach_reaches grouped by video_id."""
    if cohort:
        rows = conn.execute(
            "SELECT * FROM mousereach_reaches WHERE cohort = ? ORDER BY video_id, segment_num, reach_num",
            (cohort,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM mousereach_reaches ORDER BY video_id, segment_num, reach_num"
        ).fetchall()
    result: Dict[str, List[dict]] = {}
    for r in rows:
        d = dict(r)
        result.setdefault(d["video_id"], []).append(d)
    return result


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare old ASPA vs new mousereach results from ASPA.db."
    )

    parser.add_argument("--cohort", metavar="COHORT", default=None,
                        help="Restrict comparison to one cohort (default: all cohorts)")
    parser.add_argument("--output", metavar="FILE", default="comparison.csv",
                        help="Output CSV path (default: comparison.csv)")
    parser.add_argument("--db-path", metavar="PATH",
                        help="Override ASPA.db path")
    parser.add_argument("--min-overlap", type=float, default=0.3, metavar="RATIO",
                        help="Minimum frame overlap ratio to count as a match (default: 0.3)")

    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_db_path()

    if not db_path.exists():
        print(f"[FAIL] ASPA.db not found at: {db_path}")
        print("       Run mousereach-aspa-import and/or mousereach-aspa-sync first.")
        sys.exit(1)

    conn = get_connection(db_path)
    try:
        aspa_by_video = _fetch_aspa(conn, args.cohort)
        mr_by_video   = _fetch_mr(conn, args.cohort)
    finally:
        conn.close()

    all_video_ids = sorted(set(aspa_by_video) | set(mr_by_video))

    if not all_video_ids:
        print("[!] No data found in ASPA.db for the specified cohort.")
        print("    Run mousereach-aspa-import and mousereach-aspa-sync first.")
        sys.exit(0)

    print(f"Comparing {len(all_video_ids)} video(s)  "
          f"({len(aspa_by_video)} with ASPA results, "
          f"{len(mr_by_video)} with mousereach results)")

    rows_out = []
    for video_id in all_video_ids:
        aspa_rows = aspa_by_video.get(video_id, [])
        mr_rows   = mr_by_video.get(video_id, [])
        metrics   = compute_video_metrics(video_id, aspa_rows, mr_rows,
                                          min_overlap_ratio=args.min_overlap)
        rows_out.append(metrics)

    # Write CSV
    output_path = Path(args.output)
    fieldnames = [
        "video_id",
        "aspa_reach_count",
        "mr_reach_count",
        "matched_count",
        "unmatched_aspa_count",
        "mr_only_count",
        "mean_overlap_ratio",
        "outcome_agreement_rate",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"[OK] Comparison written to: {output_path}")

    # Summary stats
    n_both = sum(1 for r in rows_out if r["aspa_reach_count"] > 0 and r["mr_reach_count"] > 0)
    total_aspa = sum(r["aspa_reach_count"] for r in rows_out)
    total_mr   = sum(r["mr_reach_count"]   for r in rows_out)
    total_matched = sum(r["matched_count"] for r in rows_out)

    overlap_vals = [r["mean_overlap_ratio"] for r in rows_out
                    if r["mean_overlap_ratio"] != ""]
    outcome_vals = [r["outcome_agreement_rate"] for r in rows_out
                    if r["outcome_agreement_rate"] != ""]

    print()
    print("Summary")
    print("-------")
    print(f"  Videos with both results : {n_both} / {len(all_video_ids)}")
    print(f"  Total ASPA reaches       : {total_aspa}")
    print(f"  Total mousereach reaches : {total_mr}")
    print(f"  Matched pairs            : {total_matched}")
    if overlap_vals:
        mean_ov = sum(float(v) for v in overlap_vals) / len(overlap_vals)
        print(f"  Mean boundary overlap    : {mean_ov:.4f}")
    if outcome_vals:
        mean_oa = sum(float(v) for v in outcome_vals) / len(outcome_vals)
        print(f"  Mean outcome agreement   : {mean_oa:.4f}")


if __name__ == "__main__":
    main()

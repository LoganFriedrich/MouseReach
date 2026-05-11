"""GT auto-resolve for triaged segments.

When the outcome detector triages a segment AND a unified GT file
exists for the same video that has that segment's outcome scored, we
already know what human review would reveal. Pull the GT into the algo
output directly — same effect as a human running the triage clearing
tool, no manual work required.

Marker fields (consistent with the napari triage-clearing tool):

- ``flagged_for_review`` -> False
- ``triage_cleared`` -> True
- ``cleared_by`` -> ``"gt_auto_resolve"``
- ``cleared_at`` -> ISO timestamp
- ``original_triage_reason`` -> preserved from the algo's flag_reason
- ``outcome`` / ``interaction_frame`` / ``outcome_known_frame`` ->
  copied from GT
- ``causal_reach_id`` -> derived from IFR containment on the algo's
  detected reaches (same logic the GT side uses)
- ``original_outcome`` -> preserved from the algo's pre-resolution call
  (which is "triaged" / "uncertain" or whatever the cascade emitted)

The corresponding entries in the algo's ``*_reaches.json`` are also
updated so the per-reach kinematic pipeline sees the resolved causal
reach (marked ``human_corrected=True`` with reviewer ``gt_auto_resolve``
and a ``review_note`` indicating the GT source).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CLEARED_BY = "gt_auto_resolve"


@dataclass
class ResolveResult:
    """Summary of a per-video resolve pass."""

    video_id: str
    n_segments_total: int
    n_triaged_pre: int  # segments flagged before this pass
    n_resolved_from_gt: int  # of those, how many we resolved
    n_left_triaged: int  # triaged segments with no GT match
    n_already_cleared: int  # already triage_cleared=True before this pass
    resolved_segments: List[int]  # segment_nums we resolved
    skipped_segments: List[Tuple[int, str]]  # (segment_num, reason)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _gt_outcome_segments(gt_doc: dict) -> Dict[int, dict]:
    """Index unified GT outcomes by segment_num."""
    out = {}
    block = gt_doc.get("outcomes") or {}
    for s in block.get("segments") or []:
        sn = s.get("segment_num")
        if sn is not None:
            out[int(sn)] = s
    return out


def _flat_algo_reaches_with_seg(algo_reaches_doc: dict) -> Dict[int, List[dict]]:
    """Return {segment_num: [reach_dict, ...]} from a reach detector output.

    Handles v8+ flat (top-level ``reaches``, each with ``segment_num``) and
    v7.x nested (``segments[i].reaches``) layouts.
    """
    by_seg: Dict[int, List[dict]] = {}
    flat = algo_reaches_doc.get("reaches")
    if isinstance(flat, list):
        for r in flat:
            sn = r.get("segment_num")
            if sn is not None:
                by_seg.setdefault(int(sn), []).append(r)
    for seg in algo_reaches_doc.get("segments") or []:
        sn = seg.get("segment_num")
        if sn is None:
            continue
        sn = int(sn)
        for r in seg.get("reaches") or []:
            r2 = r if "segment_num" in r else {**r, "segment_num": sn}
            by_seg.setdefault(sn, []).append(r2)
    return by_seg


def _find_causal_by_ifr(reaches: List[dict], ifr: Optional[int]) -> Optional[int]:
    """Find the reach_id of the reach whose window contains the interaction
    frame. Returns None if ifr is None or no reach contains it."""
    if ifr is None:
        return None
    for r in reaches:
        sf, ef = r.get("start_frame"), r.get("end_frame")
        if sf is None or ef is None:
            continue
        if int(sf) <= int(ifr) <= int(ef):
            return r.get("reach_id")
    return None


def resolve_triaged_outcomes(
    outcome_doc: dict,
    gt_doc: Optional[dict],
    reaches_doc: Optional[dict],
    *,
    now: Optional[str] = None,
) -> Tuple[dict, ResolveResult]:
    """Pure function: lift triage on any flagged segment that has a GT entry.

    Args:
        outcome_doc: parsed ``*_pellet_outcomes.json`` dict.
        gt_doc: parsed ``*_unified_ground_truth.json`` dict (or None if
            no GT exists; nothing will be resolved).
        reaches_doc: parsed ``*_reaches.json`` dict (used to derive
            ``causal_reach_id`` via IFR containment).
        now: ISO timestamp to stamp ``cleared_at`` (defaults to current UTC).

    Returns:
        (modified outcome_doc, ResolveResult). The dict is modified
        in-place AND returned for convenience.
    """
    if now is None:
        now = _now_iso()

    video_id = (outcome_doc.get("video_id")
                or outcome_doc.get("video_name")
                or "unknown")

    gt_segs = _gt_outcome_segments(gt_doc) if gt_doc else {}
    reaches_by_seg = _flat_algo_reaches_with_seg(reaches_doc) if reaches_doc else {}

    segs = outcome_doc.get("segments") or []
    n_total = len(segs)
    n_triaged_pre = 0
    n_resolved = 0
    n_left = 0
    n_already_cleared = 0
    resolved_list: List[int] = []
    skipped_list: List[Tuple[int, str]] = []

    for seg in segs:
        sn = seg.get("segment_num")
        if sn is None:
            continue
        sn = int(sn)
        already = bool(seg.get("triage_cleared"))
        flagged = bool(seg.get("flagged_for_review"))
        if already:
            n_already_cleared += 1
            continue
        if not flagged:
            continue
        n_triaged_pre += 1

        gt = gt_segs.get(sn)
        if gt is None or not gt.get("outcome"):
            n_left += 1
            skipped_list.append((sn, "no_gt"))
            continue

        gt_outcome = gt.get("outcome")
        gt_ifr = gt.get("interaction_frame")
        gt_okf = gt.get("outcome_known_frame")

        # Derive causal_reach_id from algo's own reach detections via IFR containment
        seg_reaches = reaches_by_seg.get(sn, [])
        cid = _find_causal_by_ifr(seg_reaches, gt_ifr) if gt_ifr is not None else None

        # Preserve algo's pre-resolution state for audit
        orig_reason = seg.get("flag_reason")
        orig_outcome = seg.get("outcome")
        seg["flagged_for_review"] = False
        seg["triage_cleared"] = True
        seg["human_verified"] = True  # GT counts as human verification
        seg["cleared_by"] = CLEARED_BY
        seg["cleared_at"] = now
        if orig_reason is not None and "original_triage_reason" not in seg:
            seg["original_triage_reason"] = orig_reason
        if orig_outcome and orig_outcome != gt_outcome:
            seg["original_outcome"] = orig_outcome
        seg["outcome"] = gt_outcome
        if gt_ifr is not None:
            seg["interaction_frame"] = int(gt_ifr)
        if gt_okf is not None:
            seg["outcome_known_frame"] = int(gt_okf)
        if cid is not None:
            seg["causal_reach_id"] = cid

        n_resolved += 1
        resolved_list.append(sn)

    return outcome_doc, ResolveResult(
        video_id=video_id,
        n_segments_total=n_total,
        n_triaged_pre=n_triaged_pre,
        n_resolved_from_gt=n_resolved,
        n_left_triaged=n_left,
        n_already_cleared=n_already_cleared,
        resolved_segments=resolved_list,
        skipped_segments=skipped_list,
    )


def resolve_triaged_reaches(
    reaches_doc: dict,
    outcome_doc_after: dict,
    *,
    now: Optional[str] = None,
) -> dict:
    """Mirror outcome resolution into the reach detector output.

    For each segment that got ``triage_cleared`` in ``outcome_doc_after``,
    flip the segment's ``flagged_for_review`` to False in the reaches
    doc, add the same audit fields, mark non-causal reaches in the
    segment as ``exclude_from_analysis=True``, and mark the causal reach
    with ``human_corrected=True`` and a ``review_note``.

    Modifies the reaches doc in place and returns it.
    """
    if now is None:
        now = _now_iso()
    outcome_segs = {int(s.get("segment_num")): s
                    for s in (outcome_doc_after.get("segments") or [])
                    if s.get("segment_num") is not None
                    and bool(s.get("triage_cleared"))
                    and s.get("cleared_by") == CLEARED_BY}
    note = "Auto-resolved from unified GT (segment had GT outcome at pipeline run time)."

    for seg in reaches_doc.get("segments") or []:
        sn = seg.get("segment_num")
        if sn is None:
            continue
        sn = int(sn)
        out_seg = outcome_segs.get(sn)
        if out_seg is None:
            continue
        causal_id = out_seg.get("causal_reach_id")
        orig_reason = seg.get("flag_reason")
        seg["flagged_for_review"] = False
        seg["triage_cleared"] = True
        seg["cleared_by"] = CLEARED_BY
        seg["cleared_at"] = now
        if orig_reason is not None and "original_triage_reason" not in seg:
            seg["original_triage_reason"] = orig_reason
        for r in seg.get("reaches") or []:
            if causal_id is not None and r.get("reach_id") == causal_id:
                r["human_corrected"] = True
                r["review_note"] = note
                r["exclude_from_analysis"] = False
                r["exclude_reason"] = None
            else:
                r["exclude_from_analysis"] = True
                r["exclude_reason"] = "not causal per GT auto-resolve"
                r["human_corrected"] = True
    return reaches_doc


def resolve_dir(
    algo_dir: Path,
    *,
    gt_dir: Optional[Path] = None,
    verbose: bool = True,
) -> List[ResolveResult]:
    """Walk a directory, find every triaged segment that has a GT entry,
    and resolve it in place.

    Args:
        algo_dir: directory with ``*_pellet_outcomes.json`` and
            ``*_reaches.json`` per video.
        gt_dir: directory with ``*_unified_ground_truth.json``. If None,
            looks for GT files in ``algo_dir`` (flat-layout improvement
            corpus) and then in a sibling ``../gt/`` directory
            (quarantine layout).
        verbose: print one line per video.

    Returns:
        List of ResolveResult, one per video processed.
    """
    algo_dir = Path(algo_dir)
    if not algo_dir.is_dir():
        raise FileNotFoundError(f"algo_dir not found: {algo_dir}")
    gt_sources: List[Path] = []
    if gt_dir is not None:
        gt_sources.append(Path(gt_dir))
    else:
        gt_sources.append(algo_dir)
        gt_sources.append(algo_dir.parent / "gt")

    def _find_gt(video_id: str) -> Optional[Path]:
        for gd in gt_sources:
            cand = gd / f"{video_id}_unified_ground_truth.json"
            if cand.exists():
                return cand
        return None

    now = _now_iso()
    results: List[ResolveResult] = []
    for outcome_path in sorted(algo_dir.glob("*_pellet_outcomes.json")):
        stem = outcome_path.stem.replace("_pellet_outcomes", "")
        outcome_doc = _load_json(outcome_path) or {}
        gt_path = _find_gt(stem)
        gt_doc = _load_json(gt_path) if gt_path else None
        reaches_path = algo_dir / f"{stem}_reaches.json"
        reaches_doc = _load_json(reaches_path)

        outcome_doc, res = resolve_triaged_outcomes(
            outcome_doc, gt_doc, reaches_doc, now=now,
        )
        if reaches_doc is not None and res.n_resolved_from_gt > 0:
            resolve_triaged_reaches(reaches_doc, outcome_doc, now=now)
            reaches_path.write_text(json.dumps(reaches_doc, indent=2), encoding="utf-8")
        if res.n_resolved_from_gt > 0:
            outcome_path.write_text(json.dumps(outcome_doc, indent=2), encoding="utf-8")
        results.append(res)
        if verbose:
            print(f"  {stem:<40} pre-triaged={res.n_triaged_pre:>3}  "
                  f"resolved-from-gt={res.n_resolved_from_gt:>3}  "
                  f"left={res.n_left_triaged:>3}")
    return results


def main():
    """CLI entry: ``mousereach-resolve-triage-from-gt -i <dir> [--gt-dir <dir>]``."""
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Auto-resolve triaged segments from unified GT files. For every "
            "segment the outcome detector flagged for review, check if "
            "GT already has the answer; if yes, copy it into the algo output "
            "and lift the flag. Production-pipeline fast path for videos "
            "that have been ground-truthed."
        ),
    )
    parser.add_argument("-i", "--input", type=Path, required=True,
                        help="Directory with *_pellet_outcomes.json + "
                             "*_reaches.json per video.")
    parser.add_argument("--gt-dir", type=Path, default=None,
                        help="Directory with *_unified_ground_truth.json. "
                             "Defaults to --input dir, then sibling ../gt/.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-video output.")
    args = parser.parse_args()

    results = resolve_dir(args.input, gt_dir=args.gt_dir, verbose=not args.quiet)
    n_videos = len(results)
    total_pre = sum(r.n_triaged_pre for r in results)
    total_resolved = sum(r.n_resolved_from_gt for r in results)
    total_left = sum(r.n_left_triaged for r in results)
    pct = (100 * total_resolved / total_pre) if total_pre else 0.0
    print()
    print(f"Done. {n_videos} videos processed.")
    print(f"  Pre-triaged segments: {total_pre}")
    print(f"  Resolved from GT:     {total_resolved} ({pct:.1f}%)")
    print(f"  Left triaged:         {total_left} (need human review)")


if __name__ == "__main__":
    main()

"""Triage worklist: enumerate segments needing human review.

A segment is "triaged" when the reach detector or outcome detector
flagged it via ``flagged_for_review=True`` in ``*_reaches.json`` or
``*_pellet_outcomes.json``. The clearing-tool workflow walks through
these segments one at a time, lets a reviewer mark the causal reach
+ outcome, then writes the cleared state back to the algo JSONs and
the unified GT JSON.

This module builds and navigates that list. It does not do any of the
clearing; that lives in the widget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Sequence
import glob
import json
import os


REACH_SUFFIX = "_reaches.json"
OUTCOME_SUFFIX = "_pellet_outcomes.json"


@dataclass
class TriageEntry:
    """One triaged segment in the worklist."""

    video_name: str
    segment_num: int
    start_frame: int
    end_frame: int
    reach_flag_reason: Optional[str] = None
    outcome_flag_reason: Optional[str] = None
    already_cleared: bool = False

    @property
    def triage_reason(self) -> str:
        parts = []
        if self.outcome_flag_reason:
            parts.append(f"outcome: {self.outcome_flag_reason}")
        if self.reach_flag_reason:
            parts.append(f"reach: {self.reach_flag_reason}")
        return " | ".join(parts) if parts else "(no reason recorded)"


def _algo_dir_for_video(video_path: Path) -> Path:
    """Resolve algo output directory for a given video path.

    Two layouts are supported:
      1. Quarantine layout: ``<root>/videos/<name>.mp4`` and
         ``<root>/algo_outputs/<name>_<type>.json``.
      2. Production layout: ``*_segments.json`` etc. live next to the
         ``.mp4`` (or in the standard MouseReach_Pipeline path).
    """
    video_path = Path(video_path)
    parent = video_path.parent
    # Quarantine sibling layout
    if parent.name == "videos":
        cand = parent.parent / "algo_outputs"
        if cand.is_dir():
            return cand
    return parent


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _segment_already_cleared(unified_gt: Optional[dict], segment_num: int) -> bool:
    """A segment is 'already cleared' if the unified GT outcome entry has
    ``triage_cleared=True``.

    The unified GT schema is:
        outcomes:
          exhaustive: bool
          segments: [{segment_num: int, outcome: str, triage_cleared: bool, ...}]
    """
    if unified_gt is None:
        return False
    outcomes = unified_gt.get("outcomes")
    if not isinstance(outcomes, dict):
        return False
    segs = outcomes.get("segments") or []
    for o in segs:
        if not isinstance(o, dict):
            continue
        if o.get("segment_num") == segment_num and o.get("triage_cleared") is True:
            return True
    return False


def scan_corpus_for_triage(
    algo_dir: Path,
    *,
    video_filter: Optional[Sequence[str]] = None,
    include_cleared: bool = False,
) -> List[TriageEntry]:
    """Scan one ``algo_outputs/`` directory and return triaged segments.

    Args:
        algo_dir: directory containing ``*_segments.json``, ``*_reaches.json``,
            ``*_pellet_outcomes.json``. In a quarantine this is the literal
            ``algo_outputs/`` folder; in a production pipeline this is the
            per-video folder.
        video_filter: optional set of video stems (e.g. ``"20250624_CNT0107_P3"``)
            to restrict the scan to. None = include all videos.
        include_cleared: if True, return entries even if they've already been
            cleared (useful for "let me re-review a previous call" mode). If
            False (the normal worklist), already-cleared entries are excluded.

    Returns:
        List of TriageEntry, ordered (video_name, segment_num).
    """
    algo_dir = Path(algo_dir)
    if not algo_dir.is_dir():
        raise FileNotFoundError(f"algo_dir not found: {algo_dir}")
    entries: List[TriageEntry] = []
    outcome_files = sorted(glob.glob(str(algo_dir / f"*{OUTCOME_SUFFIX}")))
    for outcome_path_str in outcome_files:
        outcome_path = Path(outcome_path_str)
        video_name = outcome_path.name[: -len(OUTCOME_SUFFIX)]
        if video_filter is not None and video_name not in video_filter:
            continue
        outcome_data = _load_json(outcome_path) or {}
        reach_path = algo_dir / f"{video_name}{REACH_SUFFIX}"
        reach_data = _load_json(reach_path) or {}
        unified_gt = _load_json(_resolve_unified_gt(algo_dir, video_name))
        reach_segs = {s.get("segment_num"): s for s in (reach_data.get("segments") or [])}
        for out_seg in outcome_data.get("segments") or []:
            sn = out_seg.get("segment_num")
            if sn is None:
                continue
            out_flag = bool(out_seg.get("flagged_for_review"))
            rch_seg = reach_segs.get(sn, {})
            rch_flag = bool(rch_seg.get("flagged_for_review"))
            # `triage_cleared` segments (resolved by GT auto-resolve or the
            # napari clearing tool) drop `flagged_for_review` to False --
            # we still want them visible under --include-cleared.
            out_cleared = bool(out_seg.get("triage_cleared"))
            rch_cleared = bool(rch_seg.get("triage_cleared"))
            any_flagged = out_flag or rch_flag
            any_cleared = out_cleared or rch_cleared
            if include_cleared:
                if not (any_flagged or any_cleared):
                    continue
            else:
                if not any_flagged:
                    continue
            cleared = any_cleared or _segment_already_cleared(unified_gt, sn)
            if cleared and not include_cleared:
                continue
            entries.append(TriageEntry(
                video_name=video_name,
                segment_num=int(sn),
                start_frame=int(rch_seg.get("start_frame", 0)),
                end_frame=int(rch_seg.get("end_frame", 0)),
                reach_flag_reason=rch_seg.get("flag_reason") if rch_flag else None,
                outcome_flag_reason=out_seg.get("flag_reason") if out_flag else None,
                already_cleared=cleared,
            ))
    return entries


def _resolve_unified_gt(algo_dir: Path, video_name: str) -> Path:
    """Locate the unified GT JSON for a video name. Convention is
    ``<gt_dir>/<video_name>_unified_ground_truth.json`` where ``<gt_dir>`` is
    a sibling ``gt/`` directory in quarantine layout, else the algo_dir itself.
    """
    # Quarantine layout has a sibling gt/ directory
    gt_sibling = algo_dir.parent / "gt"
    if gt_sibling.is_dir():
        return gt_sibling / f"{video_name}_unified_ground_truth.json"
    return algo_dir / f"{video_name}_unified_ground_truth.json"


@dataclass
class TriageWorklist:
    """Mutable cursor over a list of triaged segments.

    Reviewer launches the clearing tool; this object tracks the
    list and the current position. Cursor moves forward (next),
    backward (previous), or jumps to a specific entry. End-of-list
    is reachable via :meth:`is_at_end`.
    """

    entries: List[TriageEntry] = field(default_factory=list)
    cursor: int = 0
    algo_dir: Optional[Path] = None
    video_root: Optional[Path] = None

    @classmethod
    def from_algo_dir(
        cls,
        algo_dir: Path,
        *,
        video_filter: Optional[Sequence[str]] = None,
        include_cleared: bool = False,
    ) -> "TriageWorklist":
        """Build a worklist by scanning an algo_outputs directory.

        Convenience constructor that calls :func:`scan_corpus_for_triage`
        and resolves the sibling ``videos/`` directory if present.
        """
        algo_dir = Path(algo_dir)
        entries = scan_corpus_for_triage(
            algo_dir, video_filter=video_filter, include_cleared=include_cleared,
        )
        video_root = algo_dir.parent / "videos"
        if not video_root.is_dir():
            video_root = algo_dir
        return cls(entries=entries, cursor=0, algo_dir=algo_dir, video_root=video_root)

    # --- query --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[TriageEntry]:
        return iter(self.entries)

    @property
    def current(self) -> Optional[TriageEntry]:
        if 0 <= self.cursor < len(self.entries):
            return self.entries[self.cursor]
        return None

    def is_at_end(self) -> bool:
        return self.cursor >= len(self.entries)

    def video_path_for(self, entry: TriageEntry) -> Optional[Path]:
        """Resolve the absolute video path for an entry."""
        if self.video_root is None:
            return None
        cand = self.video_root / f"{entry.video_name}.mp4"
        if cand.is_file():
            return cand
        # Some pipelines compress to .mkv; try that.
        cand_mkv = self.video_root / f"{entry.video_name}.mkv"
        if cand_mkv.is_file():
            return cand_mkv
        return None

    # --- navigation ---------------------------------------------------------

    def advance(self) -> Optional[TriageEntry]:
        """Move cursor to next entry, return that entry (or None at end)."""
        self.cursor += 1
        return self.current

    def retreat(self) -> Optional[TriageEntry]:
        """Move cursor back one entry, clamped at zero."""
        if self.cursor > 0:
            self.cursor -= 1
        return self.current

    def jump_to(self, video_name: str, segment_num: int) -> bool:
        """Move cursor to a specific (video, segment). Returns True if found."""
        for i, e in enumerate(self.entries):
            if e.video_name == video_name and e.segment_num == segment_num:
                self.cursor = i
                return True
        return False

    # --- summary ------------------------------------------------------------

    def progress_string(self) -> str:
        if not self.entries:
            return "no triaged segments"
        n = len(self.entries)
        i = min(self.cursor + 1, n)
        cleared = sum(1 for e in self.entries if e.already_cleared)
        return f"segment {i} of {n} ({cleared} already cleared)"

    def videos_covered(self) -> List[str]:
        seen = []
        for e in self.entries:
            if e.video_name not in seen:
                seen.append(e.video_name)
        return seen


# --- CLI entry helpers -----------------------------------------------------


def find_default_algo_dir() -> Optional[Path]:
    """Locate a default algo outputs directory.

    Resolution order:
      1. ``MOUSEREACH_TRIAGE_ALGO_DIR`` environment variable
      2. ``CONNECTOME_ROOT / Behavior / MouseReach_Pipeline / Improvement_Snapshots``
         is NOT the right place (those are snapshots, not algo outputs); skip
      3. Latest quarantine under
         ``CONNECTOME_ROOT / Validation_Runs / DLC_*/iterations/*/algo_outputs/``
    """
    env = os.environ.get("MOUSEREACH_TRIAGE_ALGO_DIR")
    if env:
        p = Path(env)
        if p.is_dir():
            return p
    cr = os.environ.get("CONNECTOME_ROOT", r"Y:\2_Connectome")
    valid = Path(cr) / "Validation_Runs"
    if not valid.is_dir():
        return None
    candidates = list(valid.glob("DLC_*/iterations/*/algo_outputs"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

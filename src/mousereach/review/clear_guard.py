"""Guard human triage-clears against being clobbered by re-processing.

The routine protocol order is: PROCESS -> TRIAGE-CLEAR (human) -> KINEMATICS.
The danger is a re-run of the detectors between clearing and kinematics: it
would regenerate ``*_reaches.json`` / ``*_pellet_outcomes.json`` and overwrite
the human's cleared segments, silently reverting to the algo's (wrong) call.

This module makes human clears authoritative across a re-run: capture the
human-locked segments from the OLD files, run the algos fresh, then re-apply the
locked segments on top. Matching is by ``segment_num``; if a locked segment no
longer exists after the re-run (e.g. a manual re-segmentation renumbered the
segments), it is reported as skipped rather than silently dropped -- the caller
decides (the stager skips preservation entirely when boundaries changed).

A segment is "human-locked" when a reviewer resolved it:
  - outcome JSON: ``triage_cleared`` or ``human_verified`` is true
  - reach   JSON: segment ``triage_cleared`` is true, or any reach in it has
    ``human_corrected`` true
"""
from __future__ import annotations

from typing import Callable, List, Tuple


def is_outcome_locked(seg: dict) -> bool:
    """True if this outcome segment carries a human resolution."""
    return bool(seg.get("triage_cleared") or seg.get("human_verified"))


def is_reach_locked(seg: dict) -> bool:
    """True if this reach segment carries a human resolution."""
    if seg.get("triage_cleared"):
        return True
    return any(r.get("human_corrected") for r in (seg.get("reaches") or []))


def human_locked_segnums(data: dict, lock_fn: Callable[[dict], bool]) -> List[int]:
    """Segment numbers in ``data`` that are human-locked per ``lock_fn``."""
    out = []
    for s in (data or {}).get("segments", []) or []:
        sn = s.get("segment_num")
        if sn is not None and lock_fn(s):
            out.append(int(sn))
    return out


def merge_preserving_clears(
    new_data: dict,
    old_data: dict,
    lock_fn: Callable[[dict], bool],
) -> Tuple[dict, List[int], List[int]]:
    """Re-apply human-locked segments from ``old_data`` onto ``new_data``.

    Segments human-locked in ``old_data`` replace the freshly-computed segment
    with the same ``segment_num`` in ``new_data`` (the human call wins). A
    locked segment whose number is absent from ``new_data`` (renumbering) is not
    applied.

    Returns ``(new_data, preserved_segnums, skipped_segnums)``. ``new_data`` is
    mutated in place and also returned.
    """
    if not old_data or not new_data:
        return new_data, [], []
    old_locked = {
        int(s["segment_num"]): s
        for s in (old_data.get("segments") or [])
        if s.get("segment_num") is not None and lock_fn(s)
    }
    if not old_locked:
        return new_data, [], []
    new_index = {
        int(s.get("segment_num")): i
        for i, s in enumerate(new_data.get("segments") or [])
        if s.get("segment_num") is not None
    }
    preserved: List[int] = []
    skipped: List[int] = []
    for sn, old_seg in sorted(old_locked.items()):
        if sn in new_index:
            new_data["segments"][new_index[sn]] = old_seg
            preserved.append(sn)
        else:
            skipped.append(sn)
    return new_data, preserved, skipped

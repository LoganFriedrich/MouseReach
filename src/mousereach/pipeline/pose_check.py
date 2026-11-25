"""One test: is this pose usable by the analysis steps?

WHY THIS EXISTS
---------------
A pose file that does not load as a pose table, or lacks the points the
steps need, would be found only when segmentation failed on it -- and a
segmentation failure is sent to the re-segmentation queue, where a person
sets cuts by hand. Nothing a person can do there fixes a pose. Its quality
check ("Could not read DLC file", "Missing SABL or SABR") would be sent to a
review queue the same way. Only what a person can fix in the review window
may hold a video in a queue; a bad pose is a problem of the POSE step, and
the remedy is a new pose.

So the watcher asks this one question before segmentation, on every tray and
every run (watcher/orchestrator.py, _pose_is_unusable), and the quality
triage's two pose-level flags carry the category POSE_UNUSABLE so they take
the same path instead of a review queue.

WHAT THE WATCHER DOES WITH AN UNUSABLE POSE -- ONLY ONE CASE IS SETTLED
----------------------------------------------------------------------
A FILED video whose unusable pose is NOT from the current (declared) model
is marked for a new pose (state 'outdated', scope 'full'): the existing
re-pose round trip asks a GPU node for a pose from the current model. Every
other case -- a filed video whose unusable pose is the current model's own,
a video not filed yet, and how many unusable poses before a video is listed
red -- is not decided here: the video goes on exactly
as it did before this check existed (orchestrator._new_pose_requestable).

WHAT "USABLE" MEANS
-------------------
  * the file loads as a pose table (DeepLabCut's .h5, or its .csv) with at
    least one frame;
  * it has the points every tray needs: SABL and SABR, the two ends of the
    ruler on the tray (the pixel scale of every distance, and the quality
    check's ruler);
  * a tray that is segmented (not an outcome-free Easy or Flat tray) also has
    BOXL and BOXR, the reference points segmentation measures the tray's
    movement against.
A file that cannot even be OPENED (permission, a missing file, a drive that
blinked) is not judged here: that is a fault of the machine, not of the pose,
and the watcher treats it as "could not run" (orchestrator.
segmentation_could_not_run).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

# The flag category the quality triage gives a pose-level problem.
POSE_UNUSABLE = "pose_unusable"

POINTS_EVERY_TRAY = ("SABL", "SABR")
POINTS_SEGMENTED_TRAY = ("BOXL", "BOXR")


def _bodyparts(df) -> set:
    cols = df.columns
    try:
        names = list(cols.names or [])
        if "bodyparts" in names:
            return set(cols.get_level_values("bodyparts"))
        if getattr(cols, "nlevels", 1) >= 2:
            return set(cols.get_level_values(1))
    except Exception:
        pass
    return set()


def check_pose(pose_path, video_id: Optional[str] = None) -> Tuple[Optional[bool], str]:
    """``(usable, why)``. ``usable`` is True when the pose is usable, False
    when it is not (``why`` says what is wrong), and None when the file could
    not be opened at all -- a fault of the machine, judged elsewhere."""
    p = Path(pose_path) if pose_path else None
    try:
        if p is None or not p.is_file():
            return None, f"no pose file at {pose_path}"
        with open(p, "rb") as fh:
            fh.read(1)
    except OSError as e:
        return None, f"the pose file could not be opened ({e})"
    try:
        import pandas as pd
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, header=[0, 1, 2], index_col=0)
        else:
            df = pd.read_hdf(p)
    except Exception as e:
        return False, f"the pose file does not load as a pose table ({type(e).__name__}: {e})"
    try:
        n = len(df)
    except Exception:
        n = 0
    if n == 0:
        return False, "the pose file holds no frames"
    parts = _bodyparts(df)
    if not parts:
        return False, "the pose file names no tracked points"
    needed = list(POINTS_EVERY_TRAY)
    try:
        from mousereach.config import is_outcome_free_tray
        segmented = not is_outcome_free_tray(video_id) if video_id else True
    except Exception:
        segmented = True
    if segmented:
        needed += list(POINTS_SEGMENTED_TRAY)
    missing = [b for b in needed if b not in parts]
    if missing:
        return False, f"the pose lacks the point(s) {', '.join(missing)} the steps need"
    return True, ""

"""Run ONE chosen video through every step -- the "Run All Steps" tab's code.

This is a thin name kept for the tab and for scripts: the work is done by
``mousereach.watcher.run_one.run_one_video``, which drives the processing
machine's watcher for that one video (take it in, analyse, quality check,
hand-score comparison, review gate, kinematics, file it), with the watcher's
own handlers. See that module for what runs, what is refused and why.

WHY no steps here any more: this module used to run its own copy of the
steps in the chosen video's own folder -- no hand-score comparison, a quality
check that passed a video when it could not run, a push into a database, and
no check of where the video was. Pointed at a filed video it rewrote the filed
results in place; pointed at a video in a review queue it rewrote the
algorithm outputs under the reviewer. Running by hand must do exactly what the
watcher does: one implementation, two ways to start it.

ASCII-only console output (Windows cp1252).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional


def run_all_steps(video_path, progress: Optional[Callable[[str, str], None]] = None) -> Dict:
    """Run one chosen video through the watcher's own steps
    (watcher.run_one.run_one_video). Returns its summary: ``refused`` (why
    nothing was run), ``held`` / ``hold_reason``, ``error``, ``done`` (filed),
    ``stages``, ``state_before`` / ``state_after``. Never raises."""
    from mousereach.watcher.run_one import run_one_video
    return run_one_video(video_path, progress=progress)

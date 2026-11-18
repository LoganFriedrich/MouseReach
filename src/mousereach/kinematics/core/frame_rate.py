"""How many frames a video holds per second, read from the video itself.

WHY THIS EXISTS. Every value in seconds (a segment's length, the time to the first
reach, the gap between reaches) and every speed per second is a count of frames
divided by the frame rate. The kinematics used to divide by a constant 30 that was
written into the code as an assumption. Videos recorded at another rate then carry
every such value off by the ratio of the two rates, silently: a 60 frames-per-second
recording reads twice as long, and its speeds per second half as fast, as they were.
A rate is a fact about each recording, so it is read from each recording.

WHERE IT COMES FROM, in this order:

1. The rate a caller states (``FeatureExtractor(frame_rate=...)``), for a lab that
   knows its rate and wants it fixed in its own configuration. Checked like the others.
2. The header of the video the pose was made from: ``<stem>.mp4`` (or .avi, .mkv,
   .mov) beside the pose file, or the video a caller names.
3. The pose tool's own record of that header: DeepLabCut writes the video's rate
   into ``<pose stem>_meta.pickle`` beside the pose file.

If the video header and the pose record disagree, the video header wins and the
disagreement is printed: the video is what the frames were counted in.

WHEN NO RATE IS FOUND. Nothing is guessed. A line says so, and the kinematics are
not written (FeatureExtractor.extract raises KinematicsProblem), so the watcher holds
the video instead of filing it with its values in seconds missing for good. The
session-only file of an Easy or Flat tray has nothing in seconds and records the rate
as missing.

WHAT THIS DOES NOT SAY. A container rate is how often the video shows a frame, not
how often the camera took a new picture. A recording program that writes frames
faster than its cameras deliver them repeats the last picture; anchored_path
explains how speeds handle those repeated frames.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Optional, Tuple, Union

# A frame rate outside this range is not a frame rate (a header can hold 0, a
# negative number or a timebase like 90000 when it is unreadable).
MIN_FRAME_RATE = 1.0
MAX_FRAME_RATE = 1000.0
# Header and pose record are the same number read twice; more than this apart
# means one of them is not about this video.
DISAGREEMENT_TOLERANCE = 0.01
VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov")

PathLike = Union[str, Path]


def valid_frame_rate(value) -> Optional[float]:
    """The value as a frame rate, or None when it cannot be one."""
    try:
        rate = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(rate) or not (MIN_FRAME_RATE <= rate <= MAX_FRAME_RATE):
        return None
    return rate


def video_stem_of_pose(pose_path: PathLike) -> str:
    """The video stem a DeepLabCut pose file was made from: everything before the
    scorer, which DeepLabCut appends as ``DLC_<network>...``."""
    name = Path(pose_path).name
    cut = name.find("DLC_")
    if cut > 0:
        return name[:cut]
    return Path(pose_path).stem


def rate_from_video_header(video_path: PathLike) -> Optional[float]:
    """The frame rate a video file's header states, or None."""
    try:
        import cv2
    except Exception:
        return None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return None
        return valid_frame_rate(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def rate_from_pose_record(pose_path: PathLike) -> Optional[float]:
    """The video frame rate DeepLabCut recorded beside the pose
    (``<pose stem>_meta.pickle``, key ``data.fps``), or None."""
    meta = Path(pose_path).with_name(Path(pose_path).stem + "_meta.pickle")
    if not meta.is_file():
        return None
    try:
        with open(meta, "rb") as f:
            record = pickle.load(f)
        return valid_frame_rate((record.get("data") or {}).get("fps"))
    except Exception:
        return None


def find_video_beside(pose_path: PathLike) -> Optional[Path]:
    """The video the pose file was made from, in the same folder, if it is there."""
    folder = Path(pose_path).parent
    stem = video_stem_of_pose(pose_path)
    for suffix in VIDEO_SUFFIXES:
        candidate = folder / (stem + suffix)
        if candidate.is_file():
            return candidate
    return None


def resolve_frame_rate(
    pose_path: PathLike,
    video_path: Optional[PathLike] = None,
    stated: Optional[float] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """``(frames per second, where it came from)`` for the video a pose file
    belongs to; ``(None, None)`` when no source gives a valid rate.

    ``stated`` is a rate the caller fixes in configuration; it is used when valid,
    and a ValueError says so when it is not (a configured rate that cannot be one
    is a mistake to fix, not to work around). ``video_path`` names the video when it
    is not beside the pose file.
    """
    if stated is not None:
        rate = valid_frame_rate(stated)
        if rate is None:
            raise ValueError(f"stated frame rate {stated!r} is not between "
                             f"{MIN_FRAME_RATE} and {MAX_FRAME_RATE} frames per second")
        return rate, "stated"

    video = Path(video_path) if video_path else find_video_beside(pose_path)
    from_header = rate_from_video_header(video) if video is not None and video.is_file() else None
    from_record = rate_from_pose_record(pose_path)

    if from_header is not None:
        if from_record is not None and abs(from_header - from_record) > DISAGREEMENT_TOLERANCE * from_header:
            print(f"[frame_rate] {video.name}: video header says {from_header:g} frames per "
                  f"second, the pose record says {from_record:g}; using the video header")
        return from_header, f"video header ({video.name})"
    if from_record is not None:
        return from_record, f"pose record ({Path(pose_path).stem}_meta.pickle)"
    return None, None

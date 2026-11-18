"""The one source a secondary export or summary reads: each video's
``<video>_features.json``.

WHY: the features file is the video's resolved result -- the algorithm's calls
with every person's answer applied (review, ground truth, ignore windows,
exclusions; review.truth_resolver). The pellet-outcomes and reaches files are
the algorithm's own calls BEFORE any person's answer. The export, summary,
quick-summary, real-kinematics and ASPA sync commands used to read those, so
a pellet a reviewer corrected was counted with the algorithm's call.

A video in the folder with algorithm files but no features file has no
resolved result yet (its kinematics have not run, or were held). It is left
out of the export and named, never filled in with the algorithm's calls.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple

FEATURES_SUFFIX = "_features.json"
_ALGO_SUFFIXES = ("_pellet_outcomes.json", "_reaches.json")


def load_resolved_videos(input_dir: Path) -> Tuple[List[Dict], List[str]]:
    """``(videos, left_out)`` for the files directly in ``input_dir``.

    ``videos``: every readable ``*_features.json``, parsed, in name order.
    ``left_out``: the names of videos that have an algorithm file
    (pellet outcomes or reaches) but no features file, or whose features file
    cannot be read -- each with the reason, for the command to print."""
    input_dir = Path(input_dir)
    videos: List[Dict] = []
    left_out: List[str] = []
    have = set()
    for fp in sorted(input_dir.glob(f"*{FEATURES_SUFFIX}")):
        stem = fp.name[: -len(FEATURES_SUFFIX)]
        have.add(stem)
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as err:
            left_out.append(f"{stem}: its features file could not be read ({err})")
            continue
        doc.setdefault("video_name", stem)
        videos.append(doc)
    algo = set()
    for suffix in _ALGO_SUFFIXES:
        for fp in input_dir.glob(f"*{suffix}"):
            algo.add(fp.name[: -len(suffix)])
    for stem in sorted(algo - have):
        left_out.append(f"{stem}: no features file yet (kinematics not run or held), "
                        f"so it has no resolved result")
    return videos, left_out


def print_left_out(left_out: List[str]) -> None:
    """Say which videos an export left out and why (ASCII only)."""
    if not left_out:
        return
    print(f"[!] {len(left_out)} video(s) left out -- only a video's features file "
          f"holds its result with every person's answer applied:")
    for line in left_out:
        print(f"    {line}")

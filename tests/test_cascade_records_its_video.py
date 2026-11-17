"""The outcome cascade says whether its video checks could run, and finds the
video by one rule.

WHY: without the video file beside the pose, the CV-artifact gate and Stage 98
do nothing, so a displaced_sa call is committed unchecked -- a different
answer from the same pose -- and the outcomes file gave no sign of it. And the
gate looked for .mp4/.avi while Stage 98 looked for .mp4 only.
"""
import numpy as np
import pandas as pd

from mousereach.outcomes.v6_cascade.detector import detect_outcomes_v6_cascade
from mousereach.outcomes.v6_cascade.stage_98_lost_in_shadow_triage import (
    Stage98LostInShadowTriage)

V = "20990101_ZZZ0101_P6"


def _pose(n=60):
    parts = ["Pellet", "SABL", "SABR", "SATL", "SATR", "RightHand", "Nose"]
    cols = pd.MultiIndex.from_tuples(
        [("s", p, c) for p in parts for c in ("x", "y", "likelihood")])
    return pd.DataFrame(np.ones((n, len(cols))), columns=cols)


def test_the_outcomes_say_the_video_was_missing(tmp_path):
    out = detect_outcomes_v6_cascade(_pose(), [], [], video_id=V,
                                     video_dir=tmp_path)
    assert out.get("cv_video") == "missing"


def test_the_outcomes_say_the_video_was_found(tmp_path):
    (tmp_path / f"{V}.mp4").write_bytes(b"")
    out = detect_outcomes_v6_cascade(_pose(), [], [], video_id=V,
                                     video_dir=tmp_path)
    assert out.get("cv_video") == "found"


def test_stage_98_finds_the_video_the_gate_finds(tmp_path):
    (tmp_path / f"{V}.avi").write_bytes(b"")
    assert Stage98LostInShadowTriage(video_dir=tmp_path)._video_path(V) is not None

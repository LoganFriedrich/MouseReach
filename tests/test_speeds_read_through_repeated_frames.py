"""Speeds read through frames that repeat the picture before them.

WHY: a recording program that writes frames faster than its cameras take pictures
repeats the last picture until a new one arrives. Step by step the hand then moves
nothing, then two frames' worth at once, so the peak speed read double, the speed at
the apex read nothing or double depending on which frame the apex fell on, and jerk
and the agreement between the paw points' speeds measured the repetition. The same
movement recorded without repeats must give the same speeds.
"""
import numpy as np
import pandas as pd
import pytest

from mousereach.kinematics.analysis.real_kinematics import compute_kinematics
from mousereach.kinematics.core.anchored_path import anchored_hand_path
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

HAND = ("RightHand", "RHLeft", "RHRight", "RHOut")
NOSE, BOXL, BOXR = (100.0, 50.0), (90.0, 70.0), (110.0, 70.0)
START = 10
# The same steady movement, 2 px per frame, seen on every frame...
EVERY_FRAME = [150.0, 152.0, 154.0, 156.0, 158.0, 160.0, 162.0, 164.0, 166.0]
# ...and seen on every other frame, each picture shown twice.
REPEATED = [150.0, 150.0, 154.0, 154.0, 158.0, 158.0, 162.0, 162.0, 166.0]
# The repeats falling on the other frames.
REPEATED_SHIFTED = [150.0, 152.0, 152.0, 156.0, 156.0, 160.0, 160.0, 164.0, 166.0]


def _pose(ys, n=40, jitter=0.0):
    cols = {}
    for name, (x, y) in {"Nose": NOSE, "BOXL": BOXL, "BOXR": BOXR}.items():
        cols[f"{name}_x"], cols[f"{name}_y"] = np.full(n, x), np.full(n, y)
        cols[f"{name}_likelihood"] = np.ones(n)
    rng = np.random.default_rng(0)
    for i, p in enumerate(HAND):
        y = np.full(n, 150.0)
        y[START:START + len(ys)] = ys
        x = np.full(n, 100.0 + i)
        if jitter:
            x[START:START + len(ys)] += rng.uniform(-jitter, jitter, len(ys))
        cols[f"{p}_x"], cols[f"{p}_y"] = x, y
        cols[f"{p}_likelihood"] = np.ones(n)
    return pd.DataFrame(cols)


def _reach(apex_offset=4):
    return {"reach_id": 1, "reach_num": 1, "start_frame": START,
            "apex_frame": START + apex_offset, "end_frame": START + len(EVERY_FRAME) - 1,
            "duration_frames": len(EVERY_FRAME)}


def _measures(ys, apex_offset=4, jitter=0.0):
    fx = FeatureExtractor(frame_rate=60.0)
    f = fx._extract_reach_features(_reach(apex_offset), _pose(ys, jitter=jitter), 34.0, {})
    e = f.extended
    return {
        "peak": f.peak_velocity_px_per_frame,
        "apex": f.velocity_at_apex_px_per_frame,
        "apex_mm_s": f.velocity_at_apex_mm_per_sec,
        "smoothness": f.trajectory_smoothness,
        "point_peak": e["righthand_peak_speed_px_per_frame"],
        "point_smoothness": e["righthand_motion_smoothness"],
        "velocity_correlation": e["paw_velocity_correlation"],
        # Not speeds: these read the positions as tracked and must not move.
        "mean": f.mean_velocity_px_per_frame,
        "straightness": f.trajectory_straightness,
        "path": e["righthand_total_path_px"],
    }


def test_a_repeated_picture_gives_the_speeds_of_the_same_movement_seen_every_frame():
    seen_every_frame = _measures(EVERY_FRAME)
    for ys in (REPEATED, REPEATED_SHIFTED):
        got = _measures(ys)
        for name, want in seen_every_frame.items():
            assert got[name] == pytest.approx(want, rel=1e-9, abs=1e-9), name


def test_peak_speed_is_not_doubled_by_repeats():
    assert _measures(REPEATED)["peak"] == pytest.approx(2.0)
    assert _measures(REPEATED)["point_peak"] == pytest.approx(2.0)


@pytest.mark.parametrize("apex_offset", [1, 2, 3, 4, 5, 6, 7])
def test_speed_at_the_apex_does_not_depend_on_which_frame_the_apex_fell_on(apex_offset):
    got = _measures(REPEATED, apex_offset=apex_offset)
    assert got["apex"] == pytest.approx(2.0)
    assert got["apex_mm_s"] == pytest.approx(2.0 * 9.0 / 34.0 * 60.0)


def test_tracking_jitter_on_a_repeated_picture_still_counts_as_a_repeat():
    got = _measures(REPEATED, jitter=0.05)
    assert got["peak"] == pytest.approx(2.0, abs=0.1)


def test_a_movement_without_repeats_is_unchanged():
    path = anchored_hand_path(_pose([150.0, 151.0, 154.0, 156.0, 160.0]), "RightHand", START, START + 4)
    v = FeatureExtractor()._compute_velocity_features(path, 1)
    assert v["peak_velocity"] == 4.0 and v["velocity_at_apex"] == 3.0


def test_real_kinematics_reads_through_repeats_too():
    def kin(ys):
        p = anchored_hand_path(_pose(ys), "RightHand", START, START + len(ys) - 1)
        return compute_kinematics(p.x, p.y, ruler_px=9.0, fps=60.0)
    assert kin(REPEATED)["peak_speed_mm_s"] == pytest.approx(kin(EVERY_FRAME)["peak_speed_mm_s"])
    assert kin(REPEATED)["peak_speed_mm_s"] == pytest.approx(2.0 * 60.0)


def test_a_hand_that_is_really_still_and_then_moves_keeps_the_move_s_speed():
    # A pause longer than a recorder repeats a picture is the hand, not the video:
    # four still frames then a 10 px step stay one 10 px step, not four of 2.5.
    from mousereach.kinematics.core.anchored_path import fill_repeated_frames
    x = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 110.0, 0.0])
    fx, _ = fill_repeated_frames(x, np.zeros_like(x))
    assert list(fx) == list(x)
    still = [150.0, 150.0, 150.0, 150.0, 160.0, 162.0, 164.0, 166.0, 168.0]
    assert _measures(still)["peak"] == pytest.approx(10.0)

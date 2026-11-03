"""Peak speed and speed at the apex come only from steps between detected frames.

WHY: the first and last steps of an anchored path join a calculated point to a detected
one and are charged to one frame, so they measure geometry, not how fast the hand moved.
Counted, they were the largest step on about half of reaches. The path, its length and
its straightness still run nose to nose, so those must still see the anchors.
"""
import numpy as np
import pandas as pd
import pytest

from mousereach.kinematics.analysis.real_kinematics import compute_kinematics
from mousereach.kinematics.core.anchored_path import (
    anchored_hand_path, tracked_steps, ANCHOR_FRACTION_FROM_NOSE)
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

HAND = ("RightHand", "RHLeft", "RHRight", "RHOut")
NOSE, BOXL, BOXR = (100.0, 50.0), (90.0, 70.0), (110.0, 70.0)
# Detected frames 10..14, moving 1, 3, 2 and 4 px along a line whose furthest point
# is the last frame. The steps to and from the anchors are about 90 px.
DETECTED_Y = [150.0, 151.0, 154.0, 156.0, 160.0]


def _pose(ys=DETECTED_Y, start=10, n=40):
    cols = {}
    for name, (x, y) in {"Nose": NOSE, "BOXL": BOXL, "BOXR": BOXR}.items():
        cols[f"{name}_x"], cols[f"{name}_y"] = np.full(n, x), np.full(n, y)
        cols[f"{name}_likelihood"] = np.ones(n)
    for p in HAND:
        y = np.full(n, 150.0)
        y[start:start + len(ys)] = ys
        cols[f"{p}_x"], cols[f"{p}_y"] = np.full(n, 100.0), y
        cols[f"{p}_likelihood"] = np.ones(n)
    return pd.DataFrame(cols)


def _path(df=None, start=10, end=14):
    return anchored_hand_path(_pose() if df is None else df, "RightHand", start, end)


def test_tracked_steps_drops_exactly_the_first_and_last_step():
    assert list(tracked_steps(np.array([90.0, 1.0, 3.0, 2.0, 4.0, 90.0]))) == [1.0, 3.0, 2.0, 4.0]


def test_the_midpoint_is_the_confirmed_fraction():
    assert ANCHOR_FRACTION_FROM_NOSE == 0.5


def test_wrist_peak_is_the_largest_step_between_detected_frames():
    fx = FeatureExtractor()
    p = _path()
    assert np.hypot(np.diff(p.x), np.diff(p.y)).max() > 80      # the jump really is on the path
    assert fx._compute_velocity_features(p, 0)["peak_velocity"] == 4.0


def test_wrist_apex_speed_is_the_step_leaving_the_apex():
    fx = FeatureExtractor()
    assert fx._compute_velocity_features(_path(), 1)["velocity_at_apex"] == 3.0
    assert fx._compute_velocity_features(_path(), 0)["velocity_at_apex"] == 1.0   # never the start jump


def test_an_apex_on_the_last_detected_frame_has_no_speed():
    fx = FeatureExtractor()
    assert fx._compute_velocity_features(_path(), 4)["velocity_at_apex"] is None


def test_a_one_frame_reach_has_no_peak_or_apex_speed():
    fx = FeatureExtractor()
    v = fx._compute_velocity_features(_path(start=12, end=12), 0)
    assert v["peak_velocity"] is None and v["velocity_at_apex"] is None
    assert v["mean_velocity"] is not None       # the whole trip still exists


def test_mean_speed_straightness_and_path_still_see_the_anchors():
    fx = FeatureExtractor()
    p = _path()
    all_steps = np.hypot(np.diff(p.x), np.diff(p.y))
    assert fx._compute_velocity_features(p, 0)["mean_velocity"] == pytest.approx(all_steps.mean())
    assert fx._compute_trajectory_features(p)["straightness"] == pytest.approx(10.0 / all_steps.sum())   # anchor to anchor
    out = fx._paw_trajectory_features(_pose(), "RightHand", 10, 14, 34.0)
    x, y = p.nose_relative()
    assert out["righthand_total_path_px"] == pytest.approx(np.hypot(np.diff(x), np.diff(y)).sum())


@pytest.mark.parametrize("point,prefix", [("RightHand", "righthand"), ("RHOut", "rhout")])
def test_per_point_peak_and_apex_speed_leave_out_the_anchor_steps(point, prefix):
    fx = FeatureExtractor()
    # The furthest point is the middle frame here, so the apex has a step leaving it.
    out = fx._paw_trajectory_features(_pose([150.0, 151.0, 170.0, 168.0, 167.0]), point, 10, 14, 34.0)
    assert out[f"{prefix}_peak_speed_px_per_frame"] == 19.0
    assert out[f"{prefix}_apex_speed_px_per_frame"] == 2.0
    assert out[f"{prefix}_peak_speed_mm_per_frame"] == pytest.approx(19.0 * 9.0 / 34.0)
    assert out[f"{prefix}_apex_speed_mm_per_frame"] == pytest.approx(2.0 * 9.0 / 34.0)


def test_per_point_apex_on_the_last_detected_frame_has_no_speed():
    out = FeatureExtractor()._paw_trajectory_features(_pose(), "RightHand", 10, 14, 34.0)
    assert out["righthand_apex_frame"] == 14
    assert out["righthand_apex_speed_px_per_frame"] is None
    assert out["righthand_apex_speed_mm_per_frame"] is None
    assert out["righthand_peak_speed_px_per_frame"] == 4.0


def test_a_reach_with_no_anchor_keeps_its_speeds_but_not_its_path():
    df = _pose()
    df["Nose_likelihood"] = 0.1
    fx = FeatureExtractor()
    v = fx._compute_velocity_features(anchored_hand_path(df, "RightHand", 10, 14), 0)
    assert v["peak_velocity"] == 4.0 and v["velocity_at_apex"] == 1.0 and v["mean_velocity"] is None
    out = fx._paw_trajectory_features(df, "RightHand", 10, 14, 34.0)
    assert out["righthand_peak_speed_px_per_frame"] == 4.0
    assert out["righthand_total_path_px"] is None


def test_the_whole_reach_row_and_its_mm_conversion_follow():
    fx = FeatureExtractor()
    reach = {"reach_id": 1, "reach_num": 1, "start_frame": 10, "apex_frame": 11,
             "end_frame": 14, "duration_frames": 5}
    f = fx._extract_reach_features(reach, _pose(), 34.0, {})
    assert f.peak_velocity_px_per_frame == 4.0
    assert f.velocity_at_apex_px_per_frame == 3.0
    assert f.velocity_at_apex_mm_per_sec == pytest.approx(3.0 * 9.0 / 34.0 * 30.0)
    g = fx._extract_reach_features(dict(reach, apex_frame=14), _pose(), 34.0, {})
    assert g.velocity_at_apex_px_per_frame is None and g.velocity_at_apex_mm_per_sec is None


def test_real_kinematics_peak_leaves_out_the_anchor_steps():
    p = _path()
    kin = compute_kinematics(p.x, p.y, ruler_px=9.0, fps=30)      # 1 px = 1 mm
    assert kin["peak_speed_mm_s"] == 4.0 * 30
    steps = np.hypot(np.diff(p.x), np.diff(p.y))
    assert kin["mean_speed_mm_s"] == pytest.approx(round(steps.mean() * 30, 2))


def test_real_kinematics_needs_a_step_between_two_tracked_positions():
    assert compute_kinematics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), ruler_px=9.0) == {}

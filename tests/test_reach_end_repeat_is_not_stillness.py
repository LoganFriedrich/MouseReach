"""A reach whose last detected frame repeats the picture before it.

WHY: the paw is seen until the first new picture that no longer shows it, so the last
detected frame can be a repeat of the picture before it. A repeated picture is never
stillness: the hand went on moving towards the end point, which it reached by the frame
after the reach (the end anchor). Repeats between two new pictures were already placed
on the line between them; a closing repeat was left where it was, so the hand read as
standing still on the last frame and then jumping to the end point. That put a zero
speed on an apex taken just before it, and a stop that never happened into both
smoothness values and the agreement between the paw points' speeds.

Peak and apex speed still read only steps the tracked hand made: the steps from the last
new picture onward are hidden travel towards a calculated point, like the anchor step.
"""
import numpy as np
import pandas as pd
import pytest

from mousereach.kinematics.analysis.real_kinematics import compute_kinematics
from mousereach.kinematics.core.anchored_path import (
    fill_repeated_frames, tracked_speeds, tracked_steps)
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

HAND = ("RightHand", "RHLeft", "RHRight", "RHOut")
NOSE, BOXL, BOXR = (100.0, 50.0), (90.0, 70.0), (110.0, 70.0)
START = 10


def _pose(ys, n=40):
    cols = {}
    for name, (x, y) in {"Nose": NOSE, "BOXL": BOXL, "BOXR": BOXR}.items():
        cols[f"{name}_x"], cols[f"{name}_y"] = np.full(n, x), np.full(n, y)
        cols[f"{name}_likelihood"] = np.ones(n)
    for i, p in enumerate(HAND):
        y = np.full(n, 150.0)
        y[START:START + len(ys)] = ys
        cols[f"{p}_x"], cols[f"{p}_y"] = np.full(n, 100.0 + i), y
        cols[f"{p}_likelihood"] = np.ones(n)
    return pd.DataFrame(cols)


def _features(ys, apex_offset):
    reach = {"reach_id": 1, "reach_num": 1, "start_frame": START,
             "apex_frame": START + apex_offset, "end_frame": START + len(ys) - 1,
             "duration_frames": len(ys)}
    return FeatureExtractor(frame_rate=60.0)._extract_reach_features(reach, _pose(ys), 34.0, {})


# Out and back, seen on every other frame; the last detected frame repeats the one
# before it. The apex (largest y) is the last new picture.
OUT_AND_BACK_ENDING_ON_A_REPEAT = [150.0, 150.0, 158.0, 158.0, 166.0, 166.0, 170.0, 170.0]


def test_a_closing_repeat_is_placed_on_the_line_to_the_end_point():
    # anchor, detected frames (last two the same picture), anchor
    x = np.array([0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 0.0])
    fx, _ = fill_repeated_frames(x, np.zeros_like(x))
    assert fx[8] == pytest.approx(20.0)      # halfway from 40 (last picture) to 0 (end point)
    assert list(fx[:8]) == pytest.approx([0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])


def test_no_step_of_the_filled_path_is_a_stop_that_did_not_happen():
    x = np.array([0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 0.0])
    fx, _ = fill_repeated_frames(x, np.zeros_like(x))
    assert np.all(np.abs(np.diff(fx)) > 0)


def test_speeds_from_the_last_picture_onward_are_left_out_of_peak_and_apex():
    x = np.array([0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0, 100.0])
    sp = tracked_speeds(x, np.zeros_like(x))
    assert len(sp) == len(tracked_steps(np.diff(x)))
    assert list(sp[:6]) == pytest.approx([5.0] * 6)
    assert np.isnan(sp[6])        # last picture -> repeat: hidden travel to a far end point
    assert np.nanmax(sp) == pytest.approx(5.0)


def test_a_path_that_ends_on_a_new_picture_is_unchanged():
    x = np.array([0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 0.0])
    fx, _ = fill_repeated_frames(x, np.zeros_like(x))
    assert list(fx) == pytest.approx([0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 0.0])
    assert not np.isnan(tracked_speeds(x, np.zeros_like(x))).any()


def test_a_hand_really_at_rest_before_the_end_stays_at_rest():
    # Three or more frames on one position is a hand that is still, not a repeat.
    x = np.array([0.0, 10.0, 20.0, 30.0, 30.0, 30.0, 30.0, 0.0])
    fx, _ = fill_repeated_frames(x, np.zeros_like(x))
    assert list(fx) == pytest.approx(list(x))
    assert not np.isnan(tracked_speeds(x, np.zeros_like(x))).any()


def test_the_apex_just_before_a_closing_repeat_has_no_speed_rather_than_zero():
    f = _features(OUT_AND_BACK_ENDING_ON_A_REPEAT, apex_offset=6)
    assert f.velocity_at_apex_px_per_frame is None
    assert f.extended["righthand_apex_speed_px_per_frame"] is None


def test_peak_speed_is_not_the_hidden_trip_to_the_end_point():
    f = _features(OUT_AND_BACK_ENDING_ON_A_REPEAT, apex_offset=6)
    assert f.peak_velocity_px_per_frame == pytest.approx(4.0)
    assert f.extended["righthand_peak_speed_px_per_frame"] == pytest.approx(4.0)


def test_real_kinematics_peak_leaves_out_the_closing_repeat():
    # Steady 4 px per frame, last frame a repeat, then a far end point.
    x = np.array([0.0, 10.0, 10.0, 18.0, 18.0, 26.0, 26.0, 200.0])
    k = compute_kinematics(x, np.zeros_like(x), ruler_px=9.0, fps=60.0)
    assert k["peak_speed_mm_s"] == pytest.approx(4.0 * 60.0)

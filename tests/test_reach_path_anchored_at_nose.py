"""A reach starts and ends at the nose, and every paw-position measurement knows it.

WHY: the hand is hidden behind the box wall in the frame before a reach is detected
and in the frame after it disappears. Path, speed and straightness measured over
detected frames alone begin and end wherever tracking did. The rule places every hand
point between the nose and a slit corner in those two frames.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mousereach.kinematics.core import anchored_path as ap
from mousereach.kinematics.core.anchored_path import anchored_hand_path
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

HAND = ("RightHand", "RHLeft", "RHRight", "RHOut")
SRC = Path(ap.__file__).resolve().parents[2]


def _pose(n=40, nose=(100.0, 50.0), boxl=(90.0, 70.0), boxr=(110.0, 70.0)):
    """A still mouse; the hand 'tracked' 40 px below the nose in every frame."""
    cols = {}
    for name, (x, y) in {"Nose": nose, "BOXL": boxl, "BOXR": boxr}.items():
        cols[f"{name}_x"], cols[f"{name}_y"] = np.full(n, x), np.full(n, y)
        cols[f"{name}_likelihood"] = np.ones(n)
    for p in HAND:
        cols[f"{p}_x"], cols[f"{p}_y"] = np.full(n, nose[0]), np.full(n, nose[1] + 40.0)
        cols[f"{p}_likelihood"] = np.ones(n)
    return pd.DataFrame(cols)


def test_the_two_ends_sit_between_the_nose_and_each_slit_corner():
    p = anchored_hand_path(_pose(), "RightHand", 10, 14)
    assert p.ok and len(p.x) == 5 + 2
    assert (p.x[0], p.y[0]) == (95.0, 60.0)      # nose and left corner
    assert (p.x[-1], p.y[-1]) == (105.0, 60.0)   # nose and right corner
    assert (p.start_source, p.end_source) == ("anchor_frame", "anchor_frame")
    x, y = p.nose_relative()
    assert (x[0], y[0], x[-1], y[-1]) == (-5.0, 10.0, 5.0, 10.0)


def test_the_anchors_use_the_frames_outside_the_reach_not_its_first_and_last():
    df = _pose()
    df.loc[9, "Nose_x"] = 200.0
    df.loc[10, "Nose_x"] = 300.0
    p = anchored_hand_path(df, "RightHand", 10, 14)
    assert p.x[0] == (200.0 + 90.0) / 2


def test_a_reach_at_the_edge_of_the_video_uses_the_nearest_frame_and_says_so():
    p = anchored_hand_path(_pose(n=20), "RightHand", 0, 19)
    assert (p.start_source, p.end_source) == ("video_edge", "video_edge")
    assert p.ok and p.x[0] == 95.0 and p.x[-1] == 105.0


def test_an_untrusted_landmark_comes_from_the_nearest_trusted_frame_and_says_so():
    df = _pose()
    df.loc[9, "Nose_likelihood"] = 0.1
    df.loc[8, "Nose_x"] = 120.0
    p = anchored_hand_path(df, "RightHand", 10, 14)
    assert p.start_source == "nearest_frame"
    assert p.x[0] == (120.0 + 90.0) / 2


def test_no_trusted_landmark_means_no_anchor_and_no_path_measurements():
    df = _pose()
    df["Nose_likelihood"] = 0.1
    p = anchored_hand_path(df, "RightHand", 10, 14)
    assert not p.ok and p.start_source == "none" and np.isnan(p.x[0])
    fx = FeatureExtractor()
    v = fx._compute_velocity_features(p, 2)
    assert v["mean_velocity"] is None
    assert v["peak_velocity"] == 0.0 and v["velocity_at_apex"] == 0.0   # detected frames only
    assert fx._compute_trajectory_features(p)["straightness"] is None
    out = fx._paw_trajectory_features(df, "RightHand", 10, 14, 34.0)
    assert out["righthand_total_path_px"] is None
    assert out["righthand_mean_speed_px_per_frame"] is None
    assert out["righthand_peak_speed_px_per_frame"] == 0.0              # detected frames only
    assert out["righthand_apex_frame"] is not None      # needs detected frames only


def test_a_neighbouring_reach_in_the_anchor_frame_keeps_the_anchor_and_is_recorded():
    p = anchored_hand_path(_pose(), "RightHand", 10, 14, other_reaches=[(5, 9), (20, 25)])
    assert p.start_in_neighbour and not p.end_in_neighbour
    assert p.start_source == "anchor_frame" and p.x[0] == 95.0


def test_the_flat_columns_measure_the_trip_to_and_from_the_slit():
    df = _pose()
    p = anchored_hand_path(df, "RightHand", 10, 14)
    fx = FeatureExtractor()
    # 30 px from the anchor out to a hand that then stays still, and 30 px or so back.
    step = float(np.hypot(5.0, 30.0))
    v = fx._compute_velocity_features(p, 2)
    assert v["mean_velocity"] == pytest.approx(2 * step / 6)   # 6 steps for 5 frames
    assert v["peak_velocity"] == 0.0                            # the jumps do not count
    assert v["velocity_at_apex"] == 0.0                         # the step after the apex
    t = fx._compute_trajectory_features(p)
    assert t["straightness"] == pytest.approx(10.0 / (2 * step))


def test_duration_stays_the_detected_frames_and_the_row_says_how_it_was_anchored():
    fx = FeatureExtractor()
    reach = {"reach_id": 1, "reach_num": 1, "start_frame": 10, "apex_frame": 12,
             "end_frame": 14, "duration_frames": 5}
    f = fx._extract_reach_features(reach, _pose(), 34.0, {}, other_reaches=[(15, 18)])
    assert f.duration_frames == 5
    assert f.peak_velocity_px_per_frame == 0.0
    assert f.mean_velocity_px_per_frame == pytest.approx(2 * float(np.hypot(5.0, 30.0)) / 6)
    assert f.extended["anchor_start_source"] == "anchor_frame"
    assert f.extended["anchor_end_in_neighbour_reach"] is True
    assert "_extended_features_error" not in f.extended
    assert f.extended["righthand_extension_past_nose_px"] == 40.0


# Reading hand coordinates is allowed only here, each for a stated reason. A new
# measurement that reads them must use anchored_hand_path or be argued onto this list.
ALLOWED_RAW_HAND_READERS = {
    ("kinematics/core/anchored_path.py", "anchored_hand_path"): "the shared function itself",
    ("kinematics/core/anchored_path.py", "_trusted_position"): "reads the nose and slit corners, not the hand",
    ("kinematics/core/anchored_path.py", "anchored_reach_frames"): "writes the anchors into a frame table",
    ("kinematics/core/feature_extractor.py", "_compute_orientation_features"): "paw shape: angle between points",
    ("kinematics/core/feature_extractor.py", "_paw_shape_features"): "paw shape: distances between points",
    ("kinematics/analysis/real_kinematics.py", "get_hand_trajectory"): "only checks the columns exist",
}
CHECKED = ("kinematics/core/anchored_path.py", "kinematics/core/feature_extractor.py",
           "kinematics/analysis/real_kinematics.py", "kinematics/analysis/reach_export.py",
           "kinematics/analysis/data_loader.py", "kinematics/analysis/group_analyzer.py",
           "kinematics/analysis/temporal_analyzer.py", "kinematics/analysis/validation_analyzer.py",
           "kinematics/analysis/odc_sci_exporter.py", "kinematics/cli.py")


def _reads_hand_coordinates(node):
    """Is this a column name for a hand point's x or y, literal or built by an f-string?"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return any(node.value == f"{p}_{a}" for p in HAND for a in "xy")
    if isinstance(node, ast.JoinedStr) and len(node.values) == 2:
        tail = node.values[1]
        return isinstance(tail, ast.Constant) and tail.value in ("_x", "_y")
    return False


def test_no_kinematic_reads_raw_hand_coordinates_around_the_shared_function():
    found = set()
    for rel in CHECKED:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if _reads_hand_coordinates(node):
                        found.add((rel, fn.name))
    assert found - set(ALLOWED_RAW_HAND_READERS) == set()


def test_the_midpoint_assumption_lives_in_one_named_constant():
    assert ap.ANCHOR_FRACTION_FROM_NOSE == 0.5
    source = (SRC / "kinematics/core/feature_extractor.py").read_text(encoding="utf-8")
    assert "/ 2.0" not in source


def test_whole_frame_tools_get_the_anchors_and_the_tracked_body():
    df = _pose()
    out = ap.anchored_reach_frames(df, 10, 14)
    assert len(out) == 5 + 2
    for p in HAND:
        assert (out[f"{p}_x"].iloc[0], out[f"{p}_y"].iloc[0]) == (95.0, 60.0)
        assert (out[f"{p}_x"].iloc[-1], out[f"{p}_y"].iloc[-1]) == (105.0, 60.0)
        assert out[f"{p}_x"].iloc[1] == 100.0
    assert out["Nose_x"].iloc[0] == 100.0
    assert (df["RightHand_x"] == 100.0).all()     # the caller's data is untouched
    bad = _pose()
    bad["Nose_likelihood"] = 0.1
    assert ap.anchored_reach_frames(bad, 10, 14) is None

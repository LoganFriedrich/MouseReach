"""Every reach carries the legacy_swipe measures (an earlier swipe-measure definition).

WHY: analyses built on an earlier program's per-swipe numbers (breadth, length, area,
path length, speed, duration in frames, path over frames) can only be continued with
new data if the same numbers are computed the same way. legacy_swipe.py reproduces
that definition on this pipeline's reach windows; these tests pin its per-frame test,
its formulas, its use of the video's own frame rate, and that the values reach the
features file.
"""
import json
import math

import cv2
import numpy as np
import pandas as pd
import pytest

from mousereach.kinematics.core.feature_extractor import FeatureExtractor
from mousereach.kinematics.core.legacy_swipe import KEYS, legacy_swipe_measures

STEM = "20990101_TEST0101_P1"
POSE_NAME = f"{STEM}DLC_resnet50_testshuffle1_100.h5"
POINTS = ("Nose", "BOXL", "BOXR", "RightHand", "RHLeft", "RHRight", "RHOut",
          "LeftEar", "RightEar", "SABL", "SABR", "Pellet", "Pillar", "Reference")
N_FRAMES = 400
# Nose at (100, 50); Reference at (20.7, 60.9) -> truncated (20, 60): the slit zone
# is Nose_y > -20 and 90 < Nose_x < 120, so the nose is in it.
REACH_X = [100.0, 102.0, 101.0, 99.0, 100.0]
REACH_Y = [150.0, 151.0, 154.0, 156.0, 160.0]


def _pose_table():
    cols = {}
    for p in POINTS:
        x, y = {"Nose": (100.0, 50.0), "BOXL": (90.0, 70.0), "BOXR": (110.0, 70.0),
                "SABL": (80.0, 200.0), "SABR": (114.0, 200.0),
                "Reference": (20.7, 60.9)}.get(p, (100.0, 150.0))
        cols[f"{p}_x"] = np.full(N_FRAMES, x)
        cols[f"{p}_y"] = np.full(N_FRAMES, y)
        cols[f"{p}_likelihood"] = np.ones(N_FRAMES)
    df = pd.DataFrame(cols)
    df.loc[100:104, "RightHand_x"] = REACH_X
    df.loc[100:104, "RightHand_y"] = REACH_Y
    return df


def _expected(xs, ys, nose_y=50.0, fps=60.0, first=100, last=104):
    x, y = np.array(xs), np.array(ys)
    path = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return {
        "legacy_swipe_duration_frames": len(x),
        "legacy_swipe_first_frame": first,
        "legacy_swipe_last_frame": last,
        "legacy_swipe_breadth_px": x.max() - x.min(),
        "legacy_swipe_length_px": y.max() - nose_y,
        "legacy_swipe_area_px2": 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))),
        "legacy_swipe_path_length_px": path,
        "legacy_swipe_speed_px_div4_per_sec": (path / 4) / ((last - first) / fps),
        "legacy_swipe_path_over_frames_px_per_frame": path / len(x),
    }


def _assert_values(got, want):
    assert set(got) == set(KEYS)
    for k, v in want.items():
        assert got[k] == pytest.approx(v), k


def test_every_frame_passing_gives_the_formulas():
    got = legacy_swipe_measures(_pose_table(), 100, 104, 40, 339, 60.0)
    _assert_values(got, _expected(REACH_X, REACH_Y))
    # Breadth 3 px and length 110 px, written out once by hand.
    assert got["legacy_swipe_breadth_px"] == 3.0
    assert got["legacy_swipe_length_px"] == 110.0


def test_speed_uses_the_video_frame_rate():
    # The earlier program divided by a constant 60; at 60 fps the value is the
    # same, at 30 fps the same frames span twice the time, so half the speed.
    at60 = legacy_swipe_measures(_pose_table(), 100, 104, 40, 339, 60.0)
    at30 = legacy_swipe_measures(_pose_table(), 100, 104, 40, 339, 30.0)
    assert at30["legacy_swipe_speed_px_div4_per_sec"] == pytest.approx(
        at60["legacy_swipe_speed_px_div4_per_sec"] / 2)
    none = legacy_swipe_measures(_pose_table(), 100, 104, 40, 339, None)
    assert none["legacy_swipe_speed_px_div4_per_sec"] is None
    assert none["legacy_swipe_path_length_px"] == at60["legacy_swipe_path_length_px"]


@pytest.mark.parametrize("change", [
    ("RightHand_likelihood", 0.7),     # must be > 0.7
    ("Nose_likelihood", 0.5),          # must be > 0.5
    ("Nose_x", 90.0),                  # must be > ref_x + 70 = 90 (ref truncated to 20)
    ("Nose_x", 120.0),                 # must be < ref_x + 100 = 120
    ("Nose_y", -20.0),                 # must be > ref_y - 80 = -20 (ref truncated to 60)
])
def test_a_frame_failing_one_condition_is_not_counted(change):
    col, value = change
    df = _pose_table()
    df.loc[102, col] = value
    got = legacy_swipe_measures(df, 100, 104, 40, 339, 60.0)
    keep = [0, 1, 3, 4]
    _assert_values(got, _expected([REACH_X[i] for i in keep], [REACH_Y[i] for i in keep]))


def test_the_paw_must_be_more_than_8_px_past_the_nose():
    df = _pose_table()
    df.loc[100, "RightHand_y"] = 58.0     # 8 px past the nose: not counted
    got = legacy_swipe_measures(df, 100, 104, 40, 339, 60.0)
    _assert_values(got, _expected(REACH_X[1:], REACH_Y[1:], first=101))


def test_the_threshold_line_is_300_px_below_the_confident_maximum():
    df = _pose_table()
    df.loc[200, "RightHand_y"] = 455.0    # confident maximum in the window: line at 155
    got = legacy_swipe_measures(df, 100, 104, 40, 339, 60.0)
    _assert_values(got, _expected(REACH_X[3:], REACH_Y[3:], first=103))
    # The same maximum at likelihood 0.95 does not set the line (must be > 0.95).
    df.loc[200, "RightHand_likelihood"] = 0.95
    _assert_values(legacy_swipe_measures(df, 100, 104, 40, 339, 60.0),
                   _expected(REACH_X, REACH_Y))


def test_one_counted_frame_gives_a_duration_and_nothing_else():
    df = _pose_table()
    df.loc[101:104, "RightHand_likelihood"] = 0.1
    got = legacy_swipe_measures(df, 100, 104, 40, 339, 60.0)
    assert got["legacy_swipe_duration_frames"] == 1
    assert got["legacy_swipe_first_frame"] == got["legacy_swipe_last_frame"] == 100
    assert all(got[k] is None for k in KEYS if not k.endswith(("_frames", "_frame")))


def test_no_reference_point_gives_empty_values():
    df = _pose_table().drop(columns=["Reference_x", "Reference_y", "Reference_likelihood"])
    got = legacy_swipe_measures(df, 100, 104, 40, 339, 60.0)
    assert got == {k: None for k in KEYS}
    df = _pose_table()
    df.loc[40, "Reference_x"] = np.nan
    assert legacy_swipe_measures(df, 100, 104, 40, 339, 60.0) == {k: None for k in KEYS}


# --- the values reach the features file -------------------------------------------

def _write_video(path, fps, n=8):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    assert out.isOpened()
    for i in range(n):
        out.write(np.full((32, 32, 3), i * 20 % 255, np.uint8))
    out.release()


def _bundle(tmp_path, fps):
    flat = _pose_table()
    df = pd.DataFrame({("s",) + tuple(c.rsplit("_", 1)): flat[c] for c in flat.columns})
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    pose = tmp_path / POSE_NAME
    df.to_hdf(pose, key="df")
    _write_video(tmp_path / f"{STEM}.mp4", fps)
    reaches = tmp_path / f"{STEM}_reaches.json"
    reaches.write_text(json.dumps({
        "video_name": STEM, "total_frames": N_FRAMES, "n_segments": 1,
        "segments": [{"segment_num": 1, "start_frame": 40, "end_frame": 339,
                      "ruler_pixels": 34.0,
                      "reaches": [{"reach_id": 1, "reach_num": 1, "start_frame": 100,
                                   "apex_frame": 104, "end_frame": 104,
                                   "duration_frames": 5}]}],
    }), encoding="utf-8")
    outcomes = tmp_path / f"{STEM}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"video_name": STEM, "segments": [
        {"segment_num": 1, "outcome": "untouched", "confidence": 1.0}]}), encoding="utf-8")
    return pose, reaches, outcomes


@pytest.mark.parametrize("fps", [60.0, 30.0])
def test_the_features_file_carries_them(tmp_path, fps):
    pose, reaches, outcomes = _bundle(tmp_path, fps)
    feats = FeatureExtractor().extract(pose, reaches, outcomes)
    d = json.loads(json.dumps(feats.to_dict()))      # what the features file holds
    ext = d["segments"][0]["reaches"][0]["extended"]
    _assert_values({k: ext.get(k) for k in KEYS}, _expected(REACH_X, REACH_Y, fps=fps))
    assert all(isinstance(ext[k], (int, float)) and math.isfinite(ext[k]) for k in KEYS)


def test_the_extractor_version_says_the_columns_are_there():
    assert FeatureExtractor.VERSION == "2.6.0"

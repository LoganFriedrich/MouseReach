"""Values in seconds use the frame rate of the video itself.

WHY: the extractor divided frame counts by a constant 30. On a video recorded at
60 frames per second every segment length, time to first reach, time to outcome and
inter-reach interval came out twice as long as it was, and the speed at the apex in
mm/s was converted on the wrong time base. A rate is a fact about each recording,
so it is read from the recording (frame_rate.py).
"""
import json
import pickle

import cv2
import numpy as np
import pandas as pd
import pytest


from mousereach.kinematics.core.feature_extractor import FeatureExtractor

STEM = "20990101_TEST0101_P1"
POSE_NAME = f"{STEM}DLC_resnet50_testshuffle1_100.h5"
POINTS = ("Nose", "BOXL", "BOXR", "RightHand", "RHLeft", "RHRight", "RHOut",
          "LeftEar", "RightEar", "SABL", "SABR", "Pellet", "Pillar")
N_FRAMES = 400


def _write_video(path, fps, n=8):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    assert out.isOpened()
    for i in range(n):
        out.write(np.full((32, 32, 3), i * 20 % 255, np.uint8))
    out.release()


def _write_pose(path):
    rows = {}
    for p in POINTS:
        x, y = {"Nose": (100.0, 50.0), "BOXL": (90.0, 70.0), "BOXR": (110.0, 70.0),
                "SABL": (80.0, 200.0), "SABR": (114.0, 200.0)}.get(p, (100.0, 150.0))
        rows[("s", p, "x")] = np.full(N_FRAMES, x)
        rows[("s", p, "y")] = np.full(N_FRAMES, y)
        rows[("s", p, "likelihood")] = np.ones(N_FRAMES)
    df = pd.DataFrame(rows)
    # One reach, frames 100..104, moving 1, 3, 2 and 4 px.
    for p in ("RightHand", "RHLeft", "RHRight", "RHOut"):
        df.loc[100:104, ("s", p, "y")] = [150.0, 151.0, 154.0, 156.0, 160.0]
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    df.to_hdf(path, key="df")


def _write_meta(pose_path, fps):
    meta = pose_path.with_name(pose_path.stem + "_meta.pickle")
    with open(meta, "wb") as f:
        pickle.dump({"data": {"fps": fps, "nframes": N_FRAMES}}, f)


def _bundle(tmp_path, video_fps=60.0, meta_fps=None):
    pose = tmp_path / POSE_NAME
    _write_pose(pose)
    if video_fps is not None:
        _write_video(tmp_path / f"{STEM}.mp4", video_fps)
    if meta_fps is not None:
        _write_meta(pose, meta_fps)
    reaches = tmp_path / f"{STEM}_reaches.json"
    reaches.write_text(json.dumps({
        "video_name": STEM, "total_frames": N_FRAMES, "n_segments": 1,
        "segments": [{"segment_num": 1, "start_frame": 40, "end_frame": 339,
                      "ruler_pixels": 34.0,
                      "reaches": [
                          {"reach_id": 1, "reach_num": 1, "start_frame": 100,
                           "apex_frame": 101, "end_frame": 104, "duration_frames": 5},
                          {"reach_id": 2, "reach_num": 2, "start_frame": 160,
                           "apex_frame": 161, "end_frame": 164, "duration_frames": 5}]}],
    }), encoding="utf-8")
    outcomes = tmp_path / f"{STEM}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"video_name": STEM, "segments": [
        {"segment_num": 1, "outcome": "untouched", "confidence": 1.0}]}), encoding="utf-8")
    return pose, reaches, outcomes


def _segment(feats):
    return feats.to_dict()["segments"][0]


def test_seconds_on_a_60_fps_video_are_frames_over_60(tmp_path):
    pose, reaches, outcomes = _bundle(tmp_path, video_fps=60.0)
    feats = FeatureExtractor().extract(pose, reaches, outcomes)
    seg = _segment(feats)
    assert feats.frame_rate == pytest.approx(60.0)
    assert feats.frame_rate_source.startswith("video header")
    assert seg["segment_duration_sec"] == pytest.approx(300 / 60.0)         # 40..339
    assert seg["time_to_first_reach_sec"] == pytest.approx(60 / 60.0)       # 40 -> 100
    assert seg["time_to_outcome_sec"] == pytest.approx(300 / 60.0)
    assert seg["mean_inter_reach_interval_sec"] == pytest.approx(56 / 60.0)  # 104 -> 160
    reach = seg["reaches"][0]
    assert reach["velocity_at_apex_px_per_frame"] == pytest.approx(3.0)
    assert reach["velocity_at_apex_mm_per_sec"] == pytest.approx(3.0 * 9.0 / 34.0 * 60.0)


def test_a_25_fps_video_gets_25(tmp_path):
    pose, reaches, outcomes = _bundle(tmp_path, video_fps=25.0)
    seg = _segment(FeatureExtractor().extract(pose, reaches, outcomes))
    assert seg["segment_duration_sec"] == pytest.approx(300 / 25.0)


def test_the_pose_record_stands_in_when_the_video_is_not_beside_the_pose(tmp_path):
    pose, reaches, outcomes = _bundle(tmp_path, video_fps=None, meta_fps=60.0)
    feats = FeatureExtractor().extract(pose, reaches, outcomes)
    assert feats.frame_rate == pytest.approx(60.0)
    assert feats.frame_rate_source.startswith("pose record")
    assert _segment(feats)["segment_duration_sec"] == pytest.approx(5.0)


def test_a_named_video_elsewhere_is_read(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    pose, reaches, outcomes = _bundle(work, video_fps=None)
    video = tmp_path / f"{STEM}.mp4"
    _write_video(video, 60.0)
    feats = FeatureExtractor().extract(pose, reaches, outcomes, video_path=video)
    assert feats.frame_rate == pytest.approx(60.0)


def test_with_no_rate_the_kinematics_are_not_written(tmp_path, capsys):
    # Filing means done for good: a video whose values in seconds cannot be
    # computed is held (KinematicsProblem), never filed with them missing.
    from mousereach.kinematics.core.feature_extractor import KinematicsProblem
    pose, reaches, outcomes = _bundle(tmp_path, video_fps=None)
    with pytest.raises(KinematicsProblem, match="no frame rate"):
        FeatureExtractor().extract(pose, reaches, outcomes)
    assert "no frame rate" in capsys.readouterr().out


def test_the_video_header_wins_over_a_disagreeing_pose_record(tmp_path, capsys):
    pose, _, _ = _bundle(tmp_path, video_fps=60.0, meta_fps=30.0)
    from mousereach.kinematics.core import frame_rate as fr
    rate, source = fr.resolve_frame_rate(pose)
    assert rate == pytest.approx(60.0) and source.startswith("video header")
    assert "using the video header" in capsys.readouterr().out


def test_a_stated_rate_is_used_and_an_impossible_one_refused(tmp_path):
    pose, reaches, outcomes = _bundle(tmp_path, video_fps=60.0)
    seg = _segment(FeatureExtractor(frame_rate=50.0).extract(pose, reaches, outcomes))
    assert seg["segment_duration_sec"] == pytest.approx(300 / 50.0)
    for bad in (0, -5, float("nan"), 90000):
        with pytest.raises(ValueError):
            FeatureExtractor(frame_rate=bad)


def test_the_rate_is_written_into_the_features_file(tmp_path):
    pose, reaches, outcomes = _bundle(tmp_path, video_fps=60.0)
    d = FeatureExtractor().extract(pose, reaches, outcomes).to_dict()
    assert d["frame_rate"] == pytest.approx(60.0)
    assert d["frame_rate_source"] == f"video header ({STEM}.mp4)"


def test_video_stem_of_pose():
    from mousereach.kinematics.core import frame_rate as fr
    assert fr.video_stem_of_pose(POSE_NAME) == STEM

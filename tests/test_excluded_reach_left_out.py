"""A reach a person marked "exclude from analysis" in the ground-truth tool is
left out of the kinematics, with the person's reason.

WHY: the exclusion removed only the ground-truth copy of the reach; the
detected reach over the same frames stayed and was measured like any other
(unless the reach set was marked complete), so the person's statement changed
no number. It now works like a reviewer's ignore window: every detected reach
it overlaps is left out and listed in the segment's ``reaches_left_out``.

Built under tmp_path with invented ids; no real folder or configuration.
"""
import pickle
import json

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as crio
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

VID = "20990101_ABC0902_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """One segment (0-99), detected reaches 1 (20-30) and 2 (50-60)."""
    monkeypatch.setattr(crio, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(crio, "durable_review_dir", lambda: None, raising=False)
    n = 100
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(n, v) + (np.arange(n) * 0.5 if c == "y" and p.startswith("R") else 0))
    pose = tmp_path / f"{VID}DLC_x.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    # The video's frame rate, recorded beside the pose as DeepLabCut does
    # (kinematics/core/frame_rate.py); without one no kinematics are written.
    with open(pose.with_name(pose.stem + "_meta.pickle"), "wb") as _f:
        pickle.dump({"data": {"fps": 60.0}}, _f)
    reaches = {"video_name": VID, "total_frames": n, "n_segments": 1,
               "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 99,
                             "ruler_pixels": 34.0, "n_reaches": 2,
                             "reaches": [
                                 {"reach_id": 1, "reach_num": 1, "start_frame": 20,
                                  "end_frame": 30, "apex_frame": 25, "duration_frames": 10},
                                 {"reach_id": 2, "reach_num": 2, "start_frame": 50,
                                  "end_frame": 60, "apex_frame": 55, "duration_frames": 10}]}]}
    (tmp_path / f"{VID}_reaches.json").write_text(json.dumps(reaches), encoding="utf-8")
    (tmp_path / f"{VID}_pellet_outcomes.json").write_text(json.dumps({
        "video_name": VID, "segments": [{"segment_num": 1, "outcome": "untouched",
                                         "start_frame": 0, "end_frame": 99}]}), encoding="utf-8")
    return tmp_path, pose


def _gt(folder, reaches, exhaustive=False):
    (folder / f"{VID}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": VID, "reaches": {"exhaustive": exhaustive, "reaches": reaches}}),
        encoding="utf-8")


def _excluded(s, e, reason="not a reach"):
    return {"reach_id": 7, "segment_num": 1, "start_frame": s, "end_frame": e,
            "exclude_from_analysis": True, "exclude_reason": reason}


def _segment(folder, pose):
    feats = FeatureExtractor().extract(pose, folder / f"{VID}_reaches.json",
                                       folder / f"{VID}_pellet_outcomes.json").to_dict()
    return feats["segments"][0]


def test_an_excluded_reach_takes_the_detected_reach_out_with_the_reason(run_dir):
    folder, pose = run_dir
    _gt(folder, [_excluded(18, 32, "tail in the slit")])
    seg = _segment(folder, pose)
    assert [r["reach_id"] for r in seg["reaches"]] == [2]
    assert seg["n_reaches"] == 1
    assert seg["n_reaches_left_out"] == 1
    out = seg["reaches_left_out"][0]
    assert (out["reach_id"], out["start_frame"], out["end_frame"]) == (1, 20, 30)
    assert out["excluded_by_person"]["reason"] == "tail in the slit"


def test_every_detected_reach_it_overlaps_is_left_out(run_dir):
    folder, pose = run_dir
    _gt(folder, [_excluded(25, 55)])
    seg = _segment(folder, pose)
    assert seg["reaches"] == []
    assert sorted(r["reach_id"] for r in seg["reaches_left_out"]) == [1, 2]


def test_with_a_complete_reach_set_the_exclusion_is_still_listed(run_dir):
    folder, pose = run_dir
    _gt(folder, [_excluded(20, 30),
                 {"reach_id": 8, "segment_num": 1, "start_frame": 50, "end_frame": 61,
                  "start_determined": True, "end_determined": True}], exhaustive=True)
    seg = _segment(folder, pose)
    assert [(r["start_frame"], r["end_frame"]) for r in seg["reaches"]] == [(50, 61)]
    assert [r["reach_id"] for r in seg["reaches_left_out"]] == [1]


def test_a_reach_nobody_excluded_is_measured(run_dir):
    folder, pose = run_dir
    _gt(folder, [_excluded(70, 80)])
    seg = _segment(folder, pose)
    assert [r["reach_id"] for r in seg["reaches"]] == [1, 2]
    assert seg["n_reaches_left_out"] == 0

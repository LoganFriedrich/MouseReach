"""The contact frame is attached to a causal reach only when it lies inside it.

WHY (owner decision D5-3, 2026-11-19): "contact frame doesn't matter. it's an
artifact" -- nothing may depend on it for a reach a person picked. The
algorithm's contact frame stayed on the pellet when a person named a different
causal reach, and was exported with that reach although it lay outside it.

Built under tmp_path with invented ids; no real folder or configuration.
"""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as crio
from mousereach.kinematics.core.feature_extractor import FeatureExtractor

VID = "20990101_ABC0901_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """One pellet (0-99), detected reaches 1 (20-30) and 2 (50-60); the
    algorithm's outcome file puts the contact at frame 25 (inside reach 1)."""
    monkeypatch.setattr(crio, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [tmp_path], raising=False)
    monkeypatch.setattr(crio, "durable_review_dir", lambda: None, raising=False)
    n = 100
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(n, v) + (np.arange(n) * 0.5 if c == "y" and p.startswith("R") else 0))
    pose = tmp_path / f"{VID}DLC_x.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    with open(pose.with_name(pose.stem + "_meta.pickle"), "wb") as f:
        pickle.dump({"data": {"fps": 60.0}}, f)
    reaches = {"video_name": VID, "total_frames": n, "n_segments": 1,
               "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 99,
                             "ruler_pixels": 34.0, "n_reaches": 2, "reaches": [
                                 {"reach_id": 1, "reach_num": 1, "start_frame": 20,
                                  "end_frame": 30, "apex_frame": 25, "duration_frames": 11},
                                 {"reach_id": 2, "reach_num": 2, "start_frame": 50,
                                  "end_frame": 60, "apex_frame": 55, "duration_frames": 11}]}]}
    (tmp_path / f"{VID}_reaches.json").write_text(json.dumps(reaches), encoding="utf-8")
    (tmp_path / f"{VID}_pellet_outcomes.json").write_text(json.dumps({
        "video_name": VID, "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 99,
                                         "outcome": "retrieved", "interaction_frame": 25}]}),
        encoding="utf-8")
    return tmp_path, pose


def _review(folder, human, answers=None):
    (folder / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "segments": [{
            "segment_num": 1, "segment_span": {"start": 0, "end": 99},
            "human": human, "answers": answers or {"reviewed": True},
            "reviewer": "reviewer_a"}]}), encoding="utf-8")


def _features(folder, pose):
    return FeatureExtractor().extract(
        pose, folder / f"{VID}_reaches.json", folder / f"{VID}_pellet_outcomes.json",
        review_path=folder / f"{VID}_causal_review.json").to_dict()["segments"][0]


def _by_id(seg):
    return {r["reach_id"]: r for r in seg["reaches"]}


def test_a_person_picked_reach_never_carries_another_reachs_contact_frame(run_dir):
    folder, pose = run_dir
    _review(folder, {"outcome": "retrieved",
                     "causal_reach": {"reach_id": 2, "start": 50, "end": 60}})
    r = _by_id(_features(folder, pose))
    assert r[2]["causal_reach"] and r[2]["causal_reach_source"] == "human_review"
    assert r[2]["interaction_frame"] is None, "frame 25 is in reach 1, not reach 2"
    assert r[2]["distance_to_interaction"] is None

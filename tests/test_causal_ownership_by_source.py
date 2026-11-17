"""Who owns a segment's causal reach in the kinematics: a PERSON, never the mere
presence of a ``causal_reach_id`` key in the outcome file.

WHY: the extractor merges the reach assignment's (algo-4) causal reach into
every segment no person decided. It used to decide "a person decided this" by
whether the outcome segment carried a ``causal_reach_id`` key at all. The
legacy outcome detector writes that key on every segment, so an old file's
guess blocked the reach assignment's answer and shipped as if a person had
made it. Ownership now comes from who the truth layering says decided the
segment (outcome_source, causal_reach_source) or an in-file human resolution
(clear_guard.is_outcome_locked).

Built under tmp_path with invented ids; no real folder or configuration.
"""
import json

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
    """One segment (0-99), two detected reaches (1: 20-30, 2: 50-60), and a
    reach assignment that names reach 2 as the causal reach."""
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
    reaches = {"video_name": VID, "total_frames": n, "n_segments": 1,
               "segments": [{"segment_num": 1, "start_frame": 0, "end_frame": 99,
                             "ruler_pixels": 34.0, "n_reaches": 2,
                             "reaches": [
                                 {"reach_id": 1, "reach_num": 1, "start_frame": 20,
                                  "end_frame": 30, "apex_frame": 25, "duration_frames": 10},
                                 {"reach_id": 2, "reach_num": 2, "start_frame": 50,
                                  "end_frame": 60, "apex_frame": 55, "duration_frames": 10}]}]}
    (tmp_path / f"{VID}_reaches.json").write_text(json.dumps(reaches), encoding="utf-8")
    (tmp_path / f"{VID}_reach_assignments.json").write_text(json.dumps({
        "reaches": [{"reach_id": 2, "segment_num": 1, "is_causal": True, "segment_ifr": 55},
                    {"reach_id": 1, "segment_num": 1, "is_causal": False}]}), encoding="utf-8")
    return tmp_path, pose


def _outcomes(folder, **seg):
    s = {"segment_num": 1, "start_frame": 0, "end_frame": 99, "outcome": "retrieved"}
    s.update(seg)
    (folder / f"{VID}_pellet_outcomes.json").write_text(
        json.dumps({"video_name": VID, "segments": [s]}), encoding="utf-8")


def _segment(folder, pose):
    feats = FeatureExtractor().extract(pose, folder / f"{VID}_reaches.json",
                                       folder / f"{VID}_pellet_outcomes.json").to_dict()
    seg = feats["segments"][0]
    causal = [r["reach_id"] for r in seg["reaches"] if r.get("causal_reach")]
    return seg, causal


def test_a_legacy_outcome_files_causal_key_does_not_block_the_assignment(run_dir):
    folder, pose = run_dir
    # The legacy detector named reach 1; no person looked at this segment.
    _outcomes(folder, causal_reach_id=1)
    seg, causal = _segment(folder, pose)
    assert seg["causal_reach_id"] == 2
    assert causal == [2]


def test_an_in_file_human_resolution_keeps_its_reach(run_dir):
    folder, pose = run_dir
    _outcomes(folder, causal_reach_id=1, human_verified=True)
    seg, causal = _segment(folder, pose)
    assert seg["causal_reach_id"] == 1
    assert causal == [1]


def test_a_reviewers_untouched_none_stays_none(run_dir, monkeypatch):
    folder, pose = run_dir
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [folder])
    _outcomes(folder)
    (folder / f"{VID}_causal_review.json").write_text(json.dumps({
        "video_name": VID, "segments": [{
            "segment_num": 1, "segment_span": {"start": 0, "end": 99},
            "human": {"outcome": "untouched"}, "answers": {}}]}), encoding="utf-8")
    fx = FeatureExtractor()
    feats = fx.extract(pose, folder / f"{VID}_reaches.json",
                       folder / f"{VID}_pellet_outcomes.json",
                       review_path=folder / f"{VID}_causal_review.json").to_dict()
    seg = feats["segments"][0]
    assert seg["outcome"] == "untouched"
    assert seg["causal_reach_id"] is None
    assert not any(r.get("causal_reach") for r in seg["reaches"])


def _gt_touched_outside_every_reach(folder):
    """Ground truth: retrieved, interaction frame 90 -- inside no reach, so the
    truth layering cannot find the causal reach by frames."""
    (folder / f"{VID}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": VID,
        "reaches": {"exhaustive": False, "reaches": []},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 90,
             "determined": True}]}}), encoding="utf-8")


def test_ground_truth_without_a_reach_at_its_frame_gets_the_assignment(run_dir):
    folder, pose = run_dir
    _outcomes(folder)
    _gt_touched_outside_every_reach(folder)
    seg, causal = _segment(folder, pose)
    assert seg["outcome"] == "retrieved"
    assert seg["causal_reach_id"] == 2
    assert causal == [2]


def test_ground_truth_does_not_turn_a_legacy_pick_into_a_persons(run_dir):
    folder, pose = run_dir
    _outcomes(folder, causal_reach_id=1)          # the legacy detector's guess
    _gt_touched_outside_every_reach(folder)
    seg, causal = _segment(folder, pose)
    assert seg["causal_reach_id"] == 2
    assert causal == [2]


def test_ground_truths_reach_at_its_interaction_frame_is_kept(run_dir):
    folder, pose = run_dir
    _outcomes(folder)
    (folder / f"{VID}_unified_ground_truth.json").write_text(json.dumps({
        "video_name": VID,
        "reaches": {"exhaustive": False, "reaches": []},
        "outcomes": {"exhaustive": False, "segments": [
            {"segment_num": 1, "outcome": "retrieved", "interaction_frame": 25,
             "determined": True}]}}), encoding="utf-8")
    seg, causal = _segment(folder, pose)
    assert seg["causal_reach_id"] == 1             # the reach holding frame 25
    assert causal == [1]

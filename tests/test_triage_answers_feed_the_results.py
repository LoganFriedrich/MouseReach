"""What a person said in triage is what the results use.

WHY (owner decisions 2026-11-19, D5-6 extension and D5-3): what a human said
is the correct result for every step after it, and nothing downstream works
out again what a person decided.

  * A causal answer includes the person checking that reach's start and end:
    the reach is measured over the person's frames.
  * Naming the causal reach answers every other reach of the pellet -- they
    all missed ("untouched": every reach missed). Each reach carries what it
    did (reach_outcome) and who settled it (reach_outcome_source).
  * The contact frame is an artifact: it is attached to the causal reach only
    when it lies inside that reach, so a person's pick never carries the
    algorithm's contact frame from another reach.

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


def test_the_causal_reach_is_measured_over_the_frames_the_person_checked(run_dir):
    folder, pose = run_dir
    _review(folder, {"outcome": "retrieved",
                     "causal_reach": {"reach_id": 2, "start": 50, "end": 60},
                     "causal_reach_bounds": {"start": 48, "end": 63}})
    r = _by_id(_features(folder, pose))
    assert (r[2]["start_frame"], r[2]["end_frame"]) == (48, 63)
    assert r[2]["duration_frames"] == 16 and r[2]["reach_source"] == "human_review"
    assert r[2]["causal_reach"]
    # A record older than the checked frames: the detector's frames stand.
    _review(folder, {"outcome": "retrieved",
                     "causal_reach": {"reach_id": 2, "start": 50, "end": 60}})
    r = _by_id(_features(folder, pose))
    assert (r[2]["start_frame"], r[2]["end_frame"]) == (50, 60)


def test_naming_the_causal_reach_makes_every_other_reach_a_known_miss(run_dir):
    folder, pose = run_dir
    _review(folder, {"outcome": "retrieved",
                     "causal_reach": {"reach_id": 2, "start": 50, "end": 60}})
    r = _by_id(_features(folder, pose))
    assert (r[2]["reach_outcome"], r[2]["reach_outcome_source"]) == ("retrieved", "human_review")
    assert (r[1]["reach_outcome"], r[1]["reach_outcome_source"]) == ("missed", "human_review")


def test_untouched_makes_every_reach_a_known_miss(run_dir):
    folder, pose = run_dir
    _review(folder, {"outcome": "untouched"})
    r = _by_id(_features(folder, pose))
    assert {(x["reach_outcome"], x["reach_outcome_source"]) for x in r.values()} == \
        {("missed", "human_review")}


def test_reach_outcome_rules():
    from mousereach.kinematics.core.feature_extractor import reach_outcome_of
    a, b = {"reach_id": 1, "start_frame": 20, "end_frame": 30}, \
        {"reach_id": 2, "start_frame": 50, "end_frame": 60}
    algo = {"outcome": "displaced_sa", "causal_reach_id": 2,
            "causal_reach_source": "reach_assignment", "outcome_source": "algo"}
    assert reach_outcome_of(b, algo) == ("displaced_sa", "reach_assignment")
    assert reach_outcome_of(a, algo) == ("missed", "reach_assignment")
    # A reach a person's record lists as a miss is a person's miss.
    assert reach_outcome_of(a, dict(algo, person_missed_reaches=[[20, 30]])) == \
        ("missed", "human_review")
    # Touched, causal reach not known: nothing is known about the reaches.
    assert reach_outcome_of(a, {"outcome": "retrieved", "causal_reach_id": None}) == (None, None)


def test_record_fields_the_results_read():
    """The field names the review window writes and the results read."""
    from mousereach.review.truth_resolver import person_missed_reaches, person_reach_bounds
    rec = {"human": {"outcome": "retrieved",
                     "causal_reach": {"reach_id": 2, "start": 50, "end": 60},
                     "causal_reach_bounds": {"start": 63, "end": 48},
                     "reach_calls": [
                         {"reach_id": 1, "start": 20, "end": 30, "call": "miss",
                          "detected": True, "bounds": "detected"},
                         {"reach_id": 2, "start": 48, "end": 63, "call": "causal",
                          "detected": True, "bounds": "human"}]}}
    assert person_reach_bounds(rec) == (48, 63)
    assert person_missed_reaches(rec) == [(20, 30)]
    only_calls = {"human": {"reach_calls": rec["human"]["reach_calls"]}}
    assert person_reach_bounds(only_calls) == (48, 63)
    assert person_reach_bounds({"human": {"causal_reach": {"start": 1, "end": 2}}}) is None

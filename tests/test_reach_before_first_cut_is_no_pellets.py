"""A reach that starts before the first cut belongs to NO pellet.

WHY: before the first cut no pellet is presented, so a
reach then cannot be one of pellet 1's. The reach step attached such a reach
to the nearest segment (segment 1) and flagged the segment with a flag nothing
read, so the reach was counted, measured, assigned and exported as pellet 1's.
Now the reach step lists it apart (``reaches_before_first_cut``) and every
reader that maps reaches to pellets keeps it out of pellet 1 -- also when an
older file still carries it inside segment 1.

Built under tmp_path with invented ids; no real folder, database or settings.
"""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

VID = "20990101_ABC0912_P1"
CUT1, CUT2, N = 100, 200, 300


def _reach(rid, s, e, num=1):
    return {"reach_id": rid, "reach_num": num, "start_frame": s, "end_frame": e,
            "apex_frame": s + 2, "duration_frames": e - s + 1}


def _old_reaches_file():
    """A reaches file as an older reach step wrote it: reach 1 starts before the
    first cut and was put into segment 1, which was flagged for it."""
    return {"video_name": VID, "total_frames": N, "n_segments": 2, "segments_flagged": 1,
            "segments": [
                {"segment_num": 1, "start_frame": CUT1, "end_frame": CUT2, "ruler_pixels": 34.0,
                 "n_reaches": 2, "flagged_for_review": True,
                 "flag_reason": "reach_start_outside_all_segments",
                 "reaches": [dict(_reach(1, 40, 60), review_note="reach_start_outside_all_segments"),
                             _reach(2, 120, 130, 2)]},
                {"segment_num": 2, "start_frame": CUT2, "end_frame": N, "ruler_pixels": 34.0,
                 "n_reaches": 1, "reaches": [_reach(3, 220, 230)]}]}


def test_an_old_file_is_read_with_the_reach_outside_pellet_1():
    from mousereach.reach.core.before_first_cut import split_before_first_cut
    doc = _old_reaches_file()
    out = split_before_first_cut(doc)
    seg1 = out["segments"][0]
    assert [r["reach_id"] for r in seg1["reaches"]] == [2]
    assert seg1["n_reaches"] == 1 and seg1["reaches"][0]["reach_num"] == 1
    assert not seg1["flagged_for_review"] and seg1["flag_reason"] is None
    assert [r["reach_id"] for r in out["reaches_before_first_cut"]] == [1]
    assert out["segments_flagged"] == 0
    assert len(doc["segments"][0]["reaches"]) == 2, "the file read is not changed"


def _pose_df():
    cols = {}
    for p in ("Nose", "BOXR", "SABL", "SABR", "RightHand"):
        cols[f"{p}_x"] = np.full(N, 100.0)
        cols[f"{p}_y"] = np.full(N, 150.0) + (np.arange(N) * 0.5 if p == "RightHand" else 0)
        cols[f"{p}_likelihood"] = np.full(N, 1.0)
    cols["SABR_x"] = np.full(N, 140.0)
    return pd.DataFrame(cols)


def test_the_reach_step_lists_it_apart_and_flags_nothing():
    from mousereach.reach.core.span_to_reaches import build_video_reaches, segments_from_boundaries
    from dataclasses import asdict
    vr = build_video_reaches(_pose_df(), segments_from_boundaries([CUT1, CUT2], N),
                             [(40, 60), (120, 130), (220, 230)], VID, N)
    d = asdict(vr)
    assert [r["start_frame"] for r in d["reaches_before_first_cut"]] == [40]
    assert [r["start_frame"] for r in d["segments"][0]["reaches"]] == [120]
    assert not d["segments"][0]["flagged_for_review"] and d["segments_flagged"] == 0
    # Ids still run in time order through the video.
    assert d["reaches_before_first_cut"][0]["reach_id"] == 1
    assert d["segments"][0]["reaches"][0]["reach_id"] == 2


def test_reach_assignment_gives_it_no_pellet(monkeypatch):
    import mousereach.assignment.v2.assign as v2
    import mousereach.lib.dlc_cleaning as clean
    import mousereach.lib.pillar_geometry as pg
    from mousereach.assignment.cli import _reaches_list
    monkeypatch.setattr(clean, "clean_dlc_bodyparts", lambda df: df)
    monkeypatch.setattr(pg, "compute_pillar_geometry_series", lambda df: None)
    monkeypatch.setattr(v2, "_compute_pellet_radius_series", lambda df, g: np.zeros(N))
    segs = [{"segment_num": 1, "start_frame": CUT1, "end_frame": CUT2 - 1, "outcome": "untouched"},
            {"segment_num": 2, "start_frame": CUT2, "end_frame": N - 1, "outcome": "untouched"}]
    out = v2.assign_reaches_v2(reaches=_reaches_list(_old_reaches_file()),
                               segments_with_outcomes=segs, dlc_df=pd.DataFrame(), video_id=VID)
    by_id = {r["reach_id"]: r for r in out["reaches"]}
    assert by_id[1]["segment_num"] is None and by_id[1]["label"] == "unassigned"
    assert by_id[1]["before_first_cut"] is True
    assert by_id[2]["segment_num"] == 1 and by_id[2]["label"] == "miss"
    # Even a reach that still says "segment 1" and ends after the cut.
    out = v2.assign_reaches_v2(
        reaches=[{"reach_id": 9, "start_frame": 90, "end_frame": 140, "segment_num": 1}],
        segments_with_outcomes=segs, dlc_df=pd.DataFrame(), video_id=VID)
    assert out["reaches"][0]["segment_num"] is None


def test_outcome_detection_input_is_unchanged():
    """The cascade still receives every reach (it places reaches by its own
    windows, which start at the first cut), so its calls do not change."""
    from mousereach.outcomes.core.batch import _extract_reaches
    from mousereach.reach.core.before_first_cut import split_before_first_cut
    old = _old_reaches_file()
    assert sorted(_extract_reaches(split_before_first_cut(old))) == sorted(_extract_reaches(old))


def test_ground_truth_reach_before_the_first_cut_joins_no_pellet():
    from mousereach.review.truth_resolver import _apply_gt_reaches
    reaches = {"video_name": VID, "segments": [
        {"segment_num": 1, "start_frame": CUT1, "end_frame": CUT2, "reaches": [_reach(2, 120, 130)]},
        {"segment_num": 2, "start_frame": CUT2, "end_frame": N, "reaches": []}]}
    gt = {"reaches": {"exhaustive": False, "reaches": [
        {"reach_id": 7, "segment_num": 1, "start_frame": 40, "end_frame": 60,
         "start_determined": True, "end_determined": True}]}}
    out = _apply_gt_reaches(reaches, gt)
    assert [r["start_frame"] for r in out["segments"][0]["reaches"]] == [120]
    assert [r["start_frame"] for r in out["reaches_before_first_cut"]] == [40]


def test_filed_decisions_do_not_count_it_as_a_change():
    from mousereach.kinematics.core.filed_decisions import (changes_not_owed,
                                                            decisions_of_features)
    filed = {"segments": [{"segment_num": 1, "start_frame": CUT1, "outcome": "untouched",
                           "causal_reach_id": None,
                           "reaches": [_reach(1, 40, 60), _reach(2, 120, 130)]}]}
    fresh = {"segments": [{"segment_num": 1, "start_frame": CUT1, "outcome": "untouched",
                           "causal_reach_id": None, "reaches": [_reach(2, 120, 130)]}]}
    assert changes_not_owed(decisions_of_features(filed), decisions_of_features(fresh)) == []


def test_exports_read_a_filed_features_file_without_it_in_pellet_1(tmp_path):
    from mousereach.export.resolved import load_resolved_videos
    from mousereach.kinematics.analysis.reach_export import _emit_features_video
    feats = {"video_name": VID, "segments": [
        {"segment_num": 1, "start_frame": CUT1, "end_frame": CUT2 - 1, "outcome": "untouched",
         "n_reaches": 2, "reaches": [dict(_reach(1, 40, 60), is_first_reach=True),
                                     dict(_reach(2, 120, 130), is_first_reach=False)]}]}
    fp = tmp_path / f"{VID}_features.json"
    fp.write_text(json.dumps(feats), encoding="utf-8")
    videos, _ = load_resolved_videos(tmp_path)
    seg1 = videos[0]["segments"][0]
    assert [r["reach_id"] for r in seg1["reaches"]] == [2] and seg1["n_reaches"] == 1
    assert seg1["reaches"][0]["is_first_reach"] is True
    assert len(_emit_features_video(fp)) == 1


# ---------------------------------------------------------------------------
# The kinematics step itself
# ---------------------------------------------------------------------------
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


@pytest.fixture
def proc(tmp_path, monkeypatch):
    import mousereach.review.causal_review_io as crio
    from mousereach.config import Paths
    monkeypatch.setattr(crio, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(crio, "review_dirs", lambda *a, **k: [], raising=False)
    monkeypatch.setattr(crio, "durable_review_dir", lambda: None, raising=False)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "triage")
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "deep_review")
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(N, v) + (np.arange(N) * 0.5 if c == "y" and p.startswith("R") else 0))
    pose = proc / f"{VID}DLC_x.h5"
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(pose, key="df")
    with open(pose.with_name(pose.stem + "_meta.pickle"), "wb") as f:
        pickle.dump({"data": {"fps": 60.0}}, f)
    return proc


def test_kinematics_measures_it_for_no_pellet_and_lists_it(proc):
    from mousereach.kinematics.core.feature_extractor import FeatureExtractor
    (proc / f"{VID}_reaches.json").write_text(json.dumps(_old_reaches_file()), encoding="utf-8")
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"video_id": VID, "segments": [
        {"segment_num": 1, "outcome": "untouched"}, {"segment_num": 2, "outcome": "untouched"}]}),
        encoding="utf-8")
    vf = FeatureExtractor().extract(proc / f"{VID}DLC_x.h5", proc / f"{VID}_reaches.json",
                                    proc / f"{VID}_pellet_outcomes.json")
    seg1 = vf.segments[0]
    assert [r.reach_id for r in seg1.reaches] == [2] and seg1.n_reaches == 1
    assert seg1.reaches[0].is_first_reach
    assert [r["reach_id"] for r in vf.reaches_before_first_cut] == [1]


def test_a_reach_a_person_credited_stays_with_its_pellet():
    """What a person said is correct: a person who named a reach starting just
    before the first cut as pellet 1's causal reach said it is pellet 1's."""
    from mousereach.reach.core.before_first_cut import split_before_first_cut
    feats = {"segments": [{"segment_num": 1, "start_frame": CUT1, "outcome": "retrieved",
                           "reaches": [dict(_reach(1, 95, 110), causal_reach=True,
                                            outcome_source="human_review"),
                                       _reach(2, 120, 130, 2)]}]}
    out = split_before_first_cut(feats)
    assert [r["reach_id"] for r in out["segments"][0]["reaches"]] == [1, 2]
    algo = {"segments": [{"segment_num": 1, "start_frame": CUT1, "outcome": "retrieved",
                          "reaches": [dict(_reach(1, 95, 110), causal_reach=True,
                                           causal_reach_source="reach_assignment"),
                                      _reach(2, 120, 130, 2)]}]}
    assert [r["reach_id"] for r in split_before_first_cut(algo)["segments"][0]["reaches"]] == [2]


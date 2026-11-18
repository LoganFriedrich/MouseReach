"""One frame convention: a window is its first and last frame, both included.

WHY: the reaches file stores each segment's end_frame as the frame AFTER its
last one (the next cut). The kinematics read it as the last frame, so every
segment-level measurement took in the next pellet's first frame and every
pellet was one frame too long. Reach durations written by the reach review
window and by the results step for ground-truth reaches were end - start, one
frame short of the detector's end - start + 1. And the reach review window's
ground-truth export read original_start_frame / original_end_frame, which
nothing writes. These tests pin the convention at every one of those readers.
Built under tmp_path.
"""
import json

import cv2
import numpy as np
import pandas as pd
import pytest

from mousereach.kinematics.core.feature_extractor import FeatureExtractor
from mousereach.reach.core.span_to_reaches import segments_from_boundaries
from mousereach.review.causal_review_io import (pellet_span, reach_duration_frames,
                                                reach_file_segment_span)
from mousereach.review.truth_resolver import _apply_gt_reaches, _gt_reach_dict

STEM = "20990101_TEST0101_P1"
POSE_NAME = f"{STEM}DLC_resnet50_testshuffle1_100.h5"
POINTS = ("Nose", "BOXL", "BOXR", "RightHand", "RHLeft", "RHRight", "RHOut",
          "LeftEar", "RightEar", "SABL", "SABR", "Pellet", "Pillar", "Reference")
N_FRAMES = 400
CUT = 200            # pellet 1 is frames 40..199; the next pellet starts at 200
FPS = 60.0


# --- the convention itself ---------------------------------------------------------

def test_a_reach_file_segment_is_exactly_the_pellet_of_the_cuts():
    cuts = [40, 200, 333, 371]
    segs = segments_from_boundaries(cuts, N_FRAMES)
    for i in range(len(cuts) - 1):
        num, start, end = segs[i]
        assert reach_file_segment_span({"start_frame": start, "end_frame": end}) \
            == pellet_span(cuts, i)
    # The stretch after the last cut runs to the video's last frame.
    _, start, end = segs[-1]
    assert reach_file_segment_span({"start_frame": start, "end_frame": end}) \
        == (371, N_FRAMES - 1)


def test_a_reach_duration_counts_both_ends():
    assert reach_duration_frames({"start_frame": 100, "end_frame": 104}) == 5
    assert reach_duration_frames({"start_frame": 100, "end_frame": 100}) == 1
    assert reach_duration_frames({"start_frame": 100}) is None


# --- the kinematics ----------------------------------------------------------------

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
    df.loc[100:104, "RightHand_x"] = [100.0, 102.0, 101.0, 99.0, 100.0]
    df.loc[100:104, "RightHand_y"] = [150.0, 151.0, 154.0, 156.0, 160.0]
    # The NEXT pellet's first frame: a confident paw far down. Taken into pellet
    # 1's window it moves the legacy_swipe threshold line to 455 - 300 = 155 and
    # drops three of the reach's five frames.
    df.loc[CUT, "RightHand_y"] = 455.0
    return df


def _write_video(path, fps, n=8):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    assert out.isOpened()
    for i in range(n):
        out.write(np.full((32, 32, 3), i * 20 % 255, np.uint8))
    out.release()


def _bundle(tmp_path, stored_duration=5):
    flat = _pose_table()
    df = pd.DataFrame({("s",) + tuple(c.rsplit("_", 1)): flat[c] for c in flat.columns})
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    pose = tmp_path / POSE_NAME
    df.to_hdf(pose, key="df")
    _write_video(tmp_path / f"{STEM}.mp4", FPS)
    # Exactly what span_to_reaches writes for cuts [40, 200]: segment 1 ends AT
    # the next cut, segment 2 is the stretch after the last cut.
    reaches = tmp_path / f"{STEM}_reaches.json"
    reaches.write_text(json.dumps({
        "video_name": STEM, "total_frames": N_FRAMES, "n_segments": 2,
        "segments": [
            {"segment_num": 1, "start_frame": 40, "end_frame": CUT, "ruler_pixels": 34.0,
             "reaches": [{"reach_id": 1, "reach_num": 1, "start_frame": 100,
                          "apex_frame": 104, "end_frame": 104,
                          "duration_frames": stored_duration}]},
            {"segment_num": 2, "start_frame": CUT, "end_frame": N_FRAMES,
             "ruler_pixels": 34.0, "reaches": []}],
    }), encoding="utf-8")
    outcomes = tmp_path / f"{STEM}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"video_name": STEM, "segments": [
        {"segment_num": 1, "start_frame": 40, "end_frame": CUT - 1,
         "outcome": "untouched", "confidence": 1.0}]}), encoding="utf-8")
    return pose, reaches, outcomes


def test_segment_windows_stop_before_the_next_pellet(tmp_path, monkeypatch):
    seen = {}
    for name in ("_compute_attention_score", "_compute_body_features_segment",
                 "_compute_quality_metrics"):
        real = getattr(FeatureExtractor, name)

        def spy(self, df, start, end, *rest, _n=name, _real=real):
            seen[_n] = (start, end)
            return _real(self, df, start, end, *rest)
        monkeypatch.setattr(FeatureExtractor, name, spy)
    pose, reaches, outcomes = _bundle(tmp_path)
    d = json.loads(json.dumps(FeatureExtractor().extract(pose, reaches, outcomes).to_dict()))
    assert seen == {n: (40, CUT - 1) for n in seen} and len(seen) == 3
    seg = d["segments"][0]
    # The features file's segment end is the pellet's last frame, as in the
    # outcomes file and pellet_span.
    assert (seg["start_frame"], seg["end_frame"]) == (40, CUT - 1)


def test_segment_duration_is_exact(tmp_path):
    pose, reaches, outcomes = _bundle(tmp_path)
    seg = FeatureExtractor().extract(pose, reaches, outcomes).to_dict()["segments"][0]
    assert seg["segment_duration_sec"] == (CUT - 40) / FPS          # 160 frames
    assert seg["time_to_outcome_sec"] == (CUT - 40) / FPS


def test_the_legacy_swipe_pellet_window_leaves_out_the_next_pellet(tmp_path):
    pose, reaches, outcomes = _bundle(tmp_path)
    r = FeatureExtractor().extract(pose, reaches, outcomes).to_dict()["segments"][0]["reaches"][0]
    # All five reach frames count: frame 200 is not in pellet 1's window.
    assert r["extended"]["legacy_swipe_duration_frames"] == 5
    assert r["extended"]["legacy_swipe_first_frame"] == 100


def test_a_reach_duration_is_counted_from_its_frames(tmp_path):
    # A stored duration one short (an older review tool wrote end - start).
    pose, reaches, outcomes = _bundle(tmp_path, stored_duration=4)
    d = FeatureExtractor().extract(pose, reaches, outcomes).to_dict()
    r = d["segments"][0]["reaches"][0]
    assert (r["start_frame"], r["end_frame"]) == (100, 104)       # frames unchanged
    assert r["duration_frames"] == 5
    assert d["summary"]["mean_duration_frames"] == 5.0


def test_the_session_only_file_uses_the_convention(tmp_path):
    pose, reaches, _ = _bundle(tmp_path, stored_duration=4)
    doc = json.loads(reaches.read_text(encoding="utf-8"))
    doc["segments"] = [{"segment_num": 1, "start_frame": 0, "end_frame": N_FRAMES,
                        "ruler_pixels": 34.0, "reaches": doc["segments"][0]["reaches"]}]
    reaches.write_text(json.dumps(doc), encoding="utf-8")
    d = FeatureExtractor().extract_session_only(pose, reaches).to_dict()
    seg = d["segments"][0]
    assert (seg["start_frame"], seg["end_frame"]) == (0, N_FRAMES - 1)
    assert seg["reaches"][0]["duration_frames"] == 5


def _pellet_seen_only_at(df, frame):
    """The pellet is tracked (likelihood 1) in ONE frame only: a window that
    holds that frame has pellet-position values, one that does not has none."""
    df["Pellet_x"], df["Pellet_y"] = 130.0, 70.0
    df["Pellet_likelihood"] = 0.0
    df.loc[frame, "Pellet_likelihood"] = 1.0
    return df


def test_the_pellet_window_ends_before_the_first_reach():
    # The first reach starts at 100, so frame 100 is the reach, not "before" it.
    df = _pellet_seen_only_at(_pose_table(), 100)
    got = FeatureExtractor()._compute_pellet_positioning(df, 40, 100, 34.0, seg_last=CUT - 1)
    assert got["pellet_position_idealness"] is None
    df = _pellet_seen_only_at(_pose_table(), 99)
    got = FeatureExtractor()._compute_pellet_positioning(df, 40, 100, 34.0, seg_last=CUT - 1)
    assert got["pellet_position_idealness"] is not None


def test_a_pellet_with_no_reach_uses_its_first_30_frames():
    # Frames 40..69 are the first 30; frame 70 is the 31st.
    df = _pellet_seen_only_at(_pose_table(), 70)
    got = FeatureExtractor()._compute_pellet_positioning(df, 40, None, 34.0, seg_last=CUT - 1)
    assert got["pellet_position_idealness"] is None
    df = _pellet_seen_only_at(_pose_table(), 69)
    got = FeatureExtractor()._compute_pellet_positioning(df, 40, None, 34.0, seg_last=CUT - 1)
    assert got["pellet_position_idealness"] is not None


def test_the_no_reach_window_never_runs_into_the_next_pellet():
    # A pellet of 5 frames (200..204): frame 205 is the next pellet's.
    df = _pellet_seen_only_at(_pose_table(), 205)
    got = FeatureExtractor()._compute_pellet_positioning(df, 200, None, 34.0, seg_last=204)
    assert got["pellet_position_idealness"] is None


def test_the_extractor_bounds_the_pellet_window_by_the_pellet(tmp_path, monkeypatch):
    seen = []
    real = FeatureExtractor._compute_pellet_positioning

    def spy(self, df, start, first_reach, ruler, *a, **k):
        seen.append((start, first_reach, k.get("seg_last")))
        return real(self, df, start, first_reach, ruler, *a, **k)
    monkeypatch.setattr(FeatureExtractor, "_compute_pellet_positioning", spy)
    pose, reaches, outcomes = _bundle(tmp_path)
    FeatureExtractor().extract(pose, reaches, outcomes)
    assert seen == [(40, 100, CUT - 1)]


def test_a_features_file_segment_is_read_by_its_version():
    from mousereach.review.causal_review_io import features_segment_span
    seg = {"start_frame": 40, "end_frame": 200}
    assert features_segment_span(seg, "2.4.0") == (40, 199)   # stored the next cut
    assert features_segment_span(seg, "2.6.0") == (40, 200)   # stores the last frame
    assert features_segment_span(seg, None) == (40, 200)


# --- ground truth (the results step) --------------------------------------------------

def test_a_ground_truth_reach_duration_is_exact():
    r = _gt_reach_dict({"reach_id": 7, "start_frame": 100, "end_frame": 104}, 1)
    assert (r["start_frame"], r["end_frame"], r["duration_frames"]) == (100, 104, 5)


def test_a_ground_truth_reach_goes_to_the_segment_holding_most_of_its_frames():
    # Reach 199..201 straddles the cut at 200: one frame in pellet 1, two in
    # pellet 2. Read as inclusive, the reaches file's stored end (200) gave
    # pellet 1 two frames too, a tie, and the tie went to pellet 1.
    reaches = {"video_name": STEM, "total_frames": N_FRAMES, "segments": [
        {"segment_num": 1, "start_frame": 40, "end_frame": CUT, "reaches": []},
        {"segment_num": 2, "start_frame": CUT, "end_frame": N_FRAMES, "reaches": []}]}
    gt = {"reaches": {"exhaustive": False, "reaches": [
        {"reach_id": 1, "segment_num": 1, "start_frame": 199, "end_frame": 201,
         "start_determined": True, "end_determined": True}]}}
    out = _apply_gt_reaches(reaches, gt)
    placed = {s["segment_num"]: [(r["start_frame"], r["end_frame"]) for r in s["reaches"]]
              for s in out["segments"]}
    assert placed == {1: [], 2: [(199, 201)]}
    assert out["segments"][1]["reaches"][0]["duration_frames"] == 3


# --- the reach review window --------------------------------------------------------

@pytest.fixture
def widget_module():
    pytest.importorskip("napari")
    from mousereach.reach import review_widget
    return review_widget


def test_moving_a_reach_keeps_an_exact_duration(widget_module):
    r = {"reach_id": 1, "start_frame": 100, "end_frame": 104, "duration_frames": 5,
         "source": "algorithm"}
    widget_module.move_reach_end(r, 110)
    assert r["duration_frames"] == 11
    widget_module.move_reach_start(r, 105)
    assert r["duration_frames"] == 6


def test_both_originals_are_kept_whichever_end_moves_first(widget_module):
    r = {"reach_id": 1, "start_frame": 100, "end_frame": 104, "source": "algorithm"}
    widget_module.move_reach_end(r, 110)
    widget_module.move_reach_start(r, 98)
    widget_module.move_reach_start(r, 97)          # a second move keeps the algorithm's
    assert (r["original_start"], r["original_end"]) == (100, 104)
    assert r["human_corrected"] is True


def test_the_ground_truth_export_carries_the_original_frames(widget_module):
    r = {"reach_id": 1, "reach_num": 1, "start_frame": 100, "end_frame": 104,
         "source": "algorithm"}
    widget_module.move_reach_start(r, 98)
    g = widget_module.gt_reach_record(r, "tester", "2099-01-01T00:00:00")
    assert g["original_start_frame"] == 100
    assert g["original_end_frame"] is None          # the end was not moved
    assert g["duration_frames"] == 7                 # 98..104


def test_a_reach_a_person_added_has_no_original(widget_module):
    r = {"reach_id": 9, "start_frame": 100, "end_frame": 104, "source": "human_added",
         "original_start": None, "original_end": None}
    widget_module.move_reach_start(r, 99)
    assert r["original_start"] is None
    assert widget_module.gt_reach_record(r, "t", "x")["duration_frames"] == 6

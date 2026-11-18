"""The kinematics step pairs each pellet's reaches with ITS outcome, by segment
number, and writes nothing when the two files do not describe the same pellets.

WHY: the reaches file and the outcomes file were paired by position (zip). If
the two files disagree on how many pellets there are, or list them in another
order, a positional pairing attaches a pellet's reaches to another pellet's
outcome and causal reach, silently, for every pellet after the first gap.
Pairing by number either pairs a pellet with its own outcome or holds the
video (KinematicsProblem: no features file, the watcher holds it).

The one expected difference still writes: the reaches file carries one more
segment than the outcomes file (the stretch after the last boundary), and that
stretch is left out as before.

Built under tmp_path with invented ids; no real folder, database or
configuration. Only names that exist on code without this change are imported
at module level, so the tests fail there by what they assert.
"""
import json
import pickle

import numpy as np
import pandas as pd
import pytest

import mousereach.review.causal_review_io as crio
from mousereach.config import Paths
from mousereach.kinematics.core.feature_extractor import FeatureExtractor, KinematicsProblem

VID = "20990101_ABC0912_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")
N = 300


@pytest.fixture
def proc(tmp_path, monkeypatch):
    """A Processing folder with a pose file (and its frame rate) and no
    review, ground truth or reach assignment anywhere."""
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


def _reach_seg(num, start, end, reach_start):
    return {"segment_num": num, "start_frame": start, "end_frame": end,
            "ruler_pixels": 34.0, "n_reaches": 1,
            "reaches": [{"reach_id": num, "reach_num": 1, "start_frame": reach_start,
                         "end_frame": reach_start + 10, "apex_frame": reach_start + 5,
                         "duration_frames": 11}]}


def _write(proc, reach_segments, outcome_segments):
    (proc / f"{VID}_reaches.json").write_text(json.dumps(
        {"video_name": VID, "total_frames": N, "n_segments": len(reach_segments),
         "segments": reach_segments}), encoding="utf-8")
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"video_id": VID, "segments": outcome_segments}), encoding="utf-8")


def _extract(proc):
    return FeatureExtractor().extract(proc / f"{VID}DLC_x.h5", proc / f"{VID}_reaches.json",
                                      proc / f"{VID}_pellet_outcomes.json")


# Two pellets, and the stretch after the last boundary (segment 3), as the
# reach detector writes them.
REACH_SEGMENTS = [_reach_seg(1, 0, 99, 20), _reach_seg(2, 100, 199, 120),
                  _reach_seg(3, 200, 299, 220)]


def test_the_usual_shape_writes_each_pellet_with_its_outcome(proc):
    """One more reach segment than outcome segments is the normal shape: both
    pellets are written with their own outcome; the trailing stretch is left
    out, as before."""
    _write(proc, REACH_SEGMENTS, [{"segment_num": 1, "outcome": "untouched"},
                                  {"segment_num": 2, "outcome": "retrieved"}])
    segs = _extract(proc).to_dict()["segments"]
    assert [(s["segment_num"], s["outcome"]) for s in segs] == [(1, "untouched"), (2, "retrieved")]
    assert [r["start_frame"] for s in segs for r in s["reaches"]] == [20, 120]


def test_outcomes_listed_in_another_order_land_on_their_own_pellet(proc):
    """The outcomes file lists pellet 2 before pellet 1. By position, pellet
    1's reaches got pellet 2's outcome; by number each keeps its own."""
    _write(proc, REACH_SEGMENTS, [{"segment_num": 2, "outcome": "retrieved"},
                                  {"segment_num": 1, "outcome": "untouched"}])
    segs = _extract(proc).to_dict()["segments"]
    by_num = {s["segment_num"]: s["outcome"] for s in segs}
    assert by_num == {1: "untouched", 2: "retrieved"}
    assert {s["segment_num"]: [r["start_frame"] for r in s["reaches"]] for s in segs} == {
        1: [20], 2: [120]}


def test_a_pellet_missing_from_the_outcomes_file_holds_the_video(proc):
    """The outcomes file has pellets 1 and 3 but not 2 (made on other cuts):
    by position pellet 2's reaches took pellet 3's outcome and pellet 3 was
    dropped. Now nothing is written and the reason names the pellet."""
    _write(proc, [_reach_seg(1, 0, 74, 20), _reach_seg(2, 75, 149, 100),
                  _reach_seg(3, 150, 224, 170), _reach_seg(4, 225, 299, 240)],
           [{"segment_num": 1, "outcome": "untouched"},
            {"segment_num": 3, "outcome": "retrieved"}])
    with pytest.raises(KinematicsProblem) as e:
        _extract(proc)
    assert "segment(s) 2 that the outcomes file has no outcome for" in str(e.value)


def test_an_outcome_with_no_reach_segment_holds_the_video(proc):
    """An outcome for a pellet the reaches file does not have would be
    dropped silently by a positional pairing. Held instead."""
    _write(proc, REACH_SEGMENTS[:2], [{"segment_num": 1, "outcome": "untouched"},
                                      {"segment_num": 2, "outcome": "retrieved"},
                                      {"segment_num": 3, "outcome": "retrieved"}])
    with pytest.raises(KinematicsProblem) as e:
        _extract(proc)
    assert "pellet(s) 3 that the reaches file has no segment for" in str(e.value)


def test_a_pellet_listed_twice_holds_the_video(proc):
    _write(proc, REACH_SEGMENTS, [{"segment_num": 1, "outcome": "untouched"},
                                  {"segment_num": 1, "outcome": "retrieved"}])
    with pytest.raises(KinematicsProblem) as e:
        _extract(proc)
    assert "lists segment(s) 1 more than once" in str(e.value)


def test_pellets_at_the_end_without_an_outcome_hold_the_video(proc):
    """Reaches file with pellets 1-3 and the stretch after the last cut (4),
    outcomes file with pellets 1 and 2 only. By position pellet 3 was dropped
    with nothing said; only the one stretch after the last cut is expected."""
    _write(proc, [_reach_seg(1, 0, 74, 20), _reach_seg(2, 75, 149, 100),
                  _reach_seg(3, 150, 224, 170), _reach_seg(4, 225, 299, 240)],
           [{"segment_num": 1, "outcome": "untouched"},
            {"segment_num": 2, "outcome": "retrieved"}])
    with pytest.raises(KinematicsProblem) as e:
        _extract(proc)
    assert "segment(s) 3 after the last outcome" in str(e.value)


def test_an_outcomes_file_without_segments_holds_the_video(proc):
    """No list to pair with: nothing is written (never an empty features file)."""
    _write(proc, REACH_SEGMENTS, [])
    doc = json.loads((proc / f"{VID}_pellet_outcomes.json").read_text(encoding="utf-8"))
    del doc["segments"]
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(Exception) as e:
        _extract(proc)
    assert "segments" in str(e.value)

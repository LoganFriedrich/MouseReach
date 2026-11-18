"""The secondary exports and summaries read each video's resolved result (its
``_features.json``), not the algorithm's own files.

WHY: the export, summary, quick-summary, real-kinematics and ASPA sync
commands read ``_pellet_outcomes.json`` / ``_reaches.json`` -- the algorithm's
calls before any person's answer -- so a pellet a reviewer corrected was
counted with the algorithm's call, and a reach whose boundaries ground truth
moved was reported at the algorithm's frames. The export and summary also
counted 'RETRIEVED' / 'DISPLACED' / 'MISSED', names no file holds, so their
counts were always 0. The reach export's fallback for a video without a
features file now says its rows are the algorithm's calls.

Every folder below is built under tmp_path with invented ids.
"""
import pickle
import csv
import json
import sys

import numpy as np
import pandas as pd
import pytest

from mousereach.config import Paths

VID = "20990101_ABC0904_P1"
POINTS = ("Nose", "BOXL", "BOXR", "SABL", "SABR", "Pellet", "Pillar",
          "RightHand", "RHLeft", "RHOut", "RHRight")


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """The algorithm said 'untouched' with a reach at 20-30; the resolved
    result (a person's answers applied) says 'retrieved', reach at 22-40."""
    proc = tmp_path / "Processing"
    proc.mkdir()
    monkeypatch.setattr(Paths, "PROCESSING", proc)
    (proc / f"{VID}_pellet_outcomes.json").write_text(json.dumps({
        "video_name": VID, "segments": [{"segment_num": 1, "outcome": "untouched"}]}),
        encoding="utf-8")
    (proc / f"{VID}_reaches.json").write_text(json.dumps({
        "video_name": VID, "segments": [{"segment_num": 1, "ruler_pixels": 34.0, "reaches": [
            {"reach_id": 1, "reach_num": 1, "start_frame": 20, "end_frame": 30,
             "apex_frame": 25, "duration_frames": 10}]}]}), encoding="utf-8")
    (proc / f"{VID}_features.json").write_text(json.dumps({
        "video_name": VID, "n_segments": 1, "segments": [{
            "segment_num": 1, "outcome": "retrieved", "outcome_confidence": 1.0,
            "ruler_pixels": 34.0, "n_reaches": 1, "causal_reach_id": 1,
            "n_reaches_left_out": 0,
            "reaches": [{"reach_id": 1, "reach_num": 1, "segment_num": 1,
                         "start_frame": 22, "end_frame": 40, "apex_frame": 31,
                         "duration_frames": 18, "outcome": "retrieved",
                         "causal_reach": True, "outcome_source": "ground_truth",
                         "reach_source": "ground_truth", "extended": {}}]}]}),
        encoding="utf-8")
    return proc


def test_the_export_shows_the_persons_outcome(folder):
    from mousereach.export.core.exporter import load_all_results, results_to_dataframe
    df = results_to_dataframe(load_all_results(folder))
    assert list(df["outcome"]) == ["retrieved"]


def test_the_summary_counts_the_persons_outcome(folder):
    from mousereach.export.core.summary import compile_results
    s = compile_results(folder)["summary"]
    assert (s["total_segments"], s["total_retrieved"], s["total_missed"]) == (1, 1, 0)


def test_a_video_without_features_is_left_out_and_named(folder):
    from mousereach.export.core.summary import compile_results
    (folder / f"{VID}_features.json").unlink()
    r = compile_results(folder)
    assert r["summary"]["total_segments"] == 0
    assert any(VID in line for line in r["metadata"]["left_out"])


def test_the_quick_summary_counts_the_persons_outcome(folder, monkeypatch):
    from mousereach.export import quick_summary
    monkeypatch.setattr(sys, "argv", ["mousereach-quick-summary"])
    quick_summary.main()
    with open(folder.parent / "summary_for_PI.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert (row["retrieved"], row["untouched"]) == ("1", "0")


def test_the_aspa_sync_rows_are_the_resolved_reaches(folder):
    from mousereach.aspa.sync import _build_reach_rows
    load = lambda n: json.loads((folder / f"{VID}_{n}.json").read_text(encoding="utf-8"))
    rows = _build_reach_rows(VID, "X", "X01", load("reaches"), load("pellet_outcomes"),
                             load("features"), None)
    assert [(r["start_frame"], r["end_frame"], r["outcome"]) for r in rows] == [(22, 40, "retrieved")]


def test_the_aspa_sync_skips_a_video_without_features(folder, tmp_path):
    from mousereach.aspa.sync import sync_video
    (folder / f"{VID}_features.json").unlink()
    assert sync_video(folder, VID, "X", "X01", tmp_path / "none.db", dry_run=True) == 0


def test_real_kinematics_uses_the_resolved_reaches_and_outcome(folder, monkeypatch):
    n = 100
    cols, data = [], []
    for p in POINTS:
        for c, v in (("x", 100.0), ("y", 150.0), ("likelihood", 1.0)):
            cols.append(("scorer", p, c))
            data.append(np.full(n, v) + (np.arange(n) * 0.5 if c in ("x", "y") and p.startswith("R") else 0))
    pd.DataFrame(np.array(data).T, columns=pd.MultiIndex.from_tuples(cols)).to_hdf(
        folder / f"{VID}DLC_x.h5", key="df")
    # The video's frame rate, recorded beside the pose as DeepLabCut does
    # (kinematics/core/frame_rate.py); without one no kinematics are written.
    with open((folder / f"{VID}DLC_x.h5").with_name((folder / f"{VID}DLC_x.h5").stem + "_meta.pickle"), "wb") as _f:
        pickle.dump({"data": {"fps": 60.0}}, _f)
    from mousereach.kinematics.analysis import real_kinematics
    monkeypatch.setattr(sys, "argv", ["mousereach-real-kinematics"])
    real_kinematics.main()
    with open(folder.parent / "real_kinematics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(r["outcome"], r["n_frames"]) for r in rows] == [("retrieved", "19")]


def test_the_reach_export_names_rows_that_are_the_algorithms_calls(folder, capsys):
    from mousereach.kinematics.analysis import reach_export
    (folder / f"{VID}_features.json").unlink()
    reach_export.main()
    out = capsys.readouterr().out
    assert "algorithm's own calls" in out and VID in out


def test_an_easy_trays_session_file_adds_no_pellet(folder):
    """An Easy or Flat tray's features file has reaches and attention but no
    pellet outcome; it must not add a pellet to the pellet counts."""
    from mousereach.export.core.summary import compile_results
    easy = "20990101_ABC0904_E1"
    (folder / f"{easy}_reaches.json").write_text(json.dumps({"video_name": easy, "segments": []}),
                                                 encoding="utf-8")
    (folder / f"{easy}_features.json").write_text(json.dumps({
        "video_name": easy, "summary": {"session_only": True},
        "segments": [{"segment_num": 1, "outcome": None, "reaches": []}]}), encoding="utf-8")
    r = compile_results(folder)
    assert (r["summary"]["n_videos"], r["summary"]["total_segments"]) == (1, 1)
    assert not any(easy in line for line in r["metadata"]["left_out"])

"""The truth layering reads a review it is handed as a FILE, through the one
read of every review copy (causal_review_io.effective_review).

WHY: the kinematics extractor passes its ``review_path`` -- the review file
itself -- to resolve_truth_layers as ``primary_dir``. A review that existed
only in the processing folder was once never read by the layering, and the
extractor quietly fell back to the algorithm's answers.
"""
import json

import pytest

import mousereach.review.causal_review_io as cio
from mousereach.review.truth_resolver import resolve_truth_layers

STEM = "20990101_CNT9901_P1"


def _answer():
    return {"segments": [{"segment_num": 1, "segment_span": {"start": 0, "end": 99},
                          "human": {"outcome": "retrieved"}, "answers": {}}]}


@pytest.fixture(autouse=True)
def _no_other_copies(monkeypatch):
    monkeypatch.setattr(cio, "durable_review_dir", lambda: None)
    monkeypatch.setattr(cio, "find_gt", lambda *a, **k: None)
    import mousereach.review.bundle_media as bm
    monkeypatch.setattr(bm, "filed_media_dirs", lambda stem: [])
    from mousereach.config import Paths
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", None, raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", None, raising=False)


def _resolve(primary):
    reaches = {"video_name": STEM, "segments": [
        {"segment_num": 1, "start_frame": 0, "end_frame": 99, "reaches": []}]}
    outcomes = {"segments": [{"segment_num": 1, "outcome": "untouched"}]}
    _, out = resolve_truth_layers(reaches, outcomes, video_stem=STEM, primary_dir=primary)
    return out["segments"][0]


def _write(p, doc):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_the_review_file_itself_is_read(tmp_path):
    seg = _resolve(_write(tmp_path / f"{STEM}_causal_review.json", _answer()))
    assert (seg["outcome"], seg["outcome_source"]) == ("retrieved", "human_review")


def test_a_folder_holding_it_is_read(tmp_path):
    _write(tmp_path / "work" / f"{STEM}_causal_review.json", _answer())
    assert _resolve(tmp_path / "work")["outcome"] == "retrieved"


def test_another_videos_file_is_never_applied(tmp_path):
    seg = _resolve(_write(tmp_path / "20990101_ABC9902_P1_causal_review.json", _answer()))
    assert (seg["outcome"], seg["outcome_source"]) == ("untouched", "algo")


def test_nothing_found_leaves_the_algorithm(tmp_path):
    assert _resolve(tmp_path / "nothing_here.json")["outcome_source"] == "algo"
    assert _resolve(None)["outcome_source"] == "algo"

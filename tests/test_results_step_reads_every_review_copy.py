"""The results step applies human answers from ONE read of every review copy.

The kinematics truth layering read the review copies by WHERE they sat -- a
queue bundle outranking the durable copy -- and never read the copy in the
video's filed folder. So an answer that lived only in a copy it did not
prefer was not applied, an older answer in a queue bundle beat a newer one in
the durable copy, and a pellet reopened ("are you sure?") in the durable copy
was still applied from the bundle. It now reads every copy through
causal_review_io.effective_review, as the review window, the release check
and the census do: per segment, the newest answer or recheck wins.
"""
import json

import pytest

import mousereach.review.bundle_media as bm
import mousereach.review.causal_review_io as cio
from mousereach.config import Paths
from mousereach.review.truth_resolver import resolve_truth_layers

STEM = "20990101_ABC9901_P1"


def _rec(outcome, when=None, recheck_at=None, placeholder=False):
    rec = {"segment_num": 1, "segment_span": {"start": 0, "end": 99},
           "human": {"outcome": outcome}, "answers": {}}
    if when:
        rec["reviewed_at"] = when
    if placeholder:
        rec["human"]["outcome"] = None
        rec["answers"] = {"reviewed": False}
    if recheck_at:
        rec["answers"] = {"reviewed": False,
                          "recheck": {"asked_at": recheck_at, "question": "sure?"}}
    return rec


@pytest.fixture
def places(tmp_path, monkeypatch):
    d = {"durable": tmp_path / "records" / "reviews",
         "triage": tmp_path / "Review" / "triage" / STEM,
         "filed": tmp_path / "Analyzed" / "group"}
    for p in d.values():
        p.mkdir(parents=True)
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(cio, "find_gt", lambda *a, **k: None)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "Review" / "triage", raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "Review" / "deep_review", raising=False)
    monkeypatch.setattr(bm, "filed_media_dirs", lambda stem: [d["filed"]])
    return d


def _put(folder, *records):
    (folder / f"{STEM}_causal_review.json").write_text(
        json.dumps({"segments": list(records)}), encoding="utf-8")


def _outcome():
    reaches = {"video_name": STEM, "segments": [
        {"segment_num": 1, "start_frame": 0, "end_frame": 99, "reaches": []}]}
    outcomes = {"segments": [{"segment_num": 1, "outcome": "untouched"}]}
    _, out = resolve_truth_layers(reaches, outcomes, video_stem=STEM)
    seg = out["segments"][0]
    return seg["outcome"], seg["outcome_source"]


def test_an_answer_only_the_filed_folder_copy_holds_is_applied(places):
    _put(places["durable"], _rec(None, placeholder=True))
    _put(places["filed"], _rec("retrieved", when="2099-01-01T10:00:00"))
    assert _outcome() == ("retrieved", "human_review")


def test_the_newest_answer_wins_wherever_it_is(places):
    _put(places["triage"], _rec("displaced_sa", when="2099-01-01T10:00:00"))
    _put(places["durable"], _rec("retrieved", when="2099-01-02T10:00:00"))
    assert _outcome() == ("retrieved", "human_review")


def test_a_pellet_reopened_in_the_durable_copy_is_not_applied(places):
    _put(places["triage"], _rec("displaced_sa", when="2099-01-01T10:00:00"))
    _put(places["durable"], _rec("displaced_sa", recheck_at="2099-01-03T10:00:00"))
    assert _outcome() == ("untouched", "algo")

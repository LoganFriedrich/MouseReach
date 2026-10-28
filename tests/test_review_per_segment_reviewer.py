"""Each segment's answer in a review file says who gave it and when.

WHY: a review file used to carry ONE "reviewer" and "reviewed_at". When a
second person saved one answer in a video a first person had answered
earlier, the file then credited the second person with every answer in it.
Now each answered segment's record carries its own "reviewer" and
"reviewed_at", set only when that segment's answer is new or changed in a
save; the file-level fields still mean "who saved the file last".

The video id is synthetic (see test_review_durability.py for why).
"""
import json

import pytest

from mousereach.review import causal_review_io as crio
from mousereach.review.causal_review_io import (
    apply_review_overrides, build_segment_record, save_causal_review,
    segment_reviewer, update_corpus_index,
)

STEM = "20990101_ZZZ9901_P1"


@pytest.fixture
def nas(tmp_path, monkeypatch):
    from mousereach.config import Paths
    root = tmp_path / "nas"
    root.mkdir()
    monkeypatch.setattr(Paths, "NAS_ROOT", root, raising=False)
    return root


def _rec(n, outcome=None):
    return build_segment_record(
        segment_num=n, pellet_num=n, algo_outcome="untouched", algo_causal_reach=None,
        algo_interaction_frame=None, human_outcome=outcome, human_causal_reach=None,
        is_phantom=False, agreed=False,
        answers={"reviewed": False} if outcome is None else {"triage_kind": "outcome_uncertain"},
        segment_span={"start": 1000 * n, "end": 1000 * n + 999})


def _save(folder, records, who, when, monkeypatch):
    monkeypatch.setattr(crio, "_get_timestamp", lambda: when)
    return save_causal_review(STEM, folder, records, provenance={}, reviewer=who)


def _segs(path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc, {s["segment_num"]: s for s in doc["segments"]}


def test_two_reviewers_each_keep_their_own_answers(tmp_path, nas, monkeypatch):
    folder = tmp_path / "bundle"
    folder.mkdir()
    # A answers segments 1 and 2; 3 is not answered yet
    _save(folder, [_rec(1, "retrieved"), _rec(2, "untouched"), _rec(3)],
          "reviewer_a", "2026-08-13T10:00:00", monkeypatch)
    # B later opens the video: 1 and 2 come back unchanged, B answers 3
    path = _save(folder, [_rec(1, "retrieved"), _rec(2, "untouched"), _rec(3, "displaced_sa")],
                 "reviewer_b", "2026-10-28T09:00:00", monkeypatch)
    doc, by = _segs(path)
    assert (by[1]["reviewer"], by[1]["reviewed_at"]) == ("reviewer_a", "2026-08-13T10:00:00")
    assert (by[2]["reviewer"], by[2]["reviewed_at"]) == ("reviewer_a", "2026-08-13T10:00:00")
    assert (by[3]["reviewer"], by[3]["reviewed_at"]) == ("reviewer_b", "2026-10-28T09:00:00")
    # the file-level fields mean "last save"
    assert (doc["reviewer"], doc["reviewed_at"]) == ("reviewer_b", "2026-10-28T09:00:00")
    # the durable copy says the same
    _, dby = _segs(crio.durable_review_path(STEM))
    assert dby[1]["reviewer"] == "reviewer_a" and dby[3]["reviewer"] == "reviewer_b"


def test_a_changed_answer_is_credited_to_whoever_changed_it(tmp_path, nas, monkeypatch):
    folder = tmp_path / "bundle"
    folder.mkdir()
    _save(folder, [_rec(1, "retrieved")], "reviewer_a", "2026-08-13T10:00:00", monkeypatch)
    path = _save(folder, [_rec(1, "displaced_sa")], "reviewer_b", "2026-10-28T09:00:00",
                 monkeypatch)
    _, by = _segs(path)
    assert by[1]["reviewer"] == "reviewer_b"


def test_an_open_placeholder_is_credited_to_nobody(tmp_path, nas, monkeypatch):
    folder = tmp_path / "bundle"
    folder.mkdir()
    path = _save(folder, [_rec(1, "retrieved"), _rec(2)], "reviewer_a",
                 "2026-08-13T10:00:00", monkeypatch)
    _, by = _segs(path)
    assert "reviewer" not in by[2] and "reviewed_at" not in by[2]


def test_an_old_file_without_per_segment_fields_still_reads(tmp_path, nas, monkeypatch):
    folder = tmp_path / "bundle"
    folder.mkdir()
    old = {"type": "causal_review", "schema_version": "1.1", "video_stem": STEM,
           "reviewer": "reviewer_a", "reviewed_at": "2026-08-13T10:00:00",
           "provenance": {}, "segments": [_rec(1, "retrieved"), _rec(2, "untouched")]}
    (folder / f"{STEM}_causal_review.json").write_text(json.dumps(old), encoding="utf-8")
    # readers fall back to the file-level fields
    assert segment_reviewer(old["segments"][0], old) == ("reviewer_a", "2026-08-13T10:00:00")
    applied = apply_review_overrides({"segments": [{"segment_num": 1, "outcome": "untouched"}]},
                                     {1: old["segments"][0]}, reviewer=old["reviewer"])
    assert applied["segments"][0]["reviewed_by"] == "reviewer_a"
    idx = update_corpus_index(STEM, folder / "x.json", old["segments"], "reviewer_a",
                              "2026-08-13T10:00:00", index_dir=tmp_path / "idx")
    entries = json.loads(idx.read_text())["entries"]
    assert entries[f"{STEM}__seg1"]["reviewer"] == "reviewer_a"
    # a later save by someone else: the unchanged old answers keep the old
    # file's reviewer, and only the new answer is credited to the new one
    path = _save(folder, [_rec(1, "retrieved"), _rec(2, "untouched"), _rec(3, "retrieved")],
                 "reviewer_b", "2026-10-28T09:00:00", monkeypatch)
    _, by = _segs(path)
    assert [by[n]["reviewer"] for n in (1, 2, 3)] == ["reviewer_a", "reviewer_a", "reviewer_b"]


def test_the_index_and_the_results_credit_each_segment_to_who_answered_it(tmp_path, nas,
                                                                         monkeypatch):
    folder = tmp_path / "bundle"
    folder.mkdir()
    _save(folder, [_rec(1, "retrieved"), _rec(2)], "reviewer_a", "2026-08-13T10:00:00",
          monkeypatch)
    path = _save(folder, [_rec(1, "retrieved"), _rec(2, "untouched")], "reviewer_b",
                 "2026-10-28T09:00:00", monkeypatch)
    doc, by = _segs(path)
    idx = update_corpus_index(STEM, path, doc["segments"], doc["reviewer"], doc["reviewed_at"],
                              index_dir=tmp_path / "idx")
    entries = json.loads(idx.read_text())["entries"]
    assert entries[f"{STEM}__seg1"]["reviewer"] == "reviewer_a"
    assert entries[f"{STEM}__seg1"]["reviewed_at"] == "2026-08-13T10:00:00"
    assert entries[f"{STEM}__seg2"]["reviewer"] == "reviewer_b"
    out = apply_review_overrides(
        {"segments": [{"segment_num": 1, "outcome": "untouched"},
                      {"segment_num": 2, "outcome": "retrieved"}]},
        by, reviewer=doc["reviewer"])
    assert [s["reviewed_by"] for s in out["segments"]] == ["reviewer_a", "reviewer_b"]

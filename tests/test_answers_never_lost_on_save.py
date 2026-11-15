"""A save can add or change a person's answers; it never drops one.

The review window used to read only the first copy of a review that had any
records (the bundle's, or the one beside the video) and never the durable
copy, and a save wrote what the window held over every copy. A bundle opened
without a working copy therefore started empty; the reviewer answered what
was open, and the save replaced the durable copy's earlier answers with
"not reviewed" placeholders -- gone from every live copy.

Now the window reads every copy and takes, per segment, the newest statement
a person made about its frames (an answer, or a recheck that reopened one),
and the save keeps every earlier answer the new save does not replace.
"""
import json

import pytest

import mousereach.review.causal_review_io as cio
from mousereach.census.review_completeness import completeness_for_stem
from mousereach.review.release_cli import _completeness

STEM = "20990101_ABC9901_P1"
SPANS = {1: (100, 199), 2: (200, 299), 3: (300, 399)}


def _span(n):
    a, b = SPANS[n]
    return {"start": a, "end": b}


def _answer(n, outcome="untouched", who="reviewer-a", when="2099-01-01T10:00:00", span=None):
    return {"segment_num": n, "segment_span": span or _span(n),
            "human": {"outcome": outcome}, "answers": {},
            "reviewer": who, "reviewed_at": when}


def _placeholder(n):
    return {"segment_num": n, "segment_span": _span(n),
            "human": {"outcome": None}, "answers": {"reviewed": False}}


def _recheck(n, asked="2099-01-03T10:00:00"):
    return {"segment_num": n, "segment_span": _span(n), "human": {"outcome": "retrieved"},
            "answers": {"reviewed": False, "recheck": {"asked_at": asked, "question": "sure?"}}}


@pytest.fixture
def durable(tmp_path, monkeypatch):
    root = tmp_path / "review_records"
    monkeypatch.setattr(cio, "_default_index_dir", lambda: root)
    (root / "reviews").mkdir(parents=True)
    return root / "reviews"


def _write(folder, segments, **extra):
    folder.mkdir(parents=True, exist_ok=True)
    doc = {"video_stem": STEM, "segments": segments}
    doc.update(extra)
    (folder / f"{STEM}_causal_review.json").write_text(json.dumps(doc), encoding="utf-8")


def _read(folder):
    return json.loads((folder / f"{STEM}_causal_review.json").read_text(encoding="utf-8"))


def _by_num(doc):
    return {r["segment_num"]: r for r in doc["segments"]}


def test_a_save_keeps_an_answer_only_the_durable_copy_held(tmp_path, durable):
    _write(durable, [_answer(1, "retrieved"), _placeholder(2)])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    # The window held no working copy: it saves only the answer it was given.
    cio.save_causal_review(STEM, bundle, [_placeholder(1), _answer(2, who="reviewer-b",
                                                                  when=None)],
                           provenance={}, reviewer="reviewer-b")
    for copy in (_read(durable), _read(bundle)):
        recs = _by_num(copy)
        assert cio.is_answered(recs[1]) and recs[1]["human"]["outcome"] == "retrieved"
        # the earlier answer keeps the name and time of the person who gave it
        assert recs[1]["reviewer"] == "reviewer-a"
        assert recs[1]["reviewed_at"] == "2099-01-01T10:00:00"
        assert cio.is_answered(recs[2])


def test_a_save_can_change_an_answer(tmp_path, durable):
    _write(durable, [_answer(1, "retrieved")])
    bundle = tmp_path / "bundle"
    cio.save_causal_review(STEM, bundle, [_answer(1, "displaced_sa", who="reviewer-b")],
                           provenance={}, reviewer="reviewer-b")
    assert _by_num(_read(durable))[1]["human"]["outcome"] == "displaced_sa"


def test_an_earlier_answer_about_frames_no_segment_has_now_is_kept_unmatched(tmp_path, durable):
    old = _answer(1, "retrieved", span={"start": 5000, "end": 5099})
    _write(durable, [old])
    bundle = tmp_path / "bundle"
    cio.save_causal_review(STEM, bundle, [_answer(1), _placeholder(2)],
                           provenance={}, reviewer="reviewer-b")
    kept = _read(durable)[cio.UNMATCHED_ANSWERS_KEY]
    assert [r["segment_span"] for r in kept] == [{"start": 5000, "end": 5099}]


def test_a_recheck_newer_than_the_answer_is_kept_open(tmp_path, durable):
    _write(durable, [_answer(1, "retrieved")])
    bundle = tmp_path / "bundle"
    _write(bundle, [_recheck(1)])
    cio.save_causal_review(STEM, bundle, [_placeholder(1), _answer(2)],
                           provenance={}, reviewer="reviewer-b")
    rec = _by_num(_read(durable))[1]
    assert not cio.is_answered(rec)
    assert rec["answers"]["recheck"]["question"] == "sure?"


def test_merge_takes_the_newest_statement_per_pellet():
    older = {"segments": [_answer(1, "retrieved", when="2099-01-01T10:00:00"),
                          _answer(2, "untouched", when="2099-01-01T10:00:00"),
                          _answer(3, "untouched", when="2099-01-01T10:00:00")]}
    newer = {"segments": [_answer(1, "displaced_sa", when="2099-01-02T10:00:00"),
                          _placeholder(2),
                          _recheck(3, asked="2099-01-02T10:00:00")]}
    current = [{"segment_num": n, "start_frame": a, "end_frame": b}
               for n, (a, b) in SPANS.items()]
    doc, _ = cio.merge_review_copies([newer, older], current)
    recs = _by_num(doc)
    assert recs[1]["human"]["outcome"] == "displaced_sa"     # newer answer wins
    assert recs[2]["human"]["outcome"] == "untouched"         # a placeholder never wins
    assert not cio.is_answered(recs[3])                       # a newer recheck reopens
    # and a newer answer ends an older recheck
    later = {"segments": [_answer(3, "retrieved", when="2099-01-04T10:00:00")]}
    doc, _ = cio.merge_review_copies([newer, later], current)
    assert cio.is_answered(_by_num(doc)[3])


def test_the_window_reads_the_durable_copy(tmp_path, durable):
    _write(durable, [_answer(1)])
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    copies = cio.review_copies(STEM, [bundle])
    assert len(copies) == 1 and copies[0]["segments"][0]["segment_num"] == 1


def test_release_check_counts_an_answer_only_the_durable_copy_holds(tmp_path, durable):
    _write(durable, [_answer(1), _answer(2)])
    bundle = tmp_path / STEM
    _write(bundle, [_answer(1), _placeholder(2)])
    assert _completeness(bundle, STEM) == (2, 2)


def test_census_sees_a_recheck_in_either_copy(tmp_path):
    bundle, store = tmp_path / "bundle", tmp_path / "store"
    _write(bundle, [_answer(1), _answer(2)])
    _write(store, [_answer(1), _recheck(2)])
    row = completeness_for_stem(STEM, bundle_dir=bundle, durable_dir=store)
    assert (row["answered"], row["total"], row["complete"]) == (1, 2, False)

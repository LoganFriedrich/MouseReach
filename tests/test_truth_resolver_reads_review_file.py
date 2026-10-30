"""The truth layering reads a review it is handed as a FILE.

WHY (2026-10-30): the kinematics extractor passes its ``review_path`` -- the
review file itself -- to resolve_truth_layers as ``primary_dir``, and the lookup
only tried ``<dir>/<stem>/<file>`` and ``<dir>/<file>``. A review that existed
only in the processing folder was therefore never read by the layering
(ISSUES.md 2026-10-27, item 2).
"""
import json

from mousereach.review.truth_resolver import _review_doc

STEM = "20990101_CNT9901_P1"
DOC = {"reviewer": "t", "segments": []}


def _write(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(DOC), encoding="utf-8")
    return p


def test_the_review_file_itself_is_read(tmp_path):
    f = _write(tmp_path / f"{STEM}_causal_review.json")
    assert _review_doc(STEM, f) == DOC


def test_another_videos_file_is_never_applied(tmp_path):
    f = _write(tmp_path / "20990101_CNT9902_P1_causal_review.json")
    assert _review_doc(STEM, f) is None


def test_the_two_directory_forms_still_work(tmp_path):
    _write(tmp_path / "bundles" / STEM / f"{STEM}_causal_review.json")
    assert _review_doc(STEM, tmp_path / "bundles") == DOC
    _write(tmp_path / "flat" / f"{STEM}_causal_review.json")
    assert _review_doc(STEM, tmp_path / "flat") == DOC


def test_missing_is_none(tmp_path):
    assert _review_doc(STEM, tmp_path / "nothing_here.json") is None
    assert _review_doc(STEM, None) is None

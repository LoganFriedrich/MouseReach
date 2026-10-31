"""Failures that could lose or misstate work must leave a trace.

WHY (2026-10-30): a review of the silent error handlers in the review and watcher
code found a save failure that still cleared the video, and unreadable files that
looked exactly like missing ones. Whatever is decided next, a person looking into
it has to be able to see that a read failed.
"""
import logging

from mousereach.review import resegmented, triage_status, truth_resolver
from mousereach.watcher import reconcile, review_routing


def _broken(tmp_path, name="x.json"):
    p = tmp_path / name
    p.write_text("{not json", encoding="utf-8")
    return p


def test_an_unreadable_truth_file_is_an_error_but_a_missing_one_is_normal(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        assert truth_resolver._read_json(tmp_path / "absent.json") is None
    assert caplog.records == []
    with caplog.at_level(logging.INFO):
        assert truth_resolver._read_json(_broken(tmp_path)) is None
    assert [r.levelname for r in caplog.records] == ["ERROR"]
    assert "human answers" in caplog.text


def test_other_unreadable_files_are_reported(tmp_path, caplog):
    bad = _broken(tmp_path, "v_routing.json")
    with caplog.at_level(logging.WARNING):
        assert resegmented._read(bad) is None
        assert reconcile._read_json(bad) is None
        assert triage_status._load(tmp_path, bad.name) is None
        assert review_routing.read_routing_manifest(tmp_path, "v") is None
    assert len(caplog.records) == 4


def test_a_failed_save_does_not_clear_the_deep_review(tmp_path, monkeypatch):
    from mousereach.review import causal_review_widget as w

    class Label:
        def setText(self, text):
            self.text = text

    class Fake:
        _bundle_dir = tmp_path
        _video_stem = "v"
        _status_label = Label()

        def _save_review(self):
            raise OSError("share went away")

    said = []
    monkeypatch.setattr(w, "show_error", said.append)
    w.CausalReviewWidget._clear_deep_review(Fake())
    assert "NOT cleared" in said[0]
    assert not (tmp_path / "v_deep_review_cleared.json").exists()

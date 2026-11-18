"""Escalate and Flag Session do not move a video whose answers did not save.

Both buttons save the review first. When that save raised, they only logged a
warning, moved the video anyway and loaded the next one, so the person's
answers -- which existed only on screen -- were lost without a word. Clear
already stopped in that case; now all three follow the same rule.
Stand-in windows under tmp_path; nothing real is moved.
"""
from types import SimpleNamespace

import mousereach.review.causal_review_widget as crw
import mousereach.watcher.review_gate as rg

STEM = "20990101_ABC9901_P1"


def _failing_save():
    raise OSError("the share is not reachable")


def _window(tmp_path):
    bundle = tmp_path / "triage" / STEM
    bundle.mkdir(parents=True)
    advanced = []
    status = []
    w = SimpleNamespace(
        _video_stem=STEM, _bundle_dir=bundle,
        _review_root=lambda: tmp_path / "root",
        _save_review=_failing_save,
        _notes_edit=SimpleNamespace(toPlainText=lambda: ""),
        _status_label=SimpleNamespace(setText=status.append),
        _load_next_video=lambda: advanced.append(True),
        _watcher_db_for_move=lambda: (None, ""),
    )
    return w, advanced, status


def _patch(monkeypatch, moved, errors):
    monkeypatch.setattr(crw, "flag_session", lambda stem, root, reason="", **kw: "key-1")
    monkeypatch.setattr(crw, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: errors.append(msg))
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: moved.append(a))
    monkeypatch.setattr(rg, "escalate_to_deep_review", lambda *a, **k: moved.append(a))
    monkeypatch.setattr(rg, "supersede_deep_review_clear", lambda *a, **k: None)


def test_escalate_stays_on_the_video_when_the_save_fails(tmp_path, monkeypatch):
    w, advanced, status = _window(tmp_path)
    moved, errors = [], []
    _patch(monkeypatch, moved, errors)
    crw.CausalReviewWidget._escalate_to_deep_review(w)
    assert moved == [] and advanced == []
    assert errors and "NOT moved" in errors[0]


def test_flag_session_stays_on_the_video_when_the_save_fails(tmp_path, monkeypatch):
    w, advanced, status = _window(tmp_path)
    moved, errors = [], []
    _patch(monkeypatch, moved, errors)
    monkeypatch.setattr(rg.Paths, "DEEP_REVIEW", tmp_path / "deep", raising=False)
    crw.CausalReviewWidget._flag_session(w)
    assert moved == [] and advanced == []
    assert errors and "NOT moved" in errors[0]

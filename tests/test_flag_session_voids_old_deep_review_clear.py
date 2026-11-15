"""Flag Session sets aside an earlier re-segmentation clear, as Escalate does.

Flagging a session means a person found a new problem. A clear marker from an
earlier visit to the re-segmentation queue travelled with the bundle, and the
return scan, which reads "cleared" from the marker, could release the video
straight back on the same cuts. Escalate already set such a marker aside; Flag
Session did not.
"""
import json
from types import SimpleNamespace

import mousereach.review.causal_review_widget as crw
import mousereach.watcher.review_gate as rg

STEM = "20990101_ABC9901_P1"


def _window(bundle, tmp_path):
    """The parts of a review window _flag_session uses."""
    return SimpleNamespace(
        _video_stem=STEM, _bundle_dir=bundle,
        _review_root=lambda: tmp_path / "root",
        _save_review=lambda: None,
        _status_label=SimpleNamespace(setText=lambda text: None),
        _load_next_video=lambda: None,
        _watcher_db_for_move=lambda: (None, ""),
    )


def _bundle_with_clear(tmp_path):
    b = tmp_path / "triage" / STEM
    b.mkdir(parents=True)
    (b / f"{STEM}_deep_review_cleared.json").write_text(
        json.dumps({"cleared_by": "reviewer-a"}), encoding="utf-8")
    return b


def test_flag_session_sets_the_old_clear_aside_before_moving(tmp_path, monkeypatch):
    b = _bundle_with_clear(tmp_path)
    seen = {}

    def route(stem, source, dest, reason, db=None, **kw):
        seen["clear_still_there"] = (source / f"{stem}_deep_review_cleared.json").exists()
        seen["kept"] = sorted(p.name for p in source.glob(f"{stem}_deep_review_cleared_superseded_*"))
        return source

    monkeypatch.setattr(crw, "flag_session", lambda stem, root, reason="", **kw: "session-key")
    monkeypatch.setattr(crw, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(crw, "show_error", lambda *a, **k: None)
    monkeypatch.setattr(rg, "route_to_queue", route)
    monkeypatch.setattr(rg.Paths, "DEEP_REVIEW", tmp_path / "deep", raising=False)

    crw.CausalReviewWidget._flag_session(_window(b, tmp_path))

    assert seen["clear_still_there"] is False
    assert len(seen["kept"]) == 1                     # kept on record, never deleted


def test_flag_session_does_not_move_a_video_whose_clear_cannot_be_set_aside(
        tmp_path, monkeypatch):
    b = _bundle_with_clear(tmp_path)
    moved, errors = [], []

    def fail(stem, source):
        raise OSError("file in use")

    monkeypatch.setattr(crw, "flag_session", lambda stem, root, reason="", **kw: "session-key")
    monkeypatch.setattr(crw, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: errors.append(msg))
    monkeypatch.setattr(rg, "supersede_deep_review_clear", fail)
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: moved.append(a))

    crw.CausalReviewWidget._flag_session(_window(b, tmp_path))

    assert moved == []
    assert errors and "NOT moved" in errors[0]

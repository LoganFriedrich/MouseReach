"""Routing a bundle onto its own folder moves nothing and deletes nothing.

Flag Session pressed in the re-segmentation window routed the deep-review
bundle into the deep-review queue -- onto itself. Each file was "moved" onto
itself, and the move's overwrite step deleted the destination first, which was
the file itself: every file of the bundle except the one the window still had
open was gone, and a fresh routing record listed them all as failed.

These tests use the real route_to_queue (no stub). Built under tmp_path with
invented video ids; no real folder or database.
"""
import json
from types import SimpleNamespace

import pytest

from mousereach.config import Paths
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.review_gate import route_to_queue
from mousereach.watcher.review_routing import _safe_move, move_video_bundle

VID = "20990101_ABC9901_P1"
FILES = (f"{VID}.mp4", f"{VID}DLC_model_x.h5", f"{VID}_segments.json",
         f"{VID}_reaches.json", f"{VID}_pellet_outcomes.json",
         f"{VID}_causal_review.json", f"{VID}_unified_ground_truth.json",
         f"{VID}_manifest.json", f"{VID}_routing.json")


@pytest.fixture
def deep_bundle(tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    bundle = deep / VID
    bundle.mkdir()
    for i, n in enumerate(FILES):
        (bundle / n).write_text(json.dumps({"n": n, "i": i}), encoding="utf-8")
    db = WatcherDB(db_path=tmp_path / "w.db")
    db.register_video(video_id=VID, source_path="x")
    return bundle, deep, db


def _contents(folder):
    return {p.name: p.read_bytes() for p in folder.iterdir()}


def test_safe_move_onto_itself_keeps_the_file(tmp_path):
    f = tmp_path / "a.json"
    f.write_text("keep me")
    _safe_move(f, tmp_path / "a.json")
    assert f.read_text() == "keep me"


def test_route_to_queue_onto_its_own_bundle_keeps_every_file(deep_bundle):
    bundle, deep, db = deep_bundle
    before = _contents(bundle)
    out = route_to_queue(VID, bundle, deep, "reviewer flagged session X",
                         db=db, db_state="deep_review")
    assert out == bundle
    # Every file kept, byte for byte -- the routing record and review manifest
    # are not rewritten either.
    assert _contents(bundle) == before
    assert db.get_video(VID)["state"] == "deep_review"


def test_move_video_bundle_onto_itself_moves_nothing(deep_bundle):
    bundle, deep, _ = deep_bundle
    before = _contents(bundle)
    out, moved = move_video_bundle(VID, bundle, deep, "again")
    assert out == bundle and moved == []
    assert _contents(bundle) == before


# --- Flag Session pressed on a bundle that is already in deep review ----------

def _window(bundle, tmp_path, said):
    """The parts of a review window _flag_session uses."""
    return SimpleNamespace(
        _video_stem=VID, _bundle_dir=bundle,
        _review_root=lambda: tmp_path / "root",
        _save_review=lambda: None,
        _status_label=SimpleNamespace(setText=lambda text: said.append(text)),
        _load_next_video=lambda: None,
        _watcher_db_for_move=lambda: (None, ""),
    )


def test_flag_session_in_the_resegmentation_queue_keeps_every_file(
        deep_bundle, tmp_path, monkeypatch):
    import mousereach.review.causal_review_widget as crw
    bundle, deep, _ = deep_bundle
    clear = bundle / f"{VID}_deep_review_cleared.json"
    clear.write_text("{}", encoding="utf-8")
    before = {k: v for k, v in _contents(bundle).items() if k != clear.name}
    said, errors = [], []
    monkeypatch.setattr(crw, "flag_session", lambda stem, root, reason="", **kw: "session-key")
    monkeypatch.setattr(crw, "show_info", lambda msg, *a, **k: said.append(msg))
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: errors.append(msg))

    crw.CausalReviewWidget._flag_session(_window(bundle, tmp_path, said))

    after = _contents(bundle)
    assert not errors
    # Every file still there, byte for byte; the earlier clear is set aside
    # (renamed, kept), so the video waits for its cuts to be set again.
    assert {k: v for k, v in after.items() if "_deep_review_cleared" not in k} == before
    assert not clear.exists()
    assert any("_deep_review_cleared_superseded_" in k for k in after)
    assert any("already in the re-segmentation queue" in m for m in said)

"""The review window and the bring-current routine no longer keep a SQLite
list of the videos waiting in the triage queue.

WHY: that list (review_records/triage_queue.db, a SQLite database in WAL
mode on the shared drive) was pushed by the bring-current routine and popped
on every save-and-next in the review window, but nothing read it -- the
window finds its next video by listing the queue folder. A database written
by several machines on a network share is a lock and corruption risk with
no reader to justify it. Built under tmp_path with invented ids.
"""
import functools
import json
import types

from mousereach.config import Paths

VID = "20990101_ABC0901_P1"


def _no_index_under(root):
    return not list(root.rglob("triage_queue.db*"))


def test_save_and_next_writes_no_queue_index(tmp_path, monkeypatch):
    from mousereach.review.causal_review_widget import CausalReviewWidget
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path, raising=False)
    bundle = tmp_path / "Processing" / "Review" / "triage" / VID
    bundle.mkdir(parents=True)
    # A bundle with nothing triaged left: every element is resolved.
    (bundle / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"segments": [{"segment_num": 1, "outcome": "retrieved"}]}), encoding="utf-8")
    loaded_next = []
    w = types.SimpleNamespace(video_path=None, _video_stem=VID, _bundle_dir=bundle,
                              _load_next_video=lambda: loaded_next.append(True))
    old_pop = getattr(CausalReviewWidget, "_pop_from_index_if_resolved", None)
    if old_pop is not None:      # the code before the index was retired
        w._pop_from_index_if_resolved = functools.partial(old_pop, w)
    CausalReviewWidget._save_and_next_video(w)
    assert loaded_next == [True]
    assert _no_index_under(tmp_path)


def test_staging_a_bring_current_bundle_writes_no_queue_index(tmp_path, monkeypatch):
    from mousereach.pipeline import reprocess_to_current as rtc
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", tmp_path, raising=False)
    triage = tmp_path / "Processing" / "Review" / "triage"
    triage.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage, raising=False)
    work = tmp_path / "work"
    work.mkdir()
    (work / f"{VID}_segments.json").write_text("{}", encoding="utf-8")
    out = rtc._stage_review_bundle(VID, work, None, work / f"{VID}DLC.h5",
                                   "triage", "a test reason")
    assert out["held"] is True
    assert _no_index_under(tmp_path)


def test_the_queue_index_module_is_gone():
    import importlib.util
    assert importlib.util.find_spec("mousereach.review.queue_index") is None

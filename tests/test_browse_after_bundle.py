"""Opening a video by hand ends the queue session; a failed load leaves nothing usable.

Two ways the review window used to pair a video with the wrong bundle:

* Browse after a queue bundle: the window kept the previous bundle's folder and
  manifest, so Escalate moved THAT bundle under the browsed video's name, and
  answers were saved into a bundle that was not the video's.
* A load that failed part-way: the new video's name was set while the previous
  video's answers, segments and bundle were still held, so a save wrote the old
  answers under the new name.

Built under tmp_path with stand-in windows; no real queue, video or database.
"""
from pathlib import Path
from types import SimpleNamespace

import mousereach.review.causal_review_widget as crw
import mousereach.watcher.review_gate as rg

OLD = "20990101_ABC9901_P1"
NEW = "20990102_ABC9902_P2"


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


def _window_on_bundle(tmp_path):
    bundle = tmp_path / "triage" / OLD
    bundle.mkdir(parents=True)
    w = SimpleNamespace(
        _bundle_dir=bundle, _manifest={"video_stem": OLD},
        _video_stem=OLD, video_path=bundle / f"{OLD}.mp4",
        _review_records={1: {"segment_num": 1}}, _segment_list=[{"segment_num": 1}],
        _status_label=_Recorder(), _notes_edit=SimpleNamespace(toPlainText=lambda: ""),
        _save_review=lambda: None, _load_next_video=lambda: None,
        _watcher_db_for_move=lambda: (None, ""),
        _update_routing_banner=lambda: None,
    )
    return w, bundle


def test_browse_ends_the_queue_session_and_escalate_moves_nothing(tmp_path, monkeypatch):
    w, bundle = _window_on_bundle(tmp_path)
    browsed = tmp_path / "elsewhere" / f"{NEW}.mp4"
    loaded = []

    def load(path):
        loaded.append((path, w._bundle_dir))
        w._video_stem, w.video_path = NEW, path

    w._load_video = load
    monkeypatch.setattr(crw.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(browsed), "")))
    crw.CausalReviewWidget._browse_video(w)

    # The browsed video was loaded with no bundle paired to it.
    assert loaded == [(browsed, None)]
    assert w._bundle_dir is None and w._manifest is None

    routed, errors = [], []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    monkeypatch.setattr(rg, "escalate_to_deep_review", lambda *a, **k: routed.append(a))
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: errors.append(msg))
    monkeypatch.setattr(crw, "show_info", lambda *a, **k: None)
    crw.CausalReviewWidget._escalate_to_deep_review(w)

    assert routed == []                               # the old bundle is never moved
    assert errors and "No bundle loaded" in errors[0]
    assert (bundle).is_dir()


def test_a_failed_load_forgets_the_previous_video(tmp_path, monkeypatch):
    w, bundle = _window_on_bundle(tmp_path)
    w._video_label = _Recorder()
    w._progress = _Recorder()
    w._enable_controls = lambda enabled: None
    errors = []
    monkeypatch.setattr(crw, "show_error", lambda msg, *a, **k: errors.append(msg))

    # A file OpenCV cannot open: the load fails after the new name is known.
    broken = tmp_path / f"{NEW}.mp4"
    broken.write_bytes(b"not a video")
    crw.CausalReviewWidget._load_video(w, broken)

    assert errors
    assert w._video_stem is None and w.video_path is None
    assert w._review_records == {} and w._segment_list == []
    assert w._bundle_dir is None
    # The bundle that failed is not offered again by Load Next this session.
    assert str(bundle) in w._reviewed_this_session

    # A save now writes nothing anywhere.
    warnings = []
    monkeypatch.setattr(crw, "show_warning", lambda msg, *a, **k: warnings.append(msg))
    w._collect_answers = lambda: (_ for _ in ()).throw(AssertionError("must not run"))
    crw.CausalReviewWidget._save_review(w)
    assert warnings == ["No video loaded"]
    assert sorted(p.name for p in tmp_path.rglob("*")
                  if p.is_file()) == [f"{NEW}.mp4"]


# ---------------------------------------------------------------------------
# The queue survives Browse and a failed load: only the bundle is forgotten.
# These call the REAL _load_next_video / _save_and_next_video / _flag_session;
# only the video decoding, the priority order and the eligibility check are
# stood in for, so the queue walk itself is what is tested.
# ---------------------------------------------------------------------------
import json

C = "20990103_ABC9903_P3"


class _Button:
    def __init__(self):
        self.enabled = False

    def setEnabled(self, on):
        self.enabled = bool(on)


class _QueueWindow:
    _browse_video = crw.CausalReviewWidget._browse_video
    load_from_manifest = crw.CausalReviewWidget.load_from_manifest
    _pending_dir = crw.CausalReviewWidget._pending_dir
    _review_root = crw.CausalReviewWidget._review_root
    _load_next_video = crw.CausalReviewWidget._load_next_video
    _save_and_next_video = crw.CausalReviewWidget._save_and_next_video
    _pick_eligible = crw.CausalReviewWidget._pick_eligible
    _review_pool_paths = crw.CausalReviewWidget._review_pool_paths
    _flag_session = crw.CausalReviewWidget._flag_session
    _real_load_video = crw.CausalReviewWidget._load_video

    def __init__(self, broken=()):
        self._bundle_dir = None
        self._manifest = None
        self._queue_pending_dir = None
        self._video_stem = None
        self.video_path = None
        self._review_records = {}
        self._segment_list = []
        self._worklist = None
        self._pool_cache = None
        self._reviewed_this_session = set()
        self._status_label = SimpleNamespace(text=lambda: "", setText=lambda t: None)
        self._video_label = _Recorder()
        self._progress = _Recorder()
        self._notes_edit = SimpleNamespace(toPlainText=lambda: "")
        self._save_next_video_btn = _Button()
        self._broken = set(broken)
        self.loaded = []
        self.saves = []

    def _enable_controls(self, enabled):
        self._save_next_video_btn.setEnabled(enabled)

    def _load_video(self, path):
        if path.stem in self._broken:
            return self._real_load_video(path)   # runs the real failure handler
        self._video_stem, self.video_path = path.stem, path
        self.loaded.append((path.stem, self._bundle_dir))
        self._enable_controls(True)

    def _update_routing_banner(self):
        pass

    def _save_review(self):
        self.saves.append(self._video_stem)

    def _pop_from_index_if_resolved(self):
        pass

    def _watcher_db_for_move(self):
        return None, ""

    def _bundle_eligible(self, bundle, root):
        return True

    def _review_pool_paths_unfiltered(self, pending_dir, root):
        return sorted(d for d in Path(pending_dir).iterdir() if d.is_dir())


def _queue(tmp_path, *stems):
    pending = tmp_path / "Review" / "triage" / "Pending"
    for s in stems:
        b = pending / s
        b.mkdir(parents=True)
        mp4 = b / f"{s}.mp4"
        mp4.write_bytes(b"not a video")
        (b / f"{s}_manifest.json").write_text(json.dumps(
            {"video_stem": s, "canonical_video_path": str(mp4)}), encoding="utf-8")
    return pending


def _manifest(bundle):
    return json.loads((bundle / f"{bundle.name}_manifest.json").read_text(encoding="utf-8"))


def _quiet(monkeypatch):
    msgs = []
    for name in ("show_error", "show_info", "show_warning"):
        monkeypatch.setattr(crw, name, lambda m, *a, _n=name, **k: msgs.append((_n, m)))
    # Deterministic pick (first in name order); no lab priority file is read.
    monkeypatch.setattr(crw, "_pick_by_priority", lambda pool: (sorted(pool)[0], None))
    return msgs


def test_next_video_after_browse_goes_back_to_the_queue(tmp_path, monkeypatch):
    _quiet(monkeypatch)
    pending = _queue(tmp_path, OLD, NEW)
    w = _QueueWindow()
    w.load_from_manifest(_manifest(pending / OLD), pending / OLD)
    browsed = tmp_path / "filed" / f"{C}.mp4"
    monkeypatch.setattr(crw.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(browsed), "")))
    w._browse_video()
    assert w._bundle_dir is None

    w._save_and_next_video()

    # The browsed video's review was saved, then the queue was taken up again:
    # the next bundle, not the one left by Browse.
    assert w.saves == [C]
    assert w.loaded[-1] == (NEW, pending / NEW)
    assert w._bundle_dir == pending / NEW


def test_after_a_failed_load_next_video_skips_the_bundle_that_failed(tmp_path, monkeypatch):
    msgs = _quiet(monkeypatch)
    pending = _queue(tmp_path, OLD, NEW)
    w = _QueueWindow(broken={OLD})
    w._pool_cache = sorted(pending.iterdir())
    w.load_from_manifest(_manifest(pending / OLD), pending / OLD)   # fails to decode

    assert w._video_stem is None and w._bundle_dir is None
    # The next-video button stays usable: the queue is still known.
    assert w._save_next_video_btn.enabled
    assert any("Save Review + Next Video" in m for _, m in msgs)

    w._save_and_next_video()

    assert w.saves == []                         # nothing was loaded, nothing saved
    assert w.loaded == [(NEW, pending / NEW)]    # the failed bundle was skipped
    assert w._bundle_dir == pending / NEW


def test_flag_session_after_browse_writes_nothing_and_moves_nothing(tmp_path, monkeypatch):
    msgs = _quiet(monkeypatch)
    pending = _queue(tmp_path, OLD)
    w = _QueueWindow()
    w.load_from_manifest(_manifest(pending / OLD), pending / OLD)
    filed = tmp_path / "filed"
    filed.mkdir()
    browsed = filed / f"{C}.mp4"
    browsed.write_bytes(b"x")
    monkeypatch.setattr(crw.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(browsed), "")))
    w._browse_video()
    before = sorted(p for p in tmp_path.rglob("*"))
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))

    w._flag_session()

    assert routed == []
    assert sorted(p for p in tmp_path.rglob("*")) == before   # no flagged_sessions.json anywhere
    assert ("show_info", "No bundle loaded: nothing was flagged or moved.") in msgs
    assert (pending / OLD).is_dir()


def test_flag_session_with_no_queue_writes_nothing_beside_the_video(tmp_path, monkeypatch):
    msgs = _quiet(monkeypatch)
    filed = tmp_path / "filed"
    filed.mkdir()
    video = filed / f"{C}.mp4"
    video.write_bytes(b"x")
    w = _QueueWindow()
    w._video_stem, w.video_path = C, video

    w._flag_session()

    assert sorted(p.name for p in filed.iterdir()) == [f"{C}.mp4"]
    assert ("show_info", "No bundle loaded: nothing was flagged or moved.") in msgs


def test_a_first_bundle_with_no_video_does_not_strand_the_queue(tmp_path, monkeypatch):
    # The first bundle of a launch has no video anywhere: the next-video button
    # must still reach the rest of the queue, without a relaunch.
    msgs = _quiet(monkeypatch)
    monkeypatch.setattr("mousereach.review.bundle_media.filed_media_dirs", lambda s: [])
    pending = _queue(tmp_path, OLD, NEW)
    (pending / OLD / f"{OLD}.mp4").unlink()
    w = _QueueWindow()
    w.load_from_manifest(_manifest(pending / OLD), pending / OLD)

    assert w.loaded == [] and w._bundle_dir is None
    assert any("Could not open review bundle" in m for _, m in msgs)
    assert w._save_next_video_btn.enabled

    w._save_and_next_video()

    assert w.saves == []
    assert w.loaded == [(NEW, pending / NEW)]

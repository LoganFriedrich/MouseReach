"""No review tool recreates a queue bundle that has left the queue.

While a person has a bundle open, the watcher's return scan (or another
window's release) can move it out of the queue. The Review Queues release
button already wrote its clear marker create-only and never into a missing
folder. The other writers went through a helper that creates missing
folders: the review window's Clear button and its clear-on-complete save,
the segmentation fixer's release after a save, and the review save itself
-- so the bundle came back as a folder holding only a clear marker (a
"cleared" video with no files, held for ever) or only a review copy, and an
existing marker was overwritten. Now every clear marker goes through one
create-only writer, the fixer refuses to save into a folder that is gone,
and the review save keeps the answers in the lasting copy and says so.

Stand-in windows under tmp_path; invented ids; nothing real is moved.
"""
import json
import os
import shutil
from types import SimpleNamespace

import pytest

import mousereach.review.causal_review_io as cio
import mousereach.review.causal_review_widget as crw
import mousereach.review.fix_segmentation_widget as fsw
import mousereach.review.resegmented as reseg
from mousereach.review.fix_segmentation_widget import FixSegmentationWidget

STEM = "20990101_ABC9931_P1"


class _Label:
    def __init__(self):
        self._t = ""

    def setText(self, t):
        self._t = t

    def text(self):
        return self._t


def _clear_window(bundle, advanced):
    return SimpleNamespace(
        _video_stem=STEM, _bundle_dir=bundle, _save_review=lambda: None,
        _status_label=_Label(), _load_next_video=lambda: advanced.append(True))


@pytest.fixture
def quiet(monkeypatch):
    said = {"error": [], "info": []}
    monkeypatch.setattr(crw, "show_error", lambda m, *a, **k: said["error"].append(m))
    monkeypatch.setattr(crw, "show_info", lambda m, *a, **k: said["info"].append(m))
    monkeypatch.setattr(reseg, "resegmented_since_entry", lambda b, s: (True, ""))
    return said


def test_clear_does_not_recreate_a_bundle_that_left_the_queue(tmp_path, quiet):
    bundle = tmp_path / "deep_review" / STEM          # moved away: does not exist
    (tmp_path / "deep_review").mkdir()
    advanced = []
    crw.CausalReviewWidget._clear_deep_review(_clear_window(bundle, advanced))
    assert not bundle.exists()
    assert quiet["error"] and "left" in quiet["error"][0]
    assert advanced == []


def test_clear_keeps_a_marker_someone_else_wrote(tmp_path, quiet):
    bundle = tmp_path / "deep_review" / STEM
    bundle.mkdir(parents=True)
    marker = bundle / f"{STEM}_deep_review_cleared.json"
    marker.write_text(json.dumps({"cleared_by": "reviewer_b"}), encoding="utf-8")
    advanced = []
    crw.CausalReviewWidget._clear_deep_review(_clear_window(bundle, advanced))
    assert json.loads(marker.read_text(encoding="utf-8")) == {"cleared_by": "reviewer_b"}
    assert advanced == [True]


def test_clear_writes_the_marker_into_a_bundle_still_there(tmp_path, quiet):
    bundle = tmp_path / "deep_review" / STEM
    bundle.mkdir(parents=True)
    advanced = []
    crw.CausalReviewWidget._clear_deep_review(_clear_window(bundle, advanced))
    doc = json.loads((bundle / f"{STEM}_deep_review_cleared.json").read_text(encoding="utf-8"))
    assert doc["type"] == "deep_review_cleared" and advanced == [True]


class _Box:
    calls = []

    @classmethod
    def warning(cls, *a, **k):
        cls.calls.append(("warning",) + a[1:])

    @classmethod
    def critical(cls, *a, **k):
        cls.calls.append(("critical",) + a[1:])


def _fixer_bundle(tmp_path):
    b = tmp_path / "Processing" / "Review" / "deep_review" / STEM
    b.mkdir(parents=True)
    (b / f"{STEM}_routing.json").write_text(json.dumps(
        {"routed_at": "2000-01-01T09:00:00", "routed_reason": "segmentation: cuts wrong"}),
        encoding="utf-8")
    (b / f"{STEM}_segments.json").write_text(json.dumps(
        {"boundaries": [100, 900, 1700], "total_frames": 2400}), encoding="utf-8")
    return b


def _fixer(bundle):
    stub = SimpleNamespace(
        seg_path=bundle / f"{STEM}_segments.json",
        seg=json.loads((bundle / f"{STEM}_segments.json").read_text(encoding="utf-8")),
        video_stem=STEM, boundaries=[120, 900, 1700], original_boundaries=[100, 900, 1700],
        n_frames=2400, _pose_frames=2400, status=_Label(),
        _next_video=lambda: None,
        # Every cut walked (hand_cuts.BOUNDARY_CALLS_KEY).
        _calls={f: {"frame": f, "action": "accepted"} for f in (120, 900, 1700)},
        _removed_calls=[])
    stub._uncalled = lambda: fsw.FixSegmentationWidget._uncalled(stub)
    return stub


def test_fixer_release_does_not_recreate_a_bundle_moved_during_the_save(
        tmp_path, monkeypatch):
    # The return scan moves the bundle out of the queue right after the cuts
    # are written (simulated: the first file swap moves the folder away).
    monkeypatch.setattr(fsw, "QMessageBox", _Box)
    bundle = _fixer_bundle(tmp_path)
    moved_to = tmp_path / "Processing" / "returned"
    real_replace = os.replace
    state = {"moved": False}

    def replace_then_move(src, dst):
        real_replace(src, dst)
        if not state["moved"]:
            state["moved"] = True
            shutil.move(str(bundle), str(moved_to))
    monkeypatch.setattr(fsw.os, "replace", replace_then_move)
    FixSegmentationWidget._save(_fixer(bundle))
    monkeypatch.setattr(fsw.os, "replace", real_replace)
    assert state["moved"]
    assert not bundle.exists()


def test_fixer_refuses_to_save_into_a_bundle_that_is_gone(tmp_path, monkeypatch):
    monkeypatch.setattr(fsw, "QMessageBox", _Box)
    _Box.calls = []
    bundle = _fixer_bundle(tmp_path)
    stub = _fixer(bundle)
    shutil.rmtree(bundle)
    FixSegmentationWidget._save(stub)
    assert not bundle.exists()
    assert _Box.calls and _Box.calls[0][0] == "warning"


def test_the_review_save_keeps_answers_in_the_lasting_copy_and_makes_no_folder(
        tmp_path, monkeypatch):
    durable = tmp_path / "review_records" / "reviews"
    monkeypatch.setattr(cio, "durable_review_dir", lambda: durable, raising=False)
    gone = tmp_path / "triage" / STEM
    with pytest.raises(cio.ReviewFolderGone):
        cio.save_causal_review(
            video_stem=STEM, output_dir=gone,
            segments=[{"segment_num": 1, "human": {"outcome": "untouched"},
                       "answers": {}}],
            provenance={}, reviewer="reviewer_a", create_folder=False)
    assert not gone.exists()
    assert (durable / f"{STEM}_causal_review.json").is_file()


def test_the_release_button_uses_the_same_writer(tmp_path):
    from mousereach.review import release_cli
    gone = tmp_path / "deep" / STEM / f"{STEM}_deep_review_cleared.json"
    assert release_cli._write_marker_once(gone, {"by": "a"}) is False
    assert not gone.parent.exists()


def test_clear_that_could_not_write_its_marker_says_so_and_stays(tmp_path, quiet, monkeypatch):
    # The folder is there, but the share refuses the marker file.
    bundle = tmp_path / "deep_review" / STEM
    bundle.mkdir(parents=True)
    monkeypatch.setattr(cio, "write_clear_marker_once", lambda path, doc: False)
    advanced = []
    crw.CausalReviewWidget._clear_deep_review(_clear_window(bundle, advanced))
    assert advanced == []
    assert quiet["error"] and "NOT" in quiet["error"][0]
    assert not (bundle / f"{STEM}_deep_review_cleared.json").exists()


def test_a_failed_lasting_copy_is_not_reported_as_kept(tmp_path, monkeypatch):
    # An older lasting copy exists, but this save's write of it fails.
    durable = tmp_path / "review_records" / "reviews"
    durable.mkdir(parents=True)
    (durable / f"{STEM}_causal_review.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cio, "durable_review_dir", lambda: durable, raising=False)
    real = cio._write_json

    def refuse_durable(path, data, **kw):
        if path.parent == durable:
            raise OSError("share unavailable (simulated)")
        return real(path, data, **kw)
    monkeypatch.setattr(cio, "_write_json", refuse_durable)
    with pytest.raises(cio.ReviewFolderGone) as gone:
        cio.save_causal_review(
            video_stem=STEM, output_dir=tmp_path / "triage" / STEM,
            segments=[{"segment_num": 1, "human": {"outcome": "untouched"}, "answers": {}}],
            provenance={}, reviewer="reviewer_a", create_folder=False)
    assert "could NOT be saved" in str(gone.value)

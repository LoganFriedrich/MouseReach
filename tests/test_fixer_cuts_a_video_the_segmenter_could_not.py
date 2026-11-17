"""A video whose segmentation crashed can be cut by hand in the fixer.

When the segmenter raises, the watcher sends the video to the
re-segmentation queue with no segments file. Every way out of that queue
needs cuts a person saved in the segmentation fixer, and the fixer skipped
any bundle without a segments file -- so no tool could make the first cuts.
Now the fixer lists such a Pillar bundle and its save writes the first
segments file, which the pipeline keeps on the re-run.
"""
import json
from types import SimpleNamespace

import mousereach.review.fix_segmentation_widget as fsw
from mousereach.review.fix_segmentation_widget import FixSegmentationWidget
from mousereach.review.resegmented import resegmented_since_entry
from mousereach.review.triage_status import segmentation_failed

STEM = "20990101_ABC9901_P1"
REASON = "segmentation_failed: the segmenter raised an error"


def _uncut_bundle(queue, stem=STEM, left_for_a_person=False):
    b = queue / stem
    b.mkdir(parents=True)
    (b / f"{stem}_routing.json").write_text(json.dumps(
        {"routed_at": "2000-01-01T09:00:00", "routed_reason": REASON}), encoding="utf-8")
    (b / f"{stem}_manifest.json").write_text(json.dumps({"video_stem": stem}),
                                             encoding="utf-8")
    if left_for_a_person:
        from mousereach.watcher.review_routing import LEFT_NOTE
        (b / LEFT_NOTE).write_text("{}", encoding="utf-8")
    return b


class _Label:
    def __init__(self):
        self._t = ""

    def setText(self, t):
        self._t = t

    def text(self):
        return self._t


def _queue_of(queue):
    """The fixer's list for this queue folder (a stub stands in for the
    window, so no Qt is started)."""
    stub = SimpleNamespace(
        queue_dir=queue, header=_Label(), status=_Label(), _queue=[],
        _load_bundle=lambda b: None,
        _needs_reseg=lambda b, s: FixSegmentationWidget._needs_reseg(None, b, s),
        _uncut_bundle=getattr(FixSegmentationWidget, "_uncut_bundle", lambda b: False))
    FixSegmentationWidget._load_queue(stub)
    return [b.name for b in stub._queue]


def test_a_bundle_with_no_segments_file_is_listed(tmp_path, monkeypatch):
    monkeypatch.setattr("mousereach.watcher.work_priority.load_lab_policy",
                        lambda: (_ for _ in ()).throw(RuntimeError("no policy")))
    queue = tmp_path / "Processing" / "Review" / "deep_review"
    _uncut_bundle(queue)
    _uncut_bundle(queue, stem="20990101_ABC9902_E1")              # Easy: Clear releases it
    _uncut_bundle(queue, stem="20990101_ABC9903_P1", left_for_a_person=True)
    assert _queue_of(queue) == [STEM]


class _Box:
    calls = []

    @classmethod
    def warning(cls, *a, **k):
        cls.calls.append(("warning",) + a[1:])

    @classmethod
    def critical(cls, *a, **k):
        cls.calls.append(("critical",) + a[1:])


def _save(bundle, boundaries, n_frames):
    stub = SimpleNamespace(
        seg_path=bundle / f"{STEM}_segments.json", seg={}, video_stem=STEM,
        boundaries=list(boundaries), original_boundaries=[], n_frames=n_frames,
        _gw_records={}, _video_fps=30.0, status=_Label(), _next_video=lambda: None)
    FixSegmentationWidget._save(stub)
    return stub


def test_saving_writes_the_first_segments_file_and_releases_it(tmp_path, monkeypatch):
    monkeypatch.setattr(fsw, "QMessageBox", _Box)
    _Box.calls = []
    b = _uncut_bundle(tmp_path / "Processing" / "Review" / "deep_review")
    _save(b, [120, 900, 1700], n_frames=2400)
    assert _Box.calls == []
    seg = json.loads((b / f"{STEM}_segments.json").read_text(encoding="utf-8"))
    assert seg["boundaries"] == [120, 900, 1700]
    assert seg["boundary_source"] == "human"
    assert seg["total_frames"] == 2400 and seg["video_name"] == STEM
    assert seg["fps"] == 30.0
    # The readers downstream take it as a person's cuts, not a failed run.
    assert not segmentation_failed(seg)
    assert resegmented_since_entry(b, STEM)[0]
    # Sent for a segmentation reason: the save releases it.
    assert (b / f"{STEM}_deep_review_cleared.json").is_file()


def test_the_segmenter_keeps_the_hand_made_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fsw, "QMessageBox", _Box)
    b = _uncut_bundle(tmp_path / "Processing" / "Review" / "deep_review")
    _save(b, [120, 900, 1700], n_frames=2400)
    from mousereach.segmentation.core.batch import process_single
    result = process_single(b / f"{STEM}DLC_resnet50_testShuffle1_1000.h5")
    assert result.get("preserved_human_boundaries") is True
    seg = json.loads((b / f"{STEM}_segments.json").read_text(encoding="utf-8"))
    assert seg["boundaries"] == [120, 900, 1700]


def test_unknown_video_length_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(fsw, "QMessageBox", _Box)
    _Box.calls = []
    b = _uncut_bundle(tmp_path / "Processing" / "Review" / "deep_review")
    _save(b, [120, 900], n_frames=0)
    assert not (b / f"{STEM}_segments.json").exists()
    assert _Box.calls and _Box.calls[0][0] == "warning"

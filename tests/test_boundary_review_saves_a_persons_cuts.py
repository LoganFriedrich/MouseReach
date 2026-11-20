"""The per-step boundary review screen ("Step 2b - Review Boundaries",
mousereach-segment-review, the Boundaries tab of "Review Tool (Tabbed)") saves
a person's cuts exactly as the segmentation fixer does.

WHY: it used to rewrite {video}_segments.json in place with new boundaries
and validation_status "validated" but WITHOUT boundary_source "human" and
corrected_at. The next segmenter run then replaced the person's cuts, the
answers given on the old cuts were not voided (a re-cut voids everything
after segmentation), and nothing after segmentation was recomputed, so the
reach and outcome files still described the old cuts.

It also never asked where it was writing: a filed video, a crystallized one
or a triage bundle had its cuts rewritten in place. Those saves now write
nothing and say why.

Each save is called with a stand-in for the widget (no window is opened),
holding exactly what the save reads. Every folder is built under tmp_path
and every id is invented.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

STEM = "20990101_ZZQ0903_P1"
OLD = [100 + 100 * i for i in range(21)]
NEW = list(OLD)
NEW[5] += 12


@pytest.fixture(autouse=True)
def _roots(monkeypatch, tmp_path):
    """Point every configured root at tmp_path and stand in the side stores
    (pipeline index, performance log), whichever version of the code runs."""
    from mousereach.config import Paths
    share = tmp_path / "share"
    review = share / "Processing" / "Review"
    # The filed tree is there (empty): a filed tree that cannot be reached
    # makes "is this video locked?" unanswerable, and the save refuses.
    (share / "Analyzed").mkdir(parents=True)
    for name, value in (("NAS_ROOT", share),
                        ("PROCESSING_ROOT", tmp_path / "local"),
                        ("REVIEW_ROOT", review),
                        ("TRIAGE_REVIEW", review / "triage"),
                        ("DEEP_REVIEW", review / "deep_review"),
                        ("ANALYZED_OUTPUT", share / "Analyzed")):
        monkeypatch.setattr(Paths, name, value, raising=False)
    import mousereach.index as index_pkg
    monkeypatch.setattr(index_pkg, "PipelineIndex", MagicMock(), raising=False)
    try:
        import mousereach.performance as perf
        monkeypatch.setattr(perf, "PerformanceLogger", MagicMock(), raising=False)
    except Exception:
        pass
    import mousereach.segmentation.review_widget as rw
    monkeypatch.setattr(rw, "show_info", lambda *a, **k: None)
    monkeypatch.setattr(rw, "show_error", lambda *a, **k: None)
    monkeypatch.setattr(rw, "show_warning", lambda *a, **k: None)


def _video_folder(folder: Path, downstream=True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{STEM}.mp4").write_bytes(b"")
    (folder / f"{STEM}_segments.json").write_text(json.dumps({
        "video_name": STEM, "boundaries": OLD, "total_frames": 2400,
        "segmenter_version": "9.9.9", "validation_status": "needs_review"}),
        encoding="utf-8")
    if downstream:
        for suffix in ("_reaches.json", "_pellet_outcomes.json",
                       "_reach_assignments.json", "_features.json"):
            (folder / f"{STEM}{suffix}").write_text('{"segments": []}', encoding="utf-8")
    return folder


def _snapshot(folder: Path):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file()}


def _save(folder: Path, reviewed=None):
    from mousereach.segmentation.review_widget import BoundaryReviewWidget
    w = MagicMock()
    w.video_path = folder / f"{STEM}.mp4"
    w.boundaries = list(NEW)
    w._original_boundaries = list(OLD)
    w.fps = 60.0
    w._algorithm_version = "9.9.9"
    # Every cut accepted or set on the screen (its boundaries_reviewed list).
    w.boundaries_reviewed = list(reviewed) if reviewed is not None else [True] * len(NEW)
    w.saved = BoundaryReviewWidget._save_validated(w)
    return w


def _status(w) -> str:
    calls = w.status_label.setText.call_args_list
    return calls[-1][0][0] if calls else ""


def test_saved_cuts_are_a_persons_cuts_and_void_later_outputs(tmp_path):
    from mousereach.review.causal_review_io import cuts_set_at, segmentation_is_human
    folder = _video_folder(tmp_path / "local" / "Processing")
    assert _save(folder).saved is True
    seg = json.loads((folder / f"{STEM}_segments.json").read_text(encoding="utf-8"))
    assert seg["boundaries"] == NEW
    assert seg.get("boundary_source") == "human"
    assert seg.get("corrected_at")
    assert seg.get("algo_boundaries") == OLD
    # A re-cut voids every answer after segmentation: the cuts carry the
    # time they were set.
    assert segmentation_is_human(seg)
    assert cuts_set_at(seg) == seg["corrected_at"]
    # Still the hand-run sign-off.
    assert seg["validation_status"] == "validated"
    # Everything computed from the old cuts is archived and removed.
    for suffix in ("_reaches.json", "_pellet_outcomes.json",
                   "_reach_assignments.json", "_features.json"):
        assert not (folder / f"{STEM}{suffix}").exists(), suffix
    archived = list((tmp_path / "share" / "Processing" / "_archived").rglob("*.json"))
    names = {p.name for p in archived}
    assert f"{STEM}_segments.json" in names
    assert f"{STEM}_reaches.json" in names
    old_copy = [p for p in archived if p.name == f"{STEM}_segments.json"][0]
    assert json.loads(old_copy.read_text(encoding="utf-8"))["boundaries"] == OLD
    assert not list(folder.glob("*.tmp"))


def test_the_segmenter_keeps_cuts_saved_here(tmp_path, monkeypatch):
    from mousereach.archive.supersede import is_human_cuts
    from mousereach.segmentation.core import batch
    folder = _video_folder(tmp_path / "local" / "Processing", downstream=False)
    _save(folder)
    seg_path = folder / f"{STEM}_segments.json"
    assert is_human_cuts(seg_path)
    before = seg_path.read_bytes()
    ran = []

    class _Ran(RuntimeError):
        pass

    def segmenter(*a, **k):
        ran.append(a)
        raise _Ran("the segmenter ran")
    monkeypatch.setattr(batch, "segment_video_multi", segmenter)
    dlc = folder / f"{STEM}DLC_resnet50_invented.h5"
    dlc.write_bytes(b"")
    try:
        batch.process_single(dlc)
    except _Ran:
        pass
    assert ran == []
    assert seg_path.read_bytes() == before


def test_a_filed_video_is_not_changed(tmp_path):
    from mousereach.pipeline.publication_lock import filed_folder_of
    folder = _video_folder(filed_folder_of(STEM))
    before = _snapshot(folder)
    w = _save(folder)
    assert w.saved is False
    assert _snapshot(folder) == before
    assert "NOT saved" in _status(w)
    assert not (tmp_path / "share" / "Processing" / "_archived").exists()


def test_a_crystallized_video_is_not_changed_anywhere(tmp_path):
    from mousereach.pipeline.publication_lock import filed_folder_of, LOCK_SUFFIX
    filed = filed_folder_of(STEM)
    filed.mkdir(parents=True, exist_ok=True)
    (filed / f"{STEM}{LOCK_SUFFIX}").write_text(json.dumps({"label": "PAPER_A"}),
                                                 encoding="utf-8")
    folder = _video_folder(tmp_path / "local" / "Processing")
    before = _snapshot(folder)
    w = _save(folder)
    assert w.saved is False
    assert _snapshot(folder) == before
    assert "crystallized" in _status(w)


def test_a_triage_bundle_is_not_recut_here(tmp_path):
    folder = _video_folder(tmp_path / "share" / "Processing" / "Review" / "triage" / STEM)
    before = _snapshot(folder)
    w = _save(folder)
    assert w.saved is False
    assert _snapshot(folder) == before
    assert "Escalate" in _status(w)


def test_a_deep_review_bundle_is_released_as_the_fixer_releases_it(tmp_path):
    from mousereach.review.resegmented import resegmented_since_entry
    folder = _video_folder(tmp_path / "share" / "Processing" / "Review"
                           / "deep_review" / STEM)
    (folder / f"{STEM}_routing.json").write_text(json.dumps(
        {"routed_at": "2000-01-01T09:00:00",
         "routed_reason": "segmentation: cuts wrong"}), encoding="utf-8")
    w = _save(folder)
    ok, why = resegmented_since_entry(folder, STEM)
    assert ok, why
    marker = folder / f"{STEM}_deep_review_cleared.json"
    assert marker.is_file()
    assert json.loads(marker.read_text(encoding="utf-8"))["gated_on"] == "boundary_source"
    assert "Released" in _status(w)


def test_a_folder_that_left_while_open_is_not_recreated(tmp_path):
    import shutil
    folder = _video_folder(tmp_path / "share" / "Processing" / "Review"
                           / "deep_review" / STEM)
    shutil.rmtree(folder)
    w = _save(folder)
    assert not folder.exists()
    assert "NOT saved" in _status(w)


def test_a_deep_review_bundle_with_cuts_not_all_accepted_is_not_released(tmp_path):
    """Owner decision MRR-08d (2026-11-19): a video leaves re-segmentation only
    when every cut was explicitly accepted or set -- on this screen, the
    boundaries it marks reviewed."""
    from mousereach.review.resegmented import resegmented_since_entry
    folder = _video_folder(tmp_path / "share" / "Processing" / "Review"
                           / "deep_review" / STEM)
    (folder / f"{STEM}_routing.json").write_text(json.dumps(
        {"routed_at": "2000-01-01T09:00:00",
         "routed_reason": "segmentation: cuts wrong"}), encoding="utf-8")
    w = _save(folder, reviewed=[True] + [False] * (len(NEW) - 1))
    assert w.saved is True
    ok, why = resegmented_since_entry(folder, STEM)
    assert not ok and "accepted or moved" in why
    assert not (folder / f"{STEM}_deep_review_cleared.json").exists()
    assert "NOT released" in _status(w)

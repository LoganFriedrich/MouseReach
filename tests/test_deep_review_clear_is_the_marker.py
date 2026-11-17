"""A video leaves the re-segmentation queue only through its clear marker.

A ground-truth file in a re-segmentation bundle used to count as a clear by
its mere presence (even a progress save with nothing determined, and with
nobody pressing Clear), and its boundaries counted as proof that a person had
re-segmented the video. Nothing writes ground-truth boundaries into the
segments file, so the released video was cut again by the segmenter exactly
as before. Now:

- the return scan reads "cleared" from the marker only, as the gate does;
- ground truth is not proof of re-segmentation (cuts are set in the fixer);
- in the re-segmentation queue the review window does not hide a video
  because it has ground truth, so every such video still has a Clear button.
"""
import json
from types import SimpleNamespace

import pytest

import mousereach.watcher.review_return as rr
from mousereach.review.causal_review_widget import CausalReviewWidget
from mousereach.review.fix_segmentation_widget import FixSegmentationWidget
from mousereach.review.resegmented import resegmented_since_entry

PILLAR = "20990101_ABC9901_P1"
EASY = "20990101_ABC9901_E1"
ROUTED = "2099-01-10T09:00:00"
AFTER = "2099-01-11T09:00:00"


def _gt(certified):
    doc = {"segmentation": {"boundaries": [
        {"index": i, "frame": f, "determined": True, "determined_at": AFTER}
        for i, f in enumerate((100, 200, 300))]}}
    if certified:
        # Certified as the pipeline computes it (triage_status.
        # gt_resolved_segments): both lists exhaustive AND every pellet's
        # outcome determined; a stored completion_status alone does not count.
        doc["completion_status"] = {"all_complete": True}
        doc["reaches"] = {"exhaustive": True}
        doc["outcomes"] = {"exhaustive": True, "segments": [
            {"segment_num": k, "determined": True, "outcome": "untouched"}
            for k in (1, 2, 3)]}
    return doc


def _bundle(root, stem=PILLAR, reason="segmentation: cuts in question",
            certified=False):
    b = root / stem
    b.mkdir(parents=True)
    (b / f"{stem}_routing.json").write_text(
        json.dumps({"routed_at": ROUTED, "routed_reason": reason}), encoding="utf-8")
    (b / f"{stem}_manifest.json").write_text(json.dumps({"video_stem": stem}),
                                             encoding="utf-8")
    (b / f"{stem}_segments.json").write_text(
        json.dumps({"boundaries": [100, 200, 300]}), encoding="utf-8")
    (b / f"{stem}_unified_ground_truth.json").write_text(
        json.dumps(_gt(certified)), encoding="utf-8")
    return b


def _scan(tmp_path, monkeypatch, bundle_root):
    # A failed return earlier in the run backs a stem off for a while; start
    # this scan with none.
    monkeypatch.setattr(rr, "_return_retry_after", {})
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", bundle_root)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", tmp_path / "Review" / "triage")
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    returned = []
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: returned.append(a[1]) or True)
    rr.scan_review_queues(None, tmp_path / "Processing")
    return returned


def test_ground_truth_alone_does_not_release_a_video(tmp_path, monkeypatch):
    deep = tmp_path / "Review" / "deep_review"
    _bundle(deep)
    assert not rr._deep_review_cleared(deep / PILLAR, PILLAR)
    assert _scan(tmp_path, monkeypatch, deep) == []


def test_ground_truth_alone_does_not_release_an_easy_tray(tmp_path, monkeypatch):
    deep = tmp_path / "Review" / "deep_review"
    _bundle(deep, stem=EASY)
    assert _scan(tmp_path, monkeypatch, deep) == []


def test_ground_truth_boundaries_are_not_proof_of_re_segmentation(tmp_path):
    b = _bundle(tmp_path)
    ok, why = resegmented_since_entry(b, PILLAR)
    assert not ok and "segmentation fixer" in why


def test_marker_and_fixer_cuts_release_it(tmp_path, monkeypatch):
    deep = tmp_path / "Review" / "deep_review"
    b = _bundle(deep)
    (b / f"{PILLAR}_segments.json").write_text(json.dumps(
        {"boundaries": [100, 210, 300], "boundary_source": "human",
         "corrected_at": AFTER}), encoding="utf-8")
    (b / f"{PILLAR}_deep_review_cleared.json").write_text("{}", encoding="utf-8")
    assert _scan(tmp_path, monkeypatch, deep) == [PILLAR]


def _window_lists(bundle):
    """What the review window decides for this bundle in the re-segmentation
    queue: its own eligibility rule, run on a stub standing in for the window
    (no Qt is started)."""
    W = CausalReviewWidget
    stub = SimpleNamespace(_deep_review=True, _triage_only=False, _worklist=None,
                           _has_triage_cache={}, _held_logged=set())
    for name in ("_bundle_reviewed", "_held_for_segmentation", "_bundle_has_triage"):
        setattr(stub, name, (lambda f: lambda *a: f(stub, *a))(getattr(W, name)))
    stub._bundle_needs_review = lambda b, r: W._bundle_needs_review(stub, b, r)
    return W._bundle_eligible(stub, bundle, bundle.parent)


def _fixer_lists(bundle):
    seg = json.loads((bundle / f"{bundle.name}_segments.json").read_text(encoding="utf-8"))
    return FixSegmentationWidget._needs_reseg(None, bundle, seg)


@pytest.mark.parametrize("stem", [PILLAR, EASY])
@pytest.mark.parametrize("certified", [True, False])
@pytest.mark.parametrize("reason", ["segmentation: cuts in question",
                                    "qc_needs_review",
                                    "reviewer flagged session ABC9901_20990101"])
def test_every_ground_truth_bundle_keeps_a_way_out(tmp_path, stem, certified, reason):
    """Whatever brought it here, a video with ground truth is listed in the
    re-segmentation window (whose Clear writes the marker); a Pillar tray is
    also listed in the fixer, where its cuts are set."""
    b = _bundle(tmp_path, stem=stem, reason=reason, certified=certified)
    assert _window_lists(b)
    if stem == PILLAR:
        assert _fixer_lists(b)


def test_triage_window_still_treats_ground_truth_as_the_answer(tmp_path):
    b = _bundle(tmp_path)
    stub = SimpleNamespace(_deep_review=False, _bundle_reviewed=lambda b: False)
    assert not CausalReviewWidget._bundle_needs_review(stub, b, tmp_path)


def _seg_failed_triage_bundle(triage, certified):
    """A triage bundle whose segmentation reports failure, its one triaged
    pellet answered, ground truth beside it and NO clear marker."""
    b = triage / PILLAR
    b.mkdir(parents=True)
    (b / f"{PILLAR}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.0, "boundaries": []}), encoding="utf-8")
    (b / f"{PILLAR}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 3, "outcome": "triaged"}]}),
        encoding="utf-8")
    (b / f"{PILLAR}_causal_review.json").write_text(json.dumps({"segments": [
        {"segment_num": 3, "human": {"outcome": "untouched"},
         "answers": {"reviewed": True}}]}), encoding="utf-8")
    (b / f"{PILLAR}_unified_ground_truth.json").write_text(
        json.dumps(_gt(certified)), encoding="utf-8")
    return b


@pytest.mark.parametrize("certified, diverted", [(True, False), (False, True)])
def test_triage_scan_honours_only_ground_truth_the_gate_honours(
        tmp_path, monkeypatch, certified, diverted):
    """Ground truth certifying the whole video passes the gate clean even with
    a failed segmentation, so the triage scan does not divert it to
    re-segmentation; partial ground truth does not stop the divert (the gate
    would send that video to re-segmentation anyway)."""
    import mousereach.watcher.review_gate as rg
    from mousereach.review import causal_review_io as cio
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    deep.mkdir(parents=True)
    _seg_failed_triage_bundle(triage, certified)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr, "_return_retry_after", {})
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append(a))
    monkeypatch.setattr(rr, "_return_to_processing", lambda *a, **k: True)

    class _DB:
        def get_video(self, vid):
            return {"video_id": vid, "state": "triage"}

        def force_state(self, *a, **k):
            pass

    rr.scan_review_queues(_DB(), tmp_path / "proc")
    assert bool(routed) is diverted


@pytest.mark.parametrize("certified", [True, False])
def test_recut_for_another_reason_with_ground_truth_still_has_clear(tmp_path, certified):
    """After the fixer save (which does not release a video sent for a quality
    hold), the window still lists it, so Clear is reachable."""
    b = _bundle(tmp_path, reason="qc_needs_review", certified=certified)
    (b / f"{PILLAR}_segments.json").write_text(json.dumps(
        {"boundaries": [100, 200, 300], "boundary_source": "human",
         "corrected_at": AFTER}), encoding="utf-8")
    assert not _fixer_lists(b)
    assert _window_lists(b)

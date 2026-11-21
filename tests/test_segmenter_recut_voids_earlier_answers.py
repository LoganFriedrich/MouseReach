"""A re-cut by the SEGMENTER voids earlier answers exactly as a person's does.

Owner ruling R2 (narrowed the same day): a re-cut voids every answer given
before it, except an answer on a pellet whose frames the re-cut did not
change. A person's re-cut in the segment fixer has followed it since the
fixer stamps its save (``corrected_at`` and the cuts it replaced). But a
triage return whose algorithm cuts are out of date resumes at segmentation,
and the SEGMENTER cuts again: its new file recorded no time, so the review
readers saw no re-cut at all and fitted the earlier answers onto the new cuts
by frame overlap -- an answer about other footage applied to a pellet nobody
judged on these cuts.

Now the segmenter records, in the file it writes, when its cuts replaced
different ones and which cuts they replaced (separate keys, never the fixer's
person-only ones), and every reader uses the same one rule
(causal_review_io.answer_survives_recut). The same cuts again are not a
re-cut. A person's mislabel statement is still settled only by a person.

Everything under tmp_path; invented ids. The segmenter's cut-finding is
replaced by a stub that returns given cuts; the file it writes is the real
save_segmentation's.
"""
import json

import pytest

import mousereach.review.causal_review_io as io
import mousereach.segmentation.core.batch as segb
import mousereach.watcher.review_gate as rg
from mousereach.segmentation.core.segmenter_robust import SegmentationDiagnostics

VID = "20990101_ABC9921_P1"
OLD_CUTS = [0, 100, 190, 300]     # the cuts the person answered on
NEW_CUTS = [0, 100, 200, 300]     # the segmenter's new cuts: pellet 1 unmoved, 2 and 3 moved
BEFORE = "2020-03-01T09:00:00"


def _diag(cuts):
    n = len(cuts)
    return SegmentationDiagnostics(
        video_name=VID, total_frames=400, fps=60.0, box_center=100.0, boxl_std=0.0,
        boxr_std=0.0, reference_quality="good", sabl_coverage=1.0, sabr_coverage=1.0,
        satl_coverage=1.0, satr_coverage=1.0, n_primary_candidates=n,
        n_secondary_candidates=0, n_fallback_candidates=0, boundaries=list(cuts),
        boundary_methods=["primary"] * n, boundary_confidences=[0.9] * n, anomalies=[],
        interval_mean=100.0, interval_std=0.0, interval_cv=0.0)


def _answer(num, when=BEFORE, cuts=OLD_CUTS):
    a, b = io.pellet_span(cuts, num - 1)
    return {"segment_num": num, "pellet_num": num, "segment_span": {"start": a, "end": b},
            "reviewer": "reviewer_a", "reviewed_at": when,
            "human": {"outcome": "untouched", "causal_reach": None}, "answers": {}}


def _folder(path, cuts=OLD_CUTS, review=True, seg_text=None):
    path.mkdir(parents=True, exist_ok=True)
    if seg_text is not None:
        (path / f"{VID}_segments.json").write_text(seg_text, encoding="utf-8")
    elif cuts is not None:
        (path / f"{VID}_segments.json").write_text(json.dumps(
            {"boundaries": list(cuts), "segmenter_version": "old",
             "segmented_at": "2020-02-01T00:00:00", "overall_confidence": 0.9}),
            encoding="utf-8")
    if review:
        (path / f"{VID}_causal_review.json").write_text(json.dumps(
            {"video_stem": VID, "reviewed_at": BEFORE,
             "segments": [_answer(n) for n in (1, 2, 3)]}), encoding="utf-8")
    return path


def _downstream(path, pellets=3):
    """The steps after segmentation ran again on the new cuts: outcomes (the
    algorithm called every pellet 'retrieved', nothing flagged) and a causal
    reach for each, written after the segments file."""
    (path / f"{VID}_pellet_outcomes.json").write_text(json.dumps({
        "video_name": VID, "segments": [
            {"segment_num": n, "outcome": "retrieved", "flagged_for_review": False}
            for n in range(1, pellets + 1)]}), encoding="utf-8")
    (path / f"{VID}_reach_assignments.json").write_text(json.dumps({"reaches": [
        {"reach_id": n, "segment_num": n, "is_causal": True, "segment_ifr": None}
        for n in range(1, pellets + 1)]}), encoding="utf-8")


@pytest.fixture
def segment(monkeypatch, tmp_path):
    monkeypatch.setattr(io, "_GT_INDEX_CACHE", {})
    monkeypatch.setattr(io, "durable_review_dir", lambda: None, raising=False)
    monkeypatch.setattr(io, "review_dirs", lambda stem, *a, **k: [tmp_path / "work"],
                        raising=False)

    def run(cuts, folder, **kw):
        monkeypatch.setattr(segb, "segment_video_multi", lambda p: (list(cuts), _diag(cuts)))
        result = segb.process_single(folder / f"{VID}DLC_x.h5", **kw)
        assert result["success"], result
        return json.loads((folder / f"{VID}_segments.json").read_text(encoding="utf-8"))
    return run


def _review(folder):
    return json.loads((folder / f"{VID}_causal_review.json").read_text(encoding="utf-8"))


def test_a_segmenter_recut_voids_answers_on_pellets_it_moved(tmp_path, segment):
    work = _folder(tmp_path / "work")
    seg = segment(NEW_CUTS, work)
    answered = io.answered_segments(_review(work), io.segments_from_segmentation(seg))
    # Pellet 1 (frames 0-99) did not move: its answer stands. Pellets 2 and 3
    # moved: their answers were given on other footage and are void.
    assert answered == {1}


def test_the_gate_drops_the_voided_answers_and_asks_nothing(tmp_path, segment):
    # owner ruling 2026-11-21: the voided answers on pellets 2 and 3 are
    # dropped; the re-run algorithm flagged nothing, so the video is clean
    work = _folder(tmp_path / "work")
    segment(NEW_CUTS, work)
    _downstream(work)
    decision, reason, _ = rg.evaluate_gate(VID, work)
    assert decision == rg.DECISION_CLEAN, reason
    assert rg.recut_voided_open(VID, work) == set()


def test_the_segmenter_never_passes_for_a_person(tmp_path, segment):
    work = _folder(tmp_path / "work")
    seg = segment(NEW_CUTS, work)
    assert seg.get("boundary_source") != "human"
    assert "corrected_at" not in seg and "algo_boundaries" not in seg


def test_the_same_cuts_again_void_nothing(tmp_path, segment):
    work = _folder(tmp_path / "work")
    seg = segment(OLD_CUTS, work)
    answered = io.answered_segments(_review(work), io.segments_from_segmentation(seg))
    assert answered == {1, 2, 3}
    _downstream(work)
    decision, reason, _ = rg.evaluate_gate(VID, work)
    assert decision == rg.DECISION_CLEAN, reason


def test_the_same_cuts_again_keep_an_earlier_recut_record(tmp_path, segment):
    work = _folder(tmp_path / "work")
    first = segment(NEW_CUTS, work)
    again = segment(NEW_CUTS, work)
    for key in (io.SEGMENTER_RECUT_AT_KEY, io.SEGMENTER_REPLACED_KEY):
        assert again.get(key) == first.get(key)
    answered = io.answered_segments(_review(work), io.segments_from_segmentation(again))
    assert answered == {1}


def test_an_answer_given_after_the_recut_stands(tmp_path, segment):
    work = _folder(tmp_path / "work")
    seg = segment(NEW_CUTS, work)
    doc = _review(work)
    doc["segments"].append(_answer(2, when="2999-01-01T00:00:00", cuts=NEW_CUTS))
    doc["segments"] = [r for r in doc["segments"]
                       if not (r["segment_num"] == 2 and r["reviewed_at"] == BEFORE)]
    assert io.answered_segments(doc, io.segments_from_segmentation(seg)) == {1, 2}


def test_unreadable_earlier_cuts_void_every_earlier_answer(tmp_path, segment):
    work = _folder(tmp_path / "work", seg_text="{ not json")
    seg = segment(NEW_CUTS, work)
    assert io.answered_segments(_review(work), io.segments_from_segmentation(seg)) == set()


def test_bring_current_passes_the_filed_cuts(tmp_path, segment):
    # An empty work folder; the earlier cuts are in the filed folder.
    filed = _folder(tmp_path / "filed", review=False)
    work = _folder(tmp_path / "work", cuts=None)
    seg = segment(NEW_CUTS, work, output_dir=work,
                  previous_cuts=filed / f"{VID}_segments.json")
    assert seg.get(io.SEGMENTER_REPLACED_KEY) == OLD_CUTS
    assert io.answered_segments(_review(work), io.segments_from_segmentation(seg)) == {1}


def test_a_segmenter_recut_does_not_settle_a_persons_mislabel(tmp_path, segment):
    # A person said frames 100-189 (segment 2) are really pellet 3. The
    # segmenter re-cutting the video afterwards is nobody's word on that.
    from mousereach.watcher.hand_scores import unresolved_mislabels
    work = _folder(tmp_path / "work", review=False)
    seg = segment(NEW_CUTS, work)
    rec = dict(_answer(2), true_segment_num=3, segmentation_wrong=True)
    assert unresolved_mislabels({"segments": [rec]}, seg) == {2: 2}

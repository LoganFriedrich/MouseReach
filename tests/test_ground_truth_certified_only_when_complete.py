"""A video's ground truth certifies the WHOLE video only when it really is
complete: reaches and outcomes both marked exhaustive AND every pellet's
outcome determined by a person. A stored ``completion_status`` is not trusted.

WHY: a fully certified video passes the review gate as clean -- even with a
failed segmentation -- and its ground truth stands in for every element. The
two "exhaustive" switches can be set in the ground-truth tool while some
outcomes are still undetermined, and a stored completion_status can be stale
against the file, so neither alone may let a video skip review.
"""
import json
import os

import pytest

VID = "20990101_XYZ0101_P1"


def _video(folder, n=3, flagged=(1, 2, 3)):
    (folder / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": k, "outcome": "triaged" if k in flagged else "untouched",
         "flagged_for_review": k in flagged} for k in range(1, n + 1)]}), encoding="utf-8")
    (folder / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [1000 * i for i in range(n + 1)]}),
        encoding="utf-8")
    # segments older than the outcomes, as a normal run leaves them (else the
    # gate sends the video round again for stale outputs)
    old = (folder / f"{VID}_pellet_outcomes.json").stat().st_mtime - 60
    os.utime(folder / f"{VID}_segments.json", (old, old))


def _gt(folder, determined, undetermined=(), rr=True, oo=True, completion=None):
    doc = {"reaches": {"exhaustive": rr},
           "outcomes": {"exhaustive": oo, "segments":
                        [{"segment_num": k, "determined": True, "outcome": "untouched"}
                         for k in determined]
                        + [{"segment_num": k, "determined": False} for k in undetermined]}}
    if completion is not None:
        doc["completion_status"] = {"all_complete": completion}
    (folder / f"{VID}_unified_ground_truth.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.fixture(autouse=True)
def empty_gt_index(monkeypatch):
    import mousereach.review.causal_review_io as cio
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})


def test_exhaustive_flags_with_an_undetermined_outcome_do_not_certify(tmp_path):
    import mousereach.watcher.review_gate as rg
    _video(tmp_path)
    _gt(tmp_path, determined=(1, 2), undetermined=(3,))
    decision, _, st = rg.evaluate_gate(VID, tmp_path)
    assert decision == rg.DECISION_TRIAGE
    assert 3 in st.unresolved


def test_a_stored_completion_status_alone_does_not_certify(tmp_path):
    import mousereach.watcher.review_gate as rg
    _video(tmp_path)
    _gt(tmp_path, determined=(), rr=False, oo=False, completion=True)
    assert rg.evaluate_gate(VID, tmp_path)[0] == rg.DECISION_TRIAGE


def test_an_outcome_missing_for_a_current_segment_does_not_certify(tmp_path):
    from mousereach.review.triage_status import gt_resolved_segments
    _video(tmp_path)
    _gt(tmp_path, determined=(1, 2))      # segment 3 not listed at all
    full, segs = gt_resolved_segments(VID, extra_dirs=(tmp_path,))
    assert not full and segs == {1, 2}


def test_flags_set_and_every_outcome_determined_certifies(tmp_path):
    import mousereach.watcher.review_gate as rg
    _video(tmp_path)
    _gt(tmp_path, determined=(1, 2, 3))
    decision, reason, _ = rg.evaluate_gate(VID, tmp_path)
    assert decision == rg.DECISION_CLEAN and reason == "ground_truth_exhaustive"


def test_a_short_segmenter_artifact_does_not_block_full_certification(tmp_path):
    """The segmenter left a 3-frame sliver between two real segments. No
    pellet presentation fits in it, so no outcome is ever mapped onto it; the
    person determined every real pellet, so the video is fully certified --
    wanting the sliver too would send it to deep review with nothing for the
    person to fix in the ground-truth tool."""
    from mousereach.review.triage_status import gt_resolved_segments
    (tmp_path / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": k, "outcome": "untouched"} for k in (1, 2, 3, 4)]}), encoding="utf-8")
    (tmp_path / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [0, 1000, 2000, 2003, 3000]}),
        encoding="utf-8")
    _gt(tmp_path, determined=(1, 2, 4))
    full, segs = gt_resolved_segments(VID, extra_dirs=(tmp_path,))
    assert full, segs

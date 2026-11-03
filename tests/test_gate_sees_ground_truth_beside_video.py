"""Ground truth beside a video is a person's answer, and every check that
decides whether to ask a person sees it: the review gate and
mousereach-route-to-queue, through triage_status.gt_resolved_segments. (The
release check of bundles already in a queue does not yet -- see the last test.)

WHY (2026-11-03): only the corpus-wide ground-truth index was read, and it is
built once per process -- a re-run copies the ground truth into the processing
folder after that, so the gate could not see it and sent videos whose open
segments ground truth had answered back to a person. Logan, 2026-11-03: "Any
segment whose boundaries and causal reach and outcome have all already been
reviewed by a human should never under any condition require a human to look
at it ever again."
"""
import json

import pytest

VID = "20250709_ABC0207_P1"


# ---------------------------------------------------------------------------
# Ground truth beside the video is a person's answer; the gate sees it
# ---------------------------------------------------------------------------

def _flagged_video(folder, flagged=(1,), n_segments=3):
    import os
    outcomes = folder / f"{VID}_pellet_outcomes.json"
    outcomes.write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "triaged" if n in flagged else "untouched",
         "flagged_for_review": n in flagged} for n in range(1, n_segments + 1)]}),
        encoding="utf-8")
    seg = folder / f"{VID}_segments.json"
    seg.write_text(json.dumps({"overall_confidence": 0.9,
                               "boundaries": [1000 * i for i in range(n_segments + 1)]}),
                   encoding="utf-8")
    old = outcomes.stat().st_mtime - 60
    os.utime(seg, (old, old))


def _gt(folder, segs, full=False):
    (folder / f"{VID}_unified_ground_truth.json").write_text(json.dumps({
        "completion_status": {"all_complete": full},
        "outcomes": {"segments": [{"segment_num": n, "determined": True} for n in segs]}}),
        encoding="utf-8")


@pytest.fixture
def empty_gt_index(monkeypatch):
    """The corpus-wide ground-truth index knows nothing -- as when it was built
    before a re-run copied the ground truth into the processing folder."""
    import mousereach.review.causal_review_io as cio
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})


def test_the_gate_counts_ground_truth_beside_the_video_as_an_answer(tmp_path, empty_gt_index):
    import mousereach.watcher.review_gate as rg
    _flagged_video(tmp_path, flagged=(2,))
    decision, _, _ = rg.evaluate_gate(VID, tmp_path)
    assert decision == rg.DECISION_TRIAGE           # nobody answered segment 2
    _gt(tmp_path, [2])
    decision, _, _ = rg.evaluate_gate(VID, tmp_path)
    assert decision == rg.DECISION_CLEAN            # ground truth answered it


def test_the_gate_still_asks_about_a_segment_ground_truth_did_not_answer(tmp_path, empty_gt_index):
    import mousereach.watcher.review_gate as rg
    _flagged_video(tmp_path, flagged=(1, 2))
    _gt(tmp_path, [2])
    decision, reason, st = rg.evaluate_gate(VID, tmp_path)
    assert decision == rg.DECISION_TRIAGE and st.unresolved - {2} == {1}


def test_exhaustive_ground_truth_beside_the_video_clears_it(tmp_path, empty_gt_index):
    import mousereach.watcher.review_gate as rg
    _flagged_video(tmp_path, flagged=(1, 2, 3))
    _gt(tmp_path, [], full=True)
    assert rg.evaluate_gate(VID, tmp_path)[0] == rg.DECISION_CLEAN


def test_route_to_queue_sees_it_too(tmp_path, empty_gt_index):
    from mousereach.watcher.route_cli import answered_segments_of
    _flagged_video(tmp_path, flagged=(2,))
    _gt(tmp_path, [2])
    assert 2 in answered_segments_of(VID, tmp_path)


def test_the_release_check_does_not_yet_read_ground_truth_inside_a_bundle(tmp_path, empty_gt_index):
    """Held back on purpose until the queue hold is lifted (see the comment in
    triage_status.triage_status): reading it would release bundles on the
    next scan."""
    from mousereach.review.triage_status import triage_status
    _flagged_video(tmp_path, flagged=(2,))
    _gt(tmp_path, [2])
    assert triage_status(tmp_path, VID).unresolved == {2}

"""An answer the pipeline cannot count is never recorded, and the reviewer is told.

WHY (2026-10-23): a reviewer answered a triaged pellet with a reach outside that
segment's frames; the pipeline dropped the answer (it cannot belong to the
pellet as the video is cut) without a word on screen, and the video came back
with the same question -- one segment was answered three times. Another video
asked about a segment four frames long, which no answer can count for. Logan's
rules: everything that holds a video must be visible in the GUI; only what a
person can fix in the GUI may hold a video; and when the event belongs to a
different pellet, the reviewer saves nothing and escalates to deep review.
"""
import json
from types import SimpleNamespace

import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.review.causal_review_io import answer_problem, plain_review_note
from mousereach.watcher.review_gate import DECISION_CLEAN, DECISION_DEEP, evaluate_gate


def _rec(span, reach=None, outcome="displaced_sa", **answers):
    return {"segment_num": 17, "segment_span": {"start": span[0], "end": span[1]},
            "human": {"outcome": outcome,
                      "causal_reach": ({"start": reach[0], "end": reach[1]} if reach else None)},
            "answers": answers or {"outcome_pick": outcome}}


def test_a_reach_outside_the_segment_cannot_count():
    msg = answer_problem(_rec((34355, 34422), (34446, 34462)))
    assert "outside segment 17" in msg and "34355 to 34422" in msg


def test_a_reach_inside_the_segment_counts():
    assert answer_problem(_rec((34355, 34422), (34400, 34410))) is None


def test_a_segment_too_short_to_be_a_pellet_cannot_be_answered():
    assert "only 4 frame(s) long" in answer_problem(_rec((3861, 3864), (3861, 3862)))
    # a person's own cuts are taken as made, however short
    assert answer_problem(_rec((3861, 3864), (3861, 3862)), human_segmentation=True) is None


def test_placeholders_and_untouched_answers_are_not_judged():
    assert answer_problem(_rec((1, 100), (500, 510), reviewed=False)) is None
    assert answer_problem(_rec((1, 100), None, outcome="untouched")) is None


def test_matching_notes_are_put_in_the_reviewers_words():
    note = ("segment 17 review names causal reach 34446-34462, outside the frames it "
            "records (34355-34422); not applied -- it is about other footage")
    assert plain_review_note(note).startswith("Segment 17: the reach given earlier "
                                              "(frames 34446-34462) is outside")
    assert plain_review_note("segment 3 review covers frames 3861-3864, shorter than any "
                             "real footage (a zero-length-segment artifact); dropped") is None
    assert plain_review_note("something new") == "something new"


def test_the_review_tool_refuses_such_an_answer_and_says_why():
    from mousereach.review.causal_review_widget import CausalReviewWidget
    stub = SimpleNamespace(
        _collect_answers_unchecked=lambda: _rec((34355, 34422), (34446, 34462)),
        _segmentation_is_human=lambda: False)
    assert CausalReviewWidget._collect_answers(stub) is None
    assert "outside segment 17" in stub._answer_problem
    stub._collect_answers_unchecked = lambda: _rec((34355, 34422), (34400, 34410))
    assert CausalReviewWidget._collect_answers(stub) is not None
    assert stub._answer_problem is None


VID = "20990101_CNT9901_P1"


def _short_segment_video(d, cleared=False):
    """Segment 2 is 4 frames long (cuts at 100 and 104) and triaged."""
    (d / f"{VID}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": [0, 100, 104, 2000]}), encoding="utf-8")
    (d / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"segments": [{"segment_num": 2, "outcome": "triaged"}]}), encoding="utf-8")
    if cleared:
        (d / f"{VID}_deep_review_cleared.json").write_text(
            json.dumps({"cleared_by": "tester"}), encoding="utf-8")


def test_the_gate_sends_a_too_short_triaged_segment_to_deep_review(tmp_path):
    _short_segment_video(tmp_path)
    decision, reason, _ = evaluate_gate(VID, tmp_path, qc_verdict="auto_approved")
    assert decision == DECISION_DEEP
    assert reason.startswith("segmentation:") and "frames 100-103" in reason


def test_cuts_a_person_kept_leave_nothing_to_ask(tmp_path):
    _short_segment_video(tmp_path, cleared=True)
    decision, _, _ = evaluate_gate(VID, tmp_path, qc_verdict="auto_approved")
    assert decision == DECISION_CLEAN


def test_a_triage_bundle_stuck_on_a_too_short_segment_moves_to_deep_review(
        tmp_path, monkeypatch):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    b = triage / VID
    b.mkdir()
    _short_segment_video(b)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: routed.append((a, k)))
    db = SimpleNamespace(get_video=lambda v: {"video_id": v, "state": "triage"},
                         force_state=lambda *a, **k: None)
    summary = rr.scan_review_queues(db, tmp_path / "proc")
    assert summary.get("diverted_to_deep") == 1
    (args, kw), = routed
    assert args[2] == deep and kw["reason"].startswith("segmentation:")


def test_a_pellet_sent_back_by_another_system_carries_its_own_reason(tmp_path):
    # mousedb's bench scan sends finished videos back with pellets whose hand
    # score disagrees; each pellet's flag reason says what the two say, and
    # the review tool shows it beside that pellet.
    from mousereach.watcher.route_cli import flag_segments
    p = tmp_path / "v_pellet_outcomes.json"
    p.write_text(json.dumps({"segments": [{"segment_num": 3, "outcome": "retrieved"},
                                          {"segment_num": 7, "outcome": "untouched"}]}),
                 encoding="utf-8")
    assert flag_segments(p, [3, 7], "bench disagreement",
                         details={"3": "hand score missed, algorithm retrieved"}) == [3, 7]
    segs = {s["segment_num"]: s for s in json.loads(p.read_text(encoding="utf-8"))["segments"]}
    assert segs[3]["flag_reason"] == "bench disagreement: hand score missed, algorithm retrieved"
    assert segs[7]["flag_reason"] == "bench disagreement"

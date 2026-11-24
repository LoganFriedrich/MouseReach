"""An answer whose causal reach lies outside its pellet sends the video to
re-segmentation.

A person who saw a reach touch the pellet, outside that pellet's frames as the
video is cut, has shown that the cuts are wrong (owner rule 2026-11-24: "if it
doesn't make sense, put it in front of the reviewer again"). The matching drops
such an answer, so before this the pellet stayed open: the bundle waited in
triage for an answer the window refused, or the results step refused it later.
Now the review gate and the return scan send the video to re-segmentation (deep
review) with the reason, the route a "this segment is another pellet" statement
takes. A person's cuts saved after the answer settle it.
"""
import json

import mousereach.review.causal_review_io as cio
import mousereach.watcher.review_gate as rg
import mousereach.watcher.review_return as rr
from mousereach.review.triage_status import triage_status

VID = "20990101_ABC9901_P2"
CUTS = [1767, 3600, 5400, 7200]          # pellet 1 is frames 1767-3599


def _answer(reach=(433, 451), span=(1767, 3599), when="2099-01-01T10:00:00"):
    return {"segment_num": 1, "segment_span": {"start": span[0], "end": span[1]},
            "human": {"outcome": "displaced_sa",
                      "causal_reach": {"start": reach[0], "end": reach[1], "reach_id": 4}},
            "answers": {"triage_kind": "both_uncertain", "outcome_pick": "displaced_sa"},
            "reviewer": "someone", "reviewed_at": when}


def _current(extra=None):
    seg = {"boundaries": CUTS}
    seg.update(extra or {})
    return cio.segments_from_segmentation(seg)


def test_a_reach_before_the_first_cut_is_found():
    found = cio.reach_outside_answers({"segments": [_answer()]}, _current())
    assert set(found) == {1}
    reason = cio.reach_outside_reason(found)
    assert reason.startswith("segmentation:")          # the re-seg tools release it
    assert "433-451" in reason and "1767-3599" in reason


def test_a_reach_inside_its_pellet_is_not():
    assert cio.reach_outside_answers({"segments": [_answer(reach=(2000, 2010))]},
                                     _current()) == {}


def test_carried_aside_answers_count_too():
    doc = {"segments": [], "unmatched_answers": [_answer()]}
    assert set(cio.reach_outside_answers(doc, _current())) == {1}


def test_a_persons_cuts_saved_after_the_answer_settle_it():
    extra = {"boundary_source": "human", "corrected_at": "2099-01-02T10:00:00",
             "algo_boundaries": CUTS}
    found = cio.reach_outside_answers({"segments": [_answer()]}, _current(extra),
                                      person_cut_at="2099-01-02T10:00:00")
    assert found == {}


def test_a_newer_answer_on_that_pellet_replaces_it():
    newer = _answer(reach=(2000, 2010), when="2099-01-03T10:00:00")
    assert cio.reach_outside_answers({"segments": [newer], "unmatched_answers": [_answer()]},
                                     _current()) == {}


def test_the_window_records_such_an_answer_instead_of_refusing_it():
    assert cio.answer_sends_to_reseg(_answer())
    assert not cio.answer_sends_to_reseg(_answer(reach=(2000, 2010)))


def _folder(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{VID}_segments.json").write_text(json.dumps({"boundaries": CUTS}),
                                               encoding="utf-8")
    (path / f"{VID}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": 1, "outcome": "triaged"},
        {"segment_num": 2, "outcome": "untouched"},
        {"segment_num": 3, "outcome": "untouched"}]}), encoding="utf-8")
    (path / f"{VID}_causal_review.json").write_text(json.dumps(
        {"reviewer": "someone", "reviewed_at": "2099-01-01T10:00:00",
         "segments": [_answer()]}), encoding="utf-8")
    return path


def _isolate(monkeypatch, tmp_path):
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr.Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    monkeypatch.setattr(cio, "durable_review_dir", lambda: None)
    import mousereach.archive.core as ac
    monkeypatch.setattr(ac, "get_archive_destination",
                        lambda stem: tmp_path / "Analyzed" / "PROJ" / "PROJ01")
    return triage, deep


def test_the_gate_sends_the_video_to_resegmentation(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    work = _folder(tmp_path / "work")
    assert set(triage_status(work, VID).reach_outside) == {1}
    decision, reason, _ = rg.evaluate_gate(VID, work)
    assert decision == rg.DECISION_DEEP
    assert reason.startswith("segmentation: a reviewer's answer names a causal reach")


class _DB:
    def get_video(self, vid):
        return {"video_id": vid, "state": "triage"}

    def force_state(self, *a, **k):
        pass


def test_the_return_scan_moves_a_triage_bundle_to_resegmentation(tmp_path, monkeypatch):
    triage, deep = _isolate(monkeypatch, tmp_path)
    _folder(triage / VID)
    routed = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda *a, **k: routed.append((a, k)))
    monkeypatch.setattr(rr, "_return_to_processing", lambda *a, **k: True)
    summary = rr.scan_review_queues(_DB(), tmp_path / "proc")
    assert len(routed) == 1
    (stem, _bundle, queue), kw = routed[0]
    assert stem == VID and queue == deep
    assert kw["reason"].startswith("segmentation:") and kw["db_state"] == "deep_review"
    assert summary["diverted_to_deep"] == 1

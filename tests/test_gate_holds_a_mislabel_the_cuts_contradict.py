"""The review gate holds a video whose review says a segment is another pellet.

A person can record that a segment is really another pellet
(``true_segment_num`` with ``segmentation_wrong``). The hand-score check
already refuses to compare such a video and sends it to manual segmentation,
but that check is skipped on a kinematics-only re-run and compares only
pellets nobody answered. The gate -- which every run passes -- ignored the
statement and passed the video CLEAN, so a re-run filed it again on the cuts
the reviewer had called wrong. Now the gate sends it to deep review with a
reason that names segmentation, until a person's re-cut settles it.
Everything under tmp_path; invented video names.
"""
import json

import mousereach.review.causal_review_io as cio
import mousereach.watcher.review_gate as rg

VID = "20990101_ABC9901_P1"
EASY = "20990101_ABC9901_E1"


def _site(tmp_path, monkeypatch, records, seg_doc=None, vid=VID):
    monkeypatch.setattr(cio, "durable_review_dir", lambda: None)
    folder = tmp_path / "work"
    folder.mkdir()
    seg_doc = seg_doc or {"boundaries": [100, 200, 300, 400]}
    (folder / f"{vid}_segments.json").write_text(json.dumps(seg_doc), encoding="utf-8")
    (folder / f"{vid}_reaches.json").write_text(json.dumps({"segments": []}),
                                                encoding="utf-8")
    (folder / f"{vid}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "untouched"} for n in (1, 2, 3)]}), encoding="utf-8")
    (folder / f"{vid}_causal_review.json").write_text(
        json.dumps({"reviewed_at": "2099-01-01T10:00:00", "segments": records}),
        encoding="utf-8")
    return folder


def _mislabel(seg_num, true_num, start, when="2099-01-01T10:00:00"):
    return {"segment_num": seg_num, "true_segment_num": true_num, "segmentation_wrong": True,
            "reviewed_at": when,
            "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": "untouched"}, "answers": {"reviewed": True}}


def test_gate_sends_a_contradicted_mislabel_to_manual_segmentation(tmp_path, monkeypatch):
    # A person said segment 2 (frames 200-299) is really pellet 3; the cuts
    # still call it 2. Nothing else is open, so the gate used to say CLEAN.
    folder = _site(tmp_path, monkeypatch, [_mislabel(2, 3, 200)])
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_DEEP
    assert reason.startswith("segmentation:") and "2" in reason


def test_the_reason_releases_through_the_re_segmentation_tools(tmp_path, monkeypatch):
    from mousereach.review.release_cli import _names_segmentation
    folder = _site(tmp_path, monkeypatch, [_mislabel(2, 3, 200)])
    _, reason, _ = rg.evaluate_gate(VID, folder)
    assert _names_segmentation(reason)


def test_run_gate_routes_it_to_deep_review(tmp_path, monkeypatch):
    folder = _site(tmp_path, monkeypatch, [_mislabel(2, 3, 200)])
    moved = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda vid, src, dest, reason, **kw: moved.append((dest, reason)))
    monkeypatch.setattr(rg.Paths, "DEEP_REVIEW", tmp_path / "deep", raising=False)
    assert rg.run_gate(VID, folder) == rg.DECISION_DEEP
    assert moved and moved[0][0] == tmp_path / "deep"
    assert moved[0][1].startswith("segmentation:")


def test_cuts_that_already_number_it_as_the_person_said_pass(tmp_path, monkeypatch):
    # One more cut in front: frames 200-299 are now segment 3, as the person said.
    folder = _site(tmp_path, monkeypatch, [_mislabel(2, 3, 200)],
                   seg_doc={"boundaries": [0, 100, 200, 300, 400]})
    decision, _, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN


def test_a_persons_re_cut_after_the_statement_settles_it(tmp_path, monkeypatch):
    # A person re-cut the video after the statement and kept these cuts: the
    # newest human word about the cuts wins, so the video is not held again.
    seg = {"boundaries": [100, 200, 300, 400], "boundary_source": "human",
           "corrected_at": "2099-01-02T09:00:00",
           "algo_boundaries": [100, 200, 300, 400]}
    folder = _site(tmp_path, monkeypatch, [_mislabel(2, 3, 200)], seg_doc=seg)
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN, reason


def test_easy_and_flat_trays_are_never_held_for_it(tmp_path, monkeypatch):
    folder = _site(tmp_path, monkeypatch, [_mislabel(1, 1, 0)], vid=EASY,
                   seg_doc={"boundaries": [0]})
    decision, _, _ = rg.evaluate_gate(EASY, folder)
    assert decision == rg.DECISION_CLEAN


def test_no_statement_no_hold(tmp_path, monkeypatch):
    folder = _site(tmp_path, monkeypatch, [])
    decision, reason, _ = rg.evaluate_gate(VID, folder)
    assert decision == rg.DECISION_CLEAN, reason

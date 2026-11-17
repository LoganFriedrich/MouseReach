"""Reopening a ground-truth file lays it over the current outputs by FRAMES.

It was laid over by reach id and pellet number. Reach ids are renumbered on
every run and a re-cut renumbers pellets, so after a re-run a determined reach
took over whichever current reach had its id, and a determined outcome landed
on another pellet's footage -- and the next save wrote both there as the
person's answers.

Algorithm outputs and the saved file are written under tmp_path; invented ids.
"""
import json

from mousereach.review.unified_gt import load_or_create_unified_gt

VID = "20990101_ABC0101_P1"


def _write(folder, name, doc):
    (folder / name).write_text(json.dumps(doc), encoding="utf-8")


def _algo(folder, boundaries, reaches):
    _write(folder, f"{VID}_segments.json", {"boundaries": boundaries})
    _write(folder, f"{VID}_reaches.json", {"video_name": VID, "segments": [
        {"segment_num": sn, "reaches": [{"reach_id": rid, "start_frame": a, "end_frame": b}
                                        for rid, a, b in rs]}
        for sn, rs in reaches.items()]})
    _write(folder, f"{VID}_pellet_outcomes.json", {"video_name": VID, "segments": [
        {"segment_num": n, "outcome": "triaged"} for n in range(1, len(boundaries))]})


def _saved(folder, boundaries, reaches, outcomes):
    _write(folder, f"{VID}_unified_ground_truth.json", {
        "video_name": VID, "schema_version": "2.0",
        "segmentation": {"boundaries": [{"index": i, "frame": f, "determined": False}
                                        for i, f in enumerate(boundaries)]},
        "reaches": {"reaches": [dict(r, start_determined=True, end_determined=True)
                                for r in reaches]},
        "outcomes": {"segments": [dict(o, determined=True) for o in outcomes]},
    })


def _open(folder):
    video = folder / f"{VID}.mp4"
    video.write_bytes(b"x")
    return load_or_create_unified_gt(video)


def test_a_determined_reach_lands_on_the_reach_with_its_frames_not_its_id(tmp_path):
    # Today's run numbered the reaches differently: id 7 is now another reach.
    _algo(tmp_path, [0, 1000], {1: [(6, 100, 120), (7, 500, 520)]})
    _saved(tmp_path, [0, 1000],
           [{"reach_id": 7, "segment_num": 1, "start_frame": 102, "end_frame": 125}], [])
    gt = _open(tmp_path)
    rows = sorted((r.reach_id, r.start_frame, r.end_frame, r.start_determined) for r in gt.reaches)
    assert rows == [(6, 102, 125, True), (7, 500, 520, False)]


def test_an_outcome_given_on_other_cuts_is_kept_but_not_put_on_another_pellet(tmp_path):
    # Saved on cuts 0/100/200/300; today a new cut at 50 renumbers the pellets.
    # A re-cut voids the answers given on the old cuts: the saved pellet-2
    # answer is put on no pellet (not today's 2, not today's 3) and is kept.
    _algo(tmp_path, [0, 50, 100, 200, 300], {})
    _saved(tmp_path, [0, 100, 200, 300], [],
           [{"segment_num": 2, "outcome": "retrieved"}])
    gt = _open(tmp_path)
    assert not any(o.determined for o in gt.outcomes)
    assert [(o.segment_num, o.outcome) for o in gt.unmatched_outcomes] == [(2, "retrieved")]


def test_on_the_same_cuts_an_outcome_comes_back_on_its_pellet(tmp_path):
    _algo(tmp_path, [0, 100, 200, 300], {})
    _saved(tmp_path, [0, 100, 200, 300], [],
           [{"segment_num": 2, "outcome": "retrieved"}])
    gt = _open(tmp_path)
    by_num = {o.segment_num: o for o in gt.outcomes}
    assert by_num[2].outcome == "retrieved" and by_num[2].determined
    assert gt.unmatched_outcomes == []


def test_unmatched_answers_survive_a_save(tmp_path):
    from mousereach.review.unified_gt import save_unified_gt
    _algo(tmp_path, [0, 50, 100, 200, 300], {})
    _saved(tmp_path, [0, 100, 200, 300], [],
           [{"segment_num": 2, "outcome": "retrieved"}])
    gt = _open(tmp_path)
    path = save_unified_gt(gt, tmp_path / f"{VID}.mp4")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert [(o["segment_num"], o["outcome"]) for o in doc["outcomes"]["unmatched"]] == [(2, "retrieved")]
    assert not any(o["determined"] for o in doc["outcomes"]["segments"])


def test_an_outcome_that_fits_no_pellet_is_kept_unapplied(tmp_path):
    _algo(tmp_path, [0, 150, 300], {})
    _saved(tmp_path, [0, 100, 200, 300], [],
           [{"segment_num": 2, "outcome": "retrieved"}])     # frames 100-199: split today
    gt = _open(tmp_path)
    assert not any(o.determined for o in gt.outcomes)
    assert [(o.segment_num, o.outcome) for o in gt.unmatched_outcomes] == [(2, "retrieved")]


def test_a_corrected_reach_on_the_same_run_stays_one_row(tmp_path):
    # Same run: detected 100-120, the person moved it to 115-140 (less than
    # half of the corrected frames still overlap the detected reach).
    _algo(tmp_path, [0, 1000], {1: [(6, 100, 120)]})
    _saved(tmp_path, [0, 1000],
           [{"reach_id": 6, "segment_num": 1, "start_frame": 115, "end_frame": 140}], [])
    gt = _open(tmp_path)
    assert [(r.reach_id, r.start_frame, r.end_frame) for r in gt.reaches] == [(6, 115, 140)]


def test_unmatched_answers_survive_the_finishing_save(tmp_path):
    from types import SimpleNamespace
    from mousereach.review import ground_truth_widget as gtw
    _algo(tmp_path, [0, 50, 100, 200, 300], {})
    _saved(tmp_path, [0, 100, 200, 300], [],
           [{"segment_num": 2, "outcome": "retrieved"}])
    gt = _open(tmp_path)
    win = SimpleNamespace(gt=gt, video_path=tmp_path / f"{VID}.mp4",
                          data_saved=SimpleNamespace(emit=lambda p: None))
    import pytest
    mp = pytest.MonkeyPatch()
    mp.setattr(gtw, "show_info", lambda m: None)
    mp.setattr(gtw, "show_error", lambda m: None)
    try:
        gtw.GroundTruthWidget._save_ground_truth(win)
    finally:
        mp.undo()
    doc = json.loads((tmp_path / f"{VID}_unified_ground_truth.json").read_text(encoding="utf-8"))
    assert [(o["segment_num"], o["outcome"]) for o in doc["outcomes"]["unmatched"]] == [(2, "retrieved")]

"""Every check that asks "has a person answered this pellet?" reads EVERY copy
of the video's review merged statement by statement (causal_review_io.
effective_review): the release check (triage_status), the review gate,
route-to-queue, the "are you sure?" reopen, and the copies the machine keeps
(the durable copy made when a bundle leaves its queue).

WHY: the review window and the results step merged every copy, while the gate,
the hand-score check and routing picked ONE copy by file time and the return
scan overwrote the durable copy with the bundle's by file time. The two sides
could disagree about what is answered -- a pellet answered in one copy was
asked again, or a recheck written into one copy was lost.
"""
import json
import os

import pytest

import mousereach.review.causal_review_io as cio
from mousereach.config import Paths

STEM = "20990101_XYZ0101_P1"
BOUNDS = [0, 100, 200, 300]          # segments 1: 0-99, 2: 100-199, 3: 200-299


def _answer(n, start, when, outcome="untouched"):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": outcome}, "answers": {},
            "reviewer": "reviewer-a", "reviewed_at": when}


def _placeholder(n, start):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": None}, "answers": {"reviewed": False}}


def _recheck(n, start, asked_at):
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": "untouched"},
            "answers": {"reviewed": False, "recheck": {"asked_at": asked_at,
                                                       "question": "sure?"}}}


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "records")
    monkeypatch.setattr(cio, "gt_index", lambda refresh=False: {})
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", tmp_path / "Review" / "triage", raising=False)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", tmp_path / "Review" / "deep_review", raising=False)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "Analyzed", raising=False)
    durable = tmp_path / "records" / "reviews"
    durable.mkdir(parents=True)
    assert cio.durable_review_path(STEM) == durable / f"{STEM}_causal_review.json"
    return tmp_path


def _video(folder, flagged=(2,)):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{STEM}_pellet_outcomes.json").write_text(json.dumps({"segments": [
        {"segment_num": n, "outcome": "triaged" if n in flagged else "untouched",
         "flagged_for_review": n in flagged} for n in (1, 2, 3)]}), encoding="utf-8")
    (folder / f"{STEM}_segments.json").write_text(json.dumps(
        {"overall_confidence": 0.9, "boundaries": BOUNDS}), encoding="utf-8")
    old = (folder / f"{STEM}_pellet_outcomes.json").stat().st_mtime - 60
    os.utime(folder / f"{STEM}_segments.json", (old, old))
    return folder


def _write(path, records, mtime=None):
    path.write_text(json.dumps({"video_stem": STEM, "segments": records}), encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


def _durable():
    return cio.durable_review_path(STEM)


def test_release_check_counts_an_answer_only_the_durable_copy_holds(site):
    from mousereach.review.triage_status import triage_status
    b = _video(site / "Review" / "triage" / STEM)
    _write(b / f"{STEM}_causal_review.json", [_placeholder(2, 100)])
    _write(_durable(), [_answer(2, 100, "2099-01-02T10:00:00")])
    assert triage_status(b, STEM).fully_resolved


def test_release_check_sees_a_recheck_only_the_durable_copy_holds(site):
    from mousereach.review.triage_status import triage_status
    b = _video(site / "Review" / "triage" / STEM)
    _write(b / f"{STEM}_causal_review.json", [_answer(2, 100, "2099-01-02T10:00:00")])
    _write(_durable(), [_recheck(2, 100, "2099-01-05T10:00:00")])
    assert triage_status(b, STEM).unresolved == {2}


def test_the_gate_does_not_pick_one_copy_by_file_time(site):
    """The copy beside the run is NEWER on disk but holds only a placeholder;
    the durable copy holds the answer. One copy picked by file time asked a
    person again; every copy merged lets the video through."""
    import mousereach.watcher.review_gate as rg
    proc = _video(site / "proc")
    t = (proc / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_answer(2, 100, "2099-01-02T10:00:00")], mtime=t - 30)
    _write(proc / f"{STEM}_causal_review.json", [_placeholder(2, 100)], mtime=t + 30)
    decision, reason, _ = rg.evaluate_gate(STEM, proc)
    assert decision == rg.DECISION_CLEAN, reason


def test_route_to_queue_counts_an_answer_in_any_copy(site):
    from mousereach.watcher.route_cli import answered_segments_of
    folder = _video(site / "Analyzed" / "PROJ" / "PROJ_01")
    t = (folder / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_answer(2, 100, "2099-01-02T10:00:00")], mtime=t - 30)
    _write(folder / f"{STEM}_causal_review.json", [_placeholder(2, 100)], mtime=t + 30)
    assert 2 in answered_segments_of(STEM, folder)


def test_leaving_the_queue_merges_into_the_durable_copy_never_overwrites_it(site):
    """The durable copy holds a recheck newer (as a statement) than the
    bundle's answer, but its FILE is older: replacing it by file time lost the
    recheck."""
    from mousereach.watcher.review_return import _ensure_durable_review
    from mousereach.review.triage_status import resolved_segments
    b = _video(site / "Review" / "triage" / STEM)
    t = (b / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_recheck(2, 100, "2099-01-05T10:00:00")], mtime=t - 30)
    _write(b / f"{STEM}_causal_review.json",
           [_answer(1, 0, "2099-01-03T10:00:00"), _answer(2, 100, "2099-01-02T10:00:00")],
           mtime=t + 30)
    _ensure_durable_review(b, STEM)
    durable = json.loads(_durable().read_text(encoding="utf-8"))
    seg_doc = {"boundaries": BOUNDS}
    assert resolved_segments(durable, seg_doc) == {1}       # 2 still reopened, 1 added


def test_reopen_decides_on_every_copy_and_writes_both_copies(site):
    """The pellet was answered only in the durable copy; the copy travelling
    with the video (in the filed folder) is newer on disk and holds a
    placeholder. The reopen must find the answer and leave the recheck in BOTH
    copies, or the window would out-vote it with the other copy."""
    from mousereach.watcher.route_cli import reopen_answered
    from mousereach.review.triage_status import resolved_segments
    folder = _video(site / "Analyzed" / "PROJ" / "PROJ_01")
    t = (folder / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_answer(2, 100, "2099-01-02T10:00:00", outcome="retrieved")],
           mtime=t - 30)
    _write(folder / f"{STEM}_causal_review.json", [_placeholder(2, 100)], mtime=t + 30)
    assert reopen_answered(STEM, folder, [2], {2: "sure?"}) == [2]
    seg_doc = {"boundaries": BOUNDS}
    for copy in (folder / f"{STEM}_causal_review.json", _durable()):
        doc = json.loads(copy.read_text(encoding="utf-8"))
        assert 2 not in resolved_segments(doc, seg_doc), copy
        rec = [r for r in doc["segments"] if r.get("answers", {}).get("recheck")]
        assert rec and rec[0]["answers"]["recheck"]["previous"]["human"]["outcome"] == "retrieved"
    merged = cio.effective_review(STEM, cio.review_dirs(STEM, folder), seg_doc)
    assert 2 not in resolved_segments(merged, seg_doc)


def test_reopen_backs_up_each_copy_under_its_own_name(site):
    """Both copies are named <video>_causal_review.json and are backed up in
    the same second: one backup must not replace the other."""
    from mousereach.watcher.route_cli import reopen_answered
    folder = _video(site / "Analyzed" / "PROJ" / "PROJ_01")
    t = (folder / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_answer(2, 100, "2099-01-02T10:00:00", outcome="retrieved")],
           mtime=t - 30)
    _write(folder / f"{STEM}_causal_review.json", [_placeholder(2, 100)], mtime=t + 30)
    before = {"travelling": (folder / f"{STEM}_causal_review.json").read_text(encoding="utf-8"),
              "durable": _durable().read_text(encoding="utf-8")}
    assert reopen_answered(STEM, folder, [2], {2: "sure?"}) == [2]
    backups = sorted((site / "Analyzed" / "Archive" / "manual_corrections").glob("*before_recheck*"))
    assert len(backups) == 2, backups
    by_role = {p.name.rsplit(".", 1)[-1]: p.read_text(encoding="utf-8") for p in backups}
    assert by_role == before


def _mislabel(n, start, true_num):
    """A "this segment is really pellet N" record, saved with the pellet open."""
    return {"segment_num": n, "segment_span": {"start": start, "end": start + 99},
            "human": {"outcome": None}, "answers": {"reviewed": False},
            "segmentation_wrong": True, "true_segment_num": true_num}


def _mislabelled(doc):
    return [(r["segment_num"], r["true_segment_num"]) for r in doc["segments"]
            if r.get("segmentation_wrong") and r.get("true_segment_num") is not None]


def test_a_mislabel_only_the_durable_copy_holds_survives_a_return(site):
    from mousereach.watcher.review_return import _ensure_durable_review
    b = _video(site / "Review" / "triage" / STEM)
    t = (b / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    (b / f"{STEM}_causal_review.json").write_text(json.dumps(
        {"video_stem": STEM, "reviewer": "reviewer-a", "reviewed_at": "2099-01-02T10:00:00",
         "segments": [_answer(1, 0, "2099-01-02T10:00:00"), _placeholder(2, 100)]}),
        encoding="utf-8")
    os.utime(b / f"{STEM}_causal_review.json", (t - 30, t - 30))
    _durable().write_text(json.dumps(
        {"video_stem": STEM, "reviewer": "reviewer-b", "reviewed_at": "2099-01-04T10:00:00",
         "segments": [_mislabel(2, 100, 3)]}), encoding="utf-8")
    durable_before = _durable().read_text(encoding="utf-8")
    _ensure_durable_review(b, STEM)
    durable = json.loads(_durable().read_text(encoding="utf-8"))
    assert _mislabelled(durable) == [(2, 3)]
    assert (durable["reviewer"], durable["reviewed_at"]) == ("reviewer-b", "2099-01-04T10:00:00")
    assert 1 in cio.index_review_by_segment(durable, None)[0]      # the bundle's answer is in
    backups = list((site / "Analyzed" / "Archive" / "manual_corrections").glob(
        f"{STEM}_causal_review.json.before_return_merge_*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == durable_before


def test_a_mislabel_only_the_durable_copy_holds_survives_a_restore_into_a_bundle(site):
    from mousereach.review.staging import _restore_review_into_bundle
    b = _video(site / "Review" / "triage" / STEM)
    t = (b / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_mislabel(2, 100, 3)], mtime=t - 30)
    _write(b / f"{STEM}_causal_review.json",
           [_answer(1, 0, "2099-01-02T10:00:00"), _placeholder(2, 100)], mtime=t + 30)
    _restore_review_into_bundle(STEM, b, lambda msg: None)
    doc = json.loads((b / f"{STEM}_causal_review.json").read_text(encoding="utf-8"))
    assert _mislabelled(doc) == [(2, 3)]


def test_route_to_queue_refuses_when_a_copy_cannot_be_read(site):
    """The copy beside the video is cut off; it may be the only copy holding a
    person's answer. Reading around it would call the pellet open and send the
    person the same question again, so the read raises instead."""
    from mousereach.watcher.route_cli import answered_segments_of
    folder = _video(site / "Analyzed" / "PROJ" / "PROJ_01")
    (folder / f"{STEM}_causal_review.json").write_text('{"video_stem": "', encoding="utf-8")
    _write(_durable(), [_placeholder(2, 100)])
    with pytest.raises(ValueError):
        answered_segments_of(STEM, folder)


@pytest.mark.parametrize("cut_off_copy_is_newer", [True, False])
def test_reopen_refuses_and_writes_nothing_when_a_copy_cannot_be_read(site, cut_off_copy_is_newer):
    """Whichever copy is newer on disk: the reopen never decides from the
    copies it happened to be able to read. (Picking one copy by file time, a
    cut-off copy that was the OLDER one was simply passed over.)"""
    from mousereach.watcher.route_cli import reopen_answered
    folder = _video(site / "Analyzed" / "PROJ" / "PROJ_01")
    t = (folder / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(folder / f"{STEM}_causal_review.json",
           [_answer(2, 100, "2099-01-02T10:00:00", outcome="retrieved")], mtime=t)
    _durable().write_text('{"segments": [', encoding="utf-8")
    os.utime(_durable(), (t + 30, t + 30) if cut_off_copy_is_newer else (t - 30, t - 30))
    before = (folder / f"{STEM}_causal_review.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        reopen_answered(STEM, folder, [2], {2: "sure?"})
    assert (folder / f"{STEM}_causal_review.json").read_text(encoding="utf-8") == before
    assert not (site / "Analyzed" / "Archive").exists()


def test_a_newer_durable_copy_keeps_its_own_top_level_fields(site):
    """The durable copy is the newer save and carries a note the older bundle
    copy lacks (and an older note of its own in the bundle). The merge must
    keep the newer copy's note, not the bundle's."""
    from mousereach.watcher.review_return import _ensure_durable_review
    b = _video(site / "Review" / "triage" / STEM)
    (b / f"{STEM}_causal_review.json").write_text(json.dumps(
        {"video_stem": STEM, "reviewer": "reviewer-a", "reviewed_at": "2099-01-02T10:00:00",
         "notes": "older note",
         "segments": [_answer(1, 0, "2099-01-02T10:00:00"), _placeholder(2, 100)]}),
        encoding="utf-8")
    _durable().write_text(json.dumps(
        {"video_stem": STEM, "reviewer": "reviewer-b", "reviewed_at": "2099-01-04T10:00:00",
         "notes": "newer note", "extra_field": 7,
         "segments": [_answer(2, 100, "2099-01-04T10:00:00")]}), encoding="utf-8")
    _ensure_durable_review(b, STEM)
    durable = json.loads(_durable().read_text(encoding="utf-8"))
    assert durable["notes"] == "newer note"
    assert durable["extra_field"] == 7
    assert (durable["reviewer"], durable["reviewed_at"]) == ("reviewer-b", "2099-01-04T10:00:00")
    assert {1, 2} <= set(cio.index_review_by_segment(durable, None)[0])


def test_restore_into_a_bundle_backs_up_the_bundle_copy_before_merging(site):
    from mousereach.review.staging import _restore_review_into_bundle
    b = _video(site / "Review" / "triage" / STEM)
    t = (b / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_answer(2, 100, "2099-01-04T10:00:00")], mtime=t - 30)
    _write(b / f"{STEM}_causal_review.json",
           [_answer(1, 0, "2099-01-02T10:00:00"), _placeholder(2, 100)], mtime=t + 30)
    before = (b / f"{STEM}_causal_review.json").read_text(encoding="utf-8")
    _restore_review_into_bundle(STEM, b, lambda msg: None)
    doc = json.loads((b / f"{STEM}_causal_review.json").read_text(encoding="utf-8"))
    assert 2 in cio.index_review_by_segment(doc, None)[0]
    backups = list((site / "Analyzed" / "Archive" / "manual_corrections").glob(
        f"{STEM}_causal_review.json.before_restore_merge_*.bundle"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == before


def test_route_video_refuses_a_video_whose_review_copy_cannot_be_read(site, monkeypatch):
    """End to end: route-to-queue reports the refusal in its result, moves
    nothing and changes no file, instead of crashing or routing around the
    unreadable copy."""
    import mousereach.watcher.review_gate as rg
    from mousereach.watcher.route_cli import route_video
    folder = _video(site / "Analyzed" / "PROJ" / "PROJ_01")
    (folder / f"{STEM}_causal_review.json").write_text('{"segments": [', encoding="utf-8")
    _write(_durable(), [_placeholder(2, 100)])
    before = {p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()}
    routed = []
    monkeypatch.setattr(rg, "route_to_queue",
                        lambda vid, src, *a, **k: routed.append(src) or src)
    res = route_video(STEM, "triage", "test", [2])
    assert not res["routed"] and not routed
    assert "its review or cuts could not be read" in (res["error"] or ""), res
    assert {p.name: p.read_text(encoding="utf-8") for p in folder.iterdir()} == before


# --- A mislabel a person's later answer replaced is not brought back --------
#
# A later answer about the same frames that no longer says "this is really
# pellet N" is the person's current word. Merging copies must not graft the
# older copy's mislabel back onto it (and then write it to disk), and the
# checks that read mislabels must not see it either.

def _old_mislabel(n, start, true_num, when):
    rec = _mislabel(n, start, true_num)
    rec["human"] = {"outcome": "retrieved"}
    rec["answers"] = {}
    rec["reviewer"], rec["reviewed_at"] = "reviewer-a", when
    return rec


def test_a_newer_answer_without_the_mislabel_is_not_given_it_back_by_a_merge(site):
    older = {"video_stem": STEM, "reviewed_at": "2099-01-01T10:00:00",
             "segments": [_old_mislabel(2, 100, 3, "2099-01-01T10:00:00")]}
    newer = {"video_stem": STEM, "reviewed_at": "2099-01-05T10:00:00",
             "segments": [_answer(1, 0, "2099-01-05T10:00:00"),
                          _answer(2, 100, "2099-01-05T10:00:00", outcome="untouched")]}
    for base, others in ((newer, [older]), (older, [newer])):
        merged = cio.merge_review_into(base, others)
        assert _mislabelled(merged) == [], (base is newer)
        rec2 = [r for r in merged["segments"] if r["segment_num"] == 2]
        assert len(rec2) == 1 and rec2[0]["human"]["outcome"] == "untouched"


def test_mislabel_records_leaves_out_a_mislabel_a_newer_answer_replaced(site):
    b = _video(site / "Review" / "triage" / STEM)
    _write(_durable(), [_old_mislabel(2, 100, 3, "2099-01-01T10:00:00")])
    _write(b / f"{STEM}_causal_review.json",
           [_answer(2, 100, "2099-01-05T10:00:00", outcome="untouched")])
    assert cio.mislabel_records(STEM, [b])["segments"] == []


def test_a_mislabel_newer_than_the_other_copys_answer_is_kept(site):
    b = _video(site / "Review" / "triage" / STEM)
    _write(_durable(), [_old_mislabel(2, 100, 3, "2099-01-05T10:00:00")])
    _write(b / f"{STEM}_causal_review.json",
           [_answer(2, 100, "2099-01-01T10:00:00", outcome="untouched")])
    assert [r["true_segment_num"] for r in
            cio.mislabel_records(STEM, [b])["segments"]] == [3]
    merged = cio.merge_review_into(json.loads(
        (b / f"{STEM}_causal_review.json").read_text(encoding="utf-8")),
        [json.loads(_durable().read_text(encoding="utf-8"))])
    assert _mislabelled(merged) == [(2, 3)]


def test_a_later_placeholder_does_not_drop_a_mislabel(site):
    """A pellet left open says nothing about it, so an open-pellet record
    saved later never cancels a mislabel (the window cannot show one)."""
    b = _video(site / "Review" / "triage" / STEM)
    _write(_durable(), [_old_mislabel(2, 100, 3, "2099-01-01T10:00:00")])
    (b / f"{STEM}_causal_review.json").write_text(json.dumps(
        {"video_stem": STEM, "reviewed_at": "2099-01-05T10:00:00",
         "segments": [_placeholder(2, 100)]}), encoding="utf-8")
    assert [r["true_segment_num"] for r in
            cio.mislabel_records(STEM, [b])["segments"]] == [3]


def test_return_does_not_write_back_a_mislabel_a_newer_answer_replaced(site):
    from mousereach.watcher.review_return import _ensure_durable_review
    b = _video(site / "Review" / "triage" / STEM)
    t = (b / f"{STEM}_pellet_outcomes.json").stat().st_mtime
    _write(_durable(), [_old_mislabel(2, 100, 3, "2099-01-01T10:00:00")], mtime=t - 30)
    (b / f"{STEM}_causal_review.json").write_text(json.dumps(
        {"video_stem": STEM, "reviewer": "reviewer-a", "reviewed_at": "2099-01-05T10:00:00",
         "segments": [_answer(2, 100, "2099-01-05T10:00:00", outcome="untouched")]}),
        encoding="utf-8")
    os.utime(b / f"{STEM}_causal_review.json", (t + 30, t + 30))
    _ensure_durable_review(b, STEM)
    durable = json.loads(_durable().read_text(encoding="utf-8"))
    assert _mislabelled(durable) == []

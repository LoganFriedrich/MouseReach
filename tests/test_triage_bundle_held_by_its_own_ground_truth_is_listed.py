"""A triage bundle held only by its own ground truth is released, not stuck.

A triage bundle whose segmentation failed and whose ground truth INSIDE the
bundle certifies the whole video is not diverted to re-segmentation (the gate
would pass it clean). A release check that counted only ground
truth it found through the ground-truth index, not ground truth inside a
bundle, so its triaged pellets stayed open, the window hid it (it has ground
truth) and the return scan could only list it for a person.

A person's answer about the correct
pellet is always true. The release check now reads ground truth inside the
bundle, so such a bundle is RELEASED by the ordinary rule, and a listing an
older scan wrote is taken off.

Built under tmp_path with invented ids.
"""
import json

import pytest

import mousereach.watcher.review_return as rr
import mousereach.watcher.review_routing as rrt

PILLAR = "20990101_ABC9941_P1"


def _gt(certified):
    doc = {"segmentation": {"boundaries": [
        {"index": i, "frame": f, "determined": True} for i, f in enumerate((100, 200, 300))]}}
    if certified:
        doc["reaches"] = {"exhaustive": True}
        doc["outcomes"] = {"exhaustive": True, "segments": [
            {"segment_num": k, "determined": True, "outcome": "untouched"} for k in (1, 2, 3)]}
    return doc


def _bundle(triage, certified=True, answered=False, marker=False):
    b = triage / PILLAR
    b.mkdir(parents=True)
    (b / f"{PILLAR}_segments.json").write_text(
        json.dumps({"overall_confidence": 0.0, "boundaries": []}), encoding="utf-8")
    (b / f"{PILLAR}_pellet_outcomes.json").write_text(
        json.dumps({"segments": [{"segment_num": 3, "outcome": "triaged"}]}), encoding="utf-8")
    (b / f"{PILLAR}_unified_ground_truth.json").write_text(json.dumps(_gt(certified)),
                                                          encoding="utf-8")
    if answered:
        (b / f"{PILLAR}_causal_review.json").write_text(json.dumps({"segments": [
            {"segment_num": 3, "human": {"outcome": "untouched"},
             "answers": {"reviewed": True}}]}), encoding="utf-8")
    if marker:
        (b / f"{PILLAR}_deep_review_cleared.json").write_text("{}", encoding="utf-8")
    return b


class _DB:
    def get_video(self, vid):
        return {"video_id": vid, "state": "triage"}

    def force_state(self, *a, **k):
        pass


@pytest.fixture
def scan(tmp_path, monkeypatch):
    import mousereach.watcher.review_gate as rg
    from mousereach.review import causal_review_io as cio
    triage = tmp_path / "Review" / "triage"
    deep = tmp_path / "Review" / "deep_review"
    deep.mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(rr.Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", tmp_path / "Review")
    monkeypatch.setattr(rr, "_return_retry_after", {})
    monkeypatch.setattr(rr, "_status_cache", {}, raising=False)
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    moved = {"routed": [], "returned": []}
    monkeypatch.setattr(rg, "route_to_queue", lambda *a, **k: moved["routed"].append(a))
    monkeypatch.setattr(rr, "_return_to_processing",
                        lambda *a, **k: moved["returned"].append(a) or True)
    listed = []

    def note(kind, video_id, folder, says):
        listed[:] = [i for i in listed if i["folder"] != str(folder)]
        listed.append({"kind": kind, "video_id": video_id, "folder": str(folder),
                       "says": says})

    def drop(video_id=None, folder=None, kind=None):
        listed[:] = [i for i in listed
                     if not ((kind is None or i["kind"] == kind)
                             and (video_id is None or i["video_id"] == video_id))]
    monkeypatch.setattr(rrt, "note_for_a_person", note)
    monkeypatch.setattr(rrt, "drop_for_a_person", drop)
    monkeypatch.setattr(rrt, "_read_items", lambda: list(listed), raising=False)

    def run():
        rr.scan_review_queues(_DB(), tmp_path / "proc")
        return listed, moved
    return triage, run


def test_a_bundle_its_own_ground_truth_certifies_is_released(scan):
    triage, run = scan
    _bundle(triage)
    listed, moved = run()
    assert listed == []
    assert [a[1] for a in moved["returned"]] == [PILLAR]
    assert moved["routed"] == []


def test_a_listing_written_by_older_code_is_taken_off(scan):
    triage, run = scan
    b = _bundle(triage)
    rrt.note_for_a_person(rr.GT_NOT_COUNTED_KIND, PILLAR, b, "older listing")
    listed, moved = run()
    assert listed == []


def test_a_bundle_partial_ground_truth_does_not_certify_is_not_listed(scan):
    triage, run = scan
    _bundle(triage, certified=False)
    listed, moved = run()
    assert listed == [] and moved["routed"]          # diverted to re-segmentation, as before


def test_a_bundle_a_person_cleared_is_not_listed(scan):
    triage, run = scan
    _bundle(triage, marker=True)
    listed, _ = run()
    assert listed == []


def test_the_resolve_window_shows_it_with_no_button():
    from mousereach.watcher.attention_actions import actions_for
    acts = actions_for({"kind": rr.GT_NOT_COUNTED_KIND, "video_id": PILLAR,
                        "folder": "x", "says": "..."})
    assert acts and all(key.startswith("note:") for key, _, _ in acts)

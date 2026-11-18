"""A triage bundle that nothing can move is listed for a person, not left unseen.

A triage bundle whose segmentation failed and whose ground truth INSIDE the
bundle certifies the whole video is not diverted to re-segmentation (the gate
would pass it clean). But the release check counts only ground truth it finds
through the ground-truth index, not ground truth inside a bundle -- whether it
may is an open decision about moving queue items -- so its triaged pellets
stay open and it is not released; and the review window does not offer it,
because it has ground truth. It sat in the queue with nothing logged.

Now the return scan lists it for a person (the attention list the Pipeline
Dashboard and the Review Queues tab show), with the reason, and takes it off
once it no longer applies. It does NOT release it.

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


def test_a_bundle_nothing_can_move_is_listed_with_its_reason(scan):
    triage, run = scan
    _bundle(triage)
    listed, moved = run()
    assert [i["video_id"] for i in listed] == [PILLAR]
    says = listed[0]["says"]
    assert "ground truth" in says and "3" in says and "triage" in says
    # Visibility only: nothing is released, returned or diverted.
    assert moved == {"routed": [], "returned": []}
    assert (triage / PILLAR).is_dir()


def test_it_is_listed_once_and_taken_off_when_it_no_longer_applies(scan):
    triage, run = scan
    b = _bundle(triage)
    run()
    listed, _ = run()
    assert len(listed) == 1
    # A person's answer for pellet 3 is found: the bundle is released by the
    # ordinary rule and no longer listed.
    (b / f"{PILLAR}_causal_review.json").write_text(json.dumps({"segments": [
        {"segment_num": 3, "human": {"outcome": "untouched"},
         "answers": {"reviewed": True}}]}), encoding="utf-8")
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

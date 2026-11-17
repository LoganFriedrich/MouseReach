"""A leftover kept for a person is listed in the review tool, with a way out.

A queue folder that is not the video's live bundle but holds a person's work
with no copy anywhere else is never deleted by the machine. It stays where it
is and is listed in the Review Queues tab under 'Folders left for a person',
read from the queue folders themselves, with one button: "Nothing in it is
needed: delete this folder". That button deletes only such a folder -- never a
live bundle, never a folder outside the queues. A route refused because the
queue holds the video's live bundle is listed there too, with what to do and
no button. Built under tmp_path with invented ids.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher import attention_actions as aa
from mousereach.watcher import review_routing as routing

LEFT_NOTE = "LEFT_FOR_A_PERSON.json"      # review_routing.LEFT_NOTE
VID = "20990101_ABC9901_P1"
OTHER = "20990101_ABC9901_P2"


@pytest.fixture
def queues(tmp_path, monkeypatch):
    nas = tmp_path / "nas"
    review = nas / "Review"
    triage, deep = review / "triage", review / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(Paths, "NAS_ROOT", nas, raising=False)
    return tmp_path, nas, triage, deep


def _leftover(queue, vid=VID):
    folder = queue / vid
    folder.mkdir()
    (folder / f"{vid}_unified_ground_truth.json").write_text('{"by": "person-a"}',
                                                             encoding="utf-8")
    (folder / LEFT_NOTE).write_text(json.dumps({
        "video_id": vid, "why": "an older copy of the video",
        "what_to_do": "look at the ground truth",
        "kept": [f"{vid}_unified_ground_truth.json"]}), encoding="utf-8")
    return folder


def _live(queue, vid=OTHER):
    folder = queue / vid
    folder.mkdir()
    (folder / f"{vid}_causal_review.json").write_text('{"by": "person-a"}',
                                                      encoding="utf-8")
    return folder


def test_every_leftover_is_listed_and_no_live_bundle_is(queues):
    _, _, triage, deep = queues
    left = _leftover(deep)
    _live(triage)
    items = routing.leftover_folders()
    assert [(i["video_id"], i["folder"]) for i in items] == [(VID, str(left))]
    assert "an older copy of the video" in items[0]["says"]
    assert "look at the ground truth" in items[0]["says"]


def test_the_listed_leftover_has_one_button_that_deletes_it(queues):
    _, _, _, deep = queues
    left = _leftover(deep)
    [item] = routing.leftover_folders()
    actions = aa.actions_for(item)
    assert [key for key, _, _ in actions] == ["delete_leftover"]
    assert f"{VID}_unified_ground_truth.json" in actions[0][2]   # says what goes
    said = aa.perform(item, "delete_leftover")
    assert not left.exists()
    assert VID in said
    assert routing.leftover_folders() == []


def test_the_button_never_deletes_a_live_bundle(queues):
    _, _, triage, _ = queues
    live = _live(triage)
    item = {"kind": "left_for_a_person", "video_id": OTHER, "folder": str(live)}
    assert not any(key == "delete_leftover" for key, _, _ in aa.actions_for(item))
    with pytest.raises(aa.ActionRefused):
        aa.perform(item, "delete_leftover")
    assert (live / f"{OTHER}_causal_review.json").is_file()


def test_the_button_never_deletes_a_folder_outside_the_queues(queues):
    tmp_path, _, _, _ = queues
    elsewhere = _leftover(tmp_path)
    item = {"kind": "left_for_a_person", "video_id": VID, "folder": str(elsewhere)}
    with pytest.raises(aa.ActionRefused):
        aa.perform(item, "delete_leftover")
    assert (elsewhere / f"{VID}_unified_ground_truth.json").is_file()


def test_a_refused_route_over_a_leftover_gets_the_same_button(queues):
    _, _, triage, _ = queues
    left = _leftover(triage)
    item = {"kind": "queue_folder_in_the_way", "video_id": VID, "folder": str(left)}
    assert [key for key, _, _ in aa.actions_for(item)] == ["delete_leftover"]


def test_the_review_queues_tab_lists_leftovers_and_refused_routes(queues):
    pytest.importorskip("qtpy")
    pytest.importorskip("napari")
    from mousereach.review.queue_launcher_widget import ReviewQueuesWidget
    _, nas, triage, deep = queues
    left = _leftover(deep)
    live = _live(triage)
    (nas / "attention").mkdir()
    (nas / "attention" / "host-a_queue_folders_for_a_person.json").write_text(
        json.dumps({"items": [{"kind": "queue_folder_in_the_way", "video_id": OTHER,
                               "folder": str(live), "says": "refused"}]}),
        encoding="utf-8")
    rows = ReviewQueuesWidget._left_items()
    assert {(r["kind"], r["folder"]) for r in rows} == {
        ("left_for_a_person", str(left)), ("queue_folder_in_the_way", str(live))}

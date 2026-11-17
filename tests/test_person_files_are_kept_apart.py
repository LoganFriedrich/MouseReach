"""Clean-up moves keep a person's files apart from machine output.

A bundle holds the video and pose, what the pipeline made, and what a person
did (a review, ground truth, a clear marker, cuts set by hand). Two clean-up
paths treated everything alike: a run moved into an existing bundle replaced
same-named files -- a person's hand-set cuts included -- and a stale bundle was
filed away to an archive folder, where no reader looks, whatever a person had
written in it. Now a stale bundle's machine outputs are deleted once a current
copy of the video exists elsewhere, and a person's file stays where it is,
listed for a person. Built under tmp_path with invented ids.
"""
import json

import pytest

import mousereach.watcher.review_return as rr
from mousereach.watcher import review_routing as routing

VID = "20990101_ABC9901_P1"


def test_which_files_are_a_persons(tmp_path):
    human = tmp_path / f"{VID}_segments.json"
    human.write_text(json.dumps({"boundary_source": "human"}), encoding="utf-8")
    algo = tmp_path / "algo" / f"{VID}_segments.json"
    algo.parent.mkdir()
    algo.write_text(json.dumps({"boundary_source": "algo"}), encoding="utf-8")
    assert routing.is_person_file(human, VID)
    assert not routing.is_person_file(algo, VID)
    for name in (f"{VID}_causal_review.json", f"{VID}_unified_ground_truth.json",
                 f"{VID}_deep_review_cleared.json", f"{VID}_seg_validation.json",
                 f"{VID}_deep_review_cleared_superseded_20990101T000000.json"):
        assert routing.is_person_file(tmp_path / name, VID), name
    for name in (f"{VID}_reaches.json", f"{VID}_pellet_outcomes.json"):
        (tmp_path / name).write_text(json.dumps({"validation_status": "needs_review"}),
                                     encoding="utf-8")
    for name in (f"{VID}_reaches.json", f"{VID}_pellet_outcomes.json",
                 f"{VID}_triage.json", f"{VID}_routing.json", f"{VID}_manifest.json",
                 f"{VID}.mp4", f"{VID}DLC_model_x.h5"):
        assert not routing.is_person_file(tmp_path / name, VID), name


def test_a_run_never_replaces_cuts_a_person_set(tmp_path):
    queue = tmp_path / "deep_review"
    bundle = queue / VID
    bundle.mkdir(parents=True)
    hand = json.dumps({"boundary_source": "human", "boundaries": [1, 2, 3]})
    (bundle / f"{VID}_segments.json").write_text(hand, encoding="utf-8")
    src = tmp_path / "Processing"
    src.mkdir()
    (src / f"{VID}_segments.json").write_text(
        json.dumps({"boundary_source": "algo", "boundaries": [9]}), encoding="utf-8")
    (src / f"{VID}_reaches.json").write_text("{}", encoding="utf-8")

    _, moved = routing.move_video_bundle(VID, src, queue, "why")

    assert (bundle / f"{VID}_segments.json").read_text(encoding="utf-8") == hand
    assert (src / f"{VID}_segments.json").is_file()          # left at the source
    assert f"{VID}_reaches.json" in moved
    record = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert [f["name"] for f in record["failed_files"]] == [f"{VID}_segments.json"]


@pytest.fixture
def review_root(tmp_path, monkeypatch):
    root = tmp_path / "Review"
    (root / "triage").mkdir(parents=True)
    monkeypatch.setattr(rr.Paths, "REVIEW_ROOT", root)
    monkeypatch.setattr(rr.Paths, "TRIAGE_REVIEW", root / "triage")
    monkeypatch.setattr(rr.Paths, "NAS_ROOT", tmp_path, raising=False)
    return root


def _filed_copy(tmp_path):
    """The video's current copy elsewhere: its filed results."""
    filed = tmp_path / "Analyzed" / "PROJ" / "PROJ01"
    filed.mkdir(parents=True)
    (filed / f"{VID}.mp4").write_text("video", encoding="utf-8")
    (filed / f"{VID}_segments.json").write_text('{"filed": 1}', encoding="utf-8")
    return {p.name: p for p in filed.iterdir()}


def test_a_stale_bundle_holding_ground_truth_stays_and_is_listed(tmp_path, review_root):
    """Never filed away among retired bundles, where nobody looks: the ground
    truth stays where it is with a note, listed for a person; the machine
    output beside it is deleted (a current copy exists)."""
    from mousereach.watcher.attention import read_items
    bundle = review_root / "triage" / VID
    bundle.mkdir()
    (bundle / f"{VID}_segments.json").write_text("{}", encoding="utf-8")
    gt = bundle / f"{VID}_unified_ground_truth.json"
    gt.write_text('{"by": "person-a"}', encoding="utf-8")

    assert rr._retire_stale_bundle(bundle, VID, "archived", live=_filed_copy(tmp_path))

    assert {p.name for p in bundle.iterdir()} == {gt.name, "LEFT_FOR_A_PERSON.json"}
    assert gt.read_text(encoding="utf-8") == '{"by": "person-a"}'
    assert [i["video_id"] for i in read_items(tmp_path)
            if i.get("kind") == "left_for_a_person"] == [VID]
    assert not (review_root / "_Problematic").exists()


def test_a_stale_machine_only_bundle_is_deleted_once_a_current_copy_exists(
        tmp_path, review_root):
    bundle = review_root / "triage" / VID
    bundle.mkdir()
    (bundle / f"{VID}_segments.json").write_text("{}", encoding="utf-8")
    (bundle / f"{VID}.mp4").write_text("video", encoding="utf-8")
    assert rr._retire_stale_bundle(bundle, VID, "archived", live=_filed_copy(tmp_path))
    assert not bundle.exists()
    assert not (review_root / "_Problematic").exists()


def test_a_stale_bundle_with_no_current_copy_anywhere_is_kept(tmp_path, review_root):
    """Nothing can be verified as copied elsewhere: nothing is deleted."""
    bundle = review_root / "triage" / VID
    bundle.mkdir()
    (bundle / f"{VID}_segments.json").write_text("{}", encoding="utf-8")
    (bundle / f"{VID}.mp4").write_text("the only video", encoding="utf-8")
    assert rr._retire_stale_bundle(bundle, VID, "processing", live={})
    assert (bundle / f"{VID}.mp4").read_text(encoding="utf-8") == "the only video"
    assert (bundle / f"{VID}_segments.json").is_file()
    assert (bundle / "LEFT_FOR_A_PERSON.json").is_file()

"""A segments, reaches or outcomes file a person checked or corrected is a
person's file, though the pipeline writes a file of the same name.

The review tools save a person's work INTO the machine-named file: the
segmentation review marks it 'validated', the re-segmentation fixer marks the
cuts 'human', the reach and outcome reviews and the ground-truth tool mark
segments and reaches human_verified / human_corrected, the triage-clearing
tools mark segments triage_cleared. A rule that judged these files by name
called them machine output, and clean-up (a leftover queue folder, a route
into a bundle, a return into Processing) deleted or replaced a person's
corrections. Built under tmp_path with invented ids.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher import review_routing as routing

VID = "20990101_ABC9901_P1"

# How each tool saves its work (only the marks that matter here).
SAVED_BY_A_PERSON = {
    "segmentation review": (f"{VID}_segments.json", {
        "boundaries": [10, 20], "validation_status": "validated",
        "validation_record": {"by": "person-a"}}),
    "re-segmentation fixer": (f"{VID}_segments.json", {
        "boundaries": [10, 20], "boundary_source": "human"}),
    "reach review": (f"{VID}_reaches.json", {
        "validation_status": "needs_review",
        "segments": [{"segment_num": 1, "reaches": [{"start": 5, "human_corrected": True}]}]}),
    "outcome review": (f"{VID}_pellet_outcomes.json", {
        "validation_status": "needs_review",
        "segments": [{"segment_num": 1, "outcome": "retrieved", "human_verified": True}]}),
    "triage clearing": (f"{VID}_pellet_outcomes.json", {
        "segments": [{"segment_num": 2, "triage_cleared": True}]}),
    "ground truth tool": (f"{VID}_reaches.json", {
        "validation_status": "validated", "segments": []}),
}
MACHINE = {
    f"{VID}_segments.json": {"boundaries": [11, 21], "validation_status": "needs_review"},
    f"{VID}_reaches.json": {"validation_status": "auto_approved", "segments": [
        {"segment_num": 1, "triage_cleared": None, "reaches": [{"human_corrected": False}]}]},
    f"{VID}_pellet_outcomes.json": {"validation_status": "needs_review", "segments": [
        {"segment_num": 1, "human_verified": False}]},
}


def _write(folder, name, doc):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(doc), encoding="utf-8")
    return folder / name


@pytest.fixture
def queues(tmp_path, monkeypatch):
    review = tmp_path / "Review"
    triage, deep = review / "triage", review / "deep_review"
    triage.mkdir(parents=True)
    deep.mkdir(parents=True)
    monkeypatch.setattr(Paths, "TRIAGE_REVIEW", triage)
    monkeypatch.setattr(Paths, "DEEP_REVIEW", deep)
    monkeypatch.setattr(Paths, "REVIEW_ROOT", review)
    monkeypatch.setattr(Paths, "NAS_ROOT", None, raising=False)
    from mousereach.review import causal_review_io as cio
    monkeypatch.setattr(cio, "_default_index_dir", lambda: tmp_path / "review_records")
    import mousereach.pipeline.fsutil as fsutil
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    return tmp_path, triage, deep


@pytest.mark.parametrize("tool", sorted(SAVED_BY_A_PERSON))
def test_each_tools_saved_file_is_a_persons(tmp_path, tool):
    name, doc = SAVED_BY_A_PERSON[tool]
    assert routing.is_person_file(_write(tmp_path, name, doc), VID)


@pytest.mark.parametrize("name", sorted(MACHINE))
def test_the_pipelines_own_file_is_not(tmp_path, name):
    assert not routing.is_person_file(_write(tmp_path, name, MACHINE[name]), VID)


def test_an_unreadable_results_file_counts_as_a_persons(tmp_path):
    (tmp_path / f"{VID}_reaches.json").write_text("{not json", encoding="utf-8")
    assert routing.is_person_file(tmp_path / f"{VID}_reaches.json", VID)


@pytest.mark.parametrize("tool", sorted(SAVED_BY_A_PERSON))
def test_a_leftover_keeps_a_persons_results(queues, tool):
    """clear_leftover with a live copy whose file of that name differs (the
    machine's): the person's file stays, the folder is kept with its note."""
    tmp_path, triage, _ = queues
    name, doc = SAVED_BY_A_PERSON[tool]
    left = _write(triage / VID, name, doc)
    live = _write(tmp_path / "live", name, MACHINE[name])
    kept = routing.clear_leftover(triage / VID, VID, {name: live},
                                  why="test", what_to_do="look")
    assert kept == [name]
    assert json.loads(left.read_text(encoding="utf-8")) == doc


def test_a_leftover_drops_a_persons_results_the_live_copy_holds_byte_for_byte(queues):
    tmp_path, triage, _ = queues
    name, doc = SAVED_BY_A_PERSON["reach review"]
    _write(triage / VID, name, doc)
    live = _write(tmp_path / "live", name, doc)
    assert routing.clear_leftover(triage / VID, VID, {name: live},
                                  why="test", what_to_do="look") == []
    assert not (triage / VID).exists()


@pytest.mark.parametrize("tool", sorted(SAVED_BY_A_PERSON))
def test_a_route_never_replaces_a_persons_results_in_the_bundle(queues, tool):
    tmp_path, triage, _ = queues
    name, doc = SAVED_BY_A_PERSON[tool]
    held = _write(triage / VID, name, doc)
    src = _write(tmp_path / "Processing", name, MACHINE[name]).parent
    routing.move_video_bundle(VID, src, triage, "why")
    assert json.loads(held.read_text(encoding="utf-8")) == doc
    assert (src / name).is_file()                    # left at the source


@pytest.mark.parametrize("tool", sorted(SAVED_BY_A_PERSON))
def test_a_return_never_replaces_a_persons_results_in_processing(queues, tool):
    import mousereach.watcher.review_return as rr
    tmp_path, triage, _ = queues
    name, doc = SAVED_BY_A_PERSON[tool]
    src = _write(triage / VID, name, MACHINE[name])
    dst = _write(tmp_path / "Processing", name, doc)
    assert rr._files_in_the_way(VID, [(src, dst)]) == [name]

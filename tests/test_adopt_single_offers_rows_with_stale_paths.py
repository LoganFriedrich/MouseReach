"""A single whose RECORDED path is stale is still offered for adoption.

This pins the boundary of the NO_FILE_HERE filter next door
(test_work_selection_skips_known_pathless.py), because the obvious extension of
that filter to the singles bucket is wrong, and wrong in a way that costs GPU work
rather than announcing itself.

WHY a path-existence test must NOT be added here: a 'validated' row's recorded path
is frequently ANOTHER node's local path. Cross-node recovery copies
``videos.current_path`` from the shared record as a breadcrumb, so a row can read
``A:\\...\\DLC_Queue\\x.mp4`` -- a path that has never existed on this machine -- while
the file itself sits in the shared singles folder, ready to be adopted. Dropping the
row because its recorded path does not resolve would refuse work that is genuinely
here. Only ``locate_video_file`` (which searches the shared folder, not just the
recorded path) can answer the question, and the handler already calls it.

The placeholder filter is sound for the opposite reason: NO_FILE_HERE is not a stale
path, it is this node's own recorded finding that the video is not here at all.

The cost of leaving the bucket unfiltered is bounded and self-clearing: every handler
path takes the row out of 'validated' -- finished or held rows are recorded as such
(and cost no poll, the handler returns True), a vanished file is marked unresolvable,
a second copy of another node's claim likewise. See test_singles_intake_claim.py.

Since 2026-10-21 the bucket IS checked against a listing of the places the handler
takes a video from (the shared folder among them) -- never against the recorded path
alone -- so a row whose file is in the shared folder is offered whatever its recorded
path says, which is what these tests pin. They put the file in a test shared folder.
"""
import pytest

import mousereach.watcher.orchestrator as orch


@pytest.fixture(autouse=True)
def shared(tmp_path, monkeypatch):
    """A test shared singles folder (never the real share)."""
    door = tmp_path / "Single_Animal"
    door.mkdir()
    for name, d in (("SINGLE_ANIMAL_OUTPUT", door), ("DLC_QUEUE", tmp_path / "q"),
                    ("PROCESSING", tmp_path / "p")):
        if d != door:
            d.mkdir()
        monkeypatch.setattr(orch.Paths, name, d)
    return door

from mousereach.watcher.orchestrator import DLCOrchestrator


class FakeDB:
    """Only what _select_work_item reads."""

    NO_FILE_HERE = "(no file on this node)"

    def __init__(self, rows):
        self._rows = rows

    def get_videos_in_state(self, state):
        return [dict(r) for r in self._rows if r["state"] == state]

    def get_collages_in_state(self, state):
        return []


class FakeConfig:
    also_process = False
    work_priority = None


def _node(rows):
    node = DLCOrchestrator.__new__(DLCOrchestrator)
    node.db = FakeDB(rows)
    node.config = FakeConfig()
    node._get_priority_animal = lambda: None
    node._pick_from_pool = lambda items, *a, **k: (items[0] if items else None)
    node._coordinator_init_error = None
    node._claim_backoff_active = lambda *a, **k: False
    node._retry_unsynced_collage_claims = lambda *a, **k: None
    node._repose_requests_to_take = lambda *a, **k: []
    # The two filters the singles bucket really does apply.
    node._singles_braked = lambda: False
    node._single_claim_backoff_active = lambda *a, **k: False
    return node


def test_a_validated_row_whose_recorded_path_is_another_nodes_is_still_offered(shared):
    (shared / "20240101_ABC0101_P1.mp4").write_bytes(b"v")
    node = _node([
        {"video_id": "20240101_ABC0101_P1", "state": "validated",
         "source_path": r"A:\Behavior\MouseReach_Pipeline\DLC_Queue\20240101_ABC0101_P1.mp4",
         "current_path": r"A:\Behavior\MouseReach_Pipeline\DLC_Queue\20240101_ABC0101_P1.mp4",
         "animal_id": "ABC0101"},
    ])
    work = node._select_work_item()
    assert work is not None, (
        "a stale recorded path must not disqualify a single: the file may be in the "
        "shared singles folder, which only locate_video_file looks in")
    assert work["type"] == "adopt_single"
    assert work["id"] == "20240101_ABC0101_P1"


def test_a_validated_row_with_no_recorded_path_at_all_is_offered(shared):
    (shared / "20240101_ABC0102_P1.mp4").write_bytes(b"v")
    node = _node([
        {"video_id": "20240101_ABC0102_P1", "state": "validated",
         "source_path": None, "current_path": None, "animal_id": "ABC0102"},
    ])
    work = node._select_work_item()
    assert work is not None and work["type"] == "adopt_single"


def test_the_two_filters_the_bucket_does_apply_still_work():
    rows = [{"video_id": "20240101_ABC0103_P1", "state": "validated",
             "source_path": None, "current_path": None, "animal_id": "ABC0103"}]

    braked = _node(rows)
    braked._singles_braked = lambda: True
    assert braked._select_work_item() is None, (
        "no single is adopted while this node's recent poses keep failing")

    backed_off = _node(rows)
    backed_off._single_claim_backoff_active = lambda *a, **k: True
    assert backed_off._select_work_item() is None, (
        "a single whose claim was just refused waits out its backoff")

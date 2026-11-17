"""Collage crop failures and stuck crop claims become visible and recoverable.

WHY: a GPU node that could not crop a collage used to leave no trace a person
would see. An error before the crop started wrote a state change the database
refuses and escaped into the main loop; a collage out of retries sat 'failed'
in the intake folder with a DEBUG line; a stale claim another node could not
take over waited for its holder for ever with a log WARNING; a 'cropped' mark
that did not reach the shared claim was forgotten at a restart; and a child
that reached neither the share nor the node's queue was deleted while its
collage was recorded as cut.

Everything lives under tmp_path with config.Paths pointed at it. Handlers run
on an orchestrator built without its constructor. All ids are invented.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import mousereach.watcher.orchestrator as orch
from mousereach.watcher.db import WatcherDB
from mousereach.watcher.orchestrator import DLCOrchestrator

HOST = "NODE-A"
OTHER = "NODE-B"

# Two animals and six blank (cohort 00) slots: two expected children.
COLLAGE = ("20250101_TST0101,TST0102,TST0001,TST0002,TST0003,TST0004,"
           "TST0005,TST0006_P1.mkv")
KID1 = "20250101_TST0101_P1"
KID2 = "20250101_TST0102_P1"

LAYOUT = {
    "NAS_ROOT": "share",
    "MULTI_ANIMAL_SOURCE": "share/Unanalyzed/Multi-Animal",
    "SINGLE_ANIMAL_OUTPUT": "share/Unanalyzed/Single_Animal",
    "DLC_STAGING": "share/Processing/Posed",
    "REVIEW_ROOT": "share/Processing/Review",
    "TRIAGE_REVIEW": "share/Processing/Review/triage",
    "DEEP_REVIEW": "share/Processing/Review/deep_review",
    "ANALYZED_OUTPUT": "share/Analyzed",
    "PROCESSING_ROOT": "node",
    "DLC_QUEUE": "node/DLC_Queue",
    "PROCESSING": "node/Processing",
}


class FakeCoordinator:
    """The shared claim table, reduced to what these handlers ask of it."""

    def __init__(self, claim=True, holder=OTHER, claim_state="cropping",
                 claimed_at=None):
        self.claim = claim
        self.holder = holder
        self.claim_state = claim_state
        self.claimed_at = claimed_at or datetime.now().isoformat()
        self.released, self.updates = [], []

    def try_claim_collage(self, filename, hostname):
        return self.claim

    def release_collage_claim(self, filename, hostname):
        self.released.append((filename, hostname))
        return True

    def update_collage_state(self, filename, state, **kw):
        self.updates.append((filename, state, kw))
        return True

    def get_collage_claim(self, filename):
        if self.claim_state is None:
            return None
        return {"filename": filename, "hostname": self.holder,
                "state": self.claim_state, "claimed_at": self.claimed_at}

    def get_all_collage_states(self):
        row = self.get_collage_claim(COLLAGE)
        return {COLLAGE: row} if row else {}


@pytest.fixture
def env(tmp_path, monkeypatch):
    dirs = {}
    for name, sub in LAYOUT.items():
        d = tmp_path / sub
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(orch.Paths, name, d, raising=False)
        dirs[name.lower()] = d
    monkeypatch.setattr(orch, "require_processing_root", lambda: dirs["processing_root"])
    o = object.__new__(DLCOrchestrator)
    o.db = WatcherDB(db_path=tmp_path / "watcher.db")
    o.hostname = HOST
    o.working_dir = dirs["processing_root"] / "watcher_working"
    o.working_dir.mkdir()
    o.config = SimpleNamespace(also_process=False, dlc_gpu_device=0,
                               dlc_config_path=None, max_retries=3,
                               work_priority=None)
    o.coordinator = FakeCoordinator()
    o._sync_to_connectome = lambda *a, **k: None
    o._backup_local_db = lambda: None
    return SimpleNamespace(o=o, db=o.db, tmp=tmp_path, **dirs)


def _collage(env, state="stable"):
    src = env.multi_animal_source / COLLAGE
    src.write_bytes(b"collage")
    env.db.register_collage(COLLAGE, str(src))
    path = {"stable": ("validated", "stable"),
            "cropping": ("validated", "stable", "cropping"),
            "failed": ("validated", "stable", "cropping", "failed"),
            "cropped": ("validated", "stable", "cropping", "cropped")}[state]
    for s in path:
        env.db.update_collage_state(COLLAGE, s)
    return {"type": "collage", "id": COLLAGE, "data": env.db.get_collage(COLLAGE)}


def _attention(env, name):
    f = env.nas_root / "attention" / f"{HOST}_{name}.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text(encoding="utf-8"))["items"]


# ------------------------------------------------------------------ MRP-02d

def test_an_error_before_the_crop_starts_never_escapes_the_dispatch(env):
    """An error while the children are looked up leaves the row 'stable'.
    'stable' -> 'failed' is refused by the database; that refusal used to leave
    the dispatch and the same collage was picked again at once."""
    work = _collage(env, "stable")

    def boom(name):
        raise OSError("share blinked")
    env.o._expected_children = boom

    assert env.o._dispatch_work(work) is False
    assert env.db.get_collage(COLLAGE)["state"] == "stable"
    assert env.o._claim_backoff_active(COLLAGE), (
        "the collage must wait before it is picked again, not spin the loop")


def test_an_error_during_the_crop_still_records_failed(env):
    """A collage already 'cropping' becomes 'failed' exactly as before."""
    work = _collage(env, "cropping")
    env.o._process_collage = lambda w: (_ for _ in ()).throw(RuntimeError("bad"))
    assert env.o._dispatch_work(work) is False
    row = env.db.get_collage(COLLAGE)
    assert row["state"] == "failed" and "bad" in (row.get("validation_error") or "")


# --------------------------------------------------------- MRP-02e / D2-8

def _scanning_node(env, monkeypatch):
    """The node's scan with everything but the collage intake stubbed out."""
    from mousereach.watcher.state import WatcherStateManager
    from mousereach.watcher.watcher import FileWatcher
    monkeypatch.setattr(orch.Paths, "SINGLE_ANIMAL_OUTPUT", None, raising=False)
    import mousereach.watcher.watcher as watcher_mod
    monkeypatch.setattr(watcher_mod.Paths, "SINGLE_ANIMAL_OUTPUT", None, raising=False)
    monkeypatch.setattr(watcher_mod.Paths, "MULTI_ANIMAL_SOURCE",
                        env.multi_animal_source, raising=False)
    o = env.o
    o.state = WatcherStateManager(o.db, o.config)
    o.file_watcher = FileWatcher(o.config, o.state)
    for name in ("_single_claim_upkeep", "_retry_failed_shared_writes",
                 "_finish_shared_record_fill", "_scan_for_dlc_completions",
                 "_adopt_untracked_queue_files", "_repose_heartbeat"):
        setattr(o, name, lambda *a, **k: None)
    o._retry_coordinator = lambda: True
    o._is_paused = lambda: True          # stop before the re-pose requests
    return o


def _out_of_retries(env):
    _collage(env, "failed")
    env.db.update_collage_state(COLLAGE, "validated", retry_count=3)
    for s in ("stable", "cropping"):
        env.db.update_collage_state(COLLAGE, s)
    env.db.update_collage_state(COLLAGE, "failed",
                                validation_error="crop failed: unreadable stream")


def test_a_collage_out_of_retries_is_listed_for_a_person(env, monkeypatch):
    _out_of_retries(env)
    env.o.coordinator = FakeCoordinator(claim_state=None)     # nobody holds it
    o = _scanning_node(env, monkeypatch)

    o._scan_phase()

    assert env.db.get_collage(COLLAGE)["state"] == "failed"
    items = _attention(env, "collages")
    assert items is not None, "a parked collage must reach the attention lists"
    assert len(items) == 1
    item = items[0]
    assert item["kind"] == "collage_crop_parked" and item["needs_person"] is True
    assert item["collage"] == COLLAGE
    assert item["cohort_id"] is None, (
        "a collage problem must never be filed under one animal's cohort")
    assert "unreadable stream" in item["says"] and "max_retries" in item["says"]


def test_a_collage_cropped_elsewhere_is_recorded_not_listed(env, monkeypatch):
    _out_of_retries(env)
    env.o.coordinator = FakeCoordinator(claim_state="cropped", holder=OTHER)
    o = _scanning_node(env, monkeypatch)

    o._scan_phase()

    assert env.db.get_collage(COLLAGE)["state"] == "cropped"
    assert _attention(env, "collages") in (None, [])


def test_a_collage_another_node_is_cropping_now_is_not_listed(env, monkeypatch):
    _out_of_retries(env)
    env.o.coordinator = FakeCoordinator(claim_state="cropping", holder=OTHER)
    o = _scanning_node(env, monkeypatch)

    o._scan_phase()

    assert env.db.get_collage(COLLAGE)["state"] == "failed"
    assert _attention(env, "collages") in (None, [])


def test_a_list_that_cannot_be_written_never_stops_the_scan(env, monkeypatch):
    _out_of_retries(env)
    env.o.coordinator = FakeCoordinator(claim_state=None)
    import mousereach.pipeline.fsutil as fsutil

    def refuse(*a, **k):
        raise PermissionError("share is read-only")
    monkeypatch.setattr(fsutil, "write_json_atomic", refuse)
    o = _scanning_node(env, monkeypatch)
    o._scan_phase()                      # must not raise


def test_the_health_line_names_a_collage_that_could_not_be_cropped(env):
    from mousereach.watcher import health
    _out_of_retries(env)
    text = "\n".join(health.health_report(env.db))
    assert COLLAGE in text and "unreadable stream" in text
    assert "Nothing is stuck" not in text


# ------------------------------------------------------------------ MRP-03b

def _hours_ago(h):
    return (datetime.now() - timedelta(hours=h)).isoformat()


def test_a_stale_claim_that_is_not_taken_over_is_listed_for_a_person(env):
    """Another node has held the claim in 'cropping' for over a day and the
    takeover is refused (its children are on record): it waits for its holder
    for ever, so a person must see it."""
    work = _collage(env, "stable")
    env.o.coordinator = FakeCoordinator(claim=False, holder=OTHER,
                                        claim_state="cropping",
                                        claimed_at=_hours_ago(30))

    assert env.o._process_collage(work) is False

    items = _attention(env, "collage_claims")
    assert items is not None and len(items) == 1
    item = items[0]
    assert item["kind"] == "collage_claim_stuck" and item["needs_person"] is True
    assert item["collage"] == COLLAGE and item["holder"] == OTHER
    assert item["cohort_id"] is None
    assert OTHER in item["says"]


def test_a_live_claim_of_another_node_is_not_listed(env):
    work = _collage(env, "stable")
    env.o.coordinator = FakeCoordinator(claim=False, holder=OTHER,
                                        claim_state="cropping",
                                        claimed_at=_hours_ago(1))
    assert env.o._process_collage(work) is False
    assert _attention(env, "collage_claims") in (None, [])


def test_a_stuck_claim_leaves_the_list_once_it_is_finished(env):
    work = _collage(env, "stable")
    coord = env.o.coordinator = FakeCoordinator(claim=False, holder=OTHER,
                                                claim_state="cropping",
                                                claimed_at=_hours_ago(30))
    env.o._process_collage(work)
    assert len(_attention(env, "collage_claims")) == 1

    coord.claim_state = "cropped"                # the holder came back and finished
    env.o._claim_backoff.clear()
    env.o._process_collage(work)

    assert _attention(env, "collage_claims") == []
    assert env.db.get_collage(COLLAGE)["state"] == "cropped"


def test_every_node_listing_the_same_stuck_claim_is_shown_once(tmp_path):
    from mousereach.watcher.attention import read_items, without_repeats, write_list
    item = {"kind": "collage_claim_stuck", "needs_person": True,
            "collage": COLLAGE, "says": "stuck"}
    write_list(tmp_path, "NODE-A", "collage_claims", [item])
    write_list(tmp_path, "NODE-C", "collage_claims", [item])
    write_list(tmp_path, "NODE-A", "other", [{"kind": "x", "says": "a"},
                                             {"kind": "x", "says": "b"}])
    shown = without_repeats(read_items(tmp_path))
    assert [i["says"] for i in shown if i["kind"] == "collage_claim_stuck"] == ["stuck"]
    assert len([i for i in shown if i["kind"] == "x"]) == 2


# ------------------------------------------------------------------ MRP-03c

@pytest.fixture
def shared(tmp_path):
    pytest.importorskip("sqlalchemy")
    from mousereach.watcher.coordination import PipelineCoordinator
    c = PipelineCoordinator(db_path=tmp_path / "central" / "watcher_central.db")
    c.ensure_tables()
    return c


def test_a_cropped_mark_lost_at_a_restart_is_sent_again(env, shared):
    """The node cropped the collage, but its 'cropped' never reached the
    shared claim before the watcher stopped. A fresh watcher (nothing in
    memory) must find and send it from the two records."""
    _collage(env, "cropped")
    assert shared.try_claim_collage(COLLAGE, HOST) is True       # left 'cropping'
    env.o.coordinator = shared                                  # a fresh process

    env.o._retry_unsynced_collage_claims()

    assert shared.get_collage_claim(COLLAGE)["state"] == "cropped"


def test_another_nodes_claim_and_an_uncropped_collage_are_left_alone(env, shared):
    _collage(env, "stable")
    assert shared.try_claim_collage(COLLAGE, HOST) is True
    other = "20250101_TST0201,TST0001,TST0002,TST0003,TST0004,TST0005,TST0006,TST0007_P1.mkv"
    assert shared.try_claim_collage(other, OTHER) is True
    env.o.coordinator = shared

    env.o._retry_unsynced_collage_claims()

    assert shared.get_collage_claim(COLLAGE)["state"] == "cropping"
    assert shared.get_collage_claim(other)["state"] == "cropping"
    assert shared.get_collage_claim(other)["hostname"] == OTHER

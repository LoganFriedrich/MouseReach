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

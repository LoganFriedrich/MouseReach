"""A node's status report says what it is doing, not just that it is alive.

WHY (plan gap G14, 2026-10-22): the report said "working" every loop turn and
nothing else, so a node churning on nothing looked exactly like a productive
one, and a node deep in a 14-minute pose looked like one whose report had gone
stale. It now carries the item in hand (written as it starts), the last one
finished, and how many finished in the past hour.
"""
import json

import mousereach.watcher.orchestrator as o
from mousereach.config import Paths
from mousereach.watcher import node_status


def _node(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    orch = object.__new__(o.ProcessingOrchestrator)
    orch.hostname = "NODE-A"
    return orch


def _report(tmp_path):
    return json.loads((tmp_path / "Processing" / node_status.DIR_NAME / "NODE-A.json")
                      .read_text(encoding="utf-8"))


def test_the_item_in_hand_is_reported_as_it_starts(tmp_path, monkeypatch):
    orch = _node(tmp_path, monkeypatch)
    orch._begin_work({"type": "pipeline", "id": "20250101_CNT0101_P1"})
    r = _report(tmp_path)
    assert r["state"] == "working"
    assert r["doing"]["type"] == "pipeline" and r["doing"]["id"] == "20250101_CNT0101_P1"
    assert r["doing"]["since"]


def test_finished_items_are_counted_and_the_last_one_named(tmp_path, monkeypatch):
    orch = _node(tmp_path, monkeypatch)
    for vid in ("A", "B"):
        work = {"type": "archive", "id": vid}
        orch._begin_work(work)
        orch._end_work(work, True)
    work = {"type": "intake", "id": "C"}
    orch._begin_work(work)
    orch._end_work(work, False)                   # got nowhere: not counted
    orch._report_node_status(node_status.WORKING)
    r = _report(tmp_path)
    assert r["doing"] is None
    assert r["done_last_hour"] == 2
    assert r["last_done"]["id"] == "B"


def test_only_the_past_hour_counts(tmp_path, monkeypatch):
    orch = _node(tmp_path, monkeypatch)
    clock = [10_000.0]
    monkeypatch.setattr(o.time, "monotonic", lambda: clock[0])
    work = {"type": "archive", "id": "A"}
    orch._begin_work(work)
    orch._end_work(work, True)
    clock[0] += 3601
    assert orch._work_report()["done_last_hour"] == 0


def test_the_per_node_line_says_what_it_is_doing(tmp_path):
    node_status.write(tmp_path, node_status.WORKING, hostname="NODE-BUSY",
                      extra={"doing": {"type": "single_dlc", "id": "20250101_CNT0101_P1",
                                       "since": "2026-10-22T05:00:00"},
                             "done_last_hour": 4})
    node_status.write(tmp_path, node_status.WORKING, hostname="NODE-IDLE",
                      extra={"doing": None, "done_last_hour": 0})
    text = node_status.describe(tmp_path)
    busy = next(l for l in text.splitlines() if "NODE-BUSY" in l)
    idle = next(l for l in text.splitlines() if "NODE-IDLE" in l)
    assert "single_dlc 20250101_CNT0101_P1 since 2026-10-22T05:00:00" in busy
    assert "4 done in the last hour" in busy
    assert "idle" in idle and "0 done in the last hour" in idle

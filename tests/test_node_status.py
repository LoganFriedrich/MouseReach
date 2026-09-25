"""A quiet node says why it is quiet, so silence means something.

The case these exist for: a node pauses whenever a recording program is open, and a
recording program is routinely left open after the day's recording is finished. So a
machine doing nothing for days is usually CORRECT. A dead watcher looks exactly the
same from every other machine. Treating those alike either raises a false alarm most
weeks -- which teaches everyone to ignore the alarm -- or hides a real outage, which
has already cost about sixty hours once.
"""
import json
from datetime import datetime, timedelta

import pytest

from mousereach.watcher import node_status as ns


def test_a_paused_node_is_not_silence(tmp_path):
    ns.write(tmp_path, ns.PAUSED, "a recording program is running", hostname="NODE-A")
    rows = ns.read_all(tmp_path)
    assert len(rows) == 1
    assert rows[0]["state"] == ns.PAUSED
    assert rows[0]["running"] is True, (
        "a paused node is still RUNNING -- it is deliberately not taking work, which "
        "is the distinction this whole module exists to make")
    assert "recording" in rows[0]["reason"]


def test_a_node_that_stopped_reporting_reads_as_not_running(tmp_path):
    ns.write(tmp_path, ns.WORKING, hostname="NODE-B")
    f = ns.status_dir(tmp_path) / "NODE-B.json"
    stale = json.loads(f.read_text())
    stale["at"] = (datetime.now() - ns.STALE_AFTER - timedelta(minutes=5)).isoformat()
    f.write_text(json.dumps(stale))
    assert ns.read_all(tmp_path)[0]["running"] is False


def test_the_description_separates_the_two_cases(tmp_path):
    ns.write(tmp_path, ns.PAUSED, "a recording program is running", hostname="NODE-PAUSED")
    ns.write(tmp_path, ns.WORKING, hostname="NODE-DEAD")
    f = ns.status_dir(tmp_path) / "NODE-DEAD.json"
    rec = json.loads(f.read_text())
    rec["at"] = (datetime.now() - timedelta(hours=60)).isoformat()
    f.write_text(json.dumps(rec))

    text = ns.describe(tmp_path)
    paused_line = next(l for l in text.splitlines() if "NODE-PAUSED" in l)
    dead_line = next(l for l in text.splitlines() if "NODE-DEAD" in l)
    assert "NOT RUNNING" not in paused_line, "a pause must never read as an outage"
    assert "recording" in paused_line, "and it must say WHY, or it is just silence again"
    assert "NOT RUNNING" in dead_line
    assert "60 h ago" in dead_line, "how long it has been gone is the whole question"


def test_writing_never_raises_however_bad_the_destination(tmp_path):
    """Nothing here may ever interrupt processing."""
    for bad in (None, "", tmp_path / "a\0b"):
        assert ns.write(bad, ns.WORKING) is False


def test_reading_an_absent_or_unreadable_folder_is_empty_not_an_error(tmp_path):
    assert ns.read_all(tmp_path / "nothing here") == []
    d = ns.status_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "NODE-C.json").write_text("this is not json")
    assert ns.read_all(tmp_path) == [], "a corrupt report is ignored, not fatal"


def test_a_reader_never_sees_a_half_written_report(tmp_path):
    """Written to a temporary name and renamed, so a concurrent read is all-or-nothing."""
    ns.write(tmp_path, ns.WORKING, hostname="NODE-D")
    leftovers = [p for p in ns.status_dir(tmp_path).iterdir() if p.suffix == ".tmp"]
    assert leftovers == []
    assert json.loads((ns.status_dir(tmp_path) / "NODE-D.json").read_text())["state"] == ns.WORKING


def test_the_latest_report_replaces_the_previous_one(tmp_path):
    ns.write(tmp_path, ns.PAUSED, "a recording program is running", hostname="NODE-E")
    ns.write(tmp_path, ns.WORKING, hostname="NODE-E")
    rows = ns.read_all(tmp_path)
    assert len(rows) == 1 and rows[0]["state"] == ns.WORKING


def test_nothing_reported_yet_says_so_plainly(tmp_path):
    assert "No node has reported" in ns.describe(tmp_path)

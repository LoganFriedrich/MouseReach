"""A list of sessions a study needs can lead within its project.

WHY (plan Step 1, 2026-10-22): a study asks for particular sessions -- one
animal's tray run on one day -- spread over many cohorts and recordings, and
the order could only say "this project, these cohorts". The lab's order needed
CNT first, then one study's sessions, then the rest of that project, and there
was no way to write it. A session list is a file of sessions in the shared
pipeline folder; the order file names it under its project.
"""
import json

import pytest

from mousereach.config import Paths
from mousereach.watcher import work_priority as wp


@pytest.fixture
def share(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    monkeypatch.setattr(wp, "_SESSION_LISTS", {})
    wp.invalidate_lab_policy()
    (tmp_path / "priority_lists").mkdir()
    # lab ids are read too: J11 is ASPA1011
    (tmp_path / "priority_lists" / "study.txt").write_text(
        "# a study's sessions\n20220811_J11_P3\n20220812_ASPA1012_P1\n", encoding="utf-8")
    yield tmp_path
    wp.invalidate_lab_policy()


def _write_order(root, **body):
    (root / wp.LAB_PRIORITY_FILENAME).write_text(json.dumps(body), encoding="utf-8")
    wp.invalidate_lab_policy()


def test_sessions_of_a_video_a_bundle_and_a_recording():
    assert wp.session_keys({"video_id": "20220811_ASPA1011_P3"}) == {"20220811_ASPA1011_P3"}
    assert wp.session_keys({"filename": "20220811_ASPA1011,ASPA1012_P3.mkv"},
                           is_collage=True) == {"20220811_ASPA1011_P3", "20220811_ASPA1012_P3"}
    assert wp.session_keys({"video_id": "not a session"}) == set()


def test_the_list_leads_within_its_project(share):
    _write_order(share, projects=["CNT", "ASPA"], cohorts={"CNT": ["01"]},
                 session_lists={"ASPA": ["priority_lists/study.txt"]},
                 tray_types=["P"], idle_only_tray_types=["E", "F"])
    policy = wp.load_lab_policy(max_age_s=0)
    assert not policy.complaints
    tier = lambda vid, **kw: policy.tier({"video_id": vid, **kw})
    cnt01 = tier("20250101_CNT0101_P1")
    listed = tier("20220811_ASPA1011_P3")                 # listed by its lab id J11
    unlisted = tier("20220811_ASPA1013_P3")
    assert cnt01 < listed < unlisted
    # a recording ranks with its listed animal: it must be cut to reach it
    recording = policy.tier({"filename": "20220812_ASPA1012,ASPA1013_P1.mkv"}, is_collage=True)
    assert recording == listed


def test_a_missing_list_and_an_unknown_key_are_named(share):
    _write_order(share, projects=["ASPA"], session_lists={"ASPA": ["priority_lists/gone.txt"]},
                 tray_types=["P"], extra_key=1)
    complaints = " | ".join(wp.load_lab_policy(max_age_s=0).complaints)
    assert "gone.txt" in complaints and "was not found" in complaints
    assert "'extra_key' is not a key" in complaints


def test_saving_keeps_what_the_panel_does_not_manage(share):
    _write_order(share, projects=["CNT"], notes="why this order", order=[{"project": ["CNT"]}])
    wp.save_lab_priority(projects=["CNT", "ASPA"], cohorts={},
                         session_lists={"ASPA": ["priority_lists/study.txt"]},
                         tray_types=["P"], idle_only_tray_types=["E", "F"])
    body = json.loads((share / wp.LAB_PRIORITY_FILENAME).read_text(encoding="utf-8"))
    assert body["projects"] == ["CNT", "ASPA"]
    assert body["session_lists"] == {"ASPA": ["priority_lists/study.txt"]}
    assert body["notes"] == "why this order"              # kept (plan gap G4)
    assert body["order"] == [{"project": ["CNT"]}]        # kept, and still in force
    assert body["schema_version"] == wp.LAB_PRIORITY_SCHEMA


def test_the_panel_reads_lists_with_spaces_and_drive_letters():
    cw = pytest.importorskip("mousereach.watcher.control_widget")
    parse = cw.WatcherControlWidget._parse_lists_text
    assert parse("ASPA: priority_lists/my study.txt, C:\\lists\\b.txt; CNT: x.txt") == {
        "ASPA": ["priority_lists/my study.txt", "C:\\lists\\b.txt"], "CNT": ["x.txt"]}

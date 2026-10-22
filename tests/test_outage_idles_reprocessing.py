"""A shared-drive outage idles re-running; it never marks videos failed.

WHY: when the shared drive dropped on 2026-09-09, the re-run step marked each
outdated video FAILED ("Archive directory not found for reprocessing"), about
one a second -- infrastructure failures recorded as the videos' fault. With
the archive out of reach nothing is offered for re-running and nothing is
marked; the work resumes when the archive is back.
"""
import logging

import pytest

from mousereach.config import Paths
from test_work_priority import make_server, video


@pytest.fixture
def outdated_only():
    return {"outdated": [video("20250101_CNT0101_P1", reprocess_scope="segmentation",
                               tray_type="P", animal_id="CNT0101")]}


def test_nothing_is_offered_while_the_archive_is_out_of_reach(tmp_path, monkeypatch,
                                                              outdated_only, caplog):
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "gone", raising=False)
    orch = make_server(outdated_only, archive_reachable=None)
    with caplog.at_level(logging.WARNING):
        assert orch._select_work_item(admit_deferred=True) is None
        assert orch._select_work_item(admit_deferred=True) is None
    warnings = [r for r in caplog.records if "not reachable" in r.getMessage()]
    assert len(warnings) == 1                          # once per outage, not every cycle


def test_re_running_resumes_when_the_archive_is_back(tmp_path, monkeypatch, outdated_only):
    archive = tmp_path / "Analyzed"
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", archive, raising=False)
    orch = make_server(outdated_only, archive_reachable=None)
    assert orch._select_work_item(admit_deferred=True) is None
    archive.mkdir()
    pick = orch._select_work_item(admit_deferred=True)
    assert pick and pick["type"] == "reprocess"


def test_the_re_run_step_declines_without_marking_the_video(tmp_path, monkeypatch):
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", tmp_path / "gone", raising=False)
    orch = make_server({}, archive_reachable=None)
    failed = []
    orch.db.mark_failed = lambda vid, msg: failed.append((vid, msg))
    orch._reprocess_video({"id": "20250101_CNT0101_P1", "data": {}})
    assert failed == []

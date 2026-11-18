"""The remaining hand step commands refuse to work in the filed tree too.

WHY: a filed video has passed every check and its files are the record of
that run. mousereach-segment, -detect-reaches, -detect-outcomes and
-grasp-analyze already refuse a folder in the filed tree; mousereach-assign-
reaches still rewrote a filed video's reach-assignment file in place, and the
triage and advance commands of each step still rewrote or moved its files.
They now refuse the same way, before anything is read or written.
Video ids are invented.
"""
import json
import sys

import pytest

V = "20200101_ABC9901_P1"


@pytest.fixture
def filed(tmp_path, monkeypatch):
    from mousereach.config import Paths
    analyzed = tmp_path / "Analyzed"
    folder = analyzed / "PRJ" / "C01"
    folder.mkdir(parents=True)
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", analyzed, raising=False)
    for name in (f"{V}_segments.json", f"{V}_reaches.json", f"{V}_pellet_outcomes.json",
                 f"{V}_reach_assignments.json"):
        (folder / name).write_text('{"filed": true}')
    (folder / f"{V}DLC_resnet101_X_shuffle3_100000.h5").write_bytes(b"pose")
    return folder


def test_assign_reaches_refuses_the_filed_tree(filed, monkeypatch):
    from mousereach.assignment import cli

    def rewrite(directory, stem, pose, *a, **k):
        (directory / f"{stem}_reach_assignments.json").write_text('{"rewritten": true}')
        return {"reaches": []}

    monkeypatch.setattr(cli, "assign_reaches_for_video", rewrite)
    monkeypatch.setattr(sys, "argv", ["mousereach-assign-reaches", "-i", str(filed)])
    with pytest.raises(SystemExit) as e:
        cli.main_batch()
    assert e.value.code == 1
    assert json.loads((filed / f"{V}_reach_assignments.json").read_text()) == {"filed": True}


@pytest.mark.parametrize("module,core,func", [
    ("segmentation", "mousereach.segmentation.core", "main_triage"),
    ("segmentation", "mousereach.segmentation.core", "main_advance"),
    ("reach", "mousereach.reach.core", "main_triage"),
    ("reach", "mousereach.reach.core", "main_advance"),
    ("outcomes", "mousereach.outcomes.core", "main_triage"),
    ("outcomes", "mousereach.outcomes.core", "main_advance"),
])
def test_triage_and_advance_commands_refuse_the_filed_tree(filed, monkeypatch, module,
                                                           core, func):
    import importlib
    cli = importlib.import_module(f"mousereach.{module}.cli")
    core_mod = importlib.import_module(core)
    ran = []
    target = "triage_results" if func == "main_triage" else "advance_videos"
    monkeypatch.setattr(core_mod, target, lambda *a, **k: ran.append(a) or {}, raising=False)
    monkeypatch.setattr(sys, "argv", ["cmd", "-i", str(filed)])
    with pytest.raises(SystemExit) as e:
        getattr(cli, func)()
    assert e.value.code == 1
    assert ran == []

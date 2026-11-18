"""The old whole-video Review Tool no longer writes a person's corrections
into the algorithm's own output files.

WHY: the Review Tool (the ground-truth widget in review mode: the launcher's
old "3 - Review Tool" tab and mousereach-review-legacy) saved straight into
{video}_segments.json, _reaches.json and _pellet_outcomes.json. The next run
of a step rewrites its file (corrected cuts were not marked as a person's,
which is the only mark the segmenter keeps), nothing after the corrected step
was redone, and the watcher's release check never read those marks. The save
now writes nothing and names the tool whose save the pipeline keeps.

The separate per-step screens (boundary, reach and outcome review) are NOT
covered: they still save, because their "validated" mark is the only sign-off
a person running the steps by hand without the watcher has.

Each save is called with a stand-in for the widget (no window is opened),
holding exactly what the old save read. Built under tmp_path, invented ids.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

VID = "20990101_ABC0903_P1"


def _write(folder, name, doc):
    p = folder / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _snapshot(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


@pytest.fixture(autouse=True)
def _no_side_stores(monkeypatch, tmp_path):
    """The old saves also updated the pipeline index and a performance log;
    point every root at tmp_path and stand those in, so nothing outside the
    test folder can be touched whichever version of the code runs."""
    from mousereach.config import Paths
    for name in ("NAS_ROOT", "PROCESSING_ROOT"):
        monkeypatch.setattr(Paths, name, tmp_path / "roots", raising=False)
    import mousereach.index as index_pkg
    monkeypatch.setattr(index_pkg, "PipelineIndex", MagicMock(), raising=False)
    try:
        import mousereach.performance as perf
        monkeypatch.setattr(perf, "PerformanceLogger", MagicMock(), raising=False)
    except Exception:
        pass


@pytest.fixture
def folder(tmp_path):
    f = tmp_path / VID
    f.mkdir()
    (f / f"{VID}.mp4").write_bytes(b"")
    return f


def _stand_in(folder, **attrs):
    w = MagicMock()
    w.video_path = folder / f"{VID}.mp4"
    w.current_segment = 1
    w.viewer.dims.current_step = (5,)
    for k, v in attrs.items():
        setattr(w, k, v)
    return w


def test_old_review_tool_save_writes_no_algorithm_file(folder):
    from mousereach.review.ground_truth_widget import GroundTruthWidget
    _write(folder, f"{VID}_segments.json", {"boundaries": [100, 400]})
    _write(folder, f"{VID}_reaches.json", {"segments": []})
    _write(folder, f"{VID}_pellet_outcomes.json", {"segments": []})
    before = _snapshot(folder)
    gt = MagicMock()
    gt.boundaries = [SimpleNamespace(frame=110), SimpleNamespace(frame=400)]
    gt.reaches = []
    gt.outcomes = []
    w = _stand_in(folder, gt=gt)
    w._algo_files_dir.return_value = folder
    GroundTruthWidget._save_to_algo_files(w)
    assert _snapshot(folder) == before


def test_the_refusal_names_the_tool_to_use(capsys):
    from mousereach.review.retired_saves import refuse_algo_file_save, CUTS, OUTCOMES, REACHES
    assert refuse_algo_file_save([CUTS, OUTCOMES, REACHES]) is False
    out = capsys.readouterr().out
    for tool in ("mousereach-fix-segmentation", "mousereach-review-tool", "mousereach-gt"):
        assert tool in out
    out.encode("ascii")   # printed to the console: ASCII only


def test_old_review_tool_command_opens_nothing(monkeypatch, capsys, folder):
    """mousereach-review-legacy prints where to go and stops (no viewer)."""
    import mousereach.review.ground_truth_widget as gtw
    monkeypatch.setattr(sys, "argv", ["mousereach-review-legacy", str(folder / f"{VID}.mp4")])
    monkeypatch.setattr(gtw.napari, "Viewer", MagicMock(side_effect=AssertionError("opened a viewer")))
    monkeypatch.setattr(gtw.napari, "run", MagicMock(side_effect=AssertionError("ran napari")))
    assert gtw.main_review() == 1
    assert "mousereach-review-tool" in capsys.readouterr().out

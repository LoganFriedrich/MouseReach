"""The old triage-clearing tool is retired: it says what to use instead,
returns an error code and writes nothing.

WHY: it saved a person's call into the algorithm's own output files
(_pellet_outcomes.json, _reaches.json), where the next run erased it; it
never wrote the reach-assignment file, so a pellet it cleared stayed open
for the watcher's release check; it wrote two files with no way back; and
it took the most recently marked reach as the causal one. The review window
(mousereach-review-tool) saves a review record instead. The command name
stays installed so an old install prints the message rather than crashing.
Built under tmp_path with invented ids.
"""
import json
import sys

VID = "20990101_ABC0902_P1"


def _files(folder):
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir())}


def test_the_old_command_refuses_and_names_the_review_window(tmp_path, monkeypatch, capsys):
    algo = tmp_path / "algo"
    algo.mkdir()
    # Nothing triaged: the old tool built an empty worklist and returned 0
    # without opening a window, so this runs headless on old code too.
    (algo / f"{VID}_pellet_outcomes.json").write_text(json.dumps(
        {"segments": [{"segment_num": 1, "outcome": "untouched"}]}), encoding="utf-8")
    (algo / f"{VID}_reaches.json").write_text(json.dumps(
        {"segments": [{"segment_num": 1, "reaches": []}]}), encoding="utf-8")
    before = _files(algo)
    monkeypatch.setattr(sys, "argv", ["mousereach-triage-clearing", "--algo-dir", str(algo)])
    from mousereach.review import triage_clearing
    rc = triage_clearing.main()
    out = capsys.readouterr().out
    assert rc == 1
    assert "mousereach-review-tool" in out
    assert _files(algo) == before


def test_the_spot_check_report_is_gone_too(tmp_path, monkeypatch, capsys):
    """--qc-report used to read a spot-check log and exit 0; the spot-check
    mode walked the retired worklist and is retired with the tool."""
    monkeypatch.setattr(sys, "argv", ["mousereach-triage-clearing", "--qc-report",
                                      "--corpus-root", str(tmp_path)])
    from mousereach.review import triage_clearing
    assert triage_clearing.main() == 1
    assert "mousereach-review-tool" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_the_command_still_points_at_a_callable():
    """An install from this version keeps the command, so a stale shortcut
    gets the message instead of an import error."""
    import re
    from pathlib import Path
    toml = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^mousereach-triage-clearing\s*=\s*"([\w.]+):(\w+)"', toml, re.M)
    assert m
    import importlib
    assert callable(getattr(importlib.import_module(m.group(1)), m.group(2)))

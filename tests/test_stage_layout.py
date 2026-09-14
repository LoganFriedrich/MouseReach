"""The stage-folder layout: the skeleton builder and config.Paths agree, and no
old folder name comes back.

WHY these tests exist: the shared data drive was migrated to a layout where a
folder is a place work RESTS (Unanalyzed/Single_Animal, Processing/Posed,
Processing/Review/deep_review, Analyzed/Archive). The old folder names were
replaced by guard FILES so stale code fails loudly. If the builder or the config
ever named an old folder again, waiting work would split across two places and
nothing would say so.

WHY config.Paths is read in a SUBPROCESS: Paths is built once, when config.py is
imported, from ~/.mousereach/config.json. Reading it in this process would test
whatever this machine happens to have configured -- or skip on a fresh clone or
CI, where nothing is configured, so the rename would go untested with no message.
The subprocess gets its own home folder holding a config that names a tmp share,
so the check runs everywhere and never reads a real config.
"""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mousereach.pipeline import pipe_structure
from mousereach.pipeline.pipe_structure import (
    RETIRED_DIRS,
    TARGET_DIRS,
    UNMIGRATED_MESSAGE,
    ensure_pipe_structure,
    retired_folders_present,
)

NEW_NAMES = [
    "Unanalyzed/Multi-Animal",
    "Unanalyzed/Single_Animal",
    "Processing/Posed",
    "Processing/Repose_Queue",
    "Processing/Review/triage",
    "Processing/Review/deep_review",
    "Processing/Quarantine",
    "Processing/Failed",
    "Analyzed",
    "Analyzed/Archive",
]

OLD_NAMES = [
    "Processing/Single_Animal",
    "Processing/DLC_Complete",
    "Processing/Review/flagged_for_review",
    "Archive",
]

# Stage attribute on config.Paths -> its path relative to the pipe root.
STAGE_PATHS = {
    "MULTI_ANIMAL_SOURCE": "Unanalyzed/Multi-Animal",
    "SINGLE_ANIMAL_OUTPUT": "Unanalyzed/Single_Animal",
    "DLC_STAGING": "Processing/Posed",
    "REPOSE_QUEUE": "Processing/Repose_Queue",
    "TRIAGE_REVIEW": "Processing/Review/triage",
    "DEEP_REVIEW": "Processing/Review/deep_review",
    "FAILED": "Processing/Failed",
    "ANALYZED_OUTPUT": "Analyzed",
}

# Stage folders derived by code rather than stored on Paths -> expected path.
DERIVED_PATHS = {
    "WatcherConfig().get_quarantine_dir()": "Processing/Quarantine",
    "supersede.default_archive_root()": "Analyzed/Archive",
}

# Runs in the subprocess: prints every stage path relative to NAS_ROOT as JSON.
_PROBE = r"""
import json, sys
from pathlib import Path
from mousereach.config import Paths, WatcherConfig
from mousereach.archive.supersede import default_archive_root

root = Path(Paths.NAS_ROOT)

def rel(p):
    return None if p is None else Path(p).relative_to(root).as_posix()

out = {"NAS_ROOT_ORIGIN": Paths.NAS_ROOT_ORIGIN}
for name in json.loads(sys.argv[1]):
    out[name] = rel(getattr(Paths, name))
out["WatcherConfig().get_quarantine_dir()"] = rel(WatcherConfig().get_quarantine_dir())
out["supersede.default_archive_root()"] = rel(default_archive_root())
print(json.dumps(out))
"""

# Environment variables that would let this machine's setup leak into the probe.
# Compared upper-case: Windows environment names are case-insensitive.
_STRIPPED_ENV = {"HOME", "USERPROFILE", "PYTHONPATH",
                 "MOUSEREACH_NAS_DRIVE", "MOUSEREACH_PROCESSING_ROOT"}


def _all_dirs(root: Path) -> set:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_dir()}


def _with_parents(rels) -> set:
    out = set()
    for rel in rels:
        parts = rel.split("/")
        for i in range(1, len(parts) + 1):
            out.add("/".join(parts[:i]))
    return out


@pytest.fixture(scope="module")
def configured_paths(tmp_path_factory):
    """Stage paths as a freshly imported config.Paths computes them, for a home
    folder whose config names a tmp share. Relative to NAS_ROOT, posix style."""
    import mousereach

    home = tmp_path_factory.mktemp("home")
    (home / ".mousereach").mkdir()
    (home / ".mousereach" / "config.json").write_text(
        json.dumps({"nas_root": str(home / "share")}), encoding="utf-8")

    # The subprocess must import THIS checkout's code, not an installed copy.
    src = str(Path(mousereach.__file__).parents[1])
    env = {k: v for k, v in os.environ.items() if k.upper() not in _STRIPPED_ENV}
    env.update(USERPROFILE=str(home), HOME=str(home), PYTHONPATH=src)

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, json.dumps(sorted(STAGE_PATHS))],
        env=env, cwd=str(home), capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


# (a) ---------------------------------------------------------------------------

def test_target_dirs_hold_new_names_only():
    for name in NEW_NAMES:
        assert name in TARGET_DIRS, name
    for name in OLD_NAMES:
        assert name not in TARGET_DIRS, name
    # Nothing beneath a retired folder either.
    for rel in TARGET_DIRS:
        for old in RETIRED_DIRS:
            assert not (rel == old or rel.startswith(old + "/")), rel


# (b) ---------------------------------------------------------------------------

def test_config_stage_paths_match_target_layout(configured_paths):
    # The probe really read the tmp config, not a fallback: otherwise the
    # expected names below could be matched by accident.
    assert configured_paths["NAS_ROOT_ORIGIN"] == "config"
    for attr, expected in {**STAGE_PATHS, **DERIVED_PATHS}.items():
        assert configured_paths[attr] == expected, attr


# (c) ---------------------------------------------------------------------------

def test_every_config_stage_path_is_in_target_dirs(configured_paths):
    for attr in {**STAGE_PATHS, **DERIVED_PATHS}:
        rel = configured_paths[attr]
        assert rel in TARGET_DIRS, f"{attr} -> {rel} missing from TARGET_DIRS"


# (d) ---------------------------------------------------------------------------

def test_ensure_pipe_structure_builds_exactly_the_new_skeleton(tmp_path):
    summary = ensure_pipe_structure(tmp_path)
    assert summary["ok"], summary
    assert summary["failed"] == []
    assert sorted(summary["created"]) == sorted(TARGET_DIRS)

    built = _all_dirs(tmp_path)
    assert built == _with_parents(TARGET_DIRS)
    for old in OLD_NAMES:
        assert not (tmp_path / old).exists(), old

    # Idempotent: a second run creates nothing and still succeeds.
    again = ensure_pipe_structure(tmp_path)
    assert again["ok"] and again["created"] == []
    assert _all_dirs(tmp_path) == built


def test_ensure_pipe_structure_leaves_guard_files_alone(tmp_path):
    # A migrated drive carries a plain file at each old folder path.
    for old in RETIRED_DIRS:
        guard = tmp_path / old
        guard.parent.mkdir(parents=True, exist_ok=True)
        guard.write_text("retired folder", encoding="ascii")

    summary = ensure_pipe_structure(tmp_path)
    assert summary["ok"], summary
    assert retired_folders_present(tmp_path) == []
    for old in RETIRED_DIRS:
        assert (tmp_path / old).is_file(), old


def test_ensure_pipe_structure_refuses_a_retired_name(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe_structure, "TARGET_DIRS",
                        list(TARGET_DIRS) + ["Processing/DLC_Complete"])
    summary = pipe_structure.ensure_pipe_structure(tmp_path)
    assert not summary["ok"]
    assert any("Processing/DLC_Complete" in f for f in summary["failed"])
    assert not (tmp_path / "Processing" / "DLC_Complete").exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_ensure_pipe_structure_reports_an_unmigrated_share(tmp_path, dry_run):
    # An unmigrated share: the retired names are still real folders, one of
    # them holding waiting work. The builder must say so, not answer "ok".
    for old in RETIRED_DIRS:
        (tmp_path / old).mkdir(parents=True)
    waiting = tmp_path / "Processing" / "DLC_Complete" / "20250101_TEST0101_P1.mp4"
    waiting.write_bytes(b"v")

    summary = ensure_pipe_structure(tmp_path, dry_run=dry_run)

    assert not summary["ok"]
    for old in RETIRED_DIRS:
        assert f"{old}: {UNMIGRATED_MESSAGE}" in summary["failed"], old
    # Reported only: nothing moved, deleted or created beneath them.
    assert waiting.read_bytes() == b"v"
    for old in RETIRED_DIRS:
        assert (tmp_path / old).is_dir(), old
    assert retired_folders_present(tmp_path) == list(RETIRED_DIRS)


def test_retired_folders_present_needs_a_root(tmp_path):
    assert retired_folders_present(None) == []
    assert retired_folders_present(tmp_path / "missing") == []


# (e) ---------------------------------------------------------------------------

def test_reverse_migration_module_stays_retired():
    # pipe_migrate mapped the current Paths back onto the OLD folder names; after
    # the repoint it would move live waiting work into the retired folders.
    with pytest.raises(ImportError):
        importlib.import_module("mousereach.pipeline.pipe_migrate")

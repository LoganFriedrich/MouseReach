"""The suite's own guard: a test that forgets to point config.Paths at
tmp_path still cannot open the machine's shared SQLite indexes
(tests/conftest.py, _no_real_shared_indexes).

WHY: writing a processing manifest updates the version index at its default
path, found from config.Paths. A test that left Paths.NAS_ROOT alone wrote a
row for its invented video into a live index.

The "real" root here is a temporary folder made OUTSIDE pytest's own
temporary folder, standing in for a configured pipeline root.
"""
import shutil
import tempfile
from pathlib import Path

import pytest

from mousereach.config import Paths

VID = "20990101_ABC0801_P1"


@pytest.fixture
def unpatched_root(monkeypatch, tmp_path_factory):
    root = Path(tempfile.mkdtemp(prefix="standin_root_",
                                 dir=tmp_path_factory.getbasetemp().parent))
    monkeypatch.setattr(Paths, "NAS_ROOT", root, raising=False)
    monkeypatch.setattr(Paths, "PROCESSING_ROOT", root, raising=False)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_the_version_index_default_is_never_under_an_unpatched_root(unpatched_root):
    from mousereach.pipeline.version_index import VersionIndex
    index = VersionIndex()
    index.upsert_from_manifest(
        VID, {"pipeline_versions": {"segmenter": "1.0"}, "created_at": "2099-01-01T10:00:00"},
        unpatched_root / f"{VID}_processing_manifest.json")
    assert unpatched_root not in Path(index.db_path).parents
    assert not (unpatched_root / "pipeline_records").exists()


def test_the_triage_queue_index_default_is_never_under_an_unpatched_root(unpatched_root):
    import mousereach.review.queue_index as queue_index
    assert unpatched_root not in Path(queue_index.default_index_path()).parents


def test_a_root_under_tmp_path_keeps_its_own_index(tmp_path, monkeypatch):
    """A test that does point Paths at tmp_path gets the path it expects."""
    import mousereach.pipeline.version_index as version_index
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    assert version_index.default_index_path() == tmp_path / "pipeline_records" / "version_index.db"

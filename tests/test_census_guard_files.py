"""Census folder listings survive a guard FILE where a folder was expected.

When the folder layout changes, a plain file is written at each retired folder
name so old code that tries to create or list it fails loudly instead of
quietly rebuilding the old folder. The census and reconcile list folders by
name, so a file there must read as "nothing here", never crash the scan.
"""
from mousereach.census.runner import bundles_in, ids_in_dir

VID = "20240101_ABC0101_P1"


def _guard_file(tmp_path, name):
    guard = tmp_path / name
    guard.write_text("retired folder -- see the new layout")
    return guard


def test_ids_in_dir_file_path_is_empty(tmp_path):
    assert ids_in_dir(_guard_file(tmp_path, "Single_Animal")) == set()


def test_ids_in_dir_missing_path_is_empty(tmp_path):
    assert ids_in_dir(tmp_path / "does_not_exist") == set()
    assert ids_in_dir(None) == set()


def test_ids_in_dir_still_lists_a_real_folder(tmp_path):
    folder = tmp_path / "Posed"
    folder.mkdir()
    (folder / f"{VID}.mp4").write_bytes(b"v")
    assert ids_in_dir(folder) == {VID}


def test_bundles_in_file_path_is_empty(tmp_path):
    assert bundles_in(_guard_file(tmp_path, "flagged_for_review")) == set()


def test_bundles_in_missing_path_is_empty(tmp_path):
    assert bundles_in(tmp_path / "does_not_exist") == set()
    assert bundles_in(None) == set()


def test_bundles_in_still_lists_a_real_queue(tmp_path):
    queue = tmp_path / "deep_review"
    queue.mkdir()
    (queue / VID).mkdir()
    (queue / "_scratch").mkdir()
    assert bundles_in(queue) == {VID}

"""Is a queue folder of this video where THIS run's results went?

A route that could not move the run's processing manifest lists it in the
routing record's failed_files, and then the folder does hold this run. Each
entry there is a dict ({"name", "error", "source"}); the check read the str()
of the dict as a file name, so it never matched and the folder was taken for
a leftover. Built under tmp_path with an invented id.
"""
import json
import os
import time

from mousereach.watcher.orchestrator import queue_folder_holds_this_run

VID = "20990101_ABC9901_P1"


def _run(tmp_path, failed):
    proc = tmp_path / "Processing"
    queue = tmp_path / "triage" / VID
    proc.mkdir()
    queue.mkdir(parents=True)
    manifest = proc / f"{VID}_processing_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    old = time.time() - 600
    os.utime(manifest, (old, old))
    (queue / f"{VID}_routing.json").write_text(json.dumps({
        "routed_at": "2099-01-01T00:00:00", "failed_files": failed}), encoding="utf-8")
    return queue, proc


def test_a_dict_entry_naming_the_manifest_means_this_run(tmp_path):
    queue, proc = _run(tmp_path, [{
        "name": f"{VID}_processing_manifest.json", "error": "in use",
        "source": str(tmp_path / "Processing" / f"{VID}_processing_manifest.json")}])
    assert queue_folder_holds_this_run(queue, proc, VID) is True


def test_a_dict_entry_naming_another_file_is_a_leftover(tmp_path):
    queue, proc = _run(tmp_path, [{"name": f"{VID}.mp4", "error": "in use"}])
    assert queue_folder_holds_this_run(queue, proc, VID) is False


def test_an_old_bare_name_entry_still_counts(tmp_path):
    queue, proc = _run(tmp_path, [f"{VID}_processing_manifest.json"])
    assert queue_folder_holds_this_run(queue, proc, VID) is True

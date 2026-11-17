"""A routing record that is rewritten is replaced in one step.

The return scan and the dashboard's Resolve dialog rewrite a bundle's
existing <video>_routing.json (crossing files off its failed_files list).
Written in place, a write cut off part-way left the record truncated:
read_routing_manifest returned None and the bundle lost its audit trail and
the list of files that never moved. A failed rewrite must leave the old
record whole.
"""
import json

import pytest

from mousereach.watcher import review_routing as rr

VID = "TST9901_P1"


def test_a_rewrite_that_fails_part_way_leaves_the_old_record_whole(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    old = {"video_id": VID, "failed_files": [{"name": f"{VID}.mp4",
                                              "source": "elsewhere"}]}
    rr._write_routing_manifest(bundle, VID, old)
    record = bundle / f"{VID}_routing.json"
    before = record.read_bytes()

    # A value that cannot be written stands in for a write cut off part-way:
    # the in-place writer has already emptied the file and written the start
    # of the new record when it fails.
    broken = {"video_id": VID, "failed_files": [], "zz": object()}
    with pytest.raises(Exception):
        rr._write_routing_manifest(bundle, VID, broken)

    assert record.read_bytes() == before
    assert rr.read_routing_manifest(bundle, VID) == old
    assert [p.name for p in bundle.iterdir()] == [record.name]   # no temp left


def test_a_rewrite_replaces_the_record(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rr._write_routing_manifest(bundle, VID, {"video_id": VID, "failed_files": [1]})
    rr._write_routing_manifest(bundle, VID, {"video_id": VID, "failed_files": []})
    data = json.loads((bundle / f"{VID}_routing.json").read_text(encoding="utf-8"))
    assert data["failed_files"] == []

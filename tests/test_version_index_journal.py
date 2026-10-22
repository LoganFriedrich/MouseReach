"""The shared version index uses the rollback journal, never WAL.

WHY (2026-10-22): SQLite documents WAL as unsafe on a network filesystem -- its
-shm index is shared memory, which machines reading the file over the share do
not share -- and version_index.db lives on the shared drive, read and written
from several machines.
"""
import sqlite3

from mousereach.pipeline import version_index as vi


def test_a_new_index_is_not_wal(tmp_path):
    con = vi._connect(tmp_path / "pipeline_records" / "version_index.db")
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    con.close()


def test_an_index_left_in_wal_by_older_code_is_converted(tmp_path):
    path = tmp_path / "version_index.db"
    old = sqlite3.connect(path)
    old.execute("PRAGMA journal_mode=WAL")
    old.execute("CREATE TABLE t (x)")
    old.close()
    con = vi._connect(path)
    assert con.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    con.close()

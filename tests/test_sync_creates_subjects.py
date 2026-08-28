"""The database sync creates an animal from its first video, never drops data.

WHY: videos arrive daily; the tracking sheet that names the animals arrives
"eventually". Until 2026-08-28 the sync silently skipped any video whose
animal the sheet import had not created yet -- 1,666 sessions across CNT_05
and ASPA went missing from connectome.db that way. Now the animal (and its
cohort and project) is created from the video name and the sheet enriches
it later.
"""
import sqlite3
from pathlib import Path

from mousereach.sync.database import DatabaseSyncer

# The live tables' NOT NULL columns, and nothing that matters beyond them.
DDL = """
CREATE TABLE projects (project_code VARCHAR(10) PRIMARY KEY, project_name VARCHAR(100) NOT NULL,
                       description TEXT, created_at DATETIME);
CREATE TABLE cohorts (cohort_id VARCHAR(20) PRIMARY KEY, project_code VARCHAR(10) NOT NULL,
                      start_date DATE NOT NULL, num_mice INTEGER, notes TEXT,
                      is_archived INTEGER, archived_at DATETIME, archived_reason TEXT,
                      created_at DATETIME, protocol_id INTEGER, protocol_version INTEGER);
CREATE TABLE subjects (subject_id VARCHAR(20) PRIMARY KEY, cohort_id VARCHAR(20) NOT NULL,
                       date_of_birth DATE, date_of_death DATE, sex VARCHAR(1), ear_tag VARCHAR(20),
                       notes TEXT, is_active INTEGER, created_at DATETIME);
"""


def _db(tmp_path: Path) -> Path:
    p = tmp_path / "scratch.db"
    con = sqlite3.connect(p)
    con.executescript(DDL)
    con.commit()
    con.close()
    return p


def test_unknown_animal_is_created_with_cohort_and_project(tmp_path):
    p = _db(tmp_path)
    s = DatabaseSyncer(db_path=p, processing_path=tmp_path)
    assert s.ensure_subject_exists("CNT_07_03", "20260901_CNT0703_P2")
    con = sqlite3.connect(p)
    assert con.execute("SELECT project_name FROM projects WHERE project_code='CNT'").fetchone()[0] == "CNT"
    cid, start, notes = con.execute(
        "SELECT cohort_id, start_date, notes FROM cohorts").fetchone()
    assert cid == "CNT_07" and start == "2026-09-01"
    assert notes.startswith(DatabaseSyncer.AUTO_CREATED_MARK)
    sid, cid2, notes2 = con.execute("SELECT subject_id, cohort_id, notes FROM subjects").fetchone()
    assert (sid, cid2) == ("CNT_07_03", "CNT_07")
    assert notes2.startswith(DatabaseSyncer.AUTO_CREATED_MARK)
    con.close()


def test_existing_records_are_left_alone(tmp_path):
    p = _db(tmp_path)
    con = sqlite3.connect(p)
    con.execute("INSERT INTO projects VALUES ('CNT','Connectome',NULL,NULL)")
    con.execute("INSERT INTO cohorts (cohort_id, project_code, start_date, notes) "
                "VALUES ('CNT_07','CNT','2026-08-15','Imported from sheet')")
    con.execute("INSERT INTO subjects (subject_id, cohort_id, notes) VALUES ('CNT_07_03','CNT_07','from sheet')")
    con.commit()
    con.close()
    s = DatabaseSyncer(db_path=p, processing_path=tmp_path)
    assert s.ensure_subject_exists("CNT_07_03", "20260901_CNT0703_P2")
    con = sqlite3.connect(p)
    assert con.execute("SELECT start_date, notes FROM cohorts").fetchone() == ("2026-08-15", "Imported from sheet")
    assert con.execute("SELECT notes FROM subjects").fetchone()[0] == "from sheet"
    assert con.execute("SELECT project_name FROM projects").fetchone()[0] == "Connectome"
    con.close()


def test_aspa_animal_is_created_the_same_way(tmp_path):
    p = _db(tmp_path)
    s = DatabaseSyncer(db_path=p, processing_path=tmp_path)
    assert s.ensure_subject_exists("ASPA_10_11", "20220811_ASPA1011_P3")
    con = sqlite3.connect(p)
    assert con.execute("SELECT cohort_id, start_date FROM cohorts").fetchone() == ("ASPA_10", "2022-08-11")
    con.close()


def test_every_parseable_video_is_syncable_even_if_unknown(tmp_path):
    p = _db(tmp_path)
    (tmp_path / "20260901_CNT0703_P2_features.json").write_text("{}")
    (tmp_path / "junk_features.json").write_text("{}")
    s = DatabaseSyncer(db_path=p, processing_path=tmp_path)
    found = s.find_syncable_files()
    assert [sid for _, sid in found] == ["CNT_07_03"]

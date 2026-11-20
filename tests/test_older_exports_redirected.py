"""The older export commands give the filed results, counted from pellets
(export/results.py): mousereach-features-csv (retired: it read a file name
nothing writes) and mousereach-build-database's Sessions and Mice sheets
(they counted reach rows). quick-summary: test_secondary_exports_read_resolved_features.
"""
import pytest

from test_results_export_levels import filed  # noqa: F401  (the fixture)
def test_the_retired_features_csv_runs_the_reach_export(filed, tmp_path, monkeypatch, capsys):
    """It looked for *_grasp_features.json, which nothing writes any more."""
    from mousereach.config import Paths
    from mousereach.export import features_csv
    import sys
    monkeypatch.setattr(Paths, "ANALYZED_OUTPUT", filed)
    monkeypatch.setattr(sys, "argv", ["mousereach-features-csv",
                                      "--out", str(tmp_path / "r.csv")])
    assert features_csv.main() == 0
    out = capsys.readouterr().out
    assert "retired" in out and "reach row(s)" in out
    assert (tmp_path / "r.csv").exists()


def test_build_database_sessions_sheet_counts_pellets(filed, tmp_path, monkeypatch):
    """Its Sessions sheet was grouped from reach rows: retrieval_rate was the
    share of reaches whose pellet was retrieved."""
    import sys
    import pandas as pd
    folder = filed / "PRJ" / "PRJ_01"
    out = tmp_path / "db.xlsx"
    monkeypatch.setattr(sys, "argv", ["mousereach-build-database", "-d", str(folder),
                                      "-o", str(out), "--force", "--no-brainglobe",
                                      "--no-surgery"])
    from mousereach.analysis.cli import main_build_database
    main_build_database()
    sessions = pd.read_excel(out, sheet_name="Sessions")
    p1 = sessions[sessions["session_id"] == "20200101_ABC0101_P1"].iloc[0]
    assert p1["session_pellets_presented"] == 3
    assert p1["session_success_rate"] == pytest.approx(1 / 3)
    assert "retrieval_rate" not in sessions.columns
    assert "animal_success_rate" in pd.read_excel(out, sheet_name="Mice").columns


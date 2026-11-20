"""The Export results tab calls the same function as mousereach-results.
"""
import pytest

from test_results_export_levels import filed  # noqa: F401  (the fixture)
def test_the_export_tab_calls_the_same_function(filed, tmp_path):
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from mousereach.export.results_widget import ResultsExportWidget, LEVEL_CHOICES
    assert {k for _t, k in LEVEL_CHOICES} == {"reach", "pellet", "session", "day", "animal"}
    w = ResultsExportWidget()
    res = w.run_export("animal", str(filed), str(tmp_path / "a.xlsx"), "xlsx")
    assert res["rows"] == 2 and res["path"].endswith(".xlsx")
    assert "Exported 2 animal row(s)" in w.status.text()


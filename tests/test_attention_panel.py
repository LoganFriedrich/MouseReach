"""What needs a person is shown in red at the top of the Pipeline Dashboard.

WHY (Logan, 2026-10-26): "There needs to be a way to alert a human to an
issue of the kind you are hunting down" -- and all a person sees is the
screen. The watchers list such issues in <NAS root>/attention/; the
dashboard shows the ones a person must fix in red, and the ones that are
only waiting as a grey line.
"""
import json

from mousereach.watcher.attention import read_items


def test_every_list_is_read_and_an_unreadable_one_is_said(tmp_path):
    d = tmp_path / "attention"
    d.mkdir()
    (d / "A_hand_scores.json").write_text(json.dumps({"items": [
        {"needs_person": True, "says": "fix me"}, {"needs_person": False, "says": "wait"}]}))
    (d / "B_hand_scores.json").write_text("{broken")
    items = read_items(tmp_path)
    assert [i["says"] for i in items if i.get("listed_in") == "A_hand_scores.json"] == ["fix me", "wait"]
    bad = [i for i in items if i.get("kind") == "unreadable_list"]
    assert bad and bad[0]["needs_person"]


def test_no_folder_is_nothing_to_show(tmp_path):
    assert read_items(tmp_path) == []


def test_the_dashboard_shows_what_needs_a_person_in_red_and_hides_when_none(tmp_path, monkeypatch):
    from qtpy.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])
    from mousereach.config import Paths
    from mousereach.dashboard.widget import PipelineDashboard
    monkeypatch.setattr(Paths, "NAS_ROOT", tmp_path, raising=False)
    d = tmp_path / "attention"
    d.mkdir()
    lst = d / "HOST_hand_scores.json"
    from types import SimpleNamespace
    dash = SimpleNamespace(attention_label=QLabel(), waiting_label=QLabel(),
                           ATTENTION_SHOWN=PipelineDashboard.ATTENTION_SHOWN)
    dash._update_attention = lambda: PipelineDashboard._update_attention(dash)

    lst.write_text(json.dumps({"items": [
        {"needs_person": True, "says": "20250909_ABC0216_P1 has no hand scores"},
        {"needs_person": False, "says": "waiting"}]}))
    dash._update_attention()
    assert not dash.attention_label.isHidden()
    assert "Needs your attention (1)" in dash.attention_label.text()
    assert "20250909_ABC0216_P1 has no hand scores" in dash.attention_label.text()
    assert "1 analysed video(s) are held before triage" in dash.waiting_label.text()

    lst.write_text(json.dumps({"items": []}))
    dash._update_attention()
    assert dash.attention_label.isHidden() and dash.waiting_label.isHidden()

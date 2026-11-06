"""The viewer reads saved files correctly and every tab renders without an error."""
from pathlib import Path

import pytest

from src.app import data
from src.config import RESULTS_DIR

VIEWER = Path(__file__).resolve().parents[1] / "src" / "app" / "viewer.py"


@pytest.mark.parametrize("decision,resolved,label", [
    ("REJECT", False, "Failure stopped"), ("ABORT", True, "Pass lost"),
    ("FLAG_FOR_REVIEW", False, "Failure missed"), ("FLAG_FOR_REVIEW", True, "Pass reviewed"),
])
def test_outcome_label(decision, resolved, label):
    assert data.outcome_label(decision, resolved) == label


@pytest.mark.parametrize("dataset", ["test", "lite"])
def test_sample_logs_match_saved_outcomes(dataset):
    runs, total = data.runs_with_logs(dataset)
    assert total > 1000 and len(runs) > 0
    for _, row in runs.head(5).iterrows():
        events, outcome = data.load_run(row["log"])
        assert outcome.uid == row["uid"]
        assert outcome.final_decision.value == row["decision"]
        assert bool(outcome.resolved) == bool(row["resolved"])
        steps = data.steps_frame(events)
        ran = steps[~steps["Decision"].isin(data.BLOCKING)]
        assert len(ran) == outcome.steps_seen
        assert steps["Action no."].is_monotonic_increasing


def test_md_section_stops_at_next_heading():
    section = data.md_section(RESULTS_DIR / "governor_eval.md", "### SWE-Gym test: runs")
    assert section.startswith("| config") and "### " not in section and "SWE-bench Lite" not in section


def test_readable_table_filters_subset():
    table = "| config | subset | x |\n|---|---|---|\n| full | all | 1 |\n| full | v2 | 2 |\n| no_hints | all | 3 |"
    out = data.readable_table(table, subset="all").splitlines()
    assert out[0] == "| Configuration | X |" and len(out) == 4
    assert out[2] == "| Full agent | 1 |" and out[3] == "| No hints | 3 |"


def test_reroute_table_keeps_the_four_compared_policies():
    table, _ = data.split_table(data.md_section(RESULTS_DIR / "governor_eval.md", "### Reroute on test: tasks solved and cost per task"))
    rows = data.reroute_table(table).splitlines()[2:]
    assert [r.split("|")[1].strip() for r in rows] == list(data.REROUTE_POLICIES.values())
    assert "10.1% ± 0.5" in rows[2] and "20094" in rows[2] and "0.53" in rows[2]


def test_app_renders_without_errors():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app = streamlit_testing.AppTest.from_file(str(VIEWER), default_timeout=60).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Replay", "Runs", "Results", "Failures"]
    outcome = next(s for s in app.selectbox if s.label == "Outcome")
    for option in outcome.options:
        outcome.set_value(option).run()
        assert not app.exception
        outcome = next(s for s in app.selectbox if s.label == "Outcome")
    dataset = next(s for s in app.selectbox if s.label == "Dataset" and s.key != "failures-dataset")
    dataset.set_value("SWE-bench Lite").run()
    assert not app.exception

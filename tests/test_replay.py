"""The step-by-step replay runs the real agent, and gives the same decisions as the saved E3 logs."""
from pathlib import Path

import pytest

from src.agent.governor import Governor
from src.agent.risk import RiskModels
from src.app import data, replay
from src.data.loader import load_meta
from src.data.parser import iter_traces
from src.schema import Decision, GovernorEvent
from tests.test_governor import _models, _trace

VIEWER = Path(__file__).resolve().parents[1] / "src" / "app" / "viewer.py"
needs_data = pytest.mark.skipif(not replay.has_agent_data(), reason="needs the SWE-Gym data")


def _session(models, trace):
    return replay.live_session(trace, models)


def _play(session):
    while not (session.finished or session.waiting):
        session.step()
    return session


def test_advances_one_event_per_step():
    s = _session(_models(), _trace())
    s.step()
    assert len(s.events) == 1 and not s.finished
    s.step()
    assert len(s.events) == 2


def test_runs_to_review_without_asking_when_trust_is_high():
    s = _play(_session(_models(), _trace()))
    assert s.finished and not s.answers
    assert s.outcome.final_decision == Decision.flag_for_review


def test_waits_for_a_person_before_an_abort_takes_effect():
    s = _play(_session(_models(p_ckpt=0.01), _trace()))
    assert s.waiting and not s.finished
    assert s.pending.decision == Decision.abort and s.pending.step == 5
    assert not any(e.decision == Decision.abort for e in s.events)


def test_approve_gives_the_same_run_as_no_human_at_all():
    models, trace = _models(p_ckpt=0.01), _trace()
    s = _play(_session(models, trace))
    s.answer("approve")
    _play(s)
    items = list(Governor(models).replay(trace))
    expected = [e for e in items if isinstance(e, GovernorEvent)]
    assert [e.model_dump(exclude={"human"}) for e in s.events] == [e.model_dump(exclude={"human"}) for e in expected]
    assert s.outcome.model_dump() == items[-1].model_dump()


def test_score_behind_a_question_is_visible_while_waiting():
    s = _play(_session(_models(p_ckpt=0.01), _trace()))
    assert s.trust_curve()[-1] == {"step": 5, "p_success": 0.01}


def test_overrule_continues_the_run_without_repeating_events():
    s = _play(_session(_models(p_ckpt=0.01), _trace()))
    shown = len(s.events)
    s.answer("override")
    _play(s)
    assert s.finished and s.outcome.final_decision == Decision.flag_for_review and s.outcome.human_overrides == 1
    assert [e.seq for e in s.events] == sorted({e.seq for e in s.events})
    assert len(s.events) > shown


def test_stop_ends_the_run():
    s = _play(_session(_models(p_ckpt=0.01), _trace()))
    s.answer("stop")
    _play(s)
    assert s.outcome.final_decision == Decision.human_stop


def test_rejected_answer_is_refused():
    s = _play(_session(_models(p_ckpt=0.01), _trace()))
    with pytest.raises(ValueError):
        s.answer("maybe")


def test_label_is_not_shown_before_the_end():
    s = _session(_models(), _trace(resolved=True))
    s.step()
    assert s.outcome is None
    assert "resolved" not in s.events[0].model_dump()


def test_curve_matches_the_agents_own_report():
    s = _play(_session(_models(), _trace()))
    assert s.trust_curve() == s.outcome.trust_curve
    assert s.hints() == s.outcome.hints


def test_saved_log_steps_through_without_asking():
    runs, _ = data.runs_with_logs("test")
    events, outcome = data.load_run(runs.iloc[0]["log"])
    s = _play(replay.saved_session(events, outcome))
    assert s.finished and not s.live and s.events == events and s.outcome == outcome


@needs_data
def test_real_agent_reproduces_the_saved_logs_exactly():
    """The committed sample logs came from E3. Approving every question must rebuild them event for event."""
    logs = sorted(data.SAMPLE.glob("test_*.jsonl"))
    saved = {o.uid: (e, o) for e, o in map(data.load_run, logs)}
    models = RiskModels.fit()
    traces = list(iter_traces(load_meta(), uids=set(saved)))
    assert saved and {t.uid for t in traces} == set(saved)
    for trace in traces:
        s = _session(models, trace)
        while not s.finished:
            _play(s)
            if s.waiting:
                s.answer("approve")
        events, outcome = saved[trace.uid]
        assert [e.model_dump() for e in s.events] == [e.model_dump() for e in events]
        assert s.outcome.model_dump() == outcome.model_dump()


def test_app_falls_back_to_a_saved_log_and_says_so(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setattr(replay, "has_agent_data", lambda: False)
    app = testing.AppTest.from_file(str(VIEWER), default_timeout=120).run()
    assert not app.exception
    assert any("isn't computing anything" in c.value for c in app.caption)
    next(b for b in app.button if b.label == "Start").click().run()
    for _ in range(12):
        assert not app.exception
        assert not [b for b in app.button if b.label == "Approve"]
        if any(m.label == "Real result" for m in app.metric):
            break
        app.run()
    assert any(m.label == "Real result" for m in app.metric)


@needs_data
def test_app_replays_a_run_through_the_gate():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file(str(VIEWER), default_timeout=120).run()
    assert not app.exception
    next(b for b in app.button if b.label == "Start").click().run()
    assert not app.exception
    for _ in range(12):
        approve = [b for b in app.button if b.label == "Approve"]
        if approve:
            approve[0].click().run()
            break
        app.run()
        assert not app.exception
    assert approve, "an aborting run should reach the gate"
    for _ in range(4):
        app.run()
        assert not app.exception
    assert any(m.label == "Real result" for m in app.metric)

"""Governance agent: uses stub models, so these tests need no dataset."""
import numpy as np
import pandas as pd

from src.agent.governor import Governor, GovernorConfig
from src.agent.patterns import PatternDetector, next_phase
from src.agent.reroute import simulate
from src.agent.risk import RiskConfig, RiskModels, ScoredModel
from src.calibration import Calibrator
from src.data.parser import parse_messages
from src.data.swebench_lite import _messages_v1
from src.prefix import PrefixState
from src.schema import Decision, GovernorEvent, RunOutcome
from tests.test_parser import MSGS, _call

PATCH = "diff --git a/x b/x\n+y\n"


class StubModel:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        if self.p is None:
            raise RuntimeError("model down")
        return np.array([[1 - self.p, self.p]])


def _scored(p, tau, cols, auroc=0.7):
    return ScoredModel(model=StubModel(p), calibrator=Calibrator("none"), cols=cols, tau=tau, dev_auroc=auroc,
                       n_dev=100, oof=pd.DataFrame())


def _models(p_ckpt=0.5, p_final=0.5, tau_abort=0.1, tau_review=0.1, auroc=0.7):
    ck = {5: _scored(p_ckpt, tau_abort, ["step"], auroc)}
    return RiskModels(RiskConfig(), ck, _scored(p_final, tau_review, ["n_steps"]))


def _trace(n_extra=8, resolved=True, patch=PATCH, cmd=None):
    extra = []
    for i in range(n_extra):
        extra += [{"role": "assistant", "content": "",
                   "tool_calls": [_call(f"e{i}", "execute_bash", command=cmd or f"ls {i}")]},
                  {"role": "tool", "name": "execute_bash", "tool_call_id": f"e{i}",
                   "content": "[Command finished with exit code 0]"}]
    return parse_messages(MSGS[:-1] + extra + [MSGS[-1]], uid="u", instance_id="i", run_id="r",
                          resolved=resolved, final_patch=patch)


def _events(gov, tr):
    items = list(gov.replay(tr))
    assert isinstance(items[-1], RunOutcome) and all(isinstance(e, GovernorEvent) for e in items[:-1])
    return items[:-1], items[-1]


class Gate:
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def ask(self, decision, uid, step, reason):
        self.asked.append(decision)
        return self.answer


def test_high_trust_run_goes_to_human_review():
    ev, out = _events(Governor(_models()), _trace())
    assert out.final_decision == Decision.flag_for_review and out.abort_step is None
    assert out.steps_seen == out.steps_total and out.tokens_used == out.tokens_total
    assert ev[-1].tool_calls[0].tool == "score_final_risk"


def test_abort_stops_before_the_next_action():
    ev, out = _events(Governor(_models(p_ckpt=0.01)), _trace())
    assert out.final_decision == Decision.abort and out.abort_step == 5 and out.steps_seen == 5
    assert out.tokens_used < out.tokens_total
    assert [c.tool for c in ev[-1].tool_calls] == ["score_prefix_risk", "abort_run"]


def test_no_abort_where_dev_shows_no_signal():
    _, out = _events(Governor(_models(p_ckpt=0.01, auroc=0.5)), _trace())
    assert out.final_decision != Decision.abort
    assert out.trust_curve[0] == {"step": 5, "p_success": 0.01}      # still scored for the curve


def test_early_warning_switch_removes_checkpoint_scoring():
    gov = Governor(_models(p_ckpt=0.01), GovernorConfig(name="x", early_warning=False))
    ev, out = _events(gov, _trace())
    assert out.final_decision == Decision.flag_for_review
    assert not any(c.tool == "score_prefix_risk" for e in ev for c in e.tool_calls)


def test_low_final_trust_is_rejected():
    _, out = _events(Governor(_models(p_final=0.01)), _trace())
    assert out.final_decision == Decision.reject and out.p_final == 0.01


def test_human_override_and_stop():
    g = Gate("override")
    _, out = _events(Governor(_models(p_ckpt=0.01, p_final=0.01), gate=g), _trace())
    assert out.final_decision == Decision.flag_for_review and out.human_overrides == 2
    assert g.asked == [Decision.abort, Decision.reject]
    ev, out = _events(Governor(_models(p_ckpt=0.01), gate=Gate("stop")), _trace())
    assert out.final_decision == Decision.human_stop and ev[-1].human == "stop"


def test_label_never_changes_a_decision():
    gov = Governor(_models())
    a, oa = _events(gov, _trace(resolved=True))
    b, ob = _events(gov, _trace(resolved=False))
    assert [e.model_dump() for e in a] == [e.model_dump() for e in b]
    assert oa.final_decision == ob.final_decision and (oa.resolved, ob.resolved) == (True, False)


def test_final_patch_only_used_at_triage():
    gov = Governor(_models())
    a, _ = _events(gov, _trace(patch=PATCH))
    b, _ = _events(gov, _trace(patch=""))
    assert [e.model_dump() for e in a[:-1]] == [e.model_dump() for e in b[:-1]]


def test_model_failure_fails_open():
    ev, out = _events(Governor(_models(p_ckpt=None)), _trace())
    bad = [c for e in ev for c in e.tool_calls if not c.ok]
    assert bad[0].tool == "score_prefix_risk" and "model down" in bad[0].error
    assert out.final_decision == Decision.flag_for_review and out.degraded_calls == 1


def test_final_scoring_failure_goes_to_a_human():
    _, out = _events(Governor(_models(p_final=None)), _trace())
    assert out.final_decision == Decision.flag_for_review and out.degraded_calls == 1


def test_loop_hint_fires_once_and_respects_limits():
    _, out = _events(Governor(_models()), _trace(cmd="ls same"))
    assert sum(h["pattern"] == "repeat_loop" for h in out.hints) == 1
    assert "repeat_loop" in out.patterns_seen
    _, off = _events(Governor(_models(), GovernorConfig(name="x", hints=False)), _trace(cmd="ls same"))
    assert off.hints == [] and "repeat_loop" in off.patterns_seen      # seen, but no hint sent


def test_finish_patterns():
    st, det = PrefixState("u", "i"), PatternDetector()
    assert det.before_finish(st) == ["finish_without_edit"]
    tr = _trace(n_extra=0)
    for s in tr.steps:
        if st.is_action(s) and s.action_type.value == "finish":
            break
        st.update(s)
        if st.is_action(s):
            det.update(s, st)
    assert det.before_finish(st) == ["finish_unverified"]     # the edit came after the last test


def test_phase_tracking():
    st = PrefixState("u", "i")
    seen, phase = [], ""
    for s in _trace(n_extra=0).steps:
        st.update(s)
        if st.is_action(s):
            phase = next_phase(phase, s, st)
            seen.append(phase)
    assert seen == ["locate", "reproduce", "fix", "submit"]


def test_normalizer_switch_changes_tool_names():
    events = [{"id": 1, "source": "agent", "action": "run_ipython", "args": {"code": "open_file('a.py')"}},
              {"id": 2, "cause": 1, "observation": "run_ipython", "content": "ok"}]
    on = _messages_v1(events, normalize=True)[1]["tool_calls"][0]["function"]["name"]
    off = _messages_v1(events, normalize=False)[1]["tool_calls"][0]["function"]["name"]
    assert (on, off) == ("str_replace_editor", "execute_ipython_cell")


def test_explain_names_the_top_factors_and_is_safe_on_other_models():
    from src.baselines.tabular import _logreg
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 1.0, 2.0, 2.0]})
    pipe = _logreg(["a", "b"])(0).fit(X, [False, False, True, True])
    m = ScoredModel(model=pipe, calibrator=Calibrator("none"), cols=["a", "b"], tau=0.1, dev_auroc=0.7, n_dev=4)
    top = m.explain({"a": 3.0, "b": 2.0}, k=1)
    assert len(top) == 1 and top[0]["feature"] in ("a", "b") and top[0]["effect"] > 0
    assert _scored(0.5, 0.1, ["step"]).explain({"step": 1}) == []      # a stub model has no coefficients


def test_float_features_are_not_truncated():
    seen = {}

    class Spy:
        def predict_proba(self, X):
            seen.update(X.iloc[0].to_dict())
            return np.array([[0.5, 0.5]])
    m = ScoredModel(model=Spy(), calibrator=Calibrator("none"), cols=["phase_entropy"], tau=0.1, dev_auroc=0.7, n_dev=1)
    m.score({"phase_entropy": 1.5})
    assert seen["phase_entropy"] == 1.5


def _runs():
    rows = [("a1", "A", "ABORT", 10, 100, True), ("a2", "A", "FLAG_FOR_REVIEW", 50, 50, True),
            ("b1", "B", "REJECT", 30, 30, False), ("c1", "C", "FLAG_FOR_REVIEW", 40, 40, False)]
    return pd.DataFrame(rows, columns=["uid", "instance_id", "decision", "tokens_used", "tokens_total", "resolved"])


def test_reroute_moves_on_after_a_stop():
    per = simulate(_runs(), seed=0)
    r = per[per["policy"] == "governor_reroute"].set_index("instance_id")
    assert r.loc["A", "solved"] and r.loc["A", "reviews"] == 1      # a stop on a1 can't lose the task
    assert r.loc["A", "tokens"] in (50, 60)                         # 60 if a1 came first and was aborted at 10
    assert not r.loc["B", "solved"] and r.loc["B", "reviews"] == 0  # every attempt stopped: nothing submitted
    assert not r.loc["C", "solved"] and r.loc["C", "reviews"] == 1


def test_review_all_is_the_upper_bound_on_tasks_solved():
    for seed in range(5):
        per = simulate(_runs(), seed=seed)
        s = per.groupby("policy")["solved"].sum()
        assert s["review_all"] >= s["governor_reroute"] >= s["governor_single"]


def test_reroute_order_depends_on_the_seed():
    orders = {simulate(_runs(), seed=s).query("policy == 'governor_single' and instance_id == 'A'")["tokens"].iloc[0]
              for s in range(10)}
    assert orders == {10.0, 50.0}

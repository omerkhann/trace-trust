from src.data.parser import parse_messages
from src.features import extract
from src.prefix import PrefixState, prefixes
from src.schema import PROGRESS_FEATURES
from tests.test_features import PATCH
from tests.test_parser import MSGS, _call


def _trace(n_extra: int = 0):
    extra = []
    for i in range(n_extra):
        extra += [{"role": "assistant", "content": "", "tool_calls": [_call(f"e{i}", "execute_bash", command=f"ls {i}")]},
                  {"role": "tool", "name": "execute_bash", "tool_call_id": f"e{i}",
                   "content": "[Command finished with exit code 0]"}]
    return parse_messages(MSGS[:-1] + extra + [MSGS[-1]], uid="u", instance_id="i", run_id="r",
                          resolved=True, final_patch=PATCH)


def test_full_prefix_matches_run_features():
    tr = _trace()
    st = PrefixState(tr.uid, tr.instance_id, tr.resolved)
    for s in tr.steps:
        st.update(s)
    p, f = st.snapshot(), extract(tr)
    assert p.step == f.n_steps
    assert p.tokens_so_far == f.total_tokens
    assert p.n_errors == f.n_errors
    assert p.n_repeated_commands == f.n_repeated_commands
    assert p.n_distinct_tools == f.n_distinct_tools
    assert p.n_edits_without_view == f.n_edits_without_view
    assert p.ran_tests == f.ran_tests


def test_prefixes_only_for_runs_still_going():
    short = _trace()                                    # 4 actions: no abs checkpoint (first is 5)
    assert [p for p in prefixes(short) if p.checkpoint == "abs"] == []
    long_ = _trace(n_extra=8)                           # 12 actions: abs at 5 and 10
    ps = prefixes(long_)
    assert [p.step for p in ps if p.checkpoint == "abs"] == [5, 10]
    fr = {p.prefix_frac: p.step for p in ps if p.checkpoint == "frac"}
    assert fr == {0.25: 3, 0.5: 6, 0.75: 9}
    assert all(p.resolved for p in ps)


def test_full_prefix_matches_new_features():
    tr = _trace()
    st = PrefixState(tr.uid, tr.instance_id, tr.resolved)
    for s in tr.steps:
        st.update(s)
    p, f = st.snapshot().model_dump(), extract(tr).model_dump()
    assert {k: p[k] for k in PROGRESS_FEATURES} == {k: f[k] for k in PROGRESS_FEATURES}

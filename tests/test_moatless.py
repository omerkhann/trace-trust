from src.data.moatless import to_steps, to_trace
from src.features import extract
from src.schema import ActionType


def _log(action, text="", error=None, prompt="<issue>Fix it</issue>"):
    return action, {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": prompt}],
                    "completion": [{"content": text}], "error": error}


LOGS = [_log("SearchCode", '{"scratch_pad": "look in a.py"}'),
        _log("IdentifyCode", "### pkg/a.py is relevant"),
        _log("DecideRelevance", '{"scratch_pad": "enough"}'),
        _log("PlanToCode", '{"action": "modify", "file_path": "pkg/a.py"}'),
        _log("EditCode", "<replace>x = 2</replace>"),
        _log("PlanToCode", '{"action": "finish"}')]
PRED = {"model_patch": "diff --git a/pkg/a.py b/pkg/a.py\n+x = 2\n", "report": {"resolved": True}}


def test_normalizer_maps_actions_onto_the_schema():
    s = to_steps(LOGS)
    assert [x.action_type for x in s] == [ActionType.bash, ActionType.view, ActionType.message, ActionType.message,
                                          ActionType.edit, ActionType.finish]
    assert s[1].path == "pkg/a.py" and s[4].path == "pkg/a.py"       # the edit's file comes from the plan
    assert s[0].thought_files == ["a.py"]


def test_off_switch_keeps_raw_action_names():
    s = to_steps(LOGS, normalize=False)
    assert {x.action_type for x in s} == {ActionType.other_tool}
    assert s[0].tool_name == "SearchCode"


def test_trace_and_features():
    tr = to_trace("r", "django__django-1", LOGS, PRED)
    assert tr.uid == "r:django__django-1" and tr.resolved and "Fix it" in tr.task
    f = extract(tr)
    assert f.n_steps == 6 and f.finished and not f.patch_empty and f.patch_files == 1
    assert not f.ran_tests and not f.last_check_passed             # Moatless has no way to run anything
    assert f.n_edits_without_view == 0                             # identified, then edited
    raw = extract(to_trace("r", "django__django-1", LOGS, PRED, normalize=False))
    assert raw.n_steps == 6 and not raw.finished and raw.n_distinct_tools == 5


def test_errors_are_carried_over():
    s = to_steps([_log("SearchCode", "", error="boom")])
    assert s[0].is_error and s[0].exit_code == 1


def test_a_search_query_cannot_count_as_a_test_run():
    f = extract(to_trace("r", "i", [_log("SearchCode", '{"query": "pytest test_foo.py"}')], PRED))
    assert not f.ran_tests and not f.checked_after_last_edit

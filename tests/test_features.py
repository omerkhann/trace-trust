from src.data.parser import parse_messages
from src.features import extract, extract_task, patch_stats
from tests.test_parser import MSGS, _call

PATCH = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1,2 @@
-x = 1
+x = 2
 y = 3
"""


def test_patch_stats():
    assert patch_stats(PATCH) == (2, 1)
    assert patch_stats("") == (0, 0)


def test_extract():
    msgs = MSGS[:-1] + [
        {"role": "assistant", "content": "", "tool_calls": [_call("x", "execute_bash", command="pytest -q")]},
        {"role": "tool", "name": "execute_bash", "tool_call_id": "x", "content": "[Command finished with exit code 0]"},
        MSGS[-1],
    ]
    f = extract(parse_messages(msgs, uid="u", instance_id="i", run_id="r", resolved=True, final_patch=PATCH))
    assert f.n_steps == 5
    assert f.n_errors == 2                 # failing pytest + editor ERROR
    assert f.n_repeated_commands == 1      # second identical `pytest -q`
    assert f.n_distinct_tools == 3
    assert f.n_edits_without_view == 1     # bar.py edited, only foo.py viewed
    assert f.ran_tests and f.finished and not f.patch_empty
    assert f.patch_lines == 2 and f.patch_files == 1


def test_extract_task():
    task = ("<uploaded_files>/workspace/x</uploaded_files>\n<pr_description>\nBug\n```python\nf()\n```\n"
            "Traceback (most recent call last):\n  ...\n</pr_description>\nPlease fix.")
    tr = parse_messages([{"role": "user", "content": task}], uid="u", instance_id="getmoto__moto-5321", run_id="r")
    t = extract_task(tr)
    assert t.repo == "getmoto__moto"
    assert t.issue_code_blocks == 1 and t.issue_has_traceback
    assert 0 < t.issue_tokens < len(task) // 4


def _progress(*actions):
    """actions: (tool, args, observation[, assistant text])"""
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i, (tool, args, obs, *text) in enumerate(actions):
        msgs += [{"role": "assistant", "content": text[0] if text else "", "tool_calls": [_call(str(i), tool, **args)]},
                 {"role": "tool", "name": tool, "tool_call_id": str(i), "content": obs}]
    return extract(parse_messages(msgs, uid="u", instance_id="i", run_id="r", resolved=False))


OK, BAD = "[Command finished with exit code 0]", "[Command finished with exit code 1]"
EDITOR = "str_replace_editor"


def test_checks_fail_then_pass():
    f = _progress(("execute_bash", {"command": "python repro.py"}, BAD),
                  (EDITOR, {"command": "str_replace", "path": "/w/a.py"}, "OBSERVATION:\nok"),
                  ("execute_bash", {"command": "pytest -q"}, OK))
    assert f.last_check_passed and f.checked_after_last_edit and f.fail_then_pass
    assert f.first_edit_step == 2


def test_edit_after_check_is_unverified():
    f = _progress(("execute_bash", {"command": "pytest"}, OK),
                  (EDITOR, {"command": "str_replace", "path": "/w/a.py"}, "OBSERVATION:\nok"))
    assert f.last_check_passed and not f.checked_after_last_edit and not f.fail_then_pass


def test_piped_check_has_unknown_outcome():
    f = _progress(("execute_bash", {"command": "pytest | tail -5"}, OK))
    assert not f.last_check_passed and f.checked_after_last_edit


def test_non_checks_are_ignored():
    f = _progress(("execute_bash", {"command": "ls"}, OK), ("execute_bash", {"command": "python -m pip install x"}, OK))
    assert not f.last_check_passed and not f.checked_after_last_edit


def test_streaks_and_responsiveness():
    same = ("execute_bash", {"command": "cat x"}, BAD)
    f = _progress(same, same, same, ("execute_bash", {"command": "ls"}, OK))
    assert f.max_identical_streak == 3 and f.max_error_streak == 3
    assert abs(f.error_responsiveness - 1 / 3) < 1e-9     # 3 errors followed; only the last changed action
    assert _progress(("execute_bash", {"command": "ls"}, OK)).error_responsiveness == 1.0


def test_no_edit_and_entropy():
    f = _progress(("execute_bash", {"command": "ls"}, OK), ("execute_bash", {"command": "ls /"}, OK))
    assert f.first_edit_step == 2 and f.phase_entropy == 0
    g = _progress((EDITOR, {"command": "view", "path": "/w"}, "OBSERVATION:\nok"),
                  (EDITOR, {"command": "str_replace", "path": "/w/a.py"}, "OBSERVATION:\nok"))
    assert abs(g.phase_entropy - 1) < 1e-9                # two phases, equal shares


def test_thought_action_mismatch():
    act = (EDITOR, {"command": "view", "path": "/w/a.py"}, "OBSERVATION:\nok")
    assert _progress((*act, "Let me open a.py")).thought_action_mismatch == 0
    assert _progress((*act, "Now look at src/b.py")).thought_action_mismatch == 1
    assert _progress(act).thought_action_mismatch == 0    # no text naming a file: not counted

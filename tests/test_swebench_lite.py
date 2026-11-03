import json

from experiments.verifier_prompts import render_turns
from src.data.swebench_lite import to_messages, to_trace
from src.schema import ActionType


def _tcm(call_id, fn, args, resp_id, content=""):
    return {"tool_call_id": call_id, "function_name": fn,
            "model_response": {"id": resp_id, "choices": [{"message": {
                "content": content, "tool_calls": [{"id": call_id, "function": {"name": fn, "arguments": json.dumps(args)}}]}}]}}


ISSUE = "<pr_description>\nfix it\n</pr_description>"


def test_v2_events_rebuild_chat_messages_with_exit_codes():
    t1 = _tcm("c1", "execute_bash", {"command": "pytest"}, "r1", "run the tests")
    t2 = _tcm("c2", "str_replace_editor", {"command": "view", "path": "/a.py"}, "r2")
    rec = {"instance_id": "x__y-1", "report": {"resolved": True}, "test_result": {"git_patch": "diff --git a/a b/a\n+x"},
           "history": [
               {"id": 0, "source": "user", "action": "message", "message": ISSUE},
               {"id": 1, "source": "agent", "action": "run", "args": {"command": "pytest"}, "tool_call_metadata": t1},
               {"id": 2, "source": "agent", "observation": "run", "cause": 1, "content": "1 failed",
                "extras": {"exit_code": 1}, "tool_call_metadata": t1},
               {"id": 3, "source": "agent", "action": "run_ipython", "args": {"code": "..."}, "tool_call_metadata": t2},
               {"id": 4, "source": "agent", "observation": "run_ipython", "cause": 3, "content": "file text"},
           ]}
    msgs = to_messages(rec)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert msgs[3]["content"].endswith("[Command finished with exit code 1]")
    assert msgs[5]["name"] == "str_replace_editor" and msgs[5]["tool_call_id"] == "c2"   # matched via cause
    tr = to_trace(rec, "claude35_v2.2")
    bash = [s for s in tr.steps if s.action_type == ActionType.bash][0]
    assert bash.exit_code == 1 and bash.is_error
    assert tr.resolved and tr.uid == "claude35_v2.2:x__y-1"


def test_v1_skills_map_onto_editor_and_bash_and_pairs_flatten():
    def act(i, code):
        return [{"id": i, "source": "agent", "action": "run_ipython", "args": {"code": code, "thought": ""}},
                {"id": i + 1, "source": "agent", "observation": "run_ipython", "cause": i, "content": "ok"}]
    rec = {"instance_id": "x__y-2", "report": {"resolved": False}, "test_result": {"git_patch": ""},
           "instance": {"problem_statement": "fix it"},
           "history": [[{"id": 0, "source": "user", "action": "message", "message": "Please fix"}, {"observation": "null"}],
                       act(1, "open_file('/r/a.py')"), act(3, "scroll_down()"),
                       act(5, "edit_file_by_replace('/r/a.py', to_replace='x', new_content='y')"),
                       act(7, "search_dir('foo', '/r')")]}
    tr = to_trace(rec, "gpt4o_v1.9")
    acts = [(s.action_type, s.path) for s in tr.steps if s.role.value == "assistant"]
    assert acts == [(ActionType.view, "/r/a.py"), (ActionType.view, "/r/a.py"),
                    (ActionType.edit, "/r/a.py"), (ActionType.bash, None)]
    assert "<pr_description>" in tr.task and "fix it" in tr.task


def test_render_turns_matches_training_conventions():
    msgs = [{"role": "system", "content": "s"},
            {"role": "user", "content": "issue"},
            {"role": "assistant", "content": "look", "tool_calls": [
                {"id": "a", "function": {"name": "str_replace_editor",
                                         "arguments": json.dumps({"command": "create", "file_text": "l1\nl2"})}}]},
            {"role": "tool", "name": "str_replace_editor", "content": "OBSERVATION:\ndone"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "b", "function": {"name": "finish", "arguments": ""}}]}]
    blocks = render_turns(msgs)
    assert blocks[0].startswith("*** Turn 1 - USER ***\nissue\n")
    assert "look\n\n<function=str_replace_editor>\n<parameter=command>create</parameter>\n" \
           "<parameter=file_text>\nl1\nl2\n</parameter>\n</function>\n" in blocks[1]
    assert blocks[2].startswith("*** Turn 2 - USER ***\nEXECUTION RESULT of [str_replace_editor]:\nOBSERVATION:\ndone")
    assert blocks[3] == "*** Turn 2 - ASSISTANT ***\n<function=finish>\n\n" + "-" * 100 + "\n"

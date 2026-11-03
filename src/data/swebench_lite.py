"""SWE-bench Lite runs from OpenHands' official evaluation outputs: a clean external test set.

Source: HF OpenHands/openhands-evaluation-outputs, outputs/SWE-bench_Lite-test/CodeActAgent/<run>/
output.jsonl. One JSON record per task: instance_id, history (OpenHands event stream),
test_result.git_patch, report.resolved, instance.problem_statement. Only the patch and the label
are read from test_result / report; test output is never used.

None of SWE-bench Lite's 12 repositories is among SWE-Gym's 11, and none of its 300 issues is in the
SWE-Gym verifier's training data (checked 3 Nov), so both our models and the verifier see these
runs for the first time.

The event stream is turned back into the OpenAI chat messages that SWE-Gym's runs use, then parsed
by `parser.parse_messages`, so features mean the same thing on both datasets:
  v2.x  function calling with the same tools as SWE-Gym (execute_bash, str_replace_editor, finish).
        Each action event carries the raw model response, which is copied as is.
  v1.9  older "agentskills" scaffold: editor helpers (open_file, edit_file_by_replace, ...) called
        from IPython. Each call is mapped onto the matching execute_bash / str_replace_editor call
        (a small scaffold normaliser), so features are comparable but not identical.

    python -m src.data.swebench_lite      # download (if needed) and print per-run stats
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Iterator, Optional

from src import config
from src.data.parser import parse_messages
from src.schema import Trace

LITE_DIR = config.DATA_DIR / "swebench_lite_oh"
BASE_URL = ("https://huggingface.co/datasets/OpenHands/openhands-evaluation-outputs/resolve/main/"
            "outputs/SWE-bench_Lite-test/CodeActAgent/{folder}/output.jsonl")

# short run name -> (folder, policy model, scaffold version)
RUNS = {
    "claude35_v2.2": ("claude-3-5-sonnet-20241022_maxiter_100_N_v2.2-no-hint", "claude-3-5-sonnet-20241022", "v2.2"),
    "claude35_v2.1": ("claude-3-5-sonnet-20241022_maxiter_100_N_v2.1-no-hint", "claude-3-5-sonnet-20241022", "v2.1"),
    "claude35_v1.9": ("claude-3-5-sonnet-20241022_maxiter_30_N_v1.9-no-hint", "claude-3-5-sonnet-20241022", "v1.9"),
    "gpt4o_v1.9": ("gpt-4o-2024-05-13_maxiter_30_N_v1.9-no-hint-eval-24sep", "gpt-4o-2024-05-13", "v1.9"),
}

# SWE-Gym's runs all start with this exact system message; the event stream doesn't record it.
SYSTEM_PROMPT = (
    "You are a helpful assistant that can interact with a computer to solve tasks.\n<IMPORTANT>\n"
    "* If user provides a path, you should NOT assume it's relative to the current working directory. "
    "Instead, you should explore the file system to find the file before working on it.\n</IMPORTANT>\n"
)
OBS = "OBSERVATION:\n"
_PR_RE = re.compile(r"<pr_description>")
_CALL_RE = re.compile(r"^\s*(?:print\()?\s*(\w+)\(")
_FIRST_STR_RE = re.compile(r"""\(\s*(?:path\s*=\s*)?['"]([^'"]+)['"]""")

# agentskills helper -> editor sub-command ("view" / "str_replace" / "create"), or "bash" for searches
_SKILLS = {
    "open_file": "view", "goto_line": "view", "scroll_down": "view", "scroll_up": "view",
    "create_file": "create",
    "edit_file_by_replace": "str_replace", "insert_content_at_line": "str_replace",
    "append_file": "str_replace", "edit_file": "str_replace",
    "search_file": "bash", "search_dir": "bash", "find_file": "bash",
}


def path(folder: str):
    return LITE_DIR / f"{folder}.jsonl"


def download(runs: Optional[list[str]] = None) -> None:
    LITE_DIR.mkdir(parents=True, exist_ok=True)
    for name in runs or RUNS:
        folder = RUNS[name][0]
        if not path(folder).exists():
            print(f"  downloading {name}")
            urllib.request.urlretrieve(BASE_URL.format(folder=folder), path(folder))


def _flatten(history: list) -> list[dict]:
    """GPT-4o v1.9 records store [action, observation] pairs; the rest store a flat event list."""
    out = []
    for e in history:
        out.extend(e if isinstance(e, list) else [e])
    return out


def _bash_content(e: dict) -> str:
    ex = e.get("extras") or {}
    code = ex.get("exit_code")
    tail = f"\n[Command finished with exit code {code}]" if code is not None else ""
    return OBS + (e.get("content") or "") + (ex.get("interpreter_details") or "") + tail


def _messages_v2(events: list[dict]) -> list[dict]:
    """Function-calling runs: copy each model response once, attach observations by tool_call_id."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    seen: set[str] = set()
    tcm_by_id = {e["id"]: e["tool_call_metadata"] for e in events if e.get("action") and e.get("tool_call_metadata")}
    for e in events:
        tcm = e.get("tool_call_metadata")
        if e.get("action"):
            if e.get("source") == "user":
                if e["action"] == "message":
                    msgs.append({"role": "user", "content": e.get("message") or ""})
                continue
            if tcm:
                resp = tcm["model_response"]
                if resp["id"] in seen:
                    continue
                seen.add(resp["id"])
                m = resp["choices"][0]["message"]
                msgs.append({"role": "assistant", "content": m.get("content") or "",
                             "tool_calls": [{"id": tc["id"], "function": {"name": tc["function"]["name"],
                                                                         "arguments": tc["function"]["arguments"]}}
                                            for tc in m.get("tool_calls") or []]})
            elif e["action"] == "message":
                msgs.append({"role": "assistant", "content": e.get("message") or ""})
        elif e.get("observation"):
            tcm = tcm or tcm_by_id.get(e.get("cause"))
            obs = e["observation"]
            if obs == "error":
                msgs.append({"role": "tool", "content": OBS + "ERROR:\n" + (e.get("content") or "")})
            elif tcm and obs in ("run", "run_ipython"):
                content = _bash_content(e) if obs == "run" else OBS + (e.get("content") or "")
                msgs.append({"role": "tool", "tool_call_id": tcm["tool_call_id"], "name": tcm["function_name"],
                             "content": content})
    return msgs


def _skill_call(code: str, current_file: Optional[str]) -> tuple[str, dict, Optional[str]]:
    """One IPython cell of the v1.9 scaffold -> (tool name, arguments, file now open)."""
    m = _CALL_RE.match(code)
    fn = m.group(1) if m else None
    kind = _SKILLS.get(fn)
    if kind is None or kind == "bash":                # plain Python or a search: a shell-like action
        return "execute_bash", {"command": code.strip()}, current_file
    p = _FIRST_STR_RE.search(code)
    target = p.group(1) if p and fn not in ("goto_line", "scroll_down", "scroll_up") else current_file
    if fn in ("open_file", "create_file"):
        current_file = target
    return "str_replace_editor", {"command": kind, "path": target, "code": code.strip()}, current_file


def _messages_v1(events: list[dict]) -> list[dict]:
    """agentskills runs: rebuild tool calls from action args, mapping editor helpers onto the editor tool."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    current_file = None
    pending: dict[int, tuple[str, str]] = {}      # action event id -> (tool_call_id, tool name)
    for e in events:
        src, act, obs = e.get("source"), e.get("action"), e.get("observation")
        if act:
            args = e.get("args") or {}
            if src == "user":
                if act == "message":
                    msgs.append({"role": "user", "content": e.get("message") or ""})
                continue
            if act == "run":
                name, call_args = "execute_bash", {"command": args.get("command", "")}
            elif act == "run_ipython":
                name, call_args, current_file = _skill_call(args.get("code", ""), current_file)
            elif act == "finish":
                name, call_args = "finish", {}
            elif act == "message":
                msgs.append({"role": "assistant", "content": e.get("message") or ""})
                continue
            else:
                continue
            call_id = f"call_{e['id']}"
            pending[e["id"]] = (call_id, name)
            msgs.append({"role": "assistant", "content": args.get("thought") or "",
                         "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(call_args)}}]})
        elif obs in ("run", "run_ipython", "error"):
            call_id, name = pending.pop(e.get("cause"), (None, None))
            content = e.get("content") or ""
            if obs == "error":
                text = OBS + "ERROR:\n" + content
            elif obs == "run":
                text = _bash_content(e)
            elif name == "execute_bash":              # IPython cell mapped to bash: derive an exit code
                text = OBS + content + f"\n[Command finished with exit code {int('Traceback (most recent call last)' in content)}]"
            else:                                     # editor helper
                failed = content.lstrip().startswith("ERROR") or "Your proposed edit has introduced" in content
                text = OBS + ("ERROR:\n" if failed else "") + content
            msgs.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": text})
    return msgs


def to_messages(record: dict) -> list[dict]:
    """Event stream -> SWE-Gym-style chat messages (v2.x copied, v1.9 mapped)."""
    events = _flatten(record.get("history") or [])
    v2 = any(e.get("tool_call_metadata") for e in events if e.get("action"))
    return _messages_v2(events) if v2 else _messages_v1(events)


def to_trace(record: dict, run: str, msgs: Optional[list[dict]] = None) -> Trace:
    msgs = msgs if msgs is not None else to_messages(record)
    trace = parse_messages(msgs, uid=f"{run}:{record['instance_id']}", instance_id=record["instance_id"], run_id=run,
                           resolved=bool((record.get("report") or {}).get("resolved")),
                           final_patch=(record.get("test_result") or {}).get("git_patch") or "")
    if not _PR_RE.search(trace.task):     # v1.9 prompt: wrap the issue the way SWE-Gym's prompt does
        issue = (record.get("instance") or {}).get("problem_statement", "")
        trace.task = f"<pr_description>\n{issue}\n</pr_description>"
    return trace


def iter_lite(runs: Optional[list[str]] = None) -> Iterator[tuple[str, Trace, list[dict]]]:
    """-> (run name, Trace, rebuilt chat messages) for every record, streamed line by line."""
    for name in runs or RUNS:
        with path(RUNS[name][0]).open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                msgs = to_messages(rec)
                yield name, to_trace(rec, name, msgs), msgs


def main() -> None:
    import pandas as pd
    download()
    rows = [{"run": r, "resolved": t.resolved, "steps": t.n_actions, "patch_empty": not t.final_patch.strip()}
            for r, t, _ in iter_lite()]
    print(pd.DataFrame(rows).groupby("run").agg(runs=("resolved", "size"), resolved=("resolved", "mean"),
                                                 median_steps=("steps", "median"),
                                                 empty_patch=("patch_empty", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()

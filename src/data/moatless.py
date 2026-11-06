"""MoatlessTools runs on SWE-bench Lite: the second agent framework for Gap 2.

Source: HF SWE-Gym/MoatlessTools-Sampled-Trajectories. Each zip holds one or more run folders with
preds.jsonl (instance_id, model_patch, report) and prompt_logs/<task>/<NN>_<Action>.json (messages,
completion, error). Only the patch and report["resolved"] are read from preds.jsonl; the evaluation logs
in the zips are never read.

MoatlessTools differs in kind from OpenHands. It is a fixed pipeline of LLM calls with no shell, so
it cannot run tests or commands, and each step's prompt is rebuilt from the task state rather than being a
growing chat. All runs use open-weight Qwen2.5-Coder models (fine-tuned by SWE-Gym; file names give size).

The scaffold normaliser (Gap 2) maps its actions onto the schema that OpenHands runs use:
  SearchCode                  -> bash (a code search)
  IdentifyCode                -> view of the first file named in the completion
  EditCode                    -> edit of the file named by the preceding plan
  PlanToCode finish / reject  -> finish
  PlanToCode other, DecideRelevance, ClarifyCodeChange -> message (reasoning with no tool)
normalize=False is the off switch: every step stays an other_tool call with its raw action name.

This mapping was fixed before any model was scored on this data.

Token counts: `tokens` is the completion, `obs_tokens` the prompt. Moatless rebuilds its prompt at each step, so
token totals aren't directly comparable to OpenHands; that shift is part of what the transfer test measures.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator, Optional

from src import config
from src.data.parser import _FILE_RE, estimate_tokens
from src.schema import ActionType, Role, Step, Trace

MOATLESS_DIR = config.DATA_DIR / "moatless"
BASE_URL = "https://huggingface.co/datasets/SWE-Gym/MoatlessTools-Sampled-Trajectories/resolve/main/"

# short name -> (file on the hub, policy model)
ZIPS = {"32b_8runs": ("32b/release-bench-lite-qwen32b-v2-it1-top2-temp_0.5-fp8_8runs.zip", "qwen2.5-coder-32b-ft"),
        "32b_zero": ("32b/release-32b-zero-bench-lite-temp_0.0-fp8_0.zip", "qwen2.5-coder-32b-ft"),
        "7b_zero": ("7b/release-bench-lite-qwen7b-zero-temp_0.0-fp8_0.zip", "qwen2.5-coder-7b-ft")}
LOCAL = {"32b_8runs": "32b_v2_it1_t05_8runs.zip", "32b_zero": "32b_zero_t0.zip", "7b_zero": "7b_zero_t0.zip"}
_LOG_RE = re.compile(r"^(.*)/prompt_logs/([^/]+)/(\d+)_(\w+)\.json$")
_ISSUE_RE = re.compile(r"<(?:issue|main_objective)>(.*?)</(?:issue|main_objective)>", re.S)
_ACTION_RE = re.compile(r'"action"\s*:\s*"(\w+)"')
_FILE_PATH_RE = re.compile(r'"file_path"\s*:\s*"([^"]+)"')
_SCRATCH_RE = re.compile(r'"scratch_pad"\s*:\s*"((?:[^"\\]|\\.)*)"', re.S)


def download(names: Optional[list[str]] = None) -> None:
    MOATLESS_DIR.mkdir(parents=True, exist_ok=True)
    for name in names or ZIPS:
        target = MOATLESS_DIR / LOCAL[name]
        if not target.exists():
            print(f"  downloading {name}")
            urllib.request.urlretrieve(BASE_URL + ZIPS[name][0], target)


def _completion_text(d: dict) -> str:
    c = d.get("completion") or []
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[0].get("content") or ""
    return str(c) if c else ""


def _prompt_chars(d: dict) -> str:
    return "\n".join(str(m.get("content") or "") for m in d.get("messages") or [])


def _thought(text: str) -> str:
    m = _SCRATCH_RE.search(text)
    return m.group(1) if m else ""


def to_steps(logs: list[tuple[str, dict]], normalize: bool = True) -> list[Step]:
    """logs: (action name, step JSON) in run order."""
    steps: list[Step] = []
    plan_file: Optional[str] = None
    for action, d in logs:
        text = _completion_text(d)
        kw = dict(index=len(steps), role=Role.assistant, tokens=estimate_tokens(text),
                  obs_tokens=estimate_tokens(_prompt_chars(d)), is_error=bool(d.get("error")),
                  args_sig=hashlib.md5(f"{action}|{text}".encode()).hexdigest()[:12],
                  thought_files=sorted({f.rsplit("/", 1)[-1] for f in _FILE_RE.findall(_thought(text))}))
        if not normalize:
            steps.append(Step(action_type=ActionType.other_tool, tool_name=action, **kw))
            continue
        if action == "SearchCode":
            steps.append(Step(action_type=ActionType.bash, tool_name="execute_bash", command="grep code search",
                              exit_code=int(kw["is_error"]), **kw))
        elif action == "IdentifyCode":
            files = _FILE_RE.findall(text)
            steps.append(Step(action_type=ActionType.view, tool_name="str_replace_editor", command="view",
                              path=files[0] if files else None, **kw))
        elif action == "EditCode":
            steps.append(Step(action_type=ActionType.edit, tool_name="str_replace_editor", command="str_replace",
                              path=plan_file, **kw))
        elif action == "PlanToCode":
            m = _FILE_PATH_RE.search(text)
            plan_file = m.group(1) if m else plan_file
            act = _ACTION_RE.search(text)
            if act and act.group(1) in ("finish", "reject"):
                steps.append(Step(action_type=ActionType.finish, tool_name="finish", **kw))
            else:
                steps.append(Step(action_type=ActionType.message, **kw))
        else:                                   # DecideRelevance, ClarifyCodeChange, anything new
            steps.append(Step(action_type=ActionType.message, **kw))
    return steps


def to_trace(run: str, instance_id: str, logs: list[tuple[str, dict]], pred: dict, normalize: bool = True) -> Trace:
    first = logs[0][1] if logs else {}
    m = _ISSUE_RE.search(_prompt_chars(first))
    issue = m.group(1).strip() if m else ""
    prompt = Step(index=0, role=Role.user, action_type=ActionType.prompt, tokens=estimate_tokens(issue))
    steps = [prompt] + [s.model_copy(update={"index": i + 1}) for i, s in enumerate(to_steps(logs, normalize))]
    return Trace(uid=f"{run}:{instance_id}", instance_id=instance_id, run_id=run,
                 resolved=bool((pred.get("report") or {}).get("resolved")),
                 task=f"<pr_description>\n{issue}\n</pr_description>", steps=steps,
                 final_patch=pred.get("model_patch") or "")


def iter_moatless(names: Optional[list[str]] = None, normalize: bool = True) -> Iterator[tuple[str, Trace]]:
    """-> (zip short name, Trace). Runs with no step logs have nothing to observe and are skipped."""
    for name in names or ZIPS:
        path: Path = MOATLESS_DIR / LOCAL[name]
        with zipfile.ZipFile(path) as z:
            logs: dict[tuple[str, str], list[tuple[int, str, str]]] = {}
            for n in z.namelist():
                m = _LOG_RE.match(n)
                if m:
                    logs.setdefault((m.group(1), m.group(2)), []).append((int(m.group(3)), m.group(4), n))
            for pf in sorted(n for n in z.namelist() if n.endswith("/preds.jsonl")):
                folder = pf.rsplit("/", 1)[0]
                run = f"{name}/{folder.rsplit('_', 1)[-1]}" if name == "32b_8runs" else name
                for line in z.read(pf).splitlines():
                    if not line.strip():
                        continue
                    pred = json.loads(line)
                    entries = sorted(logs.get((folder, pred["instance_id"]), []))
                    if not entries:
                        continue
                    steps = [(act, json.loads(z.read(n))) for _, act, n in entries]
                    yield name, to_trace(run, pred["instance_id"], steps, pred, normalize)

"""Telemetry feature extraction: Trace -> RunFeatures, cached to data/features.parquet.

    python -m src.features [--refresh]
"""
from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter

import pandas as pd

from src import config
from src.data.loader import load_meta
from src.data.parser import estimate_tokens, iter_traces
from src.schema import ActionType, Role, RunFeatures, TaskFeatures, Trace

_ISSUE_RE = re.compile(r"<pr_description>(.*?)(?:</pr_description>|$)", re.S)

# Commands that execute tests (a test runner, or a test file run directly).
_TEST_RE = re.compile(
    r"\b(pytest|py\.test|unittest|nosetests|tox|nox)\b"
    r"|manage\.py\s+test|\bmake\s+test"
    r"|\btest_\w*\.py\b|\b\w*_test\.py\b"
)
# Anything the agent runs to check its work: tests, a type checker, or a Python script.
_CHECK_RE = re.compile(_TEST_RE.pattern + r"|\bmypy\b|\bpython\d?(?:\.\d+)?\s+(?!-m\s+pip)\S")
_LOCATE_RE = re.compile(r"\b(grep|rg|find|ag|ack)\b")
_RUN_RE = re.compile(r"\bpython\d?\b")
_REPRO_PATH_RE = re.compile(r"(test|repro|reproduce|example)", re.I)
_AGENT_ACTIONS = {ActionType.bash, ActionType.view, ActionType.create, ActionType.edit,
                  ActionType.finish, ActionType.message, ActionType.other_tool}


def patch_stats(patch: str) -> tuple[int, int]:
    """-> (changed lines, files touched) of a unified git diff."""
    lines = files = 0
    for ln in patch.splitlines():
        if ln.startswith("diff --git"):
            files += 1
        elif (ln.startswith("+") and not ln.startswith("+++")) or (ln.startswith("-") and not ln.startswith("---")):
            lines += 1
    return lines, files


def phase_of(prev: str, s, n_edits: int) -> str:
    """Belief about the agent's phase after action s; n_edits includes s."""
    t = s.action_type
    if t == ActionType.finish:
        return "submit"
    if t == ActionType.create:
        return "reproduce" if n_edits <= 1 and s.path and _REPRO_PATH_RE.search(s.path) else "fix"
    if t == ActionType.edit:
        return "fix"
    if t == ActionType.bash and s.command:
        if _TEST_RE.search(s.command) or _RUN_RE.search(s.command):
            return "verify" if n_edits and prev in ("fix", "verify") else "reproduce"
        if _LOCATE_RE.search(s.command):
            return "locate"
    if t == ActionType.view:
        return "locate" if s.path and "." in s.path.rsplit("/", 1)[-1] else "explore"
    return prev or "explore"


def check_outcome(s) -> str | None:
    """-> None if s isn't a check; else "pass", "fail" or "unknown". A pipe hides the real exit code
    (`pytest | tail` reports tail's), so piped commands are unknown."""
    if s.action_type != ActionType.bash or not s.command or not _CHECK_RE.search(s.command):
        return None
    if "|" in s.command or s.exit_code is None:
        return "unknown"
    return "pass" if s.exit_code == 0 else "fail"


class ProgressSignals:
    """The features added in Phase 2, updated one agent action at a time. Live signals only: no patch,
    no total length. Shared by full-run extraction and prefix telemetry, so both mean the same thing."""

    def __init__(self):
        self.step = self.n_edits = 0
        self.first_edit = None
        self.phase = ""
        self.phases: Counter[str] = Counter()
        self.last_check = None
        self.checked_since_edit = self.seen_fail = self.fail_pass = False
        self.prev_sig, self.id_streak, self.max_id = None, 0, 0
        self.err_streak = self.max_err = 0
        self.err_sig, self.err_followed, self.err_changed = None, 0, 0
        self.file_acts = self.mismatched = 0

    def update(self, s) -> None:
        self.step += 1
        if self.err_sig is not None:                       # the action right after an error
            self.err_followed += 1
            self.err_changed += s.args_sig != self.err_sig
        self.err_sig = s.args_sig if s.is_error else None
        self.err_streak = self.err_streak + 1 if s.is_error else 0
        self.max_err = max(self.max_err, self.err_streak)
        self.id_streak = self.id_streak + 1 if s.args_sig and s.args_sig == self.prev_sig else 1
        self.prev_sig = s.args_sig
        self.max_id = max(self.max_id, self.id_streak)
        if s.action_type in (ActionType.edit, ActionType.create):
            self.n_edits += 1
            self.checked_since_edit = False
            if self.first_edit is None:
                self.first_edit = self.step
        outcome = check_outcome(s)
        if outcome:
            self.checked_since_edit = True
            self.last_check = outcome
            self.fail_pass = self.fail_pass or (outcome == "pass" and self.seen_fail)
            self.seen_fail = self.seen_fail or outcome == "fail"
        self.phase = phase_of(self.phase, s, self.n_edits)
        self.phases[self.phase] += 1
        if s.path and s.thought_files and "." in s.path.rsplit("/", 1)[-1]:
            self.file_acts += 1
            self.mismatched += s.path.rsplit("/", 1)[-1] not in s.thought_files

    def snapshot(self) -> dict:
        n = sum(self.phases.values())
        return {"last_check_passed": self.last_check == "pass", "checked_after_last_edit": self.checked_since_edit,
                "fail_then_pass": self.fail_pass, "max_identical_streak": self.max_id,
                "max_error_streak": self.max_err,
                "error_responsiveness": self.err_changed / self.err_followed if self.err_followed else 1.0,
                "first_edit_step": self.first_edit if self.first_edit is not None else self.step,
                "phase_entropy": -sum(c / n * math.log2(c / n) for c in self.phases.values()) if n else 0.0,
                "thought_action_mismatch": self.mismatched / self.file_acts if self.file_acts else 0.0}


def extract(trace: Trace) -> RunFeatures:
    actions = [s for s in trace.steps if s.role == Role.assistant and s.action_type in _AGENT_ACTIONS]
    progress = ProgressSignals()
    for s in actions:
        progress.update(s)

    seen_sigs: set[str] = set()
    repeated = 0
    viewed: set[str] = set()
    edits_blind = 0
    ran_tests = False
    for s in actions:
        if s.args_sig:
            repeated += s.args_sig in seen_sigs
            seen_sigs.add(s.args_sig)
        if s.action_type in (ActionType.view, ActionType.create) and s.path:
            viewed.add(s.path)
        elif s.action_type == ActionType.edit and s.path and s.path not in viewed:
            edits_blind += 1
        if s.action_type == ActionType.bash and s.command and _TEST_RE.search(s.command):
            ran_tests = True

    p_lines, p_files = patch_stats(trace.final_patch)
    return RunFeatures(
        uid=trace.uid,
        instance_id=trace.instance_id,
        resolved=trace.resolved,
        n_steps=len(actions),
        total_tokens=sum(s.tokens + s.obs_tokens for s in trace.steps),
        n_errors=sum(s.is_error for s in trace.steps),
        n_repeated_commands=repeated,
        n_distinct_tools=len({s.tool_name for s in actions if s.tool_name}),
        n_edits_without_view=edits_blind,
        ran_tests=ran_tests,
        patch_lines=p_lines,
        patch_files=p_files,
        patch_empty=not trace.final_patch.strip(),
        finished=any(s.action_type == ActionType.finish for s in actions),
        **progress.snapshot(),
    )


def extract_task(trace: Trace) -> TaskFeatures:
    m = _ISSUE_RE.search(trace.task)
    issue = m.group(1) if m else trace.task
    return TaskFeatures(
        uid=trace.uid,
        repo=trace.instance_id.rsplit("-", 1)[0],
        issue_tokens=estimate_tokens(issue),
        issue_code_blocks=issue.count("```") // 2,
        issue_has_traceback="Traceback (most recent call last)" in issue,
    )


def build_features(refresh: bool = False) -> pd.DataFrame:
    if config.FEATURES_PATH.exists() and not refresh:
        return pd.read_parquet(config.FEATURES_PATH)
    meta = load_meta()
    t0 = time.time()
    rows = []
    for i, tr in enumerate(iter_traces(meta), 1):
        rows.append(extract(tr).model_dump() | extract_task(tr).model_dump())
        if i % 1000 == 0:
            print(f"  features: {i}/{len(meta)} runs ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df = df.merge(meta[["uid", "run_id", "policy_model", "max_iter"]], on="uid", how="left")
    df.to_parquet(config.FEATURES_PATH, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    df = build_features(ap.parse_args().refresh)
    feats = RunFeatures.feature_names()
    print(df.groupby("resolved")[feats].mean().T.round(2).to_string())
    print(f"cached -> {config.FEATURES_PATH}")


if __name__ == "__main__":
    main()

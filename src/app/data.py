"""Read-only access to saved agent logs and results for the viewer. Nothing here fits, scores or writes.

Numbers are read from results/ and README.md, never retyped, so the viewer can't drift from the paper.
Run logs come from logs/e3_*_full/ when E3 has been run locally, otherwise from the committed logs/sample/.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src import config
from src.schema import GovernorEvent, RunOutcome

LOGS = config.ROOT / "logs"
SAMPLE = LOGS / "sample"
DATASETS = {"SWE-Gym test": "test", "SWE-bench Lite": "lite"}
STOPPED = {"ABORT", "REJECT", "HUMAN_STOP"}
BLOCKING = {"ABORT", "HUMAN_STOP"}  # logged on an action that never takes effect

OUTCOMES = {
    (True, False): "Failure stopped",
    (True, True): "Pass lost",
    (False, False): "Failure missed",
    (False, True): "Pass reviewed",
}
CONFIG_NAMES = {
    "full": "Full agent",
    "no_calibration": "No calibration",
    "no_early_warning": "No early warning",
    "no_telemetry": "No telemetry",
    "no_hints": "No hints",
    "no_normalizer": "No normalizer",
    "full_zero_shot": "Full agent, no labelled tasks",
    "empty_patch_rule": "Empty-patch rule (comparison)",
}
REROUTE_POLICIES = {
    ("full", "random_attempt"): "One random attempt",
    ("empty_patch_rule", "governor_reroute"): "Empty-patch rule + reroute",
    ("full", "governor_reroute"): "Agent + reroute",
    ("full", "review_all"): "Review every attempt",
}


def _stem(uid: str) -> str:
    return re.sub(r"[^\w.-]", "_", uid)


def log_path(dataset: str, uid: str) -> Path | None:
    """The saved log of one run: the full E3 logs if present, else the committed sample."""
    for path in (LOGS / f"e3_{dataset}_full" / f"{_stem(uid)}.jsonl", SAMPLE / f"{dataset}_{_stem(uid)}.jsonl"):
        if path.exists():
            return path
    return None


def outcome_label(decision: str, resolved: bool) -> str:
    return OUTCOMES[(decision in STOPPED, bool(resolved))]


def runs_with_logs(dataset: str) -> tuple[pd.DataFrame, int]:
    """-> (runs whose log is available, total runs in E3), from the saved E3 outcomes."""
    df = pd.read_csv(config.RESULTS_DIR / f"governor_outcomes_{dataset}.csv")
    total = len(df)
    df["log"] = [log_path(dataset, u) for u in df["uid"]]
    df = df[df["log"].notna()].sort_values("steps_total", ascending=False, kind="stable").reset_index(drop=True)
    df["outcome"] = [outcome_label(d, r) for d, r in zip(df["decision"], df["resolved"])]
    return df, total


def load_run(path: Path) -> tuple[list[GovernorEvent], RunOutcome]:
    """One JSONL log: one event per line, then the run's outcome."""
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [GovernorEvent.model_validate_json(l) for l in lines[:-1]], RunOutcome.model_validate_json(lines[-1])


def target_text(observed: dict) -> str:
    """What an action touched: its command or path, without the long workspace prefix."""
    target = observed.get("command") if observed.get("action") == "bash" else observed.get("path")
    return re.sub(r"/workspace/[^/\s]+/?", "", target or "") or ("(repository root)" if target else "")


def steps_frame(events: list[GovernorEvent]) -> pd.DataFrame:
    """One row per agent action. The end-of-run scoring event has no action and isn't included.
    An abort is logged on the next action, which it blocks, so that row is numbered as that next action."""
    rows = []
    for i, e in enumerate(events):
        if not e.observed:
            continue
        blocked = e.decision.value in BLOCKING
        rows.append({"event": i, "Action no.": e.step + blocked, "Phase": e.phase_to, "Action": e.observed.get("action") or "",
                     "Target": target_text(e.observed), "Error": bool(e.observed.get("is_error")),
                     "Trust": e.p_success, "Decision": e.decision.value})
    return pd.DataFrame(rows)


def final_event(events: list[GovernorEvent]) -> GovernorEvent | None:
    return next((e for e in reversed(events) if e.decision.value != "CONTINUE"), None) if events else None


def top_factors(events: list[GovernorEvent]) -> list[dict]:
    """The factors behind the last score that reported any (exact linear SHAP, logged by the agent)."""
    for e in reversed(events):
        for call in reversed(e.tool_calls):
            if call.output and call.output.get("top_factors"):
                return call.output["top_factors"]
    return []


def md_section(path: Path, heading: str) -> str:
    """The text under a Markdown heading (exact heading line), up to the next heading of the same or higher level."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(heading) + 1
    level = len(heading) - len(heading.lstrip("#"))
    end = next((i for i in range(start, len(lines)) if re.match(rf"#{{1,{level}}} ", lines[i])), len(lines))
    return "\n".join(lines[start:end]).strip()


def headings(path: Path, prefix: str = "### ") -> list[str]:
    return [l for l in path.read_text(encoding="utf-8").splitlines() if l.startswith(prefix)]


def split_table(section: str) -> tuple[str, str]:
    """A section -> (its Markdown table, the prose around it)."""
    lines = section.splitlines()
    return "\n".join(l for l in lines if l.startswith("|")), "\n\n".join(l for l in lines if l and not l.startswith("|"))


def readable_table(table: str, subset: str | None = None) -> str:
    """Config names in plain words; with `subset`, keep only those rows and drop the subset column."""
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in table.splitlines()]
    if subset is not None:
        rows = [r[:1] + r[2:] for i, r in enumerate(rows) if i < 2 or r[1] == subset]
    rows[0] = ["Configuration" if c == "config" else c.capitalize() for c in rows[0]]
    for r in rows[2:]:
        r[0] = CONFIG_NAMES.get(r[0], r[0])
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def reroute_table(table: str) -> str:
    """The four policies the paper compares, all tasks: one row each, from the saved reroute table."""
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in table.splitlines()]
    col = {name: i for i, name in enumerate(rows[0])}
    out = ["| Policy | Tasks solved | Tokens per task | Reviews per task |", "|---|---|---|---|"]
    for key, name in REROUTE_POLICIES.items():
        r = next(r for r in rows[2:] if (r[col["config"]], r[col["policy"]]) == key and r[col["subset"]] == "all_tasks")
        out.append(f"| {name} | {r[col['solved']]} | {r[col['tokens per task']]} | {r[col['reviews per task']]} |")
    return "\n".join(out)


def failure_notes(block: dict) -> str:
    """The lost passes and hint false alarms from failure_analysis.json, in plain words."""
    factors = ", ".join(f"`{k}` {v}" for k, v in block["rejected_pass_lowest_factor"].items())
    aborts = ", ".join(f"action {float(k):.0f} ({v})" for k, v in block["abort_steps_of_lost_passes"].items()) or "none"
    hints = block["hint_false_alarms"]
    return (f"Rejected passes, by the factor that lowered the score most: {factors}. "
            f"Aborted passes, by checkpoint: {aborts}. "
            f"Hints fired on {hints['passing_runs_hinted']} passing runs, of {hints['hinted_runs']} hinted runs.")

"""Prefix telemetry for early warning (Gap 3): features of a run after its first k actions.

Two kinds of checkpoint:
  abs   after k = 5, 10, 15, 20, 30, 40 agent actions, for runs still going (they take an action
        k+1). This is what a live monitor sees, and what the governance agent uses.
  frac  after 25 / 50 / 75 % of the run's actions (the Phase 1 proposal's framing). Knowing where
        50 % falls needs the run's total length, which a live monitor doesn't have, so these rows
        leak run length; kept only for comparison with the proposal.

    python -m src.prefix [--refresh] [--lite]
-> data/prefix_features.parquet (SWE-Gym), data/lite_prefix_features.parquet (--lite)
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Iterable

import pandas as pd

from src import config
from src.features import _AGENT_ACTIONS, _TEST_RE
from src.schema import ActionType, PrefixFeatures, Role, Trace

ABS_STEPS = (5, 10, 15, 20, 30, 40)
FRACS = (0.25, 0.5, 0.75)
RECENT = 5
PREFIX_PATH = config.DATA_DIR / "prefix_features.parquet"
LITE_PREFIX_PATH = config.DATA_DIR / "lite_prefix_features.parquet"


class PrefixState:
    """Running telemetry of one run, updated step by step. snapshot() gives the features so far.
    The governance agent feeds steps into this one at a time."""

    def __init__(self, uid: str, instance_id: str, resolved=None):
        self.uid, self.instance_id, self.resolved = uid, instance_id, resolved
        self.step = self.tokens = self.errors = self.repeated = 0
        self.edits = self.blind_edits = 0
        self.last_edit_step = 0
        self.ran_tests = False
        self.tools: set[str] = set()
        self.sigs: set[str] = set()
        self.viewed: set[str] = set()
        self.edited: set[str] = set()
        self.err_by_action: list[int] = []        # errors seen in each action's observations

    def is_action(self, s) -> bool:
        return s.role == Role.assistant and s.action_type in _AGENT_ACTIONS

    def update(self, s) -> None:
        self.tokens += s.tokens + s.obs_tokens
        if s.is_error:
            self.errors += 1
            if self.err_by_action:
                self.err_by_action[-1] += 1
        if not self.is_action(s):
            return
        self.step += 1
        self.err_by_action.append(0)
        if s.tool_name:
            self.tools.add(s.tool_name)
        if s.args_sig:
            self.repeated += s.args_sig in self.sigs
            self.sigs.add(s.args_sig)
        if s.action_type in (ActionType.view, ActionType.create) and s.path:
            self.viewed.add(s.path)
        if s.action_type in (ActionType.edit, ActionType.create):
            self.edits += 1
            self.last_edit_step = self.step
            if s.path:
                self.edited.add(s.path)
                if s.action_type == ActionType.edit and s.path not in self.viewed:
                    self.blind_edits += 1
        if s.action_type == ActionType.bash and s.command and _TEST_RE.search(s.command):
            self.ran_tests = True

    def snapshot(self, checkpoint: str = "abs", prefix_frac=None) -> PrefixFeatures:
        return PrefixFeatures(
            uid=self.uid, instance_id=self.instance_id, resolved=self.resolved,
            checkpoint=checkpoint, prefix_frac=prefix_frac, step=self.step,
            tokens_so_far=self.tokens, n_errors=self.errors,
            recent_errors=sum(self.err_by_action[-RECENT:]),
            n_repeated_commands=self.repeated, n_distinct_tools=len(self.tools),
            n_edits=self.edits, n_files_edited=len(self.edited), n_edits_without_view=self.blind_edits,
            steps_since_edit=self.step - self.last_edit_step, ran_tests=self.ran_tests,
        )


def prefixes(trace: Trace) -> list[PrefixFeatures]:
    """All abs and frac snapshots of one run. A snapshot after k actions is taken when action k+1
    starts, so it includes the k-th action's observation and only covers runs still going."""
    st = PrefixState(trace.uid, trace.instance_id, trace.resolved)
    n_actions = sum(st.is_action(s) for s in trace.steps)
    want: dict[int, list[tuple[str, float | None]]] = {}
    for k in ABS_STEPS:
        want.setdefault(k, []).append(("abs", None))
    for f in FRACS:
        k = math.ceil(f * n_actions)
        if 0 < k < n_actions:
            want.setdefault(k, []).append(("frac", f))
    out = []
    for s in trace.steps:
        if st.is_action(s) and st.step in want:
            out += [st.snapshot(c, f) for c, f in want.pop(st.step)]
        st.update(s)
    return out


def build(traces: Iterable[Trace]) -> pd.DataFrame:
    t0, rows = time.time(), []
    for i, tr in enumerate(traces, 1):
        rows += [p.model_dump() for p in prefixes(tr)]
        if i % 1000 == 0:
            print(f"  prefixes: {i} runs ({time.time() - t0:.0f}s)")
    return pd.DataFrame(rows)


def build_swegym(refresh: bool = False) -> pd.DataFrame:
    if PREFIX_PATH.exists() and not refresh:
        return pd.read_parquet(PREFIX_PATH)
    from src.data.loader import load_meta
    from src.data.parser import iter_traces
    df = build(iter_traces(load_meta()))
    df.to_parquet(PREFIX_PATH, index=False)
    return df


def build_lite(refresh: bool = False) -> pd.DataFrame:
    if LITE_PREFIX_PATH.exists() and not refresh:
        return pd.read_parquet(LITE_PREFIX_PATH)
    from src.data.swebench_lite import iter_lite
    df = build(tr for _, tr, _ in iter_lite())
    df.to_parquet(LITE_PREFIX_PATH, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--lite", action="store_true", help="also build SWE-bench Lite prefixes")
    args = ap.parse_args()
    df = build_swegym(args.refresh)
    abs_ = df[df["checkpoint"] == "abs"]
    print(abs_.groupby("step").agg(runs=("uid", "size"), resolved=("resolved", "mean")).round(3).to_string())
    if args.lite:
        lt = build_lite(args.refresh)
        print(lt[lt["checkpoint"] == "abs"].groupby("step").agg(runs=("uid", "size"), resolved=("resolved", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()

"""Abort and reroute (Phase 1 report, B.5): when a run is stopped, spend the effort on another attempt.

The agent works through a task's attempts in a random order. An ABORT or REJECT sends it to the next
attempt. The first FLAG_FOR_REVIEW goes to a human and ends the task, so a task costs at most one review.
If every attempt is stopped, nothing is submitted and the task stays unsolved.

Replay limit: "another attempt" is another logged run of the same task, so a task with one logged run can't
be rerouted. Attempts are shuffled per seed, which is where the seed-to-seed variance comes from.

Policies compared, all on the same attempt order:
  random_attempt    no agent: take the first attempt, finish it, have a human review it
  review_all        no agent: finish every attempt, review each until one passes (an upper bound on tasks
                    solved, since a reviewer is assumed always right)
  governor_single   the agent on the first attempt only
  governor_reroute  the agent, moving on after each ABORT or REJECT
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.schema import Decision

POLICIES = ("random_attempt", "review_all", "governor_single", "governor_reroute")
STOPPED = {Decision.abort.value, Decision.reject.value}


def simulate(runs: pd.DataFrame, seed: int) -> pd.DataFrame:
    """runs: uid, instance_id, decision, tokens_used, tokens_total, resolved. -> one row per task and policy."""
    rng = np.random.default_rng(seed)
    rows = []
    for task, g in runs.groupby("instance_id", sort=True):
        g = g.iloc[rng.permutation(len(g))]
        first = g.iloc[0]
        rows.append({"policy": "random_attempt", "solved": bool(first["resolved"]),
                     "tokens": float(first["tokens_total"]), "reviews": 1})
        passed = g["resolved"].to_numpy(bool)
        n_tried = int(passed.argmax()) + 1 if passed.any() else len(g)
        rows.append({"policy": "review_all", "solved": bool(passed.any()), "tokens": float(g["tokens_total"].sum()),
                     "reviews": n_tried})
        flagged = first["decision"] not in STOPPED
        rows.append({"policy": "governor_single", "solved": flagged and bool(first["resolved"]),
                     "tokens": float(first["tokens_used"]), "reviews": int(flagged)})
        tokens, solved, reviews = 0.0, False, 0
        for _, r in g.iterrows():
            tokens += float(r["tokens_used"])
            if r["decision"] not in STOPPED:
                solved, reviews = bool(r["resolved"]), 1
                break
        rows.append({"policy": "governor_reroute", "solved": solved, "tokens": tokens, "reviews": reviews})
        for row in rows[-4:]:
            row |= {"instance_id": task, "n_attempts": len(g)}
    return pd.DataFrame(rows)


def summarize(per_task: pd.DataFrame) -> pd.DataFrame:
    """One row per policy: tasks solved, cost per task, and what a solved task costs."""
    out = per_task.groupby("policy").agg(tasks=("solved", "size"), solved=("solved", "mean"),
                                         tokens_per_task=("tokens", "mean"), reviews_per_task=("reviews", "mean"))
    out["solved_per_million_tokens"] = out["solved"] / out["tokens_per_task"] * 1e6
    out["solved_per_review"] = out["solved"] / out["reviews_per_task"]
    return out.reindex(POLICIES).reset_index()


def over_seeds(runs: pd.DataFrame, seeds: tuple[int, ...]) -> pd.DataFrame:
    """Mean and std across seeds, on all tasks and on tasks with at least 2 attempts."""
    out = []
    for name, sub in (("all_tasks", runs), ("multi_attempt_tasks", runs[runs.groupby("instance_id")["uid"].transform("size") >= 2])):
        per_seed = pd.concat([summarize(simulate(sub, s)).assign(seed=s) for s in seeds])
        cols = ["solved", "tokens_per_task", "reviews_per_task", "solved_per_million_tokens", "solved_per_review"]
        agg = per_seed.groupby("policy", sort=False)[cols].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        out.append(agg.reset_index().assign(subset=name, tasks=sub["instance_id"].nunique()))
    return pd.concat(out, ignore_index=True)

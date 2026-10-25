"""Fixed dev/test split, GroupKFold folds on dev, and the fixed LLM-judge sample.

    python -m src.eval.splits             # print a summary of data/splits.csv
    python -m src.eval.splits --rebuild   # regenerate it (changes every reported number)

Output: data/splits.csv (committed) with columns uid, instance_id, resolved, split, fold, judge_sample.

- split: TEST_FRAC of tasks are held out as "test" and only used for final numbers. The rest is
  "dev", used for model selection, calibration fitting and threshold tuning. Mixed tasks (both
  resolved and unresolved runs) are split separately so test gets its share of them, since only
  mixed tasks count towards within-task AUROC.
- fold: GroupKFold(N_FOLDS) by instance_id over dev runs; -1 for test runs.
- judge_sample: LLM_JUDGE_SAMPLE/2 mixed test tasks, one resolved and one unresolved run each. If
  results/llm_judge_sample.json exists its runs are kept and their tasks pinned to test, so cached
  judge verdicts stay valid across rebuilds.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config
from src.data.loader import load_meta

JUDGE_SAMPLE_JSON = config.RESULTS_DIR / "llm_judge_sample.json"


def mixed_tasks(meta: pd.DataFrame) -> set[str]:
    per_task = meta.groupby("instance_id")["resolved"].agg(["size", "sum"])
    return set(per_task[(per_task["sum"] > 0) & (per_task["sum"] < per_task["size"])].index)


def pick_test_tasks(meta: pd.DataFrame, pinned: set[str] = frozenset(), seed: int = config.SEED) -> set[str]:
    """TEST_FRAC of tasks, drawn separately from mixed and non-mixed tasks; pinned tasks go in first."""
    mixed = mixed_tasks(meta)
    rng = np.random.default_rng(seed)
    out: set[str] = set()
    for stratum in (sorted(mixed), sorted(set(meta["instance_id"]) - mixed)):
        n_test = round(config.TEST_FRAC * len(stratum))
        pins = [t for t in stratum if t in pinned]
        rest = [t for t in rng.permutation(stratum) if t not in pinned]
        out |= set(pins + rest[:max(0, n_test - len(pins))])
    return out


def assign_folds(meta: pd.DataFrame, seed: int = config.SEED) -> np.ndarray:
    # Shuffle groups with a fixed seed so folds don't follow shard order.
    tasks = meta["instance_id"].unique()
    order = {t: i for i, t in enumerate(np.random.default_rng(seed).permutation(tasks))}
    groups = meta["instance_id"].map(order).to_numpy()
    folds = np.full(len(meta), -1)
    for k, (_, idx) in enumerate(GroupKFold(n_splits=config.N_FOLDS).split(meta, groups=groups)):
        folds[idx] = k
    return folds


def judge_sample(meta: pd.DataFrame, tasks: set[str], seed: int = config.SEED) -> list[str]:
    """One resolved and one unresolved run from each of LLM_JUDGE_SAMPLE/2 mixed tasks in `tasks`."""
    rng = np.random.default_rng(seed)
    pool = sorted(mixed_tasks(meta) & tasks)
    uids = []
    for t in sorted(rng.choice(pool, size=config.LLM_JUDGE_SAMPLE // 2, replace=False)):
        runs = meta[meta["instance_id"] == t]
        for label in (True, False):
            cands = sorted(runs.loc[runs["resolved"] == label, "uid"])
            uids.append(cands[rng.integers(len(cands))])
    return uids


def make_splits(meta: pd.DataFrame, judge_uids: list[str] | None = None) -> pd.DataFrame:
    """Pure split assignment. judge_uids: an existing judge sample to keep (its tasks go to test)."""
    pinned = set(meta.loc[meta["uid"].isin(judge_uids or []), "instance_id"])
    test = pick_test_tasks(meta, pinned)
    df = meta[["uid", "instance_id", "resolved"]].copy()
    df["split"] = np.where(df["instance_id"].isin(test), "test", "dev")
    df["fold"] = -1
    dev = (df["split"] == "dev").to_numpy()
    df.loc[dev, "fold"] = assign_folds(df[dev])
    sample = judge_uids if judge_uids else judge_sample(meta, test)
    df["judge_sample"] = df["uid"].isin(sample)
    return df


def build_splits(rebuild: bool = False) -> pd.DataFrame:
    if config.SPLITS_PATH.exists() and not rebuild:
        return pd.read_csv(config.SPLITS_PATH)
    meta = load_meta()
    judge_uids = json.loads(JUDGE_SAMPLE_JSON.read_text())["uids"] if JUDGE_SAMPLE_JSON.exists() else None
    df = make_splits(meta, judge_uids)
    df.to_csv(config.SPLITS_PATH, index=False)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    JUDGE_SAMPLE_JSON.write_text(json.dumps(
        {"seed": config.SEED, "uids": judge_uids or df.loc[df["judge_sample"], "uid"].tolist()}, indent=1))
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="regenerate data/splits.csv")
    df = build_splits(ap.parse_args().rebuild)
    mixed = mixed_tasks(df)
    df["mixed"] = df["instance_id"].isin(mixed)
    print(df.groupby(["split", "fold"]).agg(
        runs=("uid", "size"), tasks=("instance_id", "nunique"),
        mixed_tasks=("instance_id", lambda s: len(set(s) & mixed)),
        resolved_rate=("resolved", "mean")).round(3).to_string())
    js = df[df["judge_sample"]]
    print(f"judge sample: {len(js)} runs in {set(js['split'])}, resolved rate {js['resolved'].mean():.2f}")


if __name__ == "__main__":
    main()

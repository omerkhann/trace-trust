"""Discrimination and calibration metrics over a set of Predictions.

Within-task AUROC: over tasks that have both resolved and unresolved runs, the fraction of
same-task (pass, fail) pairs where the pass run gets the higher score (ties count 1/2). This
equals the per-task AUROC weighted by n_pass * n_fail.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import config
from src.schema import EvalReport

N_BINS = 15   # equal-mass bins, as in the proposal's evaluation plan


def pooled_auroc(y: np.ndarray, p: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def within_task_auroc(df: pd.DataFrame) -> tuple[Optional[float], int, int]:
    """df has columns instance_id, label, p_success. -> (auroc, n_mixed_tasks, n_pairs)"""
    wins = 0.0
    pairs = tasks = 0
    for _, g in df.groupby("instance_id"):
        pos = g.loc[g["label"], "p_success"].to_numpy()
        neg = g.loc[~g["label"], "p_success"].to_numpy()
        if len(pos) == 0 or len(neg) == 0:
            continue
        diff = pos[:, None] - neg[None, :]
        wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        pairs += diff.size
        tasks += 1
    return (wins / pairs if pairs else None), tasks, pairs


def _equal_mass_bins(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index per prediction, with bins holding (near) equal numbers of runs."""
    ranks = np.argsort(np.argsort(p, kind="stable"), kind="stable")
    return (ranks * n_bins // len(p)).astype(int)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error over equal-mass bins."""
    bins = _equal_mass_bins(p, n_bins)
    err = 0.0
    for b in np.unique(bins):
        m = bins == b
        err += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(err)


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    bins = _equal_mass_bins(p, n_bins)
    return (pd.DataFrame({"bin": bins, "y": y, "p": p}).groupby("bin")
            .agg(mean_p=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index())


def bootstrap_ci(preds: pd.DataFrame, n_boot: int = config.N_BOOT, seed: int = config.SEED,
                 level: float = 0.95) -> dict[str, Optional[tuple[float, float]]]:
    """Percentile intervals from a task-level bootstrap: draw tasks with replacement, keep all runs of
    each drawn task. A task drawn c times gets weight c, which equals duplicating its runs.
    -> {"pooled": ci, "within": ci, "gap": ci (pooled minus within)}, None where undefined."""
    task_idx, tasks = pd.factorize(preds["instance_id"])
    y = preds["label"].to_numpy(bool)
    p = preds["p_success"].to_numpy(float)
    wins = np.zeros(len(tasks))
    pairs = np.zeros(len(tasks))
    for t, g in preds.assign(_t=task_idx).groupby("_t"):
        pos = g.loc[g["label"], "p_success"].to_numpy()
        neg = g.loc[~g["label"], "p_success"].to_numpy()
        diff = pos[:, None] - neg[None, :]
        wins[t] = (diff > 0).sum() + 0.5 * (diff == 0).sum()
        pairs[t] = diff.size

    rng = np.random.default_rng(seed)
    pooled, within = [], []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
        w = counts[task_idx]
        pooled.append(roc_auc_score(y, p, sample_weight=w) if 0 < w[y].sum() < w.sum() else np.nan)
        n_pairs = counts @ pairs
        within.append(counts @ wins / n_pairs if n_pairs else np.nan)
    pooled, within = np.array(pooled), np.array(within)

    def ci(x: np.ndarray) -> Optional[tuple[float, float]]:
        x = x[~np.isnan(x)]
        if len(x) < n_boot / 2:
            return None
        lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
        return float(lo), float(hi)

    return {"pooled": ci(pooled), "within": ci(within), "gap": ci(pooled - within)}


def evaluate_seeds(seed_preds: list[pd.DataFrame], method: str, eval_set: str = "dev",
                   n_boot: int = config.N_BOOT) -> EvalReport:
    """One report over several seeds: mean of each metric, std across seeds, and bootstrap
    intervals from the first seed's predictions."""
    rs = [evaluate(p, method, eval_set) for p in seed_preds]

    def agg(field: str) -> tuple[Optional[float], Optional[float]]:
        v = [getattr(r, field) for r in rs]
        if any(x is None for x in v):
            return None, None
        return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None)

    out = rs[0].model_copy(update={"n_seeds": len(rs)})
    for f in ("pooled_auroc", "within_task_auroc", "brier", "ece"):
        mean, std = agg(f)
        setattr(out, f, mean)
        setattr(out, f"{f}_std", std)
    if n_boot:
        ci = bootstrap_ci(seed_preds[0], n_boot)
        out.pooled_auroc_ci, out.within_task_auroc_ci, out.auroc_gap_ci = ci["pooled"], ci["within"], ci["gap"]
    return out


def evaluate(preds: pd.DataFrame, method: str, eval_set: str = "dev") -> EvalReport:
    """preds: rows of schema.Prediction for one method."""
    y = preds["label"].to_numpy(bool)
    p = preds["p_success"].to_numpy(float)
    wt, n_tasks, n_pairs = within_task_auroc(preds)
    return EvalReport(
        method=method, eval_set=eval_set, n_runs=len(preds),
        pooled_auroc=pooled_auroc(y, p), within_task_auroc=wt,
        n_mixed_tasks=n_tasks, n_pairs=n_pairs,
        brier=float(brier_score_loss(y, p)), ece=ece(y, p), base_rate=float(y.mean()),
    )

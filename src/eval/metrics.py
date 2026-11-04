"""Discrimination and calibration metrics over a set of Predictions.

Within-task AUROC: over tasks that have both resolved and unresolved runs, the fraction of
same-task (pass, fail) pairs where the pass run gets the higher score (ties count 1/2). This
equals the per-task AUROC weighted by n_pass * n_fail.
"""
from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import brier_score_loss, roc_auc_score

from src import config
from src.schema import EvalReport

N_BINS = 15   # equal-mass bins, as in the proposal's evaluation plan
BEST_K = (2, 4)   # SWE-Gym runs per task: median 2, max 10; SWE-bench Lite has 4 per task


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


def within_task_wilcoxon(df: pd.DataFrame) -> tuple[Optional[float], int]:
    """One-sided Wilcoxon signed-rank test that per-task AUROC (over mixed tasks) exceeds 0.5,
    i.e. that the scores rank same-task runs better than random. Tasks at exactly 0.5 are dropped
    (zero_method="wilcox"). -> (p_value, n_mixed_tasks); p is None with fewer than 2 informative tasks."""
    aucs = []
    for _, g in df.groupby("instance_id"):
        pos = g.loc[g["label"], "p_success"].to_numpy()
        neg = g.loc[~g["label"], "p_success"].to_numpy()
        if len(pos) and len(neg):
            diff = pos[:, None] - neg[None, :]
            aucs.append(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)
    d = np.array(aucs) - 0.5
    if (d != 0).sum() < 2:
        return None, len(aucs)
    return float(wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue), len(aucs)


def best_at_k(df: pd.DataFrame, k: int) -> dict:
    """Best@k as in SWE-Gym: draw k runs of a task, submit the highest-scored one; success if it
    resolved. Computed exactly as the expectation over all k-subsets (ties broken at random), over
    tasks with at least k runs. Also returns Pass@k (oracle: any of the k resolved) and Random@k
    (= the resolve rate, submitting a random one of the k).
    -> {"k", "n_tasks", "best", "pass", "random"} (values None if no task has k runs)"""
    best, pas, rnd = [], [], []
    for _, g in df.groupby("instance_id"):
        n = len(g)
        if n < k:
            continue
        g = g.sort_values("p_success", ascending=False)
        p, y = g["p_success"].to_numpy(float), g["label"].to_numpy(float)
        total, e, i = comb(n, k), 0.0, 0
        while i < n:                                   # one tie group = ranks i..j
            j = i
            while j + 1 < n and p[j + 1] == p[i]:
                j += 1
            # P(the subset's top-scored run falls in this group) = P(none of ranks < i drawn,
            # at least one of i..j drawn); within the group each member is equally likely.
            e += (comb(n - i, k) - comb(n - j - 1, k)) / total * y[i:j + 1].mean()
            i = j + 1
        best.append(e)
        pas.append(1 - comb(n - int(y.sum()), k) / total)
        rnd.append(y.mean())
    if not best:
        return {"k": k, "n_tasks": 0, "best": None, "pass": None, "random": None}
    return {"k": k, "n_tasks": len(best), "best": float(np.mean(best)), "pass": float(np.mean(pas)),
            "random": float(np.mean(rnd))}


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


def _task_counts(preds: pd.DataFrame, task_idx: np.ndarray, n_tasks: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-task (wins, pairs) of same-task (pass, fail) comparisons."""
    wins, pairs = np.zeros(n_tasks), np.zeros(n_tasks)
    for t, g in preds.assign(_t=task_idx).groupby("_t"):
        pos = g.loc[g["label"], "p_success"].to_numpy()
        neg = g.loc[~g["label"], "p_success"].to_numpy()
        diff = pos[:, None] - neg[None, :]
        wins[t] = (diff > 0).sum() + 0.5 * (diff == 0).sum()
        pairs[t] = diff.size
    return wins, pairs


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = config.N_BOOT, seed: int = config.SEED,
                     level: float = 0.95) -> dict:
    """Difference a - b in pooled and within-task AUROC on the same runs, with a task-level bootstrap
    that resamples the same tasks for both methods.
    -> {"pooled": (diff, lo, hi, share a > b), "within": (...)}"""
    b = b.set_index("uid").loc[a["uid"]].reset_index()
    task_idx, tasks = pd.factorize(a["instance_id"])
    y = a["label"].to_numpy(bool)
    pa, pb = a["p_success"].to_numpy(float), b["p_success"].to_numpy(float)
    wa, pairs = _task_counts(a, task_idx, len(tasks))
    wb, _ = _task_counts(b.assign(label=y), task_idx, len(tasks))

    rng = np.random.default_rng(seed)
    dp, dw = [], []
    for _ in range(n_boot):
        counts = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
        w = counts[task_idx]
        if 0 < w[y].sum() < w.sum():
            dp.append(roc_auc_score(y, pa, sample_weight=w) - roc_auc_score(y, pb, sample_weight=w))
        n_pairs = counts @ pairs
        if n_pairs:
            dw.append((counts @ wa - counts @ wb) / n_pairs)

    def summ(point, x):
        if point is None or len(x) < n_boot / 2:
            return None
        lo, hi = np.quantile(x, [(1 - level) / 2, (1 + level) / 2])
        return float(point), float(lo), float(hi), float(np.mean(np.array(x) > 0))

    pooled = (pooled_auroc(y, pa) - pooled_auroc(y, pb)) if 0 < y.sum() < len(y) else None
    within = (wa.sum() - wb.sum()) / pairs.sum() if pairs.sum() else None
    return {"pooled": summ(pooled, dp), "within": summ(within, dw)}


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
        within_task_wilcoxon_p=within_task_wilcoxon(preds)[0],
        best_at_k=[best_at_k(preds, k) for k in BEST_K],
    )

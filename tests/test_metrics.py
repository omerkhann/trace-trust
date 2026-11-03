import numpy as np
import pandas as pd
import pytest

from src.eval.metrics import bootstrap_ci, ece, evaluate, evaluate_seeds, within_task_auroc


def _preds(rows):
    return pd.DataFrame(rows, columns=["instance_id", "label", "p_success"])


def test_within_task_auroc_counts_only_same_task_pairs():
    df = _preds([
        ("a", True, 0.9), ("a", False, 0.1), ("a", False, 0.95),   # 2 pairs, 1 win
        ("b", True, 0.5), ("b", False, 0.5),                       # 1 pair, tie
        ("c", False, 0.99),                                        # not mixed
    ])
    auc, tasks, pairs = within_task_auroc(df)
    assert (tasks, pairs) == (2, 3)
    assert auc == pytest.approx(1.5 / 3)


def test_within_task_auroc_none_without_mixed_tasks():
    assert within_task_auroc(_preds([("a", True, 0.3), ("b", False, 0.2)])) == (None, 0, 0)


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    p = np.array([0.25] * 4 + [0.75] * 4)
    assert ece(y, p, n_bins=2) == pytest.approx(0.0)


def test_ece_detects_overconfidence():
    y = np.zeros(10)
    p = np.full(10, 0.5)
    assert ece(y, p) == pytest.approx(0.5)


def test_evaluate_report_fields():
    df = _preds([("a", True, 0.8), ("a", False, 0.2), ("b", False, 0.1), ("b", True, 0.7)])
    r = evaluate(df, "m")
    assert r.pooled_auroc == 1.0 and r.within_task_auroc == 1.0
    assert r.base_rate == 0.5 and r.n_pairs == 2


def _noisy_preds(n_tasks=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        rows += [(f"t{t}", True, rng.uniform(0.5, 1)), (f"t{t}", False, rng.uniform(0, 0.6)), (f"t{t}", False, rng.uniform(0, 0.6))]
    return _preds(rows)


def test_bootstrap_ci_perfect_ranker_is_degenerate():
    df = _preds([(f"t{i}", lab, p) for i in range(10) for lab, p in [(True, 0.9), (False, 0.1)]])
    ci = bootstrap_ci(df, n_boot=200)
    assert ci["pooled"] == (1.0, 1.0) and ci["within"] == (1.0, 1.0) and ci["gap"] == (0.0, 0.0)


def test_bootstrap_ci_brackets_point_estimate_and_is_deterministic():
    df = _noisy_preds()
    r = evaluate(df, "m")
    ci = bootstrap_ci(df, n_boot=300, seed=1)
    assert ci["pooled"][0] <= r.pooled_auroc <= ci["pooled"][1]
    assert ci["within"][0] <= r.within_task_auroc <= ci["within"][1]
    assert ci == bootstrap_ci(df, n_boot=300, seed=1)


def test_bootstrap_ci_none_without_mixed_tasks():
    df = _preds([("a", True, 0.3), ("b", False, 0.2)])
    assert bootstrap_ci(df, n_boot=50)["within"] is None


def test_evaluate_seeds_mean_and_std():
    a = _preds([("a", True, 0.8), ("a", False, 0.2), ("b", True, 0.7), ("b", False, 0.1)])   # within 1.0
    b = a.assign(p_success=[0.2, 0.8, 0.7, 0.1])                                              # within 0.5
    r = evaluate_seeds([a, b], "m", n_boot=0)
    assert r.n_seeds == 2 and r.within_task_auroc == pytest.approx(0.75)
    assert r.within_task_auroc_std == pytest.approx(np.std([1.0, 0.5], ddof=1))
    single = evaluate_seeds([a], "m", n_boot=0)
    assert single.within_task_auroc_std is None and single.pooled_auroc_ci is None

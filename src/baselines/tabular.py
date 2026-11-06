"""Tabular baselines. Each returns schema.Prediction rows: out-of-fold on dev, or held-out on test.

  length_only     LogReg on log(n_steps), log(total_tokens)        (B.10 baseline 2)
  empty_patch     LogReg on patch_empty alone                       (one-signal check: no passing run
                                                                     has an empty final patch)
  prompt_only     LogReg on issue features + repo one-hot           (B.10 baseline 3, task difficulty)
  telemetry_lr    LogReg on all RunFeatures (log1p, standardised)   (proposed, Phase 1 subset)
  telemetry_gbm   HistGradientBoosting on all RunFeatures           (proposed, Phase 1 subset)
  telemetry20_lr / telemetry20_gbm   the same on the 20 extended features (Phase 2)

All use the split and fold columns from data/splits.csv, so every method sees the same splits.
Model factories take a seed; results are reported as mean ± std over config.SEEDS.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config
from src.schema import RunFeatures, TaskFeatures

_log1p = FunctionTransformer(lambda x: np.log1p(np.asarray(x, dtype=float)), feature_names_out="one-to-one")


def _logreg(cols: list[str], categorical: list[str] = ()) -> Callable[[int], Pipeline]:
    def make(seed: int):
        num = make_pipeline(_log1p, StandardScaler())
        ct = ColumnTransformer([("num", num, cols)] +
                               ([("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical))] if categorical else []))
        return make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000, random_state=seed))
    return make


def _gbm(cols: list[str]) -> Callable[[int], Pipeline]:
    def make(seed: int):
        ct = ColumnTransformer([("num", "passthrough", cols)])
        return make_pipeline(ct, HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed))
    return make


TELEMETRY = RunFeatures.feature_names()
TELEMETRY20 = RunFeatures.extended_feature_names()
METHODS: dict[str, tuple[Callable[[int], Pipeline], list[str]]] = {
    "length_only": (_logreg(["n_steps", "total_tokens"]), ["n_steps", "total_tokens"]),
    "empty_patch": (_logreg(["patch_empty"]), ["patch_empty"]),
    "prompt_only": (_logreg(TaskFeatures.numeric_names(), ["repo"]), TaskFeatures.numeric_names() + ["repo"]),
    "telemetry_lr": (_logreg(TELEMETRY), TELEMETRY),
    "telemetry_gbm": (_gbm(TELEMETRY), TELEMETRY),
    "telemetry20_lr": (_logreg(TELEMETRY20), TELEMETRY20),
    "telemetry20_gbm": (_gbm(TELEMETRY20), TELEMETRY20),
}


def _xy(data: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = data[cols].astype({c: int for c in cols if data[c].dtype == bool})
    return X, data["resolved"].to_numpy(bool)


def _rows(data: pd.DataFrame, method: str, p: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"uid": data["uid"], "instance_id": data["instance_id"], "method": method,
                         "fold": data["fold"], "p_success": p, "label": data["resolved"].to_numpy(bool)})


def cv_predict(data: pd.DataFrame, method: str, seed: int = config.SEED) -> pd.DataFrame:
    """Out-of-fold predictions on dev runs. data: features merged with splits."""
    make, cols = METHODS[method]
    dev = data[data["split"] == "dev"].reset_index(drop=True)
    X, y = _xy(dev, cols)
    p = np.zeros(len(dev))
    for k in sorted(dev["fold"].unique()):
        test = (dev["fold"] == k).to_numpy()
        model = make(seed).fit(X[~test], y[~test])
        p[test] = model.predict_proba(X[test])[:, 1]
    return _rows(dev, method, p)


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, method: str, seed: int = config.SEED) -> pd.DataFrame:
    """Fit on `train` runs, predict `target` runs (needs a `fold` column; -1 outside dev)."""
    make, cols = METHODS[method]
    target = target.reset_index(drop=True)
    model = make(seed).fit(*_xy(train, cols))
    return _rows(target, method, model.predict_proba(_xy(target, cols)[0])[:, 1])


def holdout_predict(data: pd.DataFrame, method: str, seed: int = config.SEED) -> pd.DataFrame:
    """Fit on all dev runs, predict the held-out test runs."""
    return fit_predict(data[data["split"] == "dev"], data[data["split"] == "test"], method, seed)


def coefficients(data: pd.DataFrame, method: str = "telemetry_lr") -> pd.Series:
    """Standardised LogReg coefficients fit on dev runs, for the explainability readout (Gap 3)."""
    make, cols = METHODS[method]
    model = make(config.SEED).fit(*_xy(data[data["split"] == "dev"], cols))
    names = model[0].get_feature_names_out()
    return pd.Series(model[-1].coef_[0], index=[n.split("__", 1)[1] for n in names]).sort_values(key=abs, ascending=False)

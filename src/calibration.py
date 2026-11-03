"""Post-hoc probability calibration (Gap 1).

A predictor can rank runs well and still give probabilities that don't match how often runs pass,
for example when it was trained at a different pass rate than it is used at (SWE-Gym ~8%,
SWE-bench Lite ~35-45%, the SWE-Gym verifier's balanced 50%). A calibrator remaps scores
monotonically, so ranking is kept (isotonic can merge close scores into ties) and probabilities
become trustworthy.

  none      raw scores; the "off" switch for the ablation
  platt     logistic regression on logit(p)
  isotonic  monotone step function fit to observed pass rates
  prior     Bayes prior-shift: p' odds = p odds * [pi_t / (1 - pi_t)] / [pi_s / (1 - pi_s)];
            no labels needed beyond the two base rates (pi_s: rate the scores were trained at,
            pi_t: rate where they are used)
  platt+prior  platt fit at the source rate, then shifted to the target rate

Calibrators are always fit on data the evaluation never sees: dev out-of-fold predictions, a
labelled slice of other tasks, or other folds of tasks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

METHODS = ("none", "platt", "isotonic", "prior", "platt+prior")
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def prior_shift(p: np.ndarray, pi_source: float, pi_target: float) -> np.ndarray:
    """Re-weight the odds from base rate pi_source to pi_target (proposal B.8, layer 4)."""
    shift = np.log(pi_target / (1 - pi_target)) - np.log(pi_source / (1 - pi_source))
    return 1 / (1 + np.exp(-(logit(p) + shift)))


@dataclass
class Calibrator:
    method: str = "platt"
    pi_target: Optional[float] = None       # for prior / platt+prior; default: the fit base rate
    pi_source: Optional[float] = None       # for prior: the scores' training rate; default: fit base rate
    _model: object = field(default=None, repr=False)
    _pi_fit: float = field(default=0.5, repr=False)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown calibration method {self.method!r}; use one of {METHODS}")

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=bool)
        self._pi_fit = float(y.mean()) if len(y) else 0.5
        if self.method in ("platt", "platt+prior"):
            self._model = LogisticRegression(C=1e4).fit(logit(p)[:, None], y)
        elif self.method == "isotonic":
            self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.method == "none":
            return p
        if self.method == "isotonic":
            return self._model.predict(p)
        if self.method == "prior":
            src = self.pi_source if self.pi_source is not None else self._pi_fit
            return prior_shift(p, src, self.pi_target if self.pi_target is not None else src)
        q = self._model.predict_proba(logit(p)[:, None])[:, 1]
        if self.method == "platt+prior" and self.pi_target is not None:
            q = prior_shift(q, self._pi_fit, self.pi_target)
        return q

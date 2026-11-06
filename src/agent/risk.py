"""The governance agent's risk models, calibrators and thresholds. All of it is fit on dev only.

  checkpoint models  one logistic regression per absolute checkpoint (after 5, 10, ... actions), on
                     prefix features only (see src/prefix.py).
  final model        logistic regression on the full-run features, for end-of-run triage. LR was chosen over
                     GBM on dev (results/metrics.md: within-task 0.865 vs 0.847, ECE 0.009 vs 0.022); with
                     the extended features they are level on dev (results/feature_eval.md).

Extended features are used only where the adoption rule in experiments/feature_eval.py passed on dev:
see EXTENDED_MODELS. The abort checkpoints (5, 10) did not pass, so they keep the Phase 1 features.

For each model, out-of-fold dev predictions train a calibrator and set the thresholds. Test and SWE-bench
Lite are never used. The agent may abort only at checkpoints whose out-of-fold dev AUROC reaches
MIN_ABORT_AUROC: the early-warning experiment found no usable signal later in a run, so an abort
there would be a guess.

Thresholds are probabilities, not ranks: "abort if calibrated p(resolved) < tau". That is why calibration
matters under shift. `retarget` moves the calibrated scores to a new base rate and leaves the thresholds
alone, whereas with calibration off the raw scores keep assuming SWE-Gym's 8 % pass rate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.baselines.tabular import METHODS, _logreg
from src.calibration import Calibrator
from src.eval.metrics import pooled_auroc
from src.prefix import ABS_STEPS, build_swegym
from src.schema import PrefixFeatures, RunFeatures

MIN_ABORT_AUROC = 0.60
PREFIX_COLS = PrefixFeatures.feature_names()
PREFIX_EXT_COLS = PrefixFeatures.extended_feature_names()
FINAL_EXT_COLS = RunFeatures.extended_feature_names()
EXTENDED_MODELS = {"final", 20, 30}       # results/feature_eval.md, table E


@dataclass
class RiskConfig:
    calibration: bool = True      # Gap 1 switch: off = raw scores
    telemetry: bool = True        # Gap 3 switch: off = length/token signals only
    far_abort: float = 0.05       # share of dev resolving runs an abort at one checkpoint may hit. Dev rule:
                                  # largest budget losing <= 10 % of passing runs overall (results/governor_tuning.md)
    far_review: float = 0.10      # share of dev resolving runs the end-of-run REJECT may hit
    seed: int = config.SEED


@dataclass
class ScoredModel:
    model: object
    calibrator: Calibrator
    cols: list[str]
    tau: float
    dev_auroc: Optional[float]
    n_dev: int
    oof: pd.DataFrame = field(repr=False, default=None)   # uid, p_raw, p_cal, resolved (dev only)

    def score(self, row: dict) -> tuple[float, float]:
        X = pd.DataFrame([{c: float(row[c]) for c in self.cols}])
        raw = float(self.model.predict_proba(X)[0, 1])
        return raw, float(self.calibrator.transform(np.array([raw]))[0])

    def explain(self, row: dict, k: int = 3) -> list[dict]:
        """Exact linear SHAP: the k features pushing this run's log-odds most, relative to an average dev
        run. Negative effect = lowers p(resolved). Only for the logistic pipelines; [] otherwise."""
        try:
            X = pd.DataFrame([{c: float(row[c]) for c in self.cols}])
            phi = self.model[0].transform(X)[0] * self.model[-1].coef_[0]
        except (TypeError, AttributeError, IndexError, KeyError):
            return []
        top = np.argsort(-np.abs(phi))[:k]
        return [{"feature": self.cols[i], "value": float(row[self.cols[i]]), "effect": round(float(phi[i]), 3)} for i in top]


def _X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype({c: int for c in cols if df[c].dtype == bool})


def _fit_one(make, cols: list[str], dev: pd.DataFrame, far: float, cfg: RiskConfig) -> ScoredModel:
    dev = dev.reset_index(drop=True)
    y = dev["resolved"].to_numpy(bool)
    oof = np.zeros(len(dev))
    for k in sorted(dev["fold"].unique()):
        m = (dev["fold"] == k).to_numpy()
        oof[m] = make(cfg.seed).fit(_X(dev[~m], cols), y[~m]).predict_proba(_X(dev[m], cols))[:, 1]
    cal = Calibrator("platt+prior" if cfg.calibration else "none").fit(oof, y)
    p_cal = cal.transform(oof)
    return ScoredModel(model=make(cfg.seed).fit(_X(dev, cols), y), calibrator=cal, cols=cols,
                       tau=float(np.quantile(p_cal[y], far)), dev_auroc=pooled_auroc(y, oof), n_dev=len(dev),
                       oof=pd.DataFrame({"uid": dev["uid"], "p_raw": oof, "p_cal": p_cal, "resolved": y}))


class RiskModels:
    def __init__(self, cfg: RiskConfig, checkpoints: dict[int, ScoredModel], final: ScoredModel):
        self.cfg, self.checkpoints, self.final = cfg, checkpoints, final

    @classmethod
    def fit(cls, cfg: Optional[RiskConfig] = None) -> "RiskModels":
        cfg = cfg or RiskConfig()
        splits = pd.read_csv(config.SPLITS_PATH)
        dev_ids = splits.loc[splits["split"] == "dev", ["uid", "fold"]]
        pre = build_swegym()
        pre = pre[pre["checkpoint"] == "abs"].merge(dev_ids, on="uid")
        def ckpt_cols(k: int) -> list[str]:
            if not cfg.telemetry:
                return ["tokens_so_far"]
            return PREFIX_EXT_COLS if k in EXTENDED_MODELS else PREFIX_COLS
        ckpts = {k: _fit_one(_logreg(ckpt_cols(k)), ckpt_cols(k), pre[pre["step"] == k], cfg.far_abort, cfg)
                 for k in ABS_STEPS}
        make, fcols = (_logreg(FINAL_EXT_COLS), FINAL_EXT_COLS) if cfg.telemetry else METHODS["length_only"]
        feats = pd.read_parquet(config.FEATURES_PATH).merge(dev_ids, on="uid")
        return cls(cfg, ckpts, _fit_one(make, list(fcols), feats, cfg.far_review, cfg))

    def can_abort(self, k: int) -> bool:
        c = self.checkpoints.get(k)
        return c is not None and c.dev_auroc is not None and c.dev_auroc >= MIN_ABORT_AUROC

    def retarget(self, pi_target: float) -> None:
        """Tell the calibrators the pass rate where the agent now works (e.g. from K labelled tasks)."""
        for m in [*self.checkpoints.values(), self.final]:
            m.calibrator.pi_target = pi_target if self.cfg.calibration else None

    def summary(self) -> pd.DataFrame:
        rows = [{"model": f"after {k} actions", "dev_runs": m.n_dev, "dev_auroc": m.dev_auroc, "tau": m.tau,
                 "acts": "ABORT" if self.can_abort(k) else "trust curve only"} for k, m in self.checkpoints.items()]
        rows.append({"model": "end of run", "dev_runs": self.final.n_dev, "dev_auroc": self.final.dev_auroc,
                     "tau": self.final.tau, "acts": "REJECT / FLAG_FOR_REVIEW"})
        return pd.DataFrame(rows).round(4)

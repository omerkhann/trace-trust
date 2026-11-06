"""Phase 1 validity check (B.12): how sensitive are the results to imperfect benchmark labels?

Wang et al. (ICSE 2026) report that 7.8 % of "resolved" patches are wrong. Simulation: flip 7.8 % of passing labels to
failing in dev, refit the end-of-run model, calibrator and threshold on those noisy labels, and score on test.
Test labels are scored twice: as released, and with 7.8 % of passing labels flipped too. Three seeds pick the flips.
This asks how fragile the numbers are; it doesn't estimate this dataset's real noise rate, which is unknown.

    python -m experiments.label_noise
-> results/label_noise.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.agent.risk import FINAL_EXT_COLS, RiskConfig, _fit_one, _X
from src.baselines.tabular import _logreg
from src.eval.metrics import ece, pooled_auroc, within_task_auroc

RATE = 0.078
OUT = config.RESULTS_DIR / "label_noise"


def flip(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = y.copy()
    passing = np.flatnonzero(y)
    y[rng.choice(passing, size=int(round(RATE * len(passing))), replace=False)] = False
    return y


def score(test: pd.DataFrame, y: np.ndarray, p: np.ndarray, tau: float) -> dict:
    wt, _, _ = within_task_auroc(pd.DataFrame({"instance_id": test["instance_id"].to_numpy(), "label": y, "p_success": p}))
    stop = p < tau
    return {"pooled": pooled_auroc(y, p), "within": wt, "ece": ece(y, p), "failures_rejected": float(stop[~y].mean()),
            "passes_rejected": float(stop[y].mean()), "reviewed": float((~stop).mean()), "pass_rate_in_review": float(y[~stop].mean())}


def main() -> None:
    gym = pd.read_parquet(config.FEATURES_PATH).merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev, test = gym[gym["split"] == "dev"].reset_index(drop=True), gym[gym["split"] == "test"].reset_index(drop=True)
    y_test = test["resolved"].to_numpy(bool)
    cols = FINAL_EXT_COLS

    def run(train: pd.DataFrame, labels: np.ndarray) -> dict:
        m = _fit_one(_logreg(cols), cols, train, RiskConfig().far_review, RiskConfig())
        p = m.calibrator.transform(m.model.predict_proba(_X(test, cols))[:, 1])
        return score(test, labels, p, m.tau)

    clean = run(dev, y_test)
    rows = {"train clean, test as released": [clean], "train noisy, test as released": [], "train noisy, test noisy": []}
    for s in config.SEEDS:
        rng = np.random.default_rng(s)
        noisy = dev.assign(resolved=flip(dev["resolved"].to_numpy(bool), rng))
        m = _fit_one(_logreg(cols), cols, noisy, RiskConfig().far_review, RiskConfig())
        p = m.calibrator.transform(m.model.predict_proba(_X(test, cols))[:, 1])
        rows["train noisy, test as released"].append(score(test, y_test, p, m.tau))
        rows["train noisy, test noisy"].append(score(test, flip(y_test, rng), p, m.tau))
    out = {k: {f: {"mean": float(np.mean([r[f] for r in v])), "std": float(np.std([r[f] for r in v], ddof=1)) if len(v) > 1 else 0.0} for f in v[0]}
           for k, v in rows.items()}
    OUT.with_suffix(".json").write_text(json.dumps({"rate": RATE, "seeds": list(config.SEEDS), "results": out}, indent=1))
    lines = [f"telemetry20_lr end-of-run model, {RATE:.1%} of passing labels flipped, seeds {list(config.SEEDS)}, mean ± std.", "",
             "| setting | pooled AUROC | within-task AUROC | ECE | failures rejected | passes rejected | reviewed | pass rate in review |", "|---|---|---|---|---|---|---|---|"]
    for k, r in out.items():
        lines.append(f"| {k} | {r['pooled']['mean']:.3f} ± {r['pooled']['std']:.3f} | {r['within']['mean']:.3f} ± {r['within']['std']:.3f} | {r['ece']['mean']:.3f} | "
                     + " | ".join(f"{100 * r[c]['mean']:.1f}%" for c in ("failures_rejected", "passes_rejected", "reviewed", "pass_rate_in_review")) + " |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

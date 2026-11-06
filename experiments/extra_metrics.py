"""AUPRC and macro-F1 (Phase 1, B.12), for every tabular method on each evaluation set.

Macro-F1 needs a yes/no decision, so the threshold is the one that maximises macro-F1 on dev out-of-fold predictions; test,
Lite and MoatlessTools never choose it. AUPRC is reported beside the base rate, which is its chance level.

    python -m experiments.extra_metrics
-> results/extra_metrics.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

from experiments.transfer_moatless import PATHS
from src import config
from src.baselines import tabular

METHODS = ["length_only", "empty_patch", "prompt_only", "telemetry_lr", "telemetry_gbm", "telemetry20_lr", "telemetry20_gbm"]
OUT = config.RESULTS_DIR / "extra_metrics"


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    return float(max(grid, key=lambda t: f1_score(y, p >= t, average="macro", zero_division=0)))


def main() -> None:
    gym = pd.read_parquet(config.FEATURES_PATH).merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev, test = gym[gym["split"] == "dev"], gym[gym["split"] == "test"]
    moat = pd.read_parquet(PATHS[True])
    sets = {"SWE-Gym test": test, "SWE-Gym test, non-empty": test[~test["patch_empty"].astype(bool)],
            "OpenHands Lite": pd.read_parquet(config.DATA_DIR / "lite_features.parquet"), "Moatless 32b_8runs": moat[moat["set"] == "32b_8runs"]}
    rows = []
    for m in METHODS:
        oof = tabular.cv_predict(gym, m)
        t = best_threshold(oof["label"].to_numpy(bool), oof["p_success"].to_numpy())
        for name, target in sets.items():
            p = tabular.fit_predict(dev, target, m, config.SEED)
            y, q = p["label"].to_numpy(bool), p["p_success"].to_numpy()
            rows.append({"method": m, "set": name, "runs": len(p), "base_rate": float(y.mean()), "auprc": float(average_precision_score(y, q)),
                         "macro_f1": float(f1_score(y, q >= t, average="macro", zero_division=0)),
                         "f1_pass": float(f1_score(y, q >= t, zero_division=0)), "threshold": t})
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    t = pd.DataFrame(rows)
    lines = ["AUPRC (chance level = base rate) and macro-F1 at the threshold that maximises it on dev out-of-fold predictions. Models fit on SWE-Gym dev.", ""]
    for name, g in t.groupby("set", sort=False):
        lines += [f"### {name}: {g['runs'].iloc[0]} runs, base rate {g['base_rate'].iloc[0]:.3f}", "", "| method | AUPRC | macro-F1 | F1 (pass class) |", "|---|---|---|---|"]
        lines += [f"| {r['method']} | {r['auprc']:.3f} | {r['macro_f1']:.3f} | {r['f1_pass']:.3f} |" for _, r in g.iterrows()] + [""]
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

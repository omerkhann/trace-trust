"""E4 / Gap 2: is the scaffold normaliser's gain real? Paired task-level bootstrap, normalised minus raw.

Same models (fit on SWE-Gym dev), same MoatlessTools runs, normaliser on vs off. Run after
experiments.transfer_moatless, which builds the feature caches.

    python -m experiments.gap2_paired
-> results/gap2_paired.{json,md}
"""
from __future__ import annotations

import json

import pandas as pd

from experiments.transfer_moatless import PATHS, SETS
from src import config
from src.baselines import tabular
from src.eval.metrics import ece, paired_bootstrap

METHODS = ["telemetry_lr", "telemetry_gbm", "telemetry20_lr", "telemetry20_gbm"]
OUT = config.RESULTS_DIR / "gap2_paired"


def main() -> None:
    norm, raw = pd.read_parquet(PATHS[True]), pd.read_parquet(PATHS[False])
    gym = pd.read_parquet(config.FEATURES_PATH).merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev = gym[gym["split"] == "dev"]
    rows = []
    for m in METHODS:
        a, b = tabular.fit_predict(dev, norm, m, config.SEED), tabular.fit_predict(dev, raw, m, config.SEED)
        for s in SETS:
            keep = set(norm.loc[norm["set"] == s, "uid"])
            pa, pb = a[a["uid"].isin(keep)].reset_index(drop=True), b[b["uid"].isin(keep)].reset_index(drop=True)
            boot = paired_bootstrap(pa, pb)
            rows.append({"method": m, "set": s, "pooled_diff": boot["pooled"][:3] if boot["pooled"] else None,
                         "within_diff": boot["within"][:3] if boot["within"] else None,
                         "ece_norm": ece(pa["label"].to_numpy(bool), pa["p_success"].to_numpy()),
                         "ece_raw": ece(pb["label"].to_numpy(bool), pb["p_success"].to_numpy())})

    def f(x):
        return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" if x else "n/a"
    lines = ["Normalised minus raw on MoatlessTools. Paired task-level bootstrap, 1,000 resamples. Models fit on SWE-Gym dev.", "",
             "| method | set | pooled AUROC diff | within-task AUROC diff | ECE normalised | ECE raw |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['method']} | {r['set']} | {f(r['pooled_diff'])} | {f(r['within_diff'])} | {r['ece_norm']:.3f} | {r['ece_raw']:.3f} |" for r in rows]
    OUT.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

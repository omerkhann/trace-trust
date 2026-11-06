"""E4 / Gap 3: telemetry vs the 32B verifier, alone and combined, on the 237 verifier-scored SWE-bench Lite runs.

H3 targets: (i) standalone telemetry reaches >= 90 % of the verifier's within-task AUROC; (ii) a hybrid improves on
the verifier's within-task AUROC by >= 0.02.

Telemetry models are fit on SWE-Gym dev. The hybrid is a logistic regression on logit(verifier) and logit(telemetry),
cross-fitted by task on these 237 runs (5 folds, 3 seeds): every score comes from a model that never saw that
task. The primary hybrid is verifier + telemetry20_lr, the agent's end-of-run model, fixed before running. The others
are shown for context, and picking the best of them afterwards would not be a fair test.

    python -m experiments.hybrid
-> results/hybrid_verifier.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src.baselines import tabular
from src.calibration import logit
from src.eval.metrics import best_at_k, ece, paired_bootstrap, pooled_auroc, within_task_auroc, within_task_wilcoxon

PARTNERS = {"verifier + telemetry20_lr (primary)": "telemetry20_lr", "verifier + telemetry_lr": "telemetry_lr",
            "verifier + telemetry_gbm": "telemetry_gbm", "verifier + length_only": "length_only"}
OUT = config.RESULTS_DIR / "hybrid_verifier"


def frame(uids, instance, y, p, method) -> pd.DataFrame:
    return pd.DataFrame({"uid": uids, "instance_id": instance, "method": method, "fold": -1, "p_success": p, "label": y})


def cross_fit(a: np.ndarray, b: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    X = np.c_[logit(a), logit(b)]
    out = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        out[te] = LogisticRegression(C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return out


def scores(p: pd.DataFrame) -> dict:
    y, q = p["label"].to_numpy(bool), p["p_success"].to_numpy(float)
    wt, _, pairs = within_task_auroc(p)
    wp, _ = within_task_wilcoxon(p)
    return {"pooled": pooled_auroc(y, q), "within": wt, "pairs": int(pairs), "wilcoxon_p": wp, "ece": ece(y, q),
            "best_at_2": best_at_k(p, 2).get("best")}


def main() -> None:
    v = pd.read_csv(config.DATA_DIR / "verifier_scores_lite.csv").drop_duplicates("uid", keep="last")
    lite = pd.read_parquet(config.DATA_DIR / "lite_features.parquet")
    gym = pd.read_parquet(config.FEATURES_PATH).merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev = gym[gym["split"] == "dev"]
    d = v[["uid", "p_yes", "version", "seconds"]].merge(lite[["uid", "instance_id", "resolved"]], on="uid", validate="1:1")
    y, groups = d["resolved"].to_numpy(bool), d["instance_id"].to_numpy()
    ver = frame(d["uid"], d["instance_id"], y, d["p_yes"].to_numpy(), "verifier")
    alone, rows, paired = {}, [], []

    target = lite[lite["uid"].isin(set(d["uid"]))]
    for name, method in PARTNERS.items():
        p = tabular.fit_predict(dev, target, method, config.SEED).set_index("uid").loc[d["uid"]]["p_success"].to_numpy()
        alone[method] = frame(d["uid"], d["instance_id"], y, p, method)
        per_seed = [cross_fit(d["p_yes"].to_numpy(), p, y, groups, s) for s in config.SEEDS]
        ms = [scores(frame(d["uid"], d["instance_id"], y, q, name)) for q in per_seed]
        rows.append({"model": name, **{k: float(np.mean([m[k] for m in ms])) for k in ("pooled", "within", "ece", "best_at_2")},
                     "within_std": float(np.std([m["within"] for m in ms], ddof=1)), "pairs": ms[0]["pairs"], "wilcoxon_p": ms[0]["wilcoxon_p"]})
        boot = paired_bootstrap(frame(d["uid"], d["instance_id"], y, per_seed[0], name), ver)
        paired.append({"model": name, "pooled_minus_verifier": boot["pooled"][:3], "within_minus_verifier": boot["within"][:3]})

    base = {"verifier": scores(ver), **{k: scores(a) for k, a in alone.items()}}
    wv = base["verifier"]["within"]
    ratios = {k: base[k]["within"] / wv for k in alone}
    OUT.with_suffix(".json").write_text(json.dumps({"alone": base, "share_of_verifier_within": ratios, "hybrids": rows, "paired_vs_verifier": paired,
                                                    "verifier_seconds_per_run": float(d["seconds"].mean())}, indent=1, default=float))
    lines = [f"237 verifier-scored SWE-bench Lite runs, {base['verifier']['pairs']} same-task pairs. Telemetry fit on SWE-Gym dev. Hybrid: LR on logits, "
             f"5-fold cross-fit by task, mean over seeds {list(config.SEEDS)}. Verifier {d['seconds'].mean():.1f} s/run.", "",
             "### Standalone", "", "| method | pooled | within-task | share of verifier's within-task | ECE | Best@2 |", "|---|---|---|---|---|---|"]
    for k, m in base.items():
        share = "" if k == "verifier" else f"{ratios[k]:.0%}"
        lines.append(f"| {k} | {m['pooled']:.3f} | {m['within']:.3f} | {share} | {m['ece']:.3f} | {m['best_at_2']:.3f} |")
    lines += ["", "### Hybrids", "", "| model | pooled | within-task | ECE | Best@2 | pooled minus verifier | within-task minus verifier |", "|---|---|---|---|---|---|---|"]

    def f(x):
        return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    for r, b in zip(rows, paired):
        lines.append(f"| {r['model']} | {r['pooled']:.3f} | {r['within']:.3f} ± {r['within_std']:.3f} | {r['ece']:.3f} | {r['best_at_2']:.3f} | "
                     f"{f(b['pooled_minus_verifier'])} | {f(b['within_minus_verifier'])} |")
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

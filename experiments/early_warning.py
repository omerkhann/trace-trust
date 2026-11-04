"""E4 / Gap 3: early warning. Can a run's first k actions tell us it will fail?

For each checkpoint, models are fit on dev runs still going at that checkpoint and evaluated on the
held-out test runs still going there. The alarm threshold is set on dev out-of-fold predictions so
that 10 % of dev runs that resolved get flagged (target false-alarm rate); test is never used to
pick it. Reported on test:
  AUROC / within-task AUROC of p(resolved)
  failure recall   share of failing runs flagged at this checkpoint
  FAR              share of resolving runs flagged (realised false-alarm rate; target 10 %)
  tokens saved     tokens the flagged runs would have used after the checkpoint, as a share of
                   all tokens the runs still going use from there on
Checkpoints: "abs" after k actions (what a live monitor sees); "frac" after 25/50/75 % of the run
(the proposal's framing; needs the run's total length, so it is offline-only and leaks length).
SWE-bench Lite: the SWE-Gym-dev models and thresholds applied unchanged.

    python -m experiments.early_warning
-> results/early_warning.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.baselines.tabular import _gbm, _logreg
from src.eval.metrics import pooled_auroc, within_task_auroc
from src.prefix import build_lite, build_swegym
from src.schema import PrefixFeatures

FEATS = PrefixFeatures.feature_names()
MODELS = {
    "prefix_lr": (_logreg(FEATS), FEATS),
    "prefix_gbm": (_gbm(FEATS), FEATS),
    "prefix_tokens_only": (_logreg(["tokens_so_far"]), ["tokens_so_far"]),
}
TARGET_FAR = 0.10
OUT = config.RESULTS_DIR / "early_warning"


def _X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype({c: int for c in cols if df[c].dtype == bool})


def threshold_for_far(p_success: np.ndarray, y: np.ndarray, far: float = TARGET_FAR) -> float:
    """Flag a run when p_success < t. t is the `far` quantile of p_success over resolving runs, so
    about `far` of them fall below it."""
    return float(np.quantile(p_success[y], far))


def alarm_metrics(df: pd.DataFrame, p: np.ndarray, t: float) -> dict:
    y = df["resolved"].to_numpy(bool)
    flag = p < t
    rest = (df["total_tokens"] - df["tokens_so_far"]).clip(lower=0).to_numpy(float)
    wt, _, pairs = within_task_auroc(pd.DataFrame({"instance_id": df["instance_id"], "label": y, "p_success": p}))
    return {"n": int(len(df)), "base_rate": float(y.mean()), "pooled_auroc": pooled_auroc(y, p),
            "within_auroc": wt, "pairs": int(pairs),
            "fail_recall": float(flag[~y].mean()) if (~y).any() else None,
            "far": float(flag[y].mean()) if y.any() else None,
            "flag_rate": float(flag.mean()),
            "tokens_saved": float(rest[flag].sum() / rest.sum()) if rest.sum() else None}


def run_checkpoint(dev: pd.DataFrame, targets: dict[str, pd.DataFrame], model: str, seed: int) -> dict[str, dict]:
    make, cols = MODELS[model]
    y = dev["resolved"].to_numpy(bool)
    oof = np.zeros(len(dev))
    for k in sorted(dev["fold"].unique()):
        m = (dev["fold"] == k).to_numpy()
        oof[m] = make(seed).fit(_X(dev[~m], cols), y[~m]).predict_proba(_X(dev[m], cols))[:, 1]
    t = threshold_for_far(oof, y)
    fit = make(seed).fit(_X(dev, cols), y)
    out = {"dev_oof": alarm_metrics(dev, oof, t) | {"threshold": t}}
    for name, df in targets.items():
        if len(df) and df["resolved"].nunique() == 2:
            out[name] = alarm_metrics(df, fit.predict_proba(_X(df, cols))[:, 1], t) | {"threshold": t}
    return out


def _mean(dicts: list[dict]) -> dict:
    out = {}
    for k, v in dicts[0].items():
        vals = [d[k] for d in dicts]
        if isinstance(v, float) and all(x is not None for x in vals):
            out[k] = float(np.mean(vals))
            if len(vals) > 1 and np.std(vals) > 0:
                out[k + "_std"] = float(np.std(vals, ddof=1))
        else:
            out[k] = v
    return out


def main() -> None:
    splits = pd.read_csv(config.SPLITS_PATH)
    feats = pd.read_parquet(config.FEATURES_PATH, columns=["uid", "total_tokens"])
    pre = build_swegym().merge(splits[["uid", "split", "fold"]], on="uid").merge(feats, on="uid")
    lite = build_lite()
    lite_tot = pd.read_parquet(config.DATA_DIR / "lite_features.parquet", columns=["uid", "total_tokens"])
    lite = lite.merge(lite_tot, on="uid")

    def key(df: pd.DataFrame, kind: str, v) -> pd.DataFrame:
        col = "step" if kind == "abs" else "prefix_frac"
        return df[(df["checkpoint"] == kind) & (df[col] == v)]

    rows = []
    for kind, values in (("abs", sorted(pre.loc[pre["checkpoint"] == "abs", "step"].unique())),
                         ("frac", sorted(pre["prefix_frac"].dropna().unique()))):
        for v in values:
            at = key(pre, kind, v)
            dev, test = at[at["split"] == "dev"].reset_index(drop=True), at[at["split"] == "test"]
            if dev["resolved"].sum() < 10:
                continue                       # too few resolving runs left to set a threshold
            targets = {"test": test, "lite": key(lite, kind, v)}
            for model in MODELS:
                per_seed = [run_checkpoint(dev, targets, model, s) for s in config.SEEDS]
                for ev in per_seed[0]:
                    rows.append({"checkpoint": kind, "at": float(v), "model": model, "eval": ev}
                                | _mean([r[ev] for r in per_seed]))
    OUT.with_suffix(".json").write_text(json.dumps({"target_far": TARGET_FAR, "seeds": list(config.SEEDS),
                                                    "rows": rows}, indent=1))
    OUT.with_suffix(".md").write_text(to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))


def to_markdown(rows: list[dict]) -> str:
    def f(r, k, pct=False):
        x = r.get(k)
        if x is None:
            return "n/a"
        return f"{100 * x:.1f}%" if pct else f"{x:.3f}"
    lines = [f"Alarm threshold set on dev out-of-fold predictions for a {TARGET_FAR:.0%} false-alarm rate "
             "(share of resolving runs flagged); test and Lite never used to pick it. Mean over seeds "
             f"{list(config.SEEDS)}. 'abs k' = after k actions, runs still going; 'frac' = share of the run "
             "(needs the total length, offline only).", ""]
    for ev in ("test", "lite", "dev_oof"):
        rs = [r for r in rows if r["eval"] == ev]
        if not rs:
            continue
        lines += [f"### {ev}", "",
                  "| checkpoint | model | runs | base rate | pooled AUROC | within-task AUROC (pairs) | "
                  "failure recall | FAR | tokens saved |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rs:
            cp = f"abs {int(r['at'])}" if r["checkpoint"] == "abs" else f"frac {r['at']:.0%}"
            lines.append(f"| {cp} | {r['model']} | {r['n']} | {f(r, 'base_rate')} | {f(r, 'pooled_auroc')} | "
                         f"{f(r, 'within_auroc')} ({r['pairs']}) | {f(r, 'fail_recall', True)} | "
                         f"{f(r, 'far', True)} | {f(r, 'tokens_saved', True)} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

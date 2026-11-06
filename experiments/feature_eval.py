"""Do the Phase 2 features help? Dev only: test and SWE-bench Lite are never read.

Compares the Phase 1 feature set (11 full-run, 12 prefix) with the extended one (20, 21), using out-of-fold
dev predictions, on all runs and on the hard subsets. Paired task-level bootstrap for extended minus base.
  A. full-run models    all dev runs; non-empty patches (the hard case); GPT-4o only
  B. checkpoint models  AUROC after k actions, and failure recall at a 10 % false-alarm rate
  C. family ablation    extended set minus one family at a time
  D. single features    each feature alone: coverage and AUROC, on all runs and on non-empty patches

Adoption rule, written before looking at the results. Use the extended set for a model only if, on dev:
  (1) pooled or within-task AUROC on the hard subset (non-empty patches for the end-of-run model, all
      runs still going for a checkpoint) rises by at least 0.01, and
  (2) neither metric on all dev runs falls by more than 0.005.
Otherwise that model stays on the base set, and that is reported.

    python -m experiments.feature_eval
-> results/feature_eval.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config
from src.baselines.tabular import _gbm, _logreg
from src.eval.metrics import paired_bootstrap, pooled_auroc, within_task_auroc
from src.prefix import ABS_STEPS, build_swegym
from src.schema import PROGRESS_FEATURES, PrefixFeatures, RunFeatures

BASE, EXT = RunFeatures.feature_names(), RunFeatures.extended_feature_names()
P_BASE, P_EXT = PrefixFeatures.feature_names(), PrefixFeatures.extended_feature_names()
FAMILIES = {"verification": ["last_check_passed", "checked_after_last_edit", "fail_then_pass"],
            "stuck": ["max_identical_streak", "max_error_streak", "error_responsiveness"],
            "pacing": ["first_edit_step", "phase_entropy"],
            "cognitive": ["thought_action_mismatch"]}
MAKERS = {"lr": _logreg, "gbm": _gbm}
RISE, DROP = 0.01, 0.005           # the adoption rule's thresholds
OUT = config.RESULTS_DIR / "feature_eval"


def _X(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].astype({c: int for c in cols if df[c].dtype == bool})


def oof(dev: pd.DataFrame, cols: list[str], kind: str) -> pd.DataFrame:
    make = MAKERS[kind](cols)
    y = dev["resolved"].to_numpy(bool)
    p = np.zeros(len(dev))
    for k in sorted(dev["fold"].unique()):
        m = (dev["fold"] == k).to_numpy()
        p[m] = make(config.SEED).fit(_X(dev[~m], cols), y[~m]).predict_proba(_X(dev[m], cols))[:, 1]
    return pd.DataFrame({"uid": dev["uid"], "instance_id": dev["instance_id"], "label": y, "p_success": p})


def scores(pred: pd.DataFrame) -> dict:
    y, p = pred["label"].to_numpy(bool), pred["p_success"].to_numpy()
    wt, tasks, pairs = within_task_auroc(pred)
    return {"runs": len(pred), "pooled": pooled_auroc(y, p), "within": wt, "pairs": int(pairs)}


def compare(dev: pd.DataFrame, base: list[str], ext: list[str], kind: str, masks: dict) -> list[dict]:
    a, b = oof(dev, ext, kind), oof(dev, base, kind)
    rows = []
    for name, m in masks.items():
        pa, pb = a[m].reset_index(drop=True), b[m].reset_index(drop=True)
        boot = paired_bootstrap(pa, pb)
        sa, sb = scores(pa), scores(pb)
        rows.append({"model": kind, "subset": name, "runs": sa["runs"], "pairs": sa["pairs"],
                     "pooled_base": sb["pooled"], "pooled_ext": sa["pooled"],
                     "pooled_diff_ci": boot["pooled"][:3] if boot["pooled"] else None,
                     "within_base": sb["within"], "within_ext": sa["within"],
                     "within_diff_ci": boot["within"][:3] if boot["within"] else None})
    return rows


def recall_at_far(pred: pd.DataFrame, far: float = 0.10) -> float:
    y, p = pred["label"].to_numpy(bool), pred["p_success"].to_numpy()
    return float((p[~y] < np.quantile(p[y], far)).mean())


def _rises(r) -> bool:
    return (r["pooled_ext"] - r["pooled_base"] >= RISE) or (r["within_ext"] - r["within_base"] >= RISE)


def _holds(r) -> bool:
    return (r["pooled_ext"] - r["pooled_base"] >= -DROP) and (r["within_ext"] - r["within_base"] >= -DROP)


def adoption(full: pd.DataFrame, ckpt: pd.DataFrame) -> pd.DataFrame:
    """The rule in the module docstring, for the logistic regressions the agent uses."""
    lr = full[full["model"] == "lr"].set_index("subset")
    rows = [{"model": "end of run", "rises_on_hard_subset": _rises(lr.loc["non_empty_patch"]),
             "holds_on_all_runs": _holds(lr.loc["all"])}]
    for _, r in ckpt[ckpt["model"] == "lr"].iterrows():
        rows.append({"model": f"after {r['after_actions']} actions", "rises_on_hard_subset": _rises(r),
                     "holds_on_all_runs": _holds(r)})
    out = pd.DataFrame(rows)
    out["use_extended"] = out["rises_on_hard_subset"] & out["holds_on_all_runs"]
    return out


def main() -> None:
    splits = pd.read_csv(config.SPLITS_PATH)
    dev_ids = splits.loc[splits["split"] == "dev", ["uid", "fold"]]
    feats = pd.read_parquet(config.FEATURES_PATH).merge(dev_ids, on="uid").reset_index(drop=True)
    pre = build_swegym()
    pre = pre[pre["checkpoint"] == "abs"].merge(dev_ids, on="uid")

    masks = {"all": np.ones(len(feats), bool), "non_empty_patch": ~feats["patch_empty"].to_numpy(bool),
             "gpt4o_only": feats["policy_model"].str.contains("gpt").to_numpy(bool)}
    full = [r for kind in MAKERS for r in compare(feats, BASE, EXT, kind, masks)]

    ckpt = []
    for k in ABS_STEPS:
        at = pre[pre["step"] == k].reset_index(drop=True)
        if at["resolved"].sum() < 10:
            continue
        for kind in MAKERS:
            a, b = oof(at, P_EXT, kind), oof(at, P_BASE, kind)
            boot = paired_bootstrap(a, b)
            sa, sb = scores(a), scores(b)
            ckpt.append({"after_actions": int(k), "model": kind, "runs": sa["runs"], "pairs": sa["pairs"],
                         "pooled_base": sb["pooled"], "pooled_ext": sa["pooled"],
                         "pooled_diff_ci": boot["pooled"][:3] if boot["pooled"] else None,
                         "within_base": sb["within"], "within_ext": sa["within"],
                         "recall_base": recall_at_far(b), "recall_ext": recall_at_far(a)})

    fam = []
    hard = masks["non_empty_patch"]
    for kind in MAKERS:
        ext_all = oof(feats, EXT, kind)
        for name, drop in {"none (extended)": [], **FAMILIES}.items():
            cols = [c for c in EXT if c not in drop]
            pred = oof(feats, cols, kind) if drop else ext_all
            s, h = scores(pred), scores(pred[hard].reset_index(drop=True))
            fam.append({"model": kind, "dropped": name, "pooled": s["pooled"], "within": s["within"],
                        "pooled_non_empty": h["pooled"], "within_non_empty": h["within"]})

    single = []
    y = feats["resolved"].to_numpy(bool)
    for f in PROGRESS_FEATURES:
        v = feats[f].astype(float).to_numpy()
        single.append({"feature": f, "share_non_default": float((v != feats[f].astype(float).mode()[0]).mean()),
                       "mean_pass": float(v[y].mean()), "mean_fail": float(v[~y].mean()),
                       "auroc_all": pooled_auroc(y, v), "auroc_non_empty": pooled_auroc(y[hard], v[hard])})
    s = feats["thought_action_mismatch"] > 0
    coverage = {"runs_with_any_mismatch": float(s.mean()),
                "by_model": feats.groupby("policy_model")["thought_action_mismatch"].apply(lambda x: float((x > 0).mean())).to_dict()}

    tables = {"A. Full-run models, extended vs base": pd.DataFrame(full),
              "B. Checkpoint models, extended vs base": pd.DataFrame(ckpt),
              "C. Family ablation": pd.DataFrame(fam), "D. Single features": pd.DataFrame(single),
              "E. Adoption rule": adoption(pd.DataFrame(full), pd.DataFrame(ckpt))}
    OUT.with_suffix(".json").write_text(json.dumps({**{k: v.to_dict("records") for k, v in tables.items()},
                                                    "mismatch_coverage": coverage}, indent=1, default=str))
    lines = [f"Dev only, out-of-fold, {len(feats)} runs. Diff CIs: extended minus base, paired task bootstrap.", ""]

    def cell(x):
        if isinstance(x, float):
            return f"{x:.3f}"
        if isinstance(x, tuple):
            return f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
        return str(x)
    for name, t in tables.items():
        lines += [f"### {name}", "", "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
        lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in t.itertuples(index=False)]
        lines.append("")
    lines += [f"Runs with any thought-action mismatch: {coverage['runs_with_any_mismatch']:.3f}; by model: "
              + ", ".join(f"{k} {v:.3f}" for k, v in coverage["by_model"].items()), ""]
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

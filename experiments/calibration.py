"""E4 / Gap 1: does post-hoc calibration make the predictors' probabilities trustworthy?

Three settings, each with calibrators fit only on data the evaluation never sees:
  A. SWE-Gym in-domain   fit on dev out-of-fold predictions, evaluate on the held-out test set
  B. SWE-bench Lite      zero-shot: the SWE-Gym-dev calibrator as is, on all Lite runs
                         few-shot: label K Lite tasks (about 4 runs each), use them to set the target
                         pass rate (platt+prior) or refit platt, evaluate on the other Lite tasks;
                         repeated over config.SEEDS draws, reported as mean ± std
  C. 32B verifier        on the runs it scored (data/verifier_scores_lite.csv, if present): 5-fold
                         cross-fitting grouped by task; prior-shift from its 50/50 training rate
Ranking metrics are reported alongside to show calibration doesn't damage them.

    python -m experiments.calibration [--k-tasks 5 20]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold

from src import config
from src.calibration import Calibrator
from src.eval import report
from src.eval.metrics import best_at_k, ece, pooled_auroc, within_task_auroc

PREDICTORS = ["telemetry_lr", "telemetry_gbm", "length_only"]
VERIFIER_SCORES = config.DATA_DIR / "verifier_scores_lite.csv"
VERIFIER_TRAIN_RATE = 0.5          # the SWE-Gym verifier was trained on a balanced pass/fail set
OUT = config.RESULTS_DIR / "calibration"


def metrics(df: pd.DataFrame, p: np.ndarray) -> dict:
    y = df["label"].to_numpy(bool)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    wt, _, pairs = within_task_auroc(df.assign(p_success=p))
    return {"n": len(df), "base_rate": float(y.mean()), "ece": ece(y, p), "brier": float(brier_score_loss(y, p)),
            "nll": float(log_loss(y, p, labels=[False, True])), "pooled_auroc": pooled_auroc(y, p),
            "within_auroc": wt, "pairs": pairs, "best_at_2": best_at_k(df.assign(p_success=p), 2)["best"]}


def row(setting: str, predictor: str, calibrator: str, m: dict) -> dict:
    return {"setting": setting, "predictor": predictor, "calibrator": calibrator} | m


def in_domain(preds: pd.DataFrame, splits: pd.DataFrame) -> list[dict]:
    split = splits.set_index("uid")["split"]
    rows = []
    for name in PREDICTORS:
        p = preds[preds["method"] == name]
        dev, test = p[p["uid"].map(split).eq("dev")], p[p["uid"].map(split).eq("test")]
        for method in ("none", "platt", "isotonic"):
            cal = Calibrator(method).fit(dev["p_success"], dev["label"])
            rows.append(row("A. SWE-Gym test", name, method, metrics(test, cal.transform(test["p_success"]))))
    return rows


def lite(preds: pd.DataFrame, splits: pd.DataFrame, lite_preds: pd.DataFrame, k_tasks: int,
         zero_shot: bool = True) -> tuple[list[dict], dict]:
    """Zero-shot rows (once, when zero_shot) and few-shot rows for K labelled tasks."""
    split = splits.set_index("uid")["split"]
    rows, curves = [], {}
    tasks = np.array(sorted(lite_preds["instance_id"].unique()))
    for name in PREDICTORS:
        dev = preds[(preds["method"] == name) & preds["uid"].map(split).eq("dev")]
        target = lite_preds[lite_preds["method"] == name]
        pi_dev = float(dev["label"].mean())
        for method in ("none", "platt", "isotonic") if zero_shot else ():   # fit on SWE-Gym dev only
            cal = Calibrator(method).fit(dev["p_success"], dev["label"])
            q = cal.transform(target["p_success"])
            rows.append(row("B. Lite zero-shot", name, f"{method} (SWE-Gym dev)", metrics(target, q)))
            if name == "telemetry_lr" and method in ("none", "platt"):
                curves[f"Lite {method}"] = target.assign(p_success=q)
        per_seed: dict[str, list[dict]] = {}
        for seed in config.SEEDS:                            # few-shot: K labelled Lite tasks
            rng = np.random.default_rng(seed)
            shot = set(rng.choice(tasks, size=k_tasks, replace=False))
            lab, rest = target[target["instance_id"].isin(shot)], target[~target["instance_id"].isin(shot)]
            pi_t = float(np.clip(lab["label"].mean(), 0.02, 0.98))
            cands = {
                "none": rest["p_success"].to_numpy(),
                "prior (dev rate -> K-task rate)": Calibrator("prior", pi_source=pi_dev, pi_target=pi_t)
                    .fit(dev["p_success"], dev["label"]).transform(rest["p_success"]),
                "platt (dev) + prior": Calibrator("platt+prior", pi_target=pi_t)
                    .fit(dev["p_success"], dev["label"]).transform(rest["p_success"]),
                "platt (refit on K tasks)": Calibrator("platt").fit(lab["p_success"], lab["label"]).transform(rest["p_success"]),
            }
            for method, q in cands.items():
                per_seed.setdefault(method, []).append(metrics(rest, q) | {"k_runs": len(lab)})
                if name == "telemetry_lr" and seed == config.SEEDS[0] and method == "platt (dev) + prior":
                    curves["Lite few-shot platt+prior"] = rest.assign(p_success=q)
        for method, ms in per_seed.items():
            agg = {k: float(np.mean([m[k] for m in ms]))
                   for k in ("ece", "brier", "nll", "pooled_auroc", "base_rate", "k_runs", "best_at_2")}
            agg |= {f"{k}_std": float(np.std([m[k] for m in ms], ddof=1)) for k in ("ece", "brier", "nll")}
            wts = [m["within_auroc"] for m in ms if m["within_auroc"] is not None]
            agg |= {"n": ms[0]["n"], "within_auroc": float(np.mean(wts)) if wts else None, "pairs": ms[0]["pairs"]}
            rows.append(row(f"B. Lite few-shot (K={k_tasks} tasks)", name, method, agg))
    return rows, curves


def verifier(lite_preds: pd.DataFrame, pi_dev: float) -> tuple[list[dict], dict]:
    """pi_dev: SWE-Gym dev pass rate, the rate our models were trained at."""
    if not VERIFIER_SCORES.exists():
        return [], {}
    v = pd.read_csv(VERIFIER_SCORES).drop_duplicates("uid", keep="last")
    feats = pd.read_parquet(config.DATA_DIR / "lite_features.parquet")[["uid", "instance_id", "resolved"]]
    v = v.merge(feats, on="uid", validate="1:1")
    sets = {"swegym_verifier_32b": pd.DataFrame({"uid": v["uid"], "instance_id": v["instance_id"],
                                                 "label": v["resolved"].astype(bool), "p_success": v["p_yes"]})}
    tl = lite_preds[(lite_preds["method"] == "telemetry_lr") & lite_preds["uid"].isin(set(v["uid"]))]
    sets["telemetry_lr"] = tl[["uid", "instance_id", "label", "p_success"]]
    rows, curves = [], {}
    for name, df in sets.items():
        df = df.reset_index(drop=True)
        pi_src = VERIFIER_TRAIN_RATE if name == "swegym_verifier_32b" else pi_dev
        out = {m: np.zeros(len(df)) for m in ("none", "prior", "platt", "isotonic")}
        for tr, te in GroupKFold(5).split(df, groups=df["instance_id"]):
            fit = df.iloc[tr]
            pi_t = float(fit["label"].mean())
            for m in out:
                cal = Calibrator(m, pi_source=pi_src, pi_target=pi_t)
                out[m][te] = cal.fit(fit["p_success"], fit["label"]).transform(df.iloc[te]["p_success"])
        for m, q in out.items():
            rows.append(row("C. verifier runs (5-fold by task)", name, m, metrics(df, q)))
            if name == "swegym_verifier_32b" and m in ("none", "platt"):
                curves[f"verifier {m}"] = df.assign(p_success=q)
    return rows, curves


def to_markdown(rows: list[dict]) -> str:
    def f(r, k, d=3):
        x = r.get(k)
        if x is None:
            return "n/a"
        s = f"{x:.{d}f}"
        return s + (f" ± {r[k + '_std']:.{d}f}" if r.get(k + "_std") else "")
    lines = ["Calibrators are fit only on data the evaluation never sees (see experiments/calibration.py). "
             "Few-shot rows: mean ± std over 3 random draws of the labelled tasks.", ""]
    for setting in dict.fromkeys(r["setting"] for r in rows):
        rs = [r for r in rows if r["setting"] == setting]
        lines += [f"### {setting}: {rs[0]['n']} runs, base rate {rs[0]['base_rate']:.3f}", "",
                  "| predictor | calibrator | ECE | Brier | NLL | pooled AUROC | within-task AUROC | Best@2 |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {r['predictor']} | {r['calibrator']} | {f(r, 'ece')} | {f(r, 'brier')} | {f(r, 'nll')} | "
                  f"{f(r, 'pooled_auroc')} | {f(r, 'within_auroc')} | {f(r, 'best_at_2')} |" for r in rs]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-tasks", type=int, nargs="+", default=[5, 20], help="labelled Lite tasks for few-shot calibration")
    ks = ap.parse_args().k_tasks
    preds = pd.read_parquet(config.DATA_DIR / "predictions.parquet")
    splits = pd.read_csv(config.SPLITS_PATH)
    lite_preds = pd.read_parquet(config.DATA_DIR / "lite_predictions.parquet")

    rows = in_domain(preds, splits)
    lite_rows, curves = [], {}
    for i, k in enumerate(ks):
        r, c = lite(preds, splits, lite_preds, k, zero_shot=i == 0)
        lite_rows += r
        curves = curves or c
    dev_rate = float(splits.loc[splits["split"] == "dev", "resolved"].mean())
    ver_rows, ver_curves = verifier(lite_preds, dev_rate)
    rows += lite_rows + ver_rows
    OUT.with_suffix(".json").write_text(json.dumps({"k_tasks": ks, "seeds": list(config.SEEDS), "rows": rows}, indent=1))
    OUT.with_suffix(".md").write_text(to_markdown(rows), encoding="utf-8")
    report.reliability_plot(curves | ver_curves, path=config.RESULTS_DIR / "calibration_reliability.png",
                            title="Calibration before / after")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()

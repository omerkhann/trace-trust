"""E4 / Gap 2: do telemetry predictors and the governance agent carry over to another agent framework?

Models are fit on SWE-Gym dev (OpenHands) only and applied zero-shot to MoatlessTools on SWE-bench Lite, with
the scaffold normaliser on and off. Nothing is tuned on MoatlessTools.

H2 (Phase 1, B.11): normalised telemetry loses <= 0.08 AUROC from its in-distribution value (>= 85 % retained);
raw, unnormalised features lose > 0.20.

Two references, because they answer different questions:
  SWE-Gym test   in-distribution, the reference H2 names. Moving to Moatless also changes the repositories.
  OpenHands Lite the same 300 SWE-bench Lite tasks under OpenHands. Tasks are fixed, so the difference is the
                 framework (and the models driving it), not the repositories. Closer to what Gap 2 is about.

Sets: 32b_8runs (8 attempts per task from one model and configuration, the main set: clean within-task pairs),
32b_zero and 7b_zero (one attempt per task).

Also: few-shot prior shift (K labelled tasks, scored on the rest) and the locked governance agent on 32b_8runs.

    python -m experiments.transfer_moatless [--refresh]
-> results/transfer_moatless.{json,md} (all metrics), results/gap2_moatless.{json,md} (H2, calibration, agent)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from experiments.governor_eval import agent, empty_rule, replay, reroute_over, run_metrics
from src import config
from src.baselines import tabular
from src.calibration import Calibrator
from src.data.moatless import download, iter_moatless
from src.eval import report
from src.eval.metrics import ece, evaluate_seeds, pooled_auroc
from src.features import extract, extract_task

METHODS = ["telemetry_lr", "telemetry_gbm", "telemetry20_lr", "telemetry20_gbm", "length_only", "empty_patch", "prompt_only"]
SETS = ["32b_8runs", "32b_zero", "7b_zero"]
K_TASKS = (3, 20)                # 3 tasks is about the proposal's "20-run calibration" with 8 attempts per task
PATHS = {True: config.DATA_DIR / "moatless_features.parquet", False: config.DATA_DIR / "moatless_features_raw.parquet"}
AGENT_CONFIGS = {"full": ({}, {}), "no_calibration": ({"calibration": False}, {}),
                 "no_early_warning": ({}, {"early_warning": False}),
                 "no_telemetry": ({"telemetry": False}, {"hints": False}), "no_hints": ({}, {"hints": False})}
OUT = config.RESULTS_DIR / "gap2_moatless"


def build(normalize: bool, refresh: bool) -> pd.DataFrame:
    path = PATHS[normalize]
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    download()
    rows = [extract(tr).model_dump() | extract_task(tr).model_dump() | {"set": name}
            for name, tr in iter_moatless(normalize=normalize)]
    df = pd.DataFrame(rows).assign(split="moatless", fold=-1)
    df.to_parquet(path, index=False)
    return df


def predict(dev: pd.DataFrame, target: pd.DataFrame, method: str) -> list[pd.DataFrame]:
    return [tabular.fit_predict(dev, target, method, s) for s in config.SEEDS]


def few_shot(dev: pd.DataFrame, target: pd.DataFrame) -> list[dict]:
    """Prior shift on telemetry20_lr: calibrator from dev out-of-fold, pass rate from K labelled tasks."""
    oof = tabular.cv_predict(dev, "telemetry20_lr")
    base = Calibrator("platt+prior").fit(oof["p_success"].to_numpy(), oof["label"].to_numpy(bool))
    p = tabular.fit_predict(dev, target, "telemetry20_lr", config.SEED)
    tasks = np.array(sorted(p["instance_id"].unique()))
    out = []
    for k in K_TASKS:
        rows = []
        for s in config.SEEDS:
            lab = set(np.random.default_rng(s).choice(tasks, size=k, replace=False))
            m = p["instance_id"].isin(lab).to_numpy()
            y, q = p["label"].to_numpy(bool)[~m], p["p_success"].to_numpy()[~m]
            base.pi_target = None
            platt_only = base.transform(q)
            base.pi_target = float(np.clip(p["label"].to_numpy(bool)[m].mean(), 0.01, 0.99))
            rows.append({"raw": ece(y, q), "platt_dev": ece(y, platt_only), "prior_shift": ece(y, base.transform(q)),
                         "pi_est": base.pi_target, "pi_true": float(y.mean())})
        base.pi_target = None
        r = pd.DataFrame(rows)
        out.append({"k_tasks": k, "ece_raw": r["raw"].mean(), "ece_platt_dev": r["platt_dev"].mean(),
                    "ece_prior_shift": r["prior_shift"].mean(), "ece_prior_shift_std": r["prior_shift"].std(),
                    "reduction": 1 - r["prior_shift"].mean() / r["raw"].mean(), "pi_true": r["pi_true"].mean(),
                    "pi_est_mean": r["pi_est"].mean()})
    return out


def agent_on_moatless(empty: dict) -> dict:
    normal = [t for _, t in iter_moatless(["32b_8runs"])]
    raw = [t for _, t in iter_moatless(["32b_8runs"], normalize=False)]
    tasks = sorted({t.instance_id for t in normal})
    y = pd.DataFrame({"instance_id": [t.instance_id for t in normal], "resolved": [t.resolved for t in normal]})
    configs = {**{k: (*v, True, normal) for k, v in AGENT_CONFIGS.items()}, "no_normalizer": ({}, {}, True, raw),
               "full_zero_shot": ({}, {}, False, normal)}
    res, kept = {}, None
    for name, (rc, gc, retarget, traces) in configs.items():
        print(f"  agent / {name}", flush=True)
        gov = agent(rc, gc, name)
        frames = []
        for s in config.SEEDS:
            lab = set(np.random.default_rng(s).choice(tasks, size=20, replace=False))
            if retarget:
                gov.models.retarget(float(y.loc[y["instance_id"].isin(lab), "resolved"].mean()))
            frames.append((s, replay(gov, [t for t in traces if t.instance_id not in lab])))
        rows = [run_metrics(df) for _, df in frames]
        res[name] = {"runs": {k: float(np.nanmean([r[k] for r in rows])) for k, v in rows[0].items() if isinstance(v, (int, float))},
                     "runs_std": {k: float(np.nanstd([r[k] for r in rows], ddof=1)) for k, v in rows[0].items() if isinstance(v, float)},
                     "reroute": reroute_over(frames)}
        if name == "full":
            kept = frames
    rule = [(s, empty_rule(df, empty)) for s, df in kept]
    rows = [run_metrics(df) for _, df in rule]
    res["empty_patch_rule"] = {"runs": {k: float(np.nanmean([r[k] for r in rows])) for k, v in rows[0].items() if isinstance(v, (int, float))},
                               "runs_std": {}, "reroute": reroute_over(rule)}
    return res


def to_markdown(h2: list[dict], fs: list[dict], ag: dict) -> str:
    def p(x):
        return f"{100 * x:.1f}%" if x is not None and not np.isnan(x) else "n/a"
    lines = ["Models fit on SWE-Gym dev (OpenHands) only, applied zero-shot. H2 target: drop <= 0.08 (>= 85 % retained) normalised, > 0.20 raw.", "",
             "### H2: pooled AUROC on 32b_8runs (2,398 runs) vs the references", "",
             "| method | SWE-Gym test | OpenHands Lite | Moatless normalised | Moatless raw | drop from test: norm / raw | retained: norm / raw | drop from OH Lite: norm / raw |",
             "|---|---|---|---|---|---|---|---|"]
    for r in h2:
        lines.append(f"| {r['method']} | {r['test']:.3f} | {r['oh_lite']:.3f} | {r['norm']:.3f} {r['norm_ci']} | {r['raw']:.3f} | "
                     f"{r['test'] - r['norm']:+.3f} / {r['test'] - r['raw']:+.3f} | {r['norm'] / r['test']:.0%} / {r['raw'] / r['test']:.0%} | "
                     f"{r['oh_lite'] - r['norm']:+.3f} / {r['oh_lite'] - r['raw']:+.3f} |")
    lines += ["", "Within-task on the same set (same model and configuration):", "",
              "| method | within-task normalised | within-task raw | Wilcoxon p (norm) |", "|---|---|---|---|"]
    for r in h2:
        lines.append(f"| {r['method']} | {r['norm_within']} | {r['raw_within']} | {r['norm_wilcoxon']} |")
    lines += ["", "### Few-shot prior shift, telemetry20_lr on 32b_8runs", "",
              "| K tasks | ECE raw | ECE Platt (dev) | ECE prior shift | reduction | pass rate true / estimated |", "|---|---|---|---|---|---|"]
    for r in fs:
        lines.append(f"| {r['k_tasks']} | {r['ece_raw']:.3f} | {r['ece_platt_dev']:.3f} | {r['ece_prior_shift']:.3f} ± {r['ece_prior_shift_std']:.3f} | "
                     f"{r['reduction']:.0%} | {r['pi_true']:.3f} / {r['pi_est_mean']:.3f} |")
    lines += ["", "### Locked governance agent on 32b_8runs (K = 20 labelled tasks; mean over seeds)", "",
              "| config | failures stopped | passes lost | passes aborted | reviewed | pass rate in review | tokens saved |", "|---|---|---|---|---|---|---|"]
    for name, r in ag.items():
        m = r["runs"]
        lines.append(f"| {name} | {p(m['failures_stopped'])} | {p(m['passes_lost'])} | {p(m['passes_aborted'])} | {p(m['reviewed'])} | "
                     f"{p(m['pass_rate_in_review'])} | {p(m['tokens_saved'])} |")
    lines += ["", "### Reroute on 32b_8runs, tasks with several attempts", "",
              "| config | policy | solved | tokens per task | reviews per task | solved per M tokens | solved per review |", "|---|---|---|---|---|---|---|"]
    for name in ("full", "no_normalizer", "empty_patch_rule"):
        for r in ag[name]["reroute"]:
            if r["subset"] != "multi_attempt_tasks" or (name != "full" and r["policy"] != "governor_reroute"):
                continue
            lines.append(f"| {name} | {r['policy']} | {100 * r['solved_mean']:.1f}% ± {100 * r['solved_std']:.1f} | {r['tokens_per_task_mean']:.0f} | "
                         f"{r['reviews_per_task_mean']:.2f} | {r['solved_per_million_tokens_mean']:.2f} | {r['solved_per_review_mean']:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    norm, raw = build(True, args.refresh), build(False, args.refresh)
    print(norm.groupby("set").agg(runs=("uid", "size"), tasks=("instance_id", "nunique"), resolved=("resolved", "mean"),
                                  empty=("patch_empty", "mean")).round(3).to_string(), flush=True)

    gym = pd.read_parquet(config.FEATURES_PATH).merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev, test = gym[gym["split"] == "dev"], gym[gym["split"] == "test"]
    lite = pd.read_parquet(config.DATA_DIR / "lite_features.parquet")

    reports, h2 = [], []
    for method in METHODS:
        print(f"  {method}", flush=True)
        p_norm, p_raw = predict(dev, norm, method), predict(dev, raw, method)
        for s in SETS:
            keep = set(norm.loc[norm["set"] == s, "uid"])
            reports.append(evaluate_seeds([x[x["uid"].isin(keep)] for x in p_norm], method, f"moatless_{s}"))
            reports.append(evaluate_seeds([x[x["uid"].isin(keep)] for x in p_raw], method, f"moatless_{s}_raw"))
        main_set = next(r for r in reports if r.method == method and r.eval_set == "moatless_32b_8runs")
        raw_set = next(r for r in reports if r.method == method and r.eval_set == "moatless_32b_8runs_raw")
        t = tabular.fit_predict(dev, test, method, config.SEED)
        l = tabular.fit_predict(dev, lite, method, config.SEED)
        ci = main_set.pooled_auroc_ci
        h2.append({"method": method, "test": pooled_auroc(t["label"].to_numpy(bool), t["p_success"].to_numpy()),
                   "oh_lite": pooled_auroc(l["label"].to_numpy(bool), l["p_success"].to_numpy()),
                   "norm": main_set.pooled_auroc, "norm_ci": f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "",
                   "raw": raw_set.pooled_auroc, "brier_norm": main_set.brier, "ece_norm": main_set.ece,
                   "norm_within": f"{main_set.within_task_auroc:.3f} {[round(x, 3) for x in main_set.within_task_auroc_ci]}" if main_set.within_task_auroc_ci else f"{main_set.within_task_auroc}",
                   "raw_within": f"{raw_set.within_task_auroc:.3f}" if raw_set.within_task_auroc is not None else "n/a",
                   "norm_wilcoxon": main_set.within_task_wilcoxon_p, "best_at_k": main_set.best_at_k})

    report.write_metrics(reports, {"seeds": list(config.SEEDS), "trained_on": "SWE-Gym dev runs (OpenHands)",
                                   "note": "_raw sets: scaffold normaliser off"}, name="transfer_moatless")
    fs = few_shot(dev, norm[norm["set"] == "32b_8runs"])
    empty = dict(zip(norm["uid"], norm["patch_empty"].astype(bool)))
    ag = agent_on_moatless(empty)
    OUT.with_suffix(".json").write_text(json.dumps({"h2": h2, "few_shot": fs, "agent": ag}, indent=1, default=float))
    md = to_markdown(h2, fs, ag)
    OUT.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()

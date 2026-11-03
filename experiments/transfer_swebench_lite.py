"""Transfer test: models trained on SWE-Gym dev runs, scored on SWE-bench Lite runs they never saw.

SWE-bench Lite (OpenHands' official eval outputs, see src/data/swebench_lite.py) shares no
repository with SWE-Gym, so this checks whether the predictors learned how runs fail in general or
habits of SWE-Gym's 11 repositories. Nothing is refit or tuned on Lite: each model is fit on the
same SWE-Gym dev runs as in run_baseline and only predicts here.

Evaluation sets:
  lite_all        all 4 runs (1,199): Claude 3.5 Sonnet v1.9 / v2.1 / v2.2 and GPT-4o v1.9
  lite_claude     the 3 Claude runs (899): one model, three OpenHands versions
  lite_claude_v2  Claude v2.1 + v2.2 (599): same tools as SWE-Gym, the closest match
  lite_gpt4o      GPT-4o (2024-05-13) v1.9 (300): one run per task, so no within-task AUROC

Also writes results/verifier_sample_lite.json: the runs of every task where the 3 Claude runs
disagree (pass and fail), v2.x runs first, for the 32B verifier on Kaggle.

    python -m experiments.transfer_swebench_lite [--refresh]
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import config
from src.baselines import tabular
from src.data import swebench_lite
from src.eval import report
from src.eval.metrics import evaluate_seeds
from src.features import extract, extract_task

FEATURES_PATH = config.DATA_DIR / "lite_features.parquet"
PREDS_PATH = config.DATA_DIR / "lite_predictions.parquet"
SAMPLE_PATH = config.RESULTS_DIR / "verifier_sample_lite.json"


def build_features(refresh: bool = False) -> pd.DataFrame:
    if FEATURES_PATH.exists() and not refresh:
        return pd.read_parquet(FEATURES_PATH)
    swebench_lite.download()
    rows = []
    for run, trace, _ in swebench_lite.iter_lite():
        _, policy, version = swebench_lite.RUNS[run]
        rows.append(extract(trace).model_dump() | extract_task(trace).model_dump()
                    | {"run": run, "policy_model": policy, "version": version})
    df = pd.DataFrame(rows).assign(split="lite", fold=-1)
    df.to_parquet(FEATURES_PATH, index=False)
    return df


def verifier_sample(lite: pd.DataFrame) -> list[str]:
    """Runs of tasks with mixed outcomes across the 3 Claude runs; v2.x first, then v1.9."""
    claude = lite[lite["run"].str.startswith("claude")]
    n = claude.groupby("instance_id")["resolved"].agg(["size", "sum"])
    mixed = n[(n["sum"] > 0) & (n["sum"] < n["size"])].index
    s = claude[claude["instance_id"].isin(mixed)]
    s = s.assign(v1=s["version"].eq("v1.9")).sort_values(["v1", "instance_id", "run"])
    return s["uid"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="rebuild Lite features")
    lite = build_features(ap.parse_args().refresh)

    gym = pd.read_parquet(config.FEATURES_PATH).merge(
        pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev = gym[gym["split"] == "dev"]

    preds = {m: [tabular.fit_predict(dev, lite, m, s) for s in config.SEEDS] for m in tabular.METHODS}
    sets = {
        "lite_all": lite["uid"],
        "lite_claude": lite.loc[lite["run"].str.startswith("claude"), "uid"],
        "lite_claude_v2": lite.loc[lite["run"].isin(["claude35_v2.1", "claude35_v2.2"]), "uid"],
        "lite_gpt4o": lite.loc[lite["run"].eq("gpt4o_v1.9"), "uid"],
    }
    reports = []
    for name, uids in sets.items():
        keep = set(uids)
        reports += [evaluate_seeds([p[p["uid"].isin(keep)] for p in ps], m, name) for m, ps in preds.items()]
    extra = {"seeds": list(config.SEEDS), "n_boot": config.N_BOOT, "trained_on": "SWE-Gym dev runs",
             "runs": {r: {"folder": f, "policy": p, "version": v} for r, (f, p, v) in swebench_lite.RUNS.items()},
             "resolved_rate_by_run": lite.groupby("run")["resolved"].mean().round(4).to_dict()}
    report.write_metrics(reports, extra, name="transfer_swebench_lite")
    pd.concat([ps[0] for ps in preds.values()]).to_parquet(PREDS_PATH, index=False)

    sample = verifier_sample(lite)
    SAMPLE_PATH.write_text(json.dumps({"description": "SWE-bench Lite runs for the 32B verifier: all runs of tasks "
                                                      "where the 3 Claude runs disagree; v2.x runs first",
                                       "n_runs": len(sample), "uids": sample}, indent=1))
    print((config.RESULTS_DIR / "transfer_swebench_lite.md").read_text(encoding="utf-8"))
    print(f"verifier sample: {len(sample)} runs -> {SAMPLE_PATH}")


if __name__ == "__main__":
    main()

"""E2: the SWE-Gym 32B verifier vs our predictors on the same SWE-bench Lite runs.

Inputs:
  data/verifier_scores_lite.csv     p(YES) per run from experiments/kaggle_verifier.ipynb
  data/lite_features.parquet        labels and task ids (experiments/transfer_swebench_lite.py)
  data/lite_predictions.parquet     our models, fit on SWE-Gym dev only (same script)

Every method is scored on exactly the runs the verifier scored. Sets:
  verifier_all          all scored runs (up to 237: every Claude run of the 79 tasks where the
                        3 Claude runs disagree)
  verifier_v2           v2.1 + v2.2 only: the same tools as the verifier's training data
  verifier_v19          v1.9 only: older scaffold, out of distribution for the verifier
  verifier_untruncated  runs that fit the 32k context with at most 2 turns dropped
  verifier_truncated    runs with more than 20 turns dropped to fit

    python -m experiments.compare_verifier_lite [--scores PATH] [--name NAME]
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.eval import report
from src.eval.metrics import evaluate_seeds, paired_bootstrap

SCORES_PATH = config.DATA_DIR / "verifier_scores_lite.csv"
VERIFIER = "swegym_verifier_32b"
TABULAR = ["telemetry_lr", "telemetry_gbm", "length_only", "empty_patch", "prompt_only"]


def load(scores_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (verifier rows with run info, all methods' Prediction rows on the scored runs)"""
    v = pd.read_csv(scores_path).drop_duplicates("uid", keep="last")
    lite = pd.read_parquet(config.DATA_DIR / "lite_features.parquet")[["uid", "instance_id", "resolved"]]
    v = v.merge(lite, on="uid", validate="1:1")
    ver = pd.DataFrame({"uid": v["uid"], "instance_id": v["instance_id"], "method": VERIFIER, "fold": -1,
                        "p_success": v["p_yes"], "label": v["resolved"].astype(bool)})
    ours = pd.read_parquet(config.DATA_DIR / "lite_predictions.parquet")
    ours = ours[ours["uid"].isin(set(v["uid"])) & ours["method"].isin(TABULAR)]
    return v, pd.concat([ver, ours], ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(SCORES_PATH))
    ap.add_argument("--name", default="verifier_vs_telemetry_lite")
    args = ap.parse_args()
    v, preds = load(args.scores)

    dropped = v["turns_dropped"]
    sets = {
        "verifier_all": v["uid"],
        "verifier_v2": v.loc[v["version"].isin(["v2.1", "v2.2"]), "uid"],
        "verifier_v19": v.loc[v["version"].eq("v1.9"), "uid"],
        "verifier_untruncated": v.loc[dropped <= 2, "uid"],
        "verifier_truncated": v.loc[dropped > 20, "uid"],
    }
    reports = []
    for name, uids in sets.items():
        keep = set(uids)
        if not keep:
            continue
        for method in [VERIFIER, *TABULAR]:
            p = preds[(preds["method"] == method) & preds["uid"].isin(keep)]
            reports.append(evaluate_seeds([p], method, name))

    # paired differences: verifier minus each of our predictors, same runs, same resampled tasks
    paired = {}
    for name in ("verifier_all", "verifier_v2"):
        keep = set(sets[name])
        ver = preds[(preds["method"] == VERIFIER) & preds["uid"].isin(keep)].reset_index(drop=True)
        for method in ("telemetry_lr", "telemetry_gbm", "length_only"):
            other = preds[(preds["method"] == method) & preds["uid"].isin(keep)]
            paired[f"{name}: verifier - {method}"] = paired_bootstrap(ver, other)

    extra = {
        "n_scored": len(v), "n_boot": config.N_BOOT,
        "note": "our models are fit on SWE-Gym dev only; nothing is fit on these runs",
        "paired_difference": {k: {m: dict(zip(("diff", "lo", "hi", "share_positive"), x)) if x else None
                                  for m, x in d.items()} for k, d in paired.items()},
        "verifier": {
            "model": "SWE-Gym/OpenHands-32B-Verifier, 4-bit GGUF Q4_K_M (mradermacher), llama.cpp, 2 x T4",
            "mean_p_yes_by_label": v.groupby("resolved")["p_yes"].mean().round(4).to_dict(),
            "share_p_yes_above_0.8": round(float((v["p_yes"] > 0.8).mean()), 4),
            "mean_seconds_per_run": round(float(v["seconds"].mean()), 1) if "seconds" in v else None,
            "mean_tokens": round(float(v["n_tokens"].mean())),
            "share_truncated_gt2_turns": round(float((dropped > 2).mean()), 4),
        },
    }
    report.write_metrics(reports, extra, name=args.name)
    print((config.RESULTS_DIR / f"{args.name}.md").read_text(encoding="utf-8"))
    print(extra["verifier"])
    for k, d in paired.items():
        print(k, {m: tuple(round(v, 3) for v in x) if x else None for m, x in d.items()})


if __name__ == "__main__":
    main()

"""E4 / Gap 3: which telemetry signals carry the end-of-run prediction? (Phase 1: H3, feature family ablation)

Exact linear SHAP for the agent's end-of-run model (telemetry20_lr, fit on SWE-Gym dev): for each run and feature,
coefficient x standardised feature value, in log-odds, with dev as the background. Reported as mean |SHAP| per
feature and per Phase 1 family, on each dataset, because a signal that matters on one framework may not on another.

H3 predicted that error-signal responsiveness, 4-gram test-edit loop counts and thought-action misalignment would
rank as the top predictors. Our loop measures are max_identical_streak and n_repeated_commands.

    python -m experiments.explain
-> results/shap_importance.{json,md}
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.transfer_moatless import PATHS
from src import config
from src.agent.risk import FINAL_EXT_COLS, _X
from src.baselines.tabular import _logreg

FAMILIES = {"volume_pacing": ["n_steps", "total_tokens", "first_edit_step"],
            "action_phase": ["n_distinct_tools", "phase_entropy"],
            "error_loop": ["n_errors", "n_repeated_commands", "max_identical_streak", "max_error_streak", "error_responsiveness"],
            "verification": ["ran_tests", "last_check_passed", "checked_after_last_edit", "fail_then_pass", "finished"],
            "cognitive": ["thought_action_mismatch", "n_edits_without_view"],
            "patch": ["patch_lines", "patch_files", "patch_empty"]}
H3_NAMED = ["error_responsiveness", "max_identical_streak", "n_repeated_commands", "thought_action_mismatch"]
OUT = config.RESULTS_DIR / "shap_importance"


def linear_shap(model, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    z = model[0].transform(_X(df, cols))
    return pd.DataFrame(z * model[-1].coef_[0], columns=cols, index=df.index)


def main() -> None:
    assert sorted(c for f in FAMILIES.values() for c in f) == sorted(FINAL_EXT_COLS), "every feature in exactly one family"
    gym = pd.read_parquet(config.FEATURES_PATH).merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev = gym[gym["split"] == "dev"]
    cols = FINAL_EXT_COLS
    model = _logreg(cols)(config.SEED).fit(_X(dev, cols), dev["resolved"].to_numpy(bool))
    moatless = pd.read_parquet(PATHS[True])
    sets = {"dev": dev, "test": gym[gym["split"] == "test"],
            "test_non_empty": gym[(gym["split"] == "test") & ~gym["patch_empty"].astype(bool)],
            "lite": pd.read_parquet(config.DATA_DIR / "lite_features.parquet"),
            "moatless_8runs": moatless[moatless["set"] == "32b_8runs"]}
    imp = pd.DataFrame({name: linear_shap(model, df, cols).abs().mean() for name, df in sets.items()})
    ranks = imp.rank(ascending=False).astype(int)
    coef = pd.Series(model[-1].coef_[0], index=cols)
    fam = pd.DataFrame({f: imp.loc[members].sum() / imp.sum() for f, members in FAMILIES.items()}).T

    t = imp.join(ranks.add_suffix("_rank")).assign(coef=coef).sort_values("test", ascending=False)
    OUT.with_suffix(".json").write_text(json.dumps({"features": t.reset_index(names="feature").to_dict("records"),
                                                    "family_share": fam.reset_index(names="family").to_dict("records"),
                                                    "h3_named": H3_NAMED}, indent=1, default=float))
    lines = ["Mean |SHAP| in log-odds, telemetry20_lr fit on SWE-Gym dev (exact linear SHAP). Rank 1 = most important of 20. "
             "coef: standardised coefficient; positive means higher values favour success.", "",
             "| feature | coef | test | rank: test | rank: test non-empty | rank: Lite | rank: Moatless |", "|---|---|---|---|---|---|---|"]
    for f, r in t.iterrows():
        mark = " (H3)" if f in H3_NAMED else ""
        lines.append(f"| {f}{mark} | {r['coef']:+.2f} | {r['test']:.3f} | {int(r['test_rank'])} | {int(r['test_non_empty_rank'])} | "
                     f"{int(r['lite_rank'])} | {int(r['moatless_8runs_rank'])} |")
    lines += ["", "### Share of total |SHAP| by family", "", "| family | " + " | ".join(fam.columns) + " |", "|---|" + "---|" * len(fam.columns)]
    lines += [f"| {f} | " + " | ".join(f"{100 * x:.1f}%" for x in r) + " |" for f, r in fam.iterrows()]
    OUT.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Failure analysis (spec 4.3): where does the locked agent go wrong, how often, and what do those runs look like?

Uses the saved E3 outcomes and logs. Nothing is tuned. Each wrong decision goes into exactly one category:
  missed failure (failing run sent to a reviewer)
    M1 looks verified   the agent's last check passed, after its last edit, and the code is still wrong
    M2 check not green  it checked after its last edit, but the last check failed or its exit code is unknown
    M3 not verified     no check after its last edit
  lost pass (passing run stopped)
    P1 aborted early    stopped at a checkpoint, before the patch exists
    P2 rejected         stopped at the end; grouped by the feature that lowered its score most
  hint false alarm      a passing run that received a hint

For each missed-failure category, the pass rate among all reviewed runs of the same kind shows whether a reviewer gets any
help from it. A pass rate close to the average means telemetry can't separate these runs.

Representative cases follow fixed rules, not hand-picking: the run nearest the median final score of its group, among runs of
at most MAX_ACTIONS actions so the trajectory can be printed. Their full logs are copied to logs/sample/.

    python -m experiments.failure_analysis
-> results/failure_analysis.{json,md}, results/failure_cases.md, logs/sample/
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from experiments.transfer_moatless import PATHS
from src import config
from src.agent.cli import LOG_DIR, _safe
from src.agent.risk import FINAL_EXT_COLS, _fit_one, RiskConfig
from src.baselines.tabular import _logreg

MAX_ACTIONS = 25
SAMPLE = LOG_DIR / "sample"
OUT = config.RESULTS_DIR / "failure_analysis"


def missed_category(r) -> str:
    if r["checked_after_last_edit"]:
        return "M1 looks verified" if r["last_check_passed"] else "M2 check not green"
    return "M3 not verified"


def lowest_factor(model, row: pd.Series, cols: list[str]) -> str:
    x = pd.DataFrame([{c: float(row[c]) for c in cols}])
    phi = model[0].transform(x)[0] * model[-1].coef_[0]
    return cols[int(np.argmin(phi))]


def analyse(out: pd.DataFrame, feats: pd.DataFrame, model) -> dict:
    d = out.merge(feats, on="uid", suffixes=("", "_f"))
    y = d["resolved"].to_numpy(bool)
    reviewed = d["decision"] == "FLAG_FOR_REVIEW"
    rev = d[reviewed].assign(category=lambda x: x.apply(missed_category, axis=1))
    missed = rev[~rev["resolved"]]
    rows = []
    for c, g in rev.groupby("category"):
        m = g[~g["resolved"]]
        rows.append({"category": c, "missed_failures": len(m), "share_of_missed": len(m) / max(len(missed), 1), "reviewed_runs_of_this_kind": len(g),
                     "pass_rate_of_this_kind": float(g["resolved"].mean()), "median_actions": float(m["n_steps"].median()) if len(m) else None,
                     "median_patch_lines": float(m["patch_lines"].median()) if len(m) else None})
    lost = d[y & d["decision"].isin(["ABORT", "REJECT"])]
    rej = lost[lost["decision"] == "REJECT"]
    cols = FINAL_EXT_COLS
    factors = rej.apply(lambda r: lowest_factor(model, r, cols), axis=1).value_counts().to_dict() if len(rej) else {}
    hinted = d[(d["n_hints"] > 0)]
    return {"runs": len(d), "correct": {"failures_stopped": int((~y & d["decision"].isin(["ABORT", "REJECT"])).sum()), "passes_reviewed": int((y & reviewed).sum())},
            "wrong": {"missed_failures": len(missed), "lost_passes": len(lost), "aborted_passes": int((lost["decision"] == "ABORT").sum()), "rejected_passes": len(rej)},
            "reviewed_pass_rate": float(rev["resolved"].mean()) if len(rev) else None, "missed_categories": rows, "rejected_pass_lowest_factor": factors,
            "abort_steps_of_lost_passes": lost.loc[lost["decision"] == "ABORT", "abort_step"].value_counts().to_dict(),
            "hint_false_alarms": {"passing_runs_hinted": int(hinted["resolved"].sum()), "hinted_runs": len(hinted),
                                  "by_pattern_on_passing": hinted[hinted["resolved"]]["hint_patterns"].fillna("").str.split(",").explode().value_counts().to_dict()}}


def pick(df: pd.DataFrame) -> pd.Series:
    """Nearest the group's median final score, among short enough runs; ties broken by uid."""
    s = df[df["steps_total"] <= MAX_ACTIONS] if (df["steps_total"] <= MAX_ACTIONS).any() else df
    s = s.dropna(subset=["p_final"]) if s["p_final"].notna().any() else s
    target = s["p_final"].median() if s["p_final"].notna().any() else 0
    return s.assign(gap=(s["p_final"].fillna(target) - target).abs()).sort_values(["gap", "uid"]).iloc[0]


def render(uid: str, log_dir) -> list[str]:
    lines = []
    for raw in (log_dir / f"{_safe(uid)}.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(raw)
        if "decision" not in e:
            continue
        ob = e.get("observed") or {}
        what = (ob.get("path") or ob.get("command") or ob.get("tool") or "") if ob else "(end of run)"
        what = str(what).splitlines()[0][:60] if what else ""
        p = f"{e['p_success']:.3f}" if e.get("p_success") is not None else "-"
        flag = " error" if ob.get("is_error") else ""
        lines.append(f"{e['step']:>3} {e['phase_to'] or '-':<9} {p:>6} {e['decision']:<15} {ob.get('action', ''):<7} {what}{flag}")
        for c in e.get("tool_calls", []):
            if c["tool"] == "score_final_risk" and c.get("output") and c["output"].get("top_factors"):
                lines.append("    top factors: " + ", ".join(f"{f['feature']} {f['effect']:+.2f}" for f in c["output"]["top_factors"]))
        if e.get("reason"):
            lines.append(f"    -> {e['reason']}")
        if e.get("hint"):
            lines.append(f"    hint: {e['hint']}")
    return lines


def main() -> None:
    gym = pd.read_parquet(config.FEATURES_PATH).merge(pd.read_csv(config.SPLITS_PATH)[["uid", "split", "fold"]], on="uid", validate="1:1")
    dev = gym[gym["split"] == "dev"]
    keep = ["uid", "n_steps", "patch_lines", "patch_files", "patch_empty", "checked_after_last_edit", "last_check_passed", "first_edit_step", "phase_entropy"]
    feat_cols = list(dict.fromkeys(keep + FINAL_EXT_COLS))
    model = _fit_one(_logreg(FINAL_EXT_COLS), FINAL_EXT_COLS, dev, RiskConfig().far_review, RiskConfig()).model

    test_out = pd.read_csv(config.RESULTS_DIR / "governor_outcomes_test.csv")
    lite_out = pd.read_csv(config.RESULTS_DIR / "governor_outcomes_lite.csv")
    res = {"test": analyse(test_out, gym[feat_cols], model),
           "lite": analyse(lite_out, pd.read_parquet(config.DATA_DIR / "lite_features.parquet")[feat_cols], model)}
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1, default=float))

    d = test_out.merge(gym[feat_cols], on="uid", suffixes=("", "_f"))
    rev = d[d["decision"] == "FLAG_FOR_REVIEW"].assign(category=lambda x: x.apply(missed_category, axis=1))
    largest = max(res["test"]["missed_categories"], key=lambda r: r["missed_failures"])["category"]
    cases = {"Success 1: failing run with a real patch, correctly rejected": d[(d["decision"] == "REJECT") & ~d["resolved"] & ~d["patch_empty"].astype(bool)],
             "Success 2: passing run, correctly sent to a reviewer": d[(d["decision"] == "FLAG_FOR_REVIEW") & d["resolved"]],
             "Success 3: failing run, correctly aborted early": d[(d["decision"] == "ABORT") & ~d["resolved"]],
             f"Failure 1: missed failure, largest category ({largest})": rev[~rev["resolved"] & (rev["category"] == largest)],
             "Failure 2: passing run wrongly rejected": d[(d["decision"] == "REJECT") & d["resolved"]],
             "Failure 3: passing run wrongly aborted": d[(d["decision"] == "ABORT") & d["resolved"]]}
    log_dir = LOG_DIR / "e3_test_full"
    SAMPLE.mkdir(parents=True, exist_ok=True)
    out_lines = ["Representative cases from the SWE-Gym test run. Each is the run nearest its group's median final score, among runs of at most "
                 f"{MAX_ACTIONS} actions (fixed rule, not hand-picked). Columns: actions so far, phase, p(resolved), decision, action, target. "
                 "Full logs: logs/sample/.", ""]
    for title, group in cases.items():
        r = pick(group)
        shutil.copy(log_dir / f"{_safe(r['uid'])}.jsonl", SAMPLE / f"test_{_safe(r['uid'])}.jsonl")
        verdict = "resolved" if r["resolved"] else "failed"
        out_lines += [f"### {title}", "", f"Run `{r['uid']}`, task `{r['instance_id']}`: {r['decision']} after {r['steps_seen']}/{r['steps_total']} actions; actually {verdict}. "
                      f"{int(r['patch_lines'])} patch lines.", "", "```"] + render(r["uid"], log_dir) + ["```", ""]
    (config.RESULTS_DIR / "failure_cases.md").write_text("\n".join(out_lines), encoding="utf-8")

    lite_dir = LOG_DIR / "e3_lite_full"
    lite = lite_out.merge(pd.read_parquet(config.DATA_DIR / "lite_features.parquet")[["uid", "patch_empty"]], on="uid")
    for group in (lite[(lite["decision"] == "FLAG_FOR_REVIEW") & lite["resolved"]], lite[(lite["decision"] == "REJECT") & ~lite["resolved"]]):
        r = pick(group)
        shutil.copy(lite_dir / f"{_safe(r['uid'])}.jsonl", SAMPLE / f"lite_{_safe(r['uid'])}.jsonl")
    (SAMPLE / "README.md").write_text(
        "# Sample agent logs\n\nOne JSON line per agent action (what it observed, tool calls with inputs and outputs, phase change, "
        "decision, reason), then the run's outcome. Runs were chosen by the fixed rule in experiments/failure_analysis.py. "
        "test_* are SWE-Gym held-out runs, lite_* are SWE-bench Lite runs. All logs are produced by "
        "`python -m experiments.governor_eval` into logs/e3_*/ (not committed; about 30 MB). results/failure_cases.md prints the test runs readably.\n",
        encoding="utf-8")

    md = ["Wrong decisions of the locked agent, from the saved E3 outcomes. Lite: seed 42 draw. Categories defined in experiments/failure_analysis.py.", ""]
    for name, r in res.items():
        w, c = r["wrong"], r["correct"]
        md += [f"### {name}: {r['runs']} runs", "", f"Correct: {c['failures_stopped']} failures stopped, {c['passes_reviewed']} passes sent to review. "
               f"Wrong: {w['missed_failures']} missed failures, {w['lost_passes']} lost passes ({w['aborted_passes']} aborted, {w['rejected_passes']} rejected). "
               f"Pass rate of all reviewed runs: {100 * r['reviewed_pass_rate']:.1f} %.", "",
               "| category | missed failures | share | reviewed runs of this kind | their pass rate | median actions | median patch lines |", "|---|---|---|---|---|---|---|"]
        md += [f"| {x['category']} | {x['missed_failures']} | {100 * x['share_of_missed']:.1f}% | {x['reviewed_runs_of_this_kind']} | {100 * x['pass_rate_of_this_kind']:.1f}% | {x['median_actions']} | {x['median_patch_lines']} |"
               for x in r["missed_categories"]]
        md += ["", f"Rejected passes, feature that lowered the score most: {r['rejected_pass_lowest_factor']}. Aborted passes, by checkpoint: {r['abort_steps_of_lost_passes']}.",
               f"Hint false alarms: {r['hint_false_alarms']['passing_runs_hinted']} passing runs hinted, of {r['hint_false_alarms']['hinted_runs']} hinted; by pattern {r['hint_false_alarms']['by_pattern_on_passing']}.", ""]
    OUT.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

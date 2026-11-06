"""E3 (full system) and E5 (ablation): the locked governance agent on data it never saw.

Models and thresholds come from SWE-Gym dev only (src/agent/risk.py); nothing here tunes anything.
  test  SWE-Gym held-out test. One agent per configuration. Decisions are deterministic, so seeds only
        change the order of attempts in the reroute simulation.
  lite  SWE-bench Lite, unseen repositories. K labelled tasks give the new pass rate (Gap 1 prior shift);
        the agent is scored on the other tasks. Seeds change which K tasks are labelled and the order.

Configurations (each Phase 1 gap's component switched off, one at a time):
  full             everything on
  no_calibration   Gap 1: raw scores. No effect expected on test (same pass rate); the thresholds are
                   probabilities, so it should matter on Lite
  no_normalizer    Gap 2: v1.9 runs keep their raw IPython calls. Lite only
  no_early_warning Gap 3: no checkpoint scoring, no aborts
  no_telemetry     Gap 3: only length signals; no hints
  no_hints         hints off; decisions unchanged by construction, shown as a check
  full_zero_shot   Lite only: calibration on but no labelled tasks

Reported per configuration: failures stopped, passing runs aborted / rejected, share reviewed and pass rate
there, tokens saved, runs hinted, reroute results, with 95 % task-level bootstrap intervals. The rule
"reject if the patch is empty" is the comparison the agent has to beat.

    python -m experiments.governor_eval            # uses the test set: run once
    python -m experiments.governor_eval --smoke    # dev runs in place of test, small Lite slice, for debugging
-> results/governor_eval.{json,md}, results/governor_outcomes_{test,lite}.csv, logs/e3_*/ (JSONL per run)
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.agent.cli import LOG_DIR, replay_one
from src.agent.governor import Governor, GovernorConfig
from src.agent.reroute import simulate, summarize
from src.agent.risk import RiskConfig, RiskModels
from src.data.loader import load_meta
from src.data.parser import iter_traces
from src.data.swebench_lite import iter_lite
from src.eval.metrics import pooled_auroc
from src.schema import Decision, Trace

K_LABELLED = 20
STOPPED = [Decision.abort.value, Decision.reject.value]
CONFIGS = {"full": ({}, {}),
           "no_calibration": ({"calibration": False}, {}),
           "no_early_warning": ({}, {"early_warning": False}),
           "no_telemetry": ({"telemetry": False}, {"hints": False}),
           "no_hints": ({}, {"hints": False})}
LITE_FEATURES = config.DATA_DIR / "lite_features.parquet"


def replay(gov: Governor, traces: list[Trace], log_dir: Optional[Path] = None) -> pd.DataFrame:
    rows = []
    for tr in traces:
        o = replay_one(gov, tr, log_dir, verbose=False)
        rows.append({"uid": o.uid, "instance_id": o.instance_id, "decision": o.final_decision.value,
                     "tokens_used": o.tokens_used, "tokens_total": o.tokens_total, "resolved": bool(o.resolved),
                     "steps_seen": o.steps_seen, "steps_total": o.steps_total, "abort_step": o.abort_step,
                     "p_final": o.p_final, "n_hints": len(o.hints), "hint_patterns": ",".join(h["pattern"] for h in o.hints),
                     "patterns_seen": ",".join(o.patterns_seen), "degraded": o.degraded_calls})
    return pd.DataFrame(rows)


def run_metrics(df: pd.DataFrame, n_boot: int = 0, seed: int = config.SEED) -> dict:
    """Run-level results. With n_boot, adds 95 % intervals from resampling whole tasks."""
    y = df["resolved"].to_numpy(bool)
    stop = df["decision"].isin(STOPPED).to_numpy()
    abort = (df["decision"] == Decision.abort.value).to_numpy()
    used, total = df["tokens_used"].to_numpy(float), df["tokens_total"].to_numpy(float)
    task, tasks = pd.factorize(df["instance_id"])

    def per_task(mask: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        return np.bincount(task, weights=mask.astype(float) if weights is None else mask * weights, minlength=len(tasks))
    parts = {"fail": per_task(~y), "fail_stopped": per_task(~y & stop), "pass": per_task(y),
             "pass_aborted": per_task(y & abort), "pass_stopped": per_task(y & stop), "reviewed": per_task(~stop),
             "reviewed_pass": per_task(~stop & y), "used": per_task(np.ones(len(y), bool), used),
             "total": per_task(np.ones(len(y), bool), total), "n": per_task(np.ones(len(y), bool))}

    def stats(c: dict) -> dict:
        safe = lambda a, b: float(a / b) if b else float("nan")
        return {"failures_stopped": safe(c["fail_stopped"], c["fail"]), "passes_aborted": safe(c["pass_aborted"], c["pass"]),
                "passes_lost": safe(c["pass_stopped"], c["pass"]), "reviewed": safe(c["reviewed"], c["n"]),
                "pass_rate_in_review": safe(c["reviewed_pass"], c["reviewed"]), "tokens_saved": safe(c["total"] - c["used"], c["total"])}
    out = stats({k: v.sum() for k, v in parts.items()})
    out |= {"runs": len(df), "tasks": len(tasks), "n_pass": int(y.sum()), "base_rate": float(y.mean()),
            "n_aborted": int(abort.sum()), "n_rejected": int((df["decision"] == Decision.reject.value).sum()),
            "degraded_calls": int(df["degraded"].sum()) if "degraded" in df else 0}
    if "n_hints" in df:
        hinted = (df["n_hints"] > 0).to_numpy()
        out |= {"runs_hinted": float(hinted.mean()),
                "pass_rate_hinted": float(y[hinted].mean()) if hinted.any() else None,
                "pass_rate_not_hinted": float(y[~hinted].mean()) if (~hinted).any() else None}
    if "p_final" in df and df["p_final"].notna().any():
        m = df["p_final"].notna().to_numpy()
        out["final_auroc_on_finished"] = pooled_auroc(y[m], df["p_final"].to_numpy(float)[m])
    if n_boot:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
            draws.append(stats({k: v @ w for k, v in parts.items()}))
        for k in draws[0]:
            lo, hi = np.nanquantile([d[k] for d in draws], [0.025, 0.975])
            out[k + "_ci"] = (float(lo), float(hi))
    return out


def empty_rule(df: pd.DataFrame, empty: dict) -> pd.DataFrame:
    """The comparison: reject exactly the runs with an empty patch, abort nothing, save no tokens."""
    e = df["uid"].map(empty).to_numpy(bool)
    return df.assign(decision=np.where(e, Decision.reject.value, Decision.flag_for_review.value),
                     tokens_used=df["tokens_total"]).drop(columns=["n_hints", "p_final", "degraded"], errors="ignore")


def on_non_empty(df: pd.DataFrame, empty: dict) -> dict:
    """Does the agent add anything where the empty-patch rule says nothing?"""
    return run_metrics(df[~df["uid"].map(empty).to_numpy(bool)], n_boot=config.N_BOOT)


def reroute_over(frames: list[tuple[int, pd.DataFrame]]) -> list[dict]:
    """frames: (seed, runs). Mean and std across seeds, on all tasks and on tasks with >= 2 attempts."""
    out = []
    for subset in ("all_tasks", "multi_attempt_tasks"):
        per_seed = []
        for s, runs in frames:
            sub = runs if subset == "all_tasks" else runs[runs.groupby("instance_id")["uid"].transform("size") >= 2]
            per_seed.append(summarize(simulate(sub, s)).assign(seed=s))
        all_ = pd.concat(per_seed)
        cols = ["tasks", "solved", "tokens_per_task", "reviews_per_task", "solved_per_million_tokens", "solved_per_review"]
        agg = all_.groupby("policy", sort=False)[cols].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        out += agg.reset_index().assign(subset=subset).to_dict("records")
    return out


def agent(rc: dict, gc: dict, name: str) -> Governor:
    return Governor(RiskModels.fit(RiskConfig(**rc)), GovernorConfig(name=name, **gc))


def eval_test(traces: list[Trace], empty: dict, out_dir: Path, log: bool) -> dict:
    res = {}
    outcomes = None
    for name, (rc, gc) in CONFIGS.items():
        print(f"  test / {name}", flush=True)
        df = replay(agent(rc, gc, name), traces, LOG_DIR / "e3_test_full" if (log and name == "full") else None)
        res[name] = {"runs": run_metrics(df, n_boot=config.N_BOOT),
                     "reroute": reroute_over([(s, df) for s in config.SEEDS])}
        if name == "full":
            outcomes = df
            res[name]["non_empty_patch"] = on_non_empty(df, empty)
            res[name]["share_of_stops_empty"] = float(df.loc[df["decision"].isin(STOPPED), "uid"].map(empty).mean())
    rule = empty_rule(outcomes, empty)
    res["empty_patch_rule"] = {"runs": run_metrics(rule, n_boot=config.N_BOOT),
                               "reroute": reroute_over([(s, rule) for s in config.SEEDS])}
    outcomes.to_csv(out_dir / "governor_outcomes_test.csv", index=False)
    return res


def eval_lite(normal: list[Trace], raw: list[Trace], versions: dict, empty: dict, out_dir: Path, log: bool) -> dict:
    configs = {**{k: (v[0], v[1], True, normal) for k, v in CONFIGS.items()},
               "full_zero_shot": ({}, {}, False, normal), "no_normalizer": ({}, {}, True, raw)}
    y_by_task = pd.DataFrame({"instance_id": [t.instance_id for t in normal], "resolved": [t.resolved for t in normal]})
    all_tasks = sorted(y_by_task["instance_id"].unique())
    res, kept = {}, None
    for name, (rc, gc, retarget, traces) in configs.items():
        print(f"  lite / {name}", flush=True)
        gov = agent(rc, gc, name)
        frames = []
        for s in config.SEEDS:
            labelled = set(np.random.default_rng(s).choice(all_tasks, size=min(K_LABELLED, len(all_tasks) // 2), replace=False))
            if retarget:
                gov.models.retarget(float(y_by_task.loc[y_by_task["instance_id"].isin(labelled), "resolved"].mean()))
            keep = [t for t in traces if t.instance_id not in labelled]
            write = log and name == "full" and s == config.SEEDS[0]
            frames.append((s, replay(gov, keep, LOG_DIR / "e3_lite_full" if write else None)))
        subsets = {"all": lambda df: df, "v1.9": lambda df: df[df["uid"].map(versions) == "v1.9"],
                   "v2": lambda df: df[df["uid"].map(versions) != "v1.9"]}
        res[name] = {"reroute": reroute_over(frames)}
        for sname, pick in subsets.items():
            rows = [run_metrics(pick(df)) for _, df in frames]
            res[name][sname] = {k: float(np.nanmean([r[k] for r in rows])) for k, v in rows[0].items()
                                if isinstance(v, (int, float))}
            res[name][sname] |= {k + "_std": float(np.nanstd([r[k] for r in rows], ddof=1)) for k, v in rows[0].items()
                                 if isinstance(v, float)}
        if name == "full":
            kept = frames
    frames = [(s, empty_rule(df, empty)) for s, df in kept]
    rows = [run_metrics(df) for _, df in frames]
    res["empty_patch_rule"] = {"all": {k: float(np.nanmean([r[k] for r in rows])) for k, v in rows[0].items()
                                       if isinstance(v, (int, float))},
                               "reroute": reroute_over(frames)}
    kept[0][1].to_csv(out_dir / "governor_outcomes_lite.csv", index=False)
    return res


def to_markdown(results: dict, seeds, commit: str) -> str:
    def pct(x):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"

    def ci(r, k):
        c = r.get(k + "_ci")
        return f"{pct(r.get(k))} [{100 * c[0]:.1f}, {100 * c[1]:.1f}]" if c else pct(r.get(k))

    def sd(r, k):
        s = r.get(k + "_std")
        return f"{pct(r.get(k))} ± {100 * s:.1f}" if s else pct(r.get(k))
    lines = [f"Governance agent at commit {commit}. Models and thresholds from SWE-Gym dev only. Brackets: 95 % task-level "
             f"bootstrap. ± : std over seeds {list(seeds)}. Lite: K = {K_LABELLED} labelled tasks, scored on the others.", ""]
    if "test" in results:
        lines += ["### SWE-Gym test: runs", "", "| config | failures stopped | passes lost | passes aborted | reviewed | pass rate in review | tokens saved | runs hinted |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in results["test"].items():
            m = r["runs"]
            lines.append(f"| {name} | {ci(m, 'failures_stopped')} | {ci(m, 'passes_lost')} | {ci(m, 'passes_aborted')} | {ci(m, 'reviewed')} | "
                         f"{ci(m, 'pass_rate_in_review')} | {ci(m, 'tokens_saved')} | {pct(m.get('runs_hinted'))} |")
        f = results["test"]["full"]
        n = f["non_empty_patch"]
        lines += ["", f"full, on non-empty patches only ({n['runs']} runs, {n['n_pass']} passing): failures stopped {ci(n, 'failures_stopped')}, "
                      f"passes lost {ci(n, 'passes_lost')}. Share of the agent's stops that are empty patches: {pct(f['share_of_stops_empty'])}.",
                  f"Pass rate of runs hinted vs not: {pct(f['runs'].get('pass_rate_hinted'))} vs {pct(f['runs'].get('pass_rate_not_hinted'))}. "
                  f"Base rate {pct(f['runs']['base_rate'])}; {f['runs']['n_pass']} passing runs of {f['runs']['runs']}. Failed tool calls: {f['runs']['degraded_calls']}.", ""]
    if "lite" in results:
        lines += ["### SWE-bench Lite: runs", "", "| config | subset | failures stopped | passes lost | passes aborted | reviewed | pass rate in review | tokens saved |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, r in results["lite"].items():
            for sub in ("all", "v1.9", "v2"):
                if sub in r:
                    m = r[sub]
                    lines.append(f"| {name} | {sub} | {sd(m, 'failures_stopped')} | {sd(m, 'passes_lost')} | {sd(m, 'passes_aborted')} | "
                                 f"{sd(m, 'reviewed')} | {sd(m, 'pass_rate_in_review')} | {sd(m, 'tokens_saved')} |")
        lines.append("")
    for ds in ("test", "lite"):
        if ds not in results:
            continue
        lines += [f"### Reroute on {ds}: tasks solved and cost per task", "",
                  "| config | subset | policy | tasks | solved | tokens per task | reviews per task | solved per M tokens | solved per review |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name in ("full", "empty_patch_rule"):
            for r in results[ds][name]["reroute"]:
                if name == "empty_patch_rule" and r["policy"] not in ("governor_single", "governor_reroute"):
                    continue
                lines.append(f"| {name} | {r['subset']} | {r['policy']} | {r['tasks_mean']:.0f} | {100 * r['solved_mean']:.1f}% ± {100 * r['solved_std']:.1f} | "
                             f"{r['tokens_per_task_mean']:.0f} | {r['reviews_per_task_mean']:.2f} | {r['solved_per_million_tokens_mean']:.2f} ± {r['solved_per_million_tokens_std']:.2f} | "
                             f"{r['solved_per_review_mean']:.3f} ± {r['solved_per_review_std']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="dev runs in place of test, small Lite slice, no logs, scratch output")
    ap.add_argument("--out", help="output directory (default results/)")
    ap.add_argument("--only", choices=["test", "lite"])
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else config.RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log = not args.smoke

    results, seeds = {}, config.SEEDS
    if args.only != "lite":
        splits = pd.read_csv(config.SPLITS_PATH)
        which = "dev" if args.smoke else "test"
        uids = set(splits.loc[splits["split"] == which, "uid"])
        if args.smoke:
            uids = set(sorted(uids)[:150])
        feats = pd.read_parquet(config.FEATURES_PATH, columns=["uid", "patch_empty"])
        empty = dict(zip(feats["uid"], feats["patch_empty"].astype(bool)))
        traces = list(iter_traces(load_meta(), uids=uids))
        print(f"{which}: {len(traces)} runs", flush=True)
        results["test"] = eval_test(traces, empty, out_dir, log)
    if args.only != "test":
        lf = pd.read_parquet(LITE_FEATURES, columns=["uid", "patch_empty", "version"])
        normal = [t for _, t, _ in iter_lite()]
        raw = [t for _, t, _ in iter_lite(normalize=False)]
        if args.smoke:
            keep = set(sorted({t.instance_id for t in normal})[:40])
            normal, raw = [t for t in normal if t.instance_id in keep], [t for t in raw if t.instance_id in keep]
        print(f"lite: {len(normal)} runs", flush=True)
        results["lite"] = eval_lite(normal, raw, dict(zip(lf["uid"], lf["version"])),
                                    dict(zip(lf["uid"], lf["patch_empty"].astype(bool))), out_dir, log)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    (out_dir / "governor_eval.json").write_text(json.dumps({"commit": commit, "seeds": list(seeds), "k_labelled": K_LABELLED,
                                                            "results": results}, indent=1, default=float))
    md = to_markdown(results, seeds, commit)
    (out_dir / "governor_eval.md").write_text(md, encoding="utf-8")
    if args.smoke:
        print("smoke OK:", {k: list(v) for k, v in results.items()})
    else:
        print(md)


if __name__ == "__main__":
    main()

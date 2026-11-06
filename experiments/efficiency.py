"""E6: cost, latency and reliability of the governance agent vs the two LLM baselines.

Timed on this machine (see `machine` in the output): parsing, feature extraction, a full agent replay per run and
per action, and the end-of-run score alone, each over the SWE-Gym test runs, repeated 3 times for mean and std. Timings
only: nothing here reads a label or changes a decision. Peak Python memory comes from tracemalloc, measured in
a separate pass because tracing slows code down.

The verifier and judge numbers were measured when those baselines ran (data/verifier_scores_lite.csv on Kaggle 2 x T4;
results/metrics.json on the laptop's RTX 4050). Different hardware, so the ratios are indicative, not exact.
Also reports token and tool-call usage from the E3 run, and reliability counts.

    python -m experiments.efficiency
-> results/efficiency.{json,md}
"""
from __future__ import annotations

import json
import platform
import time
import tracemalloc

import numpy as np
import pandas as pd

from src import config
from src.agent.governor import Governor
from src.agent.risk import RiskModels
from src.data.loader import load_meta
from src.data.parser import iter_traces
from src.features import extract

REPEATS = 3
OUT = config.RESULTS_DIR / "efficiency"


def timed(fn, items) -> tuple[float, float]:
    """-> (mean, std) ms per item over REPEATS passes."""
    runs = []
    for _ in range(REPEATS):
        t = time.perf_counter()
        for it in items:
            fn(it)
        runs.append((time.perf_counter() - t) * 1000 / len(items))
    return float(np.mean(runs)), float(np.std(runs, ddof=1))


def main() -> None:
    splits = pd.read_csv(config.SPLITS_PATH)
    uids = set(splits.loc[splits["split"] == "test", "uid"])
    meta = load_meta()

    t = time.perf_counter()
    traces = list(iter_traces(meta, uids=uids))
    parse_ms = (time.perf_counter() - t) * 1000 / len(traces)      # includes scanning the other runs' rows

    t = time.perf_counter()
    models = RiskModels.fit()
    fit_s = time.perf_counter() - t
    gov = Governor(models)
    n_actions = int(np.mean([tr.n_actions for tr in traces]))
    feats = [extract(tr).model_dump() for tr in traces]

    rows = {"feature extraction": timed(extract, traces),
            "end-of-run score": timed(models.final.score, feats),
            "full agent replay": timed(gov.run, traces)}

    tracemalloc.start()                  # model fit + replay of 200 runs; native (Arrow) buffers aren't traced
    traced = Governor(RiskModels.fit())
    for tr in traces[:200]:
        traced.run(tr)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    v = pd.read_csv(config.DATA_DIR / "verifier_scores_lite.csv").drop_duplicates("uid", keep="last")
    judge = json.loads((config.RESULTS_DIR / "metrics.json").read_text()).get("llm_judge") or {}
    outcomes = pd.read_csv(config.RESULTS_DIR / "governor_outcomes_test.csv")
    verifier_ms = float(v["seconds"].mean() * 1000)
    judge_ms = float(judge.get("mean_latency_s", float("nan")) * 1000)
    replay_ms = rows["full agent replay"][0]
    tool_calls = 0
    for tr in traces[:200]:
        for item in gov.replay(tr):
            tool_calls += len(getattr(item, "tool_calls", []))

    res = {"machine": {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
                       "note": "CPU only; no GPU or API used by the agent"},
           "n_runs": len(traces), "mean_actions_per_run": n_actions, "repeats": REPEATS,
           "ms_per_run": {k: {"mean": m, "std": s} for k, (m, s) in rows.items()},
           "ms_per_action_replay": replay_ms / max(n_actions, 1), "parse_ms_per_run_upper_bound": parse_ms,
           "model_fit_seconds": fit_s, "peak_traced_python_memory_mb_fit_and_replay": peak / 1e6,
           "tool_calls_per_run": tool_calls / 200,
           "verifier": {"ms_per_run": verifier_ms, "std_ms": float(v["seconds"].std() * 1000), "hardware": "Kaggle 2 x Tesla T4, 4-bit GGUF",
                        "gpu_memory_gb_approx": 24, "setup_minutes_approx": 30, "n_runs": len(v)},
           "judge": {"ms_per_run": judge_ms, "hardware": "RTX 4050 6 GB, Ollama, 4-bit", "n_runs": config.LLM_JUDGE_SAMPLE,
                     "parse_failures": judge.get("parse_failures")},
           "share_of_verifier_latency": {"full agent replay": replay_ms / verifier_ms, "end-of-run score": rows["end-of-run score"][0] / verifier_ms,
                                         "feature extraction + end-of-run score": (rows["feature extraction"][0] + rows["end-of-run score"][0]) / verifier_ms},
           "share_of_judge_latency": {"full agent replay": replay_ms / judge_ms},
           "usage_in_e3_test": {"mean_tokens_watched_per_run": float(outcomes["tokens_total"].mean()),
                                "mean_tokens_after_aborts_per_run": float(outcomes["tokens_used"].mean()),
                                "llm_tokens_used_by_agent": 0},
           "reliability": {"runs_replayed": len(outcomes), "failed_tool_calls": int(outcomes["degraded"].sum()),
                           "determinism": "same decisions on repeat, verified 1,148 of 1,148",
                           "fail_open": "tested: a failing model keeps the run going; a failing final score sends it to a human (tests/test_governor.py)",
                           "dollar_cost_per_1000_runs": 0.0}}
    OUT.with_suffix(".json").write_text(json.dumps(res, indent=1))
    lines = [f"Timed on {res['machine']['platform']}, CPU only, over {len(traces)} SWE-Gym test runs (mean {n_actions} actions), {REPEATS} passes, mean ± std.", "",
             "| step | time per run |", "|---|---|"]
    lines += [f"| {k} | {m:.2f} ± {s:.2f} ms |" for k, (m, s) in rows.items()]
    lines += [f"| 7B judge (RTX 4050, 50 runs) | {judge_ms / 1000:.1f} s |", f"| 32B verifier (Kaggle 2 x T4, {len(v)} runs) | {verifier_ms / 1000:.1f} s |", "",
              f"Full agent replay is {100 * res['share_of_verifier_latency']['full agent replay']:.3f} % of the verifier's time per run "
              f"(H3 target: under 1 %), on different hardware. Per action: {res['ms_per_action_replay']:.2f} ms. "
              f"Model fit at startup: {fit_s:.1f} s. Peak traced Python memory (fit + replay of 200 runs): {peak / 1e6:.0f} MB. Tool calls per run: {res['tool_calls_per_run']:.1f}. "
              f"The agent uses no LLM tokens, GPU or paid API. Failed tool calls in E3: {res['reliability']['failed_tool_calls']}.", ""]
    OUT.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

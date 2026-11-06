# Architecture

Project G2-09, *Observability and Trust in Agentic AI for Software Engineering* (AI4015 Agentic AI). Base paper: SWE-Gym (Pan et al., ICML 2025). The diagram is in the [README](../README.md#architecture) and `results/figures/fig_architecture.png`.

## Problem

Given a software-engineering agent's execution trace, estimate p(resolved | trace), where resolved means the final patch passes the hidden tests, and decide what to do about the run. The trace is observed step by step, so the agent can only use what a live monitor could see, plus the final patch at the end.

## Data flow

```
OpenHands (SWE-Gym parquet)   -> src/data/parser.py          \
OpenHands (Lite event stream) -> src/data/swebench_lite.py    -> schema.Trace of schema.Step -> src/features.py  -> RunFeatures
MoatlessTools (Lite zips)     -> src/data/moatless.py        /                               -> src/prefix.py    -> PrefixFeatures
                                                                                                      |
data/splits.csv (fixed) -> src/agent/risk.py: fit on dev, out-of-fold calibration and thresholds  <---+
                                      |
                              src/agent/governor.py -> JSONL logs -> experiments/ -> results/
```

Pydantic models in `src/schema.py` cross every module boundary.

## Components

| Module | Role |
|---|---|
| `src/data/parser.py` | Streams OpenHands chat messages into typed `Step`s, without loading the full `messages` column |
| `src/data/swebench_lite.py` | Rebuilds Lite event streams as chat; `normalize=False` keeps raw v1.9 calls (Gap 2 switch) |
| `src/data/moatless.py` | MoatlessTools adapter: search to bash, identify to view, edit to edit, plan finish to finish; `normalize=False` keeps raw action names (Gap 2 switch) |
| `src/features.py` | 11 Phase 1 features, plus `ProgressSignals`: 9 live Phase 2 features shared with prefixes |
| `src/prefix.py` | `PrefixState`, updated one step at a time |
| `src/calibration.py` | none, Platt, isotonic, prior shift, Platt + prior (Gap 1) |
| `src/agent/risk.py` | One LR per checkpoint and one end-of-run LR; thresholds are probabilities, set on dev; `retarget` for a new pass rate; `explain` gives exact linear SHAP |
| `src/agent/patterns.py` | Rule-based patterns, hints, phase belief |
| `src/agent/governor.py` | The loop; a human gate; tool calls logged; fails open |
| `src/agent/reroute.py` | Moves to the next logged attempt; task-level comparison of policies |
| `src/agent/cli.py` | `models`, `replay`, `task`, and the ablation flags |
| `src/baselines/` | Length, empty patch, prompt, telemetry LR and GBM, 7B judge |
| `src/eval/` | Fixed splits; pooled and within-task AUROC, Brier, ECE, Wilcoxon, Best@k, task bootstrap |

## Decision rules

All chosen on dev; the evidence is in `results/governor_tuning.md` and `results/feature_eval.md`.

| Rule | Value | Why |
|---|---|---|
| Abort checkpoints | 5 and 10 actions | Only those reach out-of-fold AUROC ≥ 0.60 |
| Abort budget | 5% per checkpoint | Largest losing ≤ 10% of passing runs overall |
| End-of-run `REJECT` budget | 10% of passing runs | Phase 1's false-alarm budget |
| Hints | `finish_without_edit`, `repeat_loop` (4 in 8 actions) | Pass rate when fired ≤ half of when not |
| Features | 20 at end of run and at 20, 30 actions; original set elsewhere | Adoption rule fixed before results |

## Features

Original 11: `n_steps`, `total_tokens`, `n_errors`, `n_repeated_commands`, `n_distinct_tools`, `n_edits_without_view`, `ran_tests`, `patch_lines`, `patch_files`, `patch_empty`, `finished`.

Phase 2, 9: `last_check_passed`, `checked_after_last_edit`, `fail_then_pass`, `max_identical_streak`, `max_error_streak`, `error_responsiveness`, `first_edit_step`, `phase_entropy`, `thought_action_mismatch`.

Leakage rule: `test_result.report`, `test_output` and `apply_patch_output` are never features. In MoatlessTools, `report.resolved` is read only as the label, and evaluation logs are never read.

## Log format

One JSON line per agent action: `uid`, `seq`, `step`, `observed` (action, tool, path, command, error), `tool_calls` (tool, input, output, ok, error), `phase_from`, `phase_to`, `p_success`, `decision`, `reason`, `hint`, `human`. The last line is the `RunOutcome`. See `logs/sample/`.

## Replay limits

Logged runs can't react: `inject_hint` records `delivered: false`, so only trigger accuracy is measurable. Reroute draws on other logged attempts, so a task with one logged run can't be rerouted.

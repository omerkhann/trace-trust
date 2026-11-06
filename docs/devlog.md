# Development log

Each entry: what failed or was limiting, what changed, and the measured effect. Numbers come from `results/`.

## 2026-10-17 to 10-20: Phase 1 baseline

- **Limitation:** no way to turn OpenHands traces into anything measurable.
- **Change:** streaming parser, Pydantic contracts, 11 telemetry features, baselines, 7B judge.
- **Effect:** telemetry beat length-only and prompt-only baselines (pooled AUROC about 0.81-0.84 vs 0.70).

## 2026-10-25: No data was held back

- **Failure:** every number came from 5-fold CV over all runs. Phase 2 tuning on the same runs would inflate results.
- **Change:** fixed test set of 20% of tasks, sampled separately for mixed tasks; `GroupKFold` on dev only; assignment committed.
- **Effect:** conclusions held, but within-task AUROC on test (0.76) fell below dev (0.87), an early warning about optimism.

## 2026-10-26: The base-paper verifier couldn't be fairly tested

- **Failure:** the verifier's training data overlaps the runs we meant to score it on.
- **Change:** matched issue text and step fingerprints against its training set (`experiments/verifier_overlap.py`).
- **Effect:** it saw 53.4% of tasks, and every task with a passing run; 969 of our runs are verbatim copies of its training examples. Any score there would be inflated, so E2 moved to clean data.

## 2026-11-03: One signal explained much of the result

- **Failure:** sanity checks showed no passing run has an empty patch, and that "is it Claude" alone gets 0.74 within-task on test.
- **Change:** added an `empty_patch` baseline, GPT-4o-only and non-empty subsets, task-level bootstrap intervals, 3 seeds.
- **Effect:** results hold within GPT-4o (within-task 0.830), but on runs that changed code within-task falls to 0.603 [0.394, 0.767], which includes chance. This set the direction for everything after. Seeds change nothing for deterministic models, so the bootstrap carries the uncertainty.

## 2026-11-03 to 11-04: A clean test set and the verifier run

- **Limitation:** no clean data for E2, and no unseen repositories.
- **Change:** SWE-bench Lite adapter (OpenHands event streams to chat format); exact prompt renderer (50 of 50 training examples reproduced); 4-bit verifier on Kaggle. Plain `pip` silently built CPU-only, fixed by building with `GGML_CUDA=on`.
- **Effect:** verifier 0.734 / 0.741 on 237 runs. A partial read of 132 runs had suggested it was level with telemetry; the full sample showed it isn't. Lesson: don't read partial samples.

## 2026-11-04: Calibration and early warning

- **Failure:** on Lite, ECE was 0.22-0.25, because models expected an 8% pass rate.
- **Change:** calibration module with an off switch; few-shot prior shift.
- **Effect:** ECE -53% with 20 labelled tasks, about 0.10 remains; verifier 0.290 to 0.100. Target ≤ 0.06 not met.
- **Failure:** 25/50/75% prefixes need total run length, which leaks it (a tokens-only model reached 0.81 within-task at "50%").
- **Change:** absolute-step checkpoints as the main result.
- **Effect:** the honest number is weak: 24.1% recall after 5 actions, near chance from 15.

## 2026-11-06: Governance agent, end-of-run logic was backwards

- **Failure:** first design flagged low-trust runs for review and accepted the rest. On dev, a 10% false-alarm budget marks 61% of runs low-trust, and the rest still mostly fail.
- **Change:** nothing is auto-accepted; low trust gets `REJECT`, the rest goes to a human.
- **Effect (dev):** 65.7% of failures rejected, 10.2% of passes lost, 39% of runs reviewed.

## 2026-11-06: Hint patterns didn't discriminate

- **Failure:** of 6 patterns, `error_streak` had no signal (pass rate 0.079 vs 0.082), `finish_unverified` fired more on passing runs (0.187 vs 0.039), and `repeat_loop` fired on 76% of runs.
- **Change:** a rule: keep a pattern only if runs where it fires pass at most half as often. Tightened `repeat_loop` to 4 repeats in 8 actions.
- **Effect (dev):** 2 patterns kept. `repeat_loop` fires on 53% of runs, 3.2% vs 13.8% pass.

## 2026-11-06: A suspected environment failure that wasn't

- **Failure:** 28% of dev runs, all GPT-4o, open with 4 identical root views and an empty-looking listing. These might have been broken environments, which would have invalidated those labels.
- **Change:** measured it (`experiments/empty_workspace_check.py`).
- **Effect:** hypothesis rejected: runs with that listing pass at 7.4%, vs 8.1% otherwise. It's a tool quirk.

## 2026-11-06: 9 new features, with a rule fixed in advance

- **Limitation:** 11 of 20 promised features; weak on runs that changed code.
- **Change:** `ProgressSignals`, shared by live and full-run extraction. Adoption rule written before seeing results.
- **Bug found:** the live scorer cast features to `int`, truncating decimals. Fixed; live scores now equal offline scores exactly (846 scores compared).
- **Effect (test):** LR pooled 0.812 to 0.836; on non-empty patches 0.672 to 0.714, within-task 0.603 to 0.629 [0.500, 0.753]. No gain at the 5 and 10 action checkpoints, or for GBM on test, though dev suggested one.

## 2026-11-06: Evaluation order corrected, then the agent locked

- **Failure:** the plan was to run E3 on test, then add features and rerun. That is tuning on test.
- **Change:** improvements judged on dev first, agent locked at `4a3b1a6`, E3/E5 run once.
- **Effect (test):** held close to dev: 68.8% of failures stopped, 12.4% of passes lost, 19.1% pass rate in review.

## 2026-11-06: Gap 2 had no real test

- **Failure:** the Lite transfer changed repositories and scaffold version together, and the plan had dropped MoatlessTools. The project brief asks for results across frameworks.
- **Change:** checked the release (labels and step logs exist), built an adapter, fixed the mapping before scoring anything.
- **Effect:** predictors keep 85-89% of AUROC, within-task 0.66-0.69 with one model and config. Drop target narrowly missed (0.089 at best). Abort doesn't transfer, and the agent lost to the empty-patch rule there. The agent stayed locked; this is reported as a limitation.

## 2026-11-06: Reproducibility fixes

- **Failure:** verifier scores (about 4.5 hours on Kaggle) were gitignored, so E2 and the hybrid couldn't be reproduced without Kaggle. MoatlessTools had no download step.
- **Change:** committed the scores and judge cache; added `moatless.download()`.
- **Effect:** every reported number can now be rebuilt from the repository.

## Things that didn't help

| Tried | Result |
|---|---|
| Hybrid of verifier and telemetry | Within-task -0.063, CI includes 0; ECE improved |
| Few-shot calibration on MoatlessTools | No improvement (K = 3), worse (K = 20) |
| New features at early checkpoints | Flat |
| Reroute as a way to save compute | 43% more tokens than one attempt; the gain is in reviews |

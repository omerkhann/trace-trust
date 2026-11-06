Governance agent at commit 4a3b1a6. Models and thresholds from SWE-Gym dev only. Brackets: 95 % task-level bootstrap. ± : std over seeds [42, 43, 44]. Lite: K = 20 labelled tasks, scored on the others.

### SWE-Gym test: runs

| config | failures stopped | passes lost | passes aborted | reviewed | pass rate in review | tokens saved | runs hinted |
|---|---|---|---|---|---|---|---|
| full | 68.8% [65.1, 72.3] | 12.4% [3.9, 20.3] | 4.5% [0.0, 9.7] | 35.5% [32.0, 39.3] | 19.1% [14.5, 23.9] | 9.8% [7.6, 12.2] | 50.3% |
| no_calibration | 68.8% [65.1, 72.3] | 12.4% [3.9, 20.3] | 4.5% [0.0, 9.7] | 35.5% [32.0, 39.3] | 19.1% [14.5, 23.9] | 9.8% [7.6, 12.2] | 50.3% |
| no_early_warning | 65.8% [62.1, 69.3] | 11.2% [3.9, 18.4] | 0.0% [0.0, 0.0] | 38.4% [35.0, 42.0] | 17.9% [13.7, 22.4] | 0.0% [0.0, 0.0] | 52.6% |
| no_telemetry | 37.9% [33.6, 42.1] | 6.7% [2.3, 12.5] | 0.0% [0.0, 0.0] | 64.5% [60.7, 68.6] | 11.2% [8.5, 14.4] | 0.0% [0.0, 0.0] | 0.0% |
| no_hints | 68.8% [65.1, 72.3] | 12.4% [3.9, 20.3] | 4.5% [0.0, 9.7] | 35.5% [32.0, 39.3] | 19.1% [14.5, 23.9] | 9.8% [7.6, 12.2] | 0.0% |
| empty_patch_rule | 42.7% [38.8, 46.4] | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 60.6% [57.0, 64.4] | 12.8% [9.7, 16.2] | 0.0% [0.0, 0.0] | n/a |

full, on non-empty patches only (696 runs, 89 passing): failures stopped 45.6% [40.6, 50.9], passes lost 12.4% [4.8, 20.5]. Share of the agent's stops that are empty patches: 61.1%.
Pass rate of runs hinted vs not: 4.0% vs 11.6%. Base rate 7.8%; 89 passing runs of 1148. Failed tool calls: 0.

### SWE-bench Lite: runs

| config | subset | failures stopped | passes lost | passes aborted | reviewed | pass rate in review | tokens saved |
|---|---|---|---|---|---|---|---|
| full | all | 26.2% ± 5.6 | 8.3% ± 3.5 | 0.0% | 80.0% ± 4.8 | 39.8% ± 1.6 | 0.0% ± 0.0 |
| full | v1.9 | 37.6% ± 6.8 | 20.4% ± 7.7 | 0.0% | 66.9% ± 6.9 | 31.1% ± 1.1 | 0.1% ± 0.0 |
| full | v2 | 11.3% ± 4.1 | 1.0% ± 0.9 | 0.0% | 93.2% ± 2.6 | 46.1% ± 1.6 | 0.0% |
| no_calibration | all | 48.9% ± 0.1 | 20.9% ± 0.3 | 2.7% ± 0.3 | 60.8% ± 0.2 | 45.2% ± 0.6 | 4.8% ± 0.9 |
| no_calibration | v1.9 | 67.0% ± 0.3 | 44.6% ± 0.1 | 7.1% ± 0.9 | 38.8% ± 0.4 | 37.3% ± 0.6 | 13.0% ± 2.5 |
| no_calibration | v2 | 25.2% ± 0.1 | 6.6% ± 0.4 | 0.0% | 82.9% ± 0.1 | 48.8% ± 0.7 | 0.0% |
| no_early_warning | all | 25.9% ± 5.6 | 8.3% ± 3.5 | 0.0% | 80.2% ± 4.8 | 39.8% ± 1.6 | 0.0% |
| no_early_warning | v1.9 | 37.1% ± 6.8 | 20.4% ± 7.7 | 0.0% | 67.3% ± 6.9 | 31.0% ± 1.0 | 0.0% |
| no_early_warning | v2 | 11.3% ± 4.1 | 1.0% ± 0.9 | 0.0% | 93.2% ± 2.6 | 46.1% ± 1.6 | 0.0% |
| no_telemetry | all | 0.0% ± 0.1 | 0.0% | 0.0% | 100.0% ± 0.1 | 34.7% ± 0.6 | 0.0% |
| no_telemetry | v1.9 | 0.1% ± 0.1 | 0.0% | 0.0% | 99.9% ± 0.1 | 26.1% ± 0.7 | 0.0% |
| no_telemetry | v2 | 0.0% | 0.0% | 0.0% | 100.0% | 43.4% ± 0.6 | 0.0% |
| no_hints | all | 26.2% ± 5.6 | 8.3% ± 3.5 | 0.0% | 80.0% ± 4.8 | 39.8% ± 1.6 | 0.0% ± 0.0 |
| no_hints | v1.9 | 37.6% ± 6.8 | 20.4% ± 7.7 | 0.0% | 66.9% ± 6.9 | 31.1% ± 1.1 | 0.1% ± 0.0 |
| no_hints | v2 | 11.3% ± 4.1 | 1.0% ± 0.9 | 0.0% | 93.2% ± 2.6 | 46.1% ± 1.6 | 0.0% |
| full_zero_shot | all | 48.9% ± 0.1 | 20.9% ± 0.3 | 2.7% ± 0.3 | 60.8% ± 0.2 | 45.2% ± 0.6 | 4.8% ± 0.9 |
| full_zero_shot | v1.9 | 67.0% ± 0.3 | 44.6% ± 0.1 | 7.1% ± 0.9 | 38.8% ± 0.4 | 37.3% ± 0.6 | 13.0% ± 2.5 |
| full_zero_shot | v2 | 25.2% ± 0.1 | 6.6% ± 0.4 | 0.0% | 82.9% ± 0.1 | 48.8% ± 0.7 | 0.0% |
| no_normalizer | all | 30.0% ± 6.5 | 10.8% ± 4.2 | 0.0% | 76.7% ± 5.6 | 40.5% ± 1.8 | 0.0% |
| no_normalizer | v1.9 | 44.3% ± 8.4 | 27.2% ± 9.6 | 0.0% | 60.2% ± 8.6 | 31.7% ± 1.1 | 0.0% |
| no_normalizer | v2 | 11.3% ± 4.1 | 1.0% ± 0.9 | 0.0% | 93.2% ± 2.6 | 46.1% ± 1.6 | 0.0% |
| empty_patch_rule | all | 5.0% | 0.0% | 0.0% | 96.7% | 35.9% | 0.0% |

### Reroute on test: tasks solved and cost per task

| config | subset | policy | tasks | solved | tokens per task | reviews per task | solved per M tokens | solved per review |
|---|---|---|---|---|---|---|---|---|
| full | all_tasks | random_attempt | 488 | 8.6% ± 0.7 | 14099 | 1.00 | 6.12 ± 0.71 | 0.086 ± 0.007 |
| full | all_tasks | review_all | 488 | 12.9% ± 0.0 | 31511 | 2.15 | 4.10 ± 0.00 | 0.060 ± 0.001 |
| full | all_tasks | governor_single | 488 | 7.9% ± 0.8 | 12782 | 0.37 | 6.20 ± 0.70 | 0.214 ± 0.025 |
| full | all_tasks | governor_reroute | 488 | 10.1% ± 0.5 | 20094 | 0.53 | 5.03 ± 0.21 | 0.189 ± 0.009 |
| full | multi_attempt_tasks | random_attempt | 362 | 8.3% ± 1.0 | 13075 | 1.00 | 6.38 ± 1.04 | 0.083 ± 0.010 |
| full | multi_attempt_tasks | review_all | 362 | 14.1% ± 0.0 | 36548 | 2.55 | 3.85 ± 0.00 | 0.055 ± 0.001 |
| full | multi_attempt_tasks | governor_single | 362 | 7.4% ± 1.1 | 11678 | 0.37 | 6.32 ± 1.02 | 0.201 ± 0.034 |
| full | multi_attempt_tasks | governor_reroute | 362 | 10.3% ± 0.6 | 21534 | 0.59 | 4.79 ± 0.27 | 0.175 ± 0.011 |
| empty_patch_rule | all_tasks | governor_single | 488 | 8.6% ± 0.7 | 14099 | 0.63 | 6.12 ± 0.71 | 0.136 ± 0.015 |
| empty_patch_rule | all_tasks | governor_reroute | 488 | 10.1% ± 0.6 | 18875 | 0.82 | 5.36 ± 0.38 | 0.123 ± 0.008 |
| empty_patch_rule | multi_attempt_tasks | governor_single | 362 | 8.3% ± 1.0 | 13075 | 0.62 | 6.38 ± 1.04 | 0.135 ± 0.021 |
| empty_patch_rule | multi_attempt_tasks | governor_reroute | 362 | 10.3% ± 0.8 | 19514 | 0.87 | 5.29 ± 0.50 | 0.119 ± 0.010 |

### Reroute on lite: tasks solved and cost per task

| config | subset | policy | tasks | solved | tokens per task | reviews per task | solved per M tokens | solved per review |
|---|---|---|---|---|---|---|---|---|
| full | all_tasks | random_attempt | 280 | 35.1% ± 0.8 | 48214 | 1.00 | 7.28 ± 0.16 | 0.351 ± 0.008 |
| full | all_tasks | review_all | 280 | 55.1% ± 0.8 | 190037 | 2.66 | 2.90 ± 0.03 | 0.207 ± 0.004 |
| full | all_tasks | governor_single | 280 | 32.7% ± 2.5 | 48198 | 0.80 | 6.79 ± 0.48 | 0.408 ± 0.010 |
| full | all_tasks | governor_reroute | 280 | 40.2% ± 1.1 | 61768 | 1.00 | 6.55 ± 0.55 | 0.404 ± 0.011 |
| full | multi_attempt_tasks | random_attempt | 280 | 35.1% ± 0.8 | 48214 | 1.00 | 7.28 ± 0.16 | 0.351 ± 0.008 |
| full | multi_attempt_tasks | review_all | 280 | 55.1% ± 0.8 | 190037 | 2.66 | 2.90 ± 0.03 | 0.207 ± 0.004 |
| full | multi_attempt_tasks | governor_single | 280 | 32.7% ± 2.5 | 48198 | 0.80 | 6.79 ± 0.48 | 0.408 ± 0.010 |
| full | multi_attempt_tasks | governor_reroute | 280 | 40.2% ± 1.1 | 61768 | 1.00 | 6.55 ± 0.55 | 0.404 ± 0.011 |
| empty_patch_rule | all_tasks | governor_single | 280 | 35.1% ± 0.8 | 48214 | 0.97 | 7.28 ± 0.16 | 0.360 ± 0.010 |
| empty_patch_rule | all_tasks | governor_reroute | 280 | 35.6% ± 0.7 | 49190 | 1.00 | 7.24 ± 0.08 | 0.356 ± 0.007 |
| empty_patch_rule | multi_attempt_tasks | governor_single | 280 | 35.1% ± 0.8 | 48214 | 0.97 | 7.28 ± 0.16 | 0.360 ± 0.010 |
| empty_patch_rule | multi_attempt_tasks | governor_reroute | 280 | 35.6% ± 0.7 | 49190 | 1.00 | 7.24 ± 0.08 | 0.356 ± 0.007 |

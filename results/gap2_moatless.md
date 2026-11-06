Models fit on SWE-Gym dev (OpenHands) only, applied zero-shot. H2 target: drop <= 0.08 (>= 85 % retained) normalised, > 0.20 raw.

### H2: pooled AUROC on 32b_8runs (2,398 runs) vs the references

| method | SWE-Gym test | OpenHands Lite | Moatless normalised | Moatless raw | drop from test: norm / raw | retained: norm / raw | drop from OH Lite: norm / raw |
|---|---|---|---|---|---|---|---|
| telemetry_lr | 0.812 | 0.655 | 0.723 [0.681, 0.764] | 0.657 | +0.089 / +0.155 | 89% / 81% | -0.068 / -0.002 |
| telemetry_gbm | 0.835 | 0.636 | 0.740 [0.698, 0.777] | 0.731 | +0.095 / +0.104 | 89% / 88% | -0.104 / -0.095 |
| telemetry20_lr | 0.836 | 0.659 | 0.712 [0.670, 0.756] | 0.683 | +0.124 / +0.153 | 85% / 82% | -0.053 / -0.024 |
| telemetry20_gbm | 0.835 | 0.655 | 0.725 [0.685, 0.767] | 0.679 | +0.110 / +0.157 | 87% / 81% | -0.071 / -0.024 |
| length_only | 0.695 | 0.591 | 0.347 [0.299, 0.402] | 0.347 | +0.348 / +0.348 | 50% / 50% | +0.244 / +0.244 |
| empty_patch | 0.713 | 0.525 | 0.555 [0.545, 0.565] | 0.555 | +0.158 / +0.158 | 78% / 78% | -0.030 / -0.030 |
| prompt_only | 0.657 | 0.439 | 0.348 [0.288, 0.411] | 0.348 | +0.309 / +0.309 | 53% / 53% | +0.091 / +0.091 |

Within-task on the same set (same model and configuration):

| method | within-task normalised | within-task raw | Wilcoxon p (norm) |
|---|---|---|---|
| telemetry_lr | 0.671 [0.606, 0.739] | 0.628 | 4.799728132415067e-07 |
| telemetry_gbm | 0.686 [0.629, 0.743] | 0.652 | 2.968264271016224e-08 |
| telemetry20_lr | 0.664 [0.6, 0.729] | 0.627 | 1.4210565101382053e-06 |
| telemetry20_gbm | 0.659 [0.587, 0.728] | 0.624 | 2.5419255246849147e-06 |
| length_only | 0.475 [0.409, 0.542] | 0.475 | 0.7755877167364746 |
| empty_patch | 0.618 [0.581, 0.659] | 0.618 | 3.310961473382679e-08 |
| prompt_only | 0.500 [0.5, 0.5] | 0.500 | None |

### Few-shot prior shift, telemetry20_lr on 32b_8runs

| K tasks | ECE raw | ECE Platt (dev) | ECE prior shift | reduction | pass rate true / estimated |
|---|---|---|---|---|---|
| 3 | 0.072 | 0.071 | 0.070 ± 0.001 | 3% | 0.191 / 0.083 |
| 20 | 0.071 | 0.070 | 0.087 ± 0.067 | -21% | 0.191 / 0.179 |

### Locked governance agent on 32b_8runs (K = 20 labelled tasks; mean over seeds)

| config | failures stopped | passes lost | passes aborted | reviewed | pass rate in review | tokens saved |
|---|---|---|---|---|---|---|
| full | 28.7% | 18.0% | 16.7% | 73.3% | 21.4% | 2.9% |
| no_calibration | 43.0% | 23.9% | 16.7% | 60.7% | 24.0% | 2.9% |
| no_early_warning | 16.1% | 1.4% | 0.0% | 86.7% | 21.7% | 0.0% |
| no_telemetry | 0.0% | 0.0% | 0.0% | 100.0% | 19.1% | 0.0% |
| no_hints | 28.7% | 18.0% | 16.7% | 73.3% | 21.4% | 2.9% |
| no_normalizer | 14.7% | 0.8% | 0.0% | 87.9% | 21.6% | 0.0% |
| full_zero_shot | 43.0% | 23.9% | 16.7% | 60.7% | 24.0% | 2.9% |
| empty_patch_rule | 11.0% | 0.0% | 0.0% | 91.1% | 21.0% | 0.0% |

### Reroute on 32b_8runs, tasks with several attempts

| config | policy | solved | tokens per task | reviews per task | solved per M tokens | solved per review |
|---|---|---|---|---|---|---|
| full | random_attempt | 18.3% ± 1.4 | 44078 | 1.00 | 4.16 | 0.183 |
| full | review_all | 33.7% ± 0.7 | 345663 | 5.98 | 0.97 | 0.056 |
| full | governor_single | 14.8% ± 0.7 | 42983 | 0.75 | 3.44 | 0.199 |
| full | governor_reroute | 19.2% ± 0.5 | 65458 | 0.99 | 2.94 | 0.194 |
| no_normalizer | governor_reroute | 20.7% ± 1.3 | 53435 | 1.00 | 3.88 | 0.207 |
| empty_patch_rule | governor_reroute | 20.7% ± 1.3 | 51002 | 1.00 | 4.08 | 0.207 |

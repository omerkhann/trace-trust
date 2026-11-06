Point estimates are the mean over seeds [42, 43, 44] (± std across seeds, shown only where non-zero). Brackets: 95% task-level bootstrap interval (1000 resamples of tasks). gap = pooled minus within-task AUROC.

### lite_all: 1199 runs, base rate 0.344, 124 mixed tasks / 404 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 300 tasks, Random@k 0.343, Pass@k (oracle) 0.456; k=4: 299 tasks, Random@k 0.344, Pass@k (oracle) 0.548

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 | Best@4 |
|---|---|---|---|---|---|---|---|---|
| length_only | 0.591 [0.556, 0.627] | 0.666 [0.601, 0.724] | < 0.001 | -0.075 [-0.126, -0.024] | 0.2416 | 0.1314 | 0.381 | 0.388 |
| empty_patch | 0.525 [0.518, 0.533] | 0.545 [0.524, 0.568] | < 0.001 | -0.019 [-0.039, 0.000] | 0.2686 | 0.2135 | 0.353 | 0.355 |
| prompt_only | 0.439 [0.386, 0.493] | 0.498 [0.492, 0.500] | n/a | -0.059 [-0.112, -0.005] | 0.3067 | 0.2801 | 0.343 | 0.344 |
| telemetry_lr | 0.655 [0.617, 0.690] | 0.696 [0.632, 0.756] | < 0.001 | -0.041 [-0.097, 0.012] | 0.2614 | 0.2178 | 0.387 | 0.405 |
| telemetry_gbm | 0.636 [0.598, 0.670] | 0.720 [0.666, 0.774] | < 0.001 | -0.084 [-0.136, -0.034] | 0.2754 | 0.2401 | 0.393 | 0.415 |
| telemetry20_lr | 0.659 [0.622, 0.692] | 0.738 [0.682, 0.796] | < 0.001 | -0.079 [-0.135, -0.028] | 0.2520 | 0.1953 | 0.397 | 0.408 |
| telemetry20_gbm | 0.655 [0.619, 0.687] | 0.745 [0.686, 0.800] | < 0.001 | -0.091 [-0.147, -0.040] | 0.2536 | 0.2003 | 0.398 | 0.408 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### lite_claude: 899 runs, base rate 0.387, 79 mixed tasks / 158 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 300 tasks, Random@k 0.387, Pass@k (oracle) 0.474

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 |
|---|---|---|---|---|---|---|---|
| length_only | 0.601 [0.562, 0.641] | 0.696 [0.615, 0.778] | < 0.001 | -0.095 [-0.167, -0.019] | 0.2538 | 0.1469 | 0.421 |
| empty_patch | 0.508 [0.503, 0.514] | 0.509 [0.500, 0.524] | 0.090 | -0.001 [-0.014, 0.009] | 0.3008 | 0.2538 | 0.388 |
| prompt_only | 0.446 [0.388, 0.504] | 0.497 [0.489, 0.500] | n/a | -0.051 [-0.109, 0.008] | 0.3443 | 0.3236 | 0.386 |
| telemetry_lr | 0.623 [0.576, 0.664] | 0.589 [0.493, 0.682] | 0.033 | 0.035 [-0.052, 0.130] | 0.2838 | 0.2349 | 0.402 |
| telemetry_gbm | 0.604 [0.558, 0.649] | 0.665 [0.576, 0.743] | < 0.001 | -0.060 [-0.139, 0.030] | 0.3002 | 0.2547 | 0.416 |
| telemetry20_lr | 0.627 [0.581, 0.669] | 0.627 [0.534, 0.719] | 0.004 | 0.001 [-0.092, 0.093] | 0.2695 | 0.2069 | 0.409 |
| telemetry20_gbm | 0.622 [0.577, 0.663] | 0.658 [0.566, 0.750] | < 0.001 | -0.037 [-0.122, 0.050] | 0.2712 | 0.1991 | 0.414 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### lite_claude_v2: 599 runs, base rate 0.427, 40 mixed tasks / 40 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 299 tasks, Random@k 0.428, Pass@k (oracle) 0.495

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 |
|---|---|---|---|---|---|---|---|
| length_only | 0.618 [0.561, 0.679] | 0.450 [0.310, 0.615] | 0.736 | 0.168 [-0.002, 0.322] | 0.2562 | 0.1386 | 0.421 |
| empty_patch | 0.510 [0.503, 0.519] | 0.512 [0.500, 0.542] | n/a | -0.002 [-0.031, 0.015] | 0.3302 | 0.2943 | 0.430 |
| prompt_only | 0.436 [0.375, 0.496] | 0.500 [0.500, 0.500] | n/a | -0.064 [-0.125, -0.004] | 0.3799 | 0.3639 | 0.428 |
| telemetry_lr | 0.616 [0.558, 0.667] | 0.550 [0.394, 0.711] | 0.264 | 0.066 [-0.093, 0.214] | 0.3091 | 0.2697 | 0.435 |
| telemetry_gbm | 0.579 [0.525, 0.633] | 0.550 [0.389, 0.708] | 0.264 | 0.029 [-0.124, 0.179] | 0.3300 | 0.2857 | 0.435 |
| telemetry20_lr | 0.619 [0.563, 0.670] | 0.550 [0.395, 0.710] | 0.264 | 0.069 [-0.089, 0.210] | 0.2876 | 0.2278 | 0.435 |
| telemetry20_gbm | 0.589 [0.537, 0.640] | 0.500 [0.362, 0.649] | 0.500 | 0.089 [-0.061, 0.235] | 0.2934 | 0.2217 | 0.428 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### lite_gpt4o: 300 runs, base rate 0.213, 0 mixed tasks / 0 same-task pairs

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE |
|---|---|---|---|---|---|---|
| length_only | 0.468 [0.387, 0.548] | n/a | n/a | n/a | 0.2051 | 0.1510 |
| empty_patch | 0.566 [0.544, 0.586] | n/a | n/a | n/a | 0.1721 | 0.1167 |
| prompt_only | 0.401 [0.330, 0.471] | n/a | n/a | n/a | 0.1939 | 0.1498 |
| telemetry_lr | 0.607 [0.536, 0.669] | n/a | n/a | n/a | 0.1941 | 0.1666 |
| telemetry_gbm | 0.593 [0.524, 0.659] | n/a | n/a | n/a | 0.2013 | 0.1840 |
| telemetry20_lr | 0.622 [0.552, 0.692] | n/a | n/a | n/a | 0.1996 | 0.1816 |
| telemetry20_gbm | 0.616 [0.546, 0.684] | n/a | n/a | n/a | 0.2009 | 0.1868 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

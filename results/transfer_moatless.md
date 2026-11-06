Point estimates are the mean over seeds [42, 43, 44] (± std across seeds, shown only where non-zero). Brackets: 95% task-level bootstrap interval (None resamples of tasks). gap = pooled minus within-task AUROC.

### moatless_32b_8runs: 2398 runs, base rate 0.190, 78 mixed tasks / 896 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 300 tasks, Random@k 0.190, Pass@k (oracle) 0.243; k=4: 300 tasks, Random@k 0.190, Pass@k (oracle) 0.294

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 | Best@4 |
|---|---|---|---|---|---|---|---|---|
| telemetry_lr | 0.723 [0.681, 0.764] | 0.671 [0.606, 0.739] | < 0.001 | 0.052 [-0.015, 0.122] | 0.1614 | 0.1342 | 0.208 | 0.219 |
| telemetry_gbm | 0.740 [0.698, 0.777] | 0.686 [0.629, 0.743] | < 0.001 | 0.054 [-0.009, 0.117] | 0.2171 | 0.2255 | 0.210 | 0.218 |
| telemetry20_lr | 0.712 [0.670, 0.756] | 0.664 [0.600, 0.729] | < 0.001 | 0.047 [-0.021, 0.114] | 0.1485 | 0.0718 | 0.207 | 0.219 |
| telemetry20_gbm | 0.725 [0.685, 0.767] | 0.659 [0.587, 0.728] | < 0.001 | 0.066 [-0.006, 0.140] | 0.1593 | 0.1272 | 0.207 | 0.215 |
| length_only | 0.347 [0.299, 0.402] | 0.475 [0.409, 0.542] | 0.776 | -0.128 [-0.213, -0.043] | 0.3477 | 0.4009 | 0.187 | 0.184 |
| empty_patch | 0.555 [0.545, 0.565] | 0.618 [0.581, 0.659] | < 0.001 | -0.063 [-0.101, -0.029] | 0.1555 | 0.0678 | 0.203 | 0.206 |
| prompt_only | 0.348 [0.288, 0.411] | 0.500 [0.500, 0.500] | n/a | -0.152 [-0.212, -0.089] | 0.1791 | 0.1509 | 0.190 | 0.190 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### moatless_32b_8runs_raw: 2398 runs, base rate 0.190, 78 mixed tasks / 896 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 300 tasks, Random@k 0.190, Pass@k (oracle) 0.243; k=4: 300 tasks, Random@k 0.190, Pass@k (oracle) 0.294

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 | Best@4 |
|---|---|---|---|---|---|---|---|---|
| telemetry_lr | 0.657 [0.608, 0.704] | 0.628 [0.556, 0.700] | < 0.001 | 0.028 [-0.045, 0.099] | 0.4959 | 0.5657 | 0.204 | 0.205 |
| telemetry_gbm | 0.731 [0.684, 0.775] | 0.652 [0.590, 0.709] | < 0.001 | 0.079 [0.009, 0.144] | 0.1778 | 0.1734 | 0.206 | 0.210 |
| telemetry20_lr | 0.683 [0.634, 0.730] | 0.627 [0.557, 0.701] | < 0.001 | 0.056 [-0.017, 0.127] | 0.1525 | 0.0807 | 0.204 | 0.204 |
| telemetry20_gbm | 0.679 [0.636, 0.722] | 0.624 [0.552, 0.695] | < 0.001 | 0.054 [-0.017, 0.134] | 0.1838 | 0.1769 | 0.203 | 0.205 |
| length_only | 0.347 [0.299, 0.402] | 0.475 [0.409, 0.542] | 0.776 | -0.128 [-0.213, -0.043] | 0.3477 | 0.4009 | 0.187 | 0.184 |
| empty_patch | 0.555 [0.545, 0.565] | 0.618 [0.581, 0.659] | < 0.001 | -0.063 [-0.101, -0.029] | 0.1555 | 0.0678 | 0.203 | 0.206 |
| prompt_only | 0.348 [0.288, 0.411] | 0.500 [0.500, 0.500] | n/a | -0.152 [-0.212, -0.089] | 0.1791 | 0.1509 | 0.190 | 0.190 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### moatless_32b_zero: 300 runs, base rate 0.190, 0 mixed tasks / 0 same-task pairs

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE |
|---|---|---|---|---|---|---|
| telemetry_lr | 0.739 [0.676, 0.799] | n/a | n/a | n/a | 0.1459 | 0.0868 |
| telemetry_gbm | 0.769 [0.709, 0.823] | n/a | n/a | n/a | 0.1924 | 0.1888 |
| telemetry20_lr | 0.743 [0.683, 0.802] | n/a | n/a | n/a | 0.1505 | 0.0983 |
| telemetry20_gbm | 0.753 [0.692, 0.809] | n/a | n/a | n/a | 0.1654 | 0.1471 |
| length_only | 0.299 [0.235, 0.371] | n/a | n/a | n/a | 0.3708 | 0.4217 |
| empty_patch | 0.650 [0.621, 0.677] | n/a | n/a | n/a | 0.1526 | 0.0883 |
| prompt_only | 0.420 [0.340, 0.501] | n/a | n/a | n/a | 0.1780 | 0.1510 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### moatless_32b_zero_raw: 300 runs, base rate 0.190, 0 mixed tasks / 0 same-task pairs

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE |
|---|---|---|---|---|---|---|
| telemetry_lr | 0.697 [0.641, 0.755] | n/a | n/a | n/a | 0.4009 | 0.4497 |
| telemetry_gbm | 0.792 [0.740, 0.844] | n/a | n/a | n/a | 0.1504 | 0.1250 |
| telemetry20_lr | 0.733 [0.676, 0.788] | n/a | n/a | n/a | 0.1516 | 0.1046 |
| telemetry20_gbm | 0.783 [0.724, 0.840] | n/a | n/a | n/a | 0.1839 | 0.1804 |
| length_only | 0.299 [0.235, 0.371] | n/a | n/a | n/a | 0.3708 | 0.4217 |
| empty_patch | 0.650 [0.621, 0.677] | n/a | n/a | n/a | 0.1526 | 0.0883 |
| prompt_only | 0.420 [0.340, 0.501] | n/a | n/a | n/a | 0.1780 | 0.1510 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### moatless_7b_zero: 299 runs, base rate 0.070, 0 mixed tasks / 0 same-task pairs

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE |
|---|---|---|---|---|---|---|
| telemetry_lr | 0.872 [0.817, 0.922] | n/a | n/a | n/a | 0.0571 | 0.0381 |
| telemetry_gbm | 0.865 [0.810, 0.915] | n/a | n/a | n/a | 0.0651 | 0.0382 |
| telemetry20_lr | 0.873 [0.820, 0.923] | n/a | n/a | n/a | 0.0597 | 0.0388 |
| telemetry20_gbm | 0.855 [0.805, 0.907] | n/a | n/a | n/a | 0.0684 | 0.0648 |
| length_only | 0.372 [0.258, 0.491] | n/a | n/a | n/a | 0.3579 | 0.5234 |
| empty_patch | 0.804 [0.777, 0.835] | n/a | n/a | n/a | 0.0592 | 0.0204 |
| prompt_only | 0.403 [0.294, 0.520] | n/a | n/a | n/a | 0.0674 | 0.0658 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### moatless_7b_zero_raw: 299 runs, base rate 0.070, 0 mixed tasks / 0 same-task pairs

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE |
|---|---|---|---|---|---|---|
| telemetry_lr | 0.856 [0.804, 0.906] | n/a | n/a | n/a | 0.2400 | 0.2867 |
| telemetry_gbm | 0.875 [0.826, 0.922] | n/a | n/a | n/a | 0.0851 | 0.1036 |
| telemetry20_lr | 0.867 [0.813, 0.918] | n/a | n/a | n/a | 0.0577 | 0.0315 |
| telemetry20_gbm | 0.852 [0.800, 0.902] | n/a | n/a | n/a | 0.0679 | 0.0646 |
| length_only | 0.372 [0.258, 0.491] | n/a | n/a | n/a | 0.3579 | 0.5234 |
| empty_patch | 0.804 [0.777, 0.835] | n/a | n/a | n/a | 0.0592 | 0.0204 |
| prompt_only | 0.403 [0.294, 0.520] | n/a | n/a | n/a | 0.0674 | 0.0658 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### verifier_all: 237 runs, base rate 0.494, 79 mixed tasks / 158 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 79 tasks, Random@k 0.494, Pass@k (oracle) 0.827

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 |
|---|---|---|---|---|---|---|---|
| swegym_verifier_32b | 0.734 [0.674, 0.792] | 0.741 [0.665, 0.816] | < 0.001 | -0.007 [-0.057, 0.038] | 0.3127 | 0.2904 | 0.654 |
| telemetry_lr | 0.604 [0.538, 0.670] | 0.589 [0.500, 0.684] | 0.033 | 0.015 [-0.050, 0.080] | 0.3755 | 0.3606 | 0.553 |
| telemetry_gbm | 0.609 [0.551, 0.671] | 0.665 [0.582, 0.741] | < 0.001 | -0.055 [-0.111, -0.004] | 0.3837 | 0.3625 | 0.603 |
| length_only | 0.691 [0.621, 0.757] | 0.696 [0.614, 0.785] | < 0.001 | -0.005 [-0.056, 0.045] | 0.2844 | 0.2414 | 0.624 |
| empty_patch | 0.508 [0.500, 0.521] | 0.509 [0.500, 0.525] | 0.090 | -0.001 [-0.006, 0.003] | 0.3787 | 0.3699 | 0.500 |
| prompt_only | 0.508 [0.464, 0.552] | 0.497 [0.491, 0.500] | n/a | 0.011 [-0.032, 0.056] | 0.4359 | 0.4313 | 0.492 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### verifier_v2: 158 runs, base rate 0.646, 40 mixed tasks / 40 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 79 tasks, Random@k 0.646, Pass@k (oracle) 0.899

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 |
|---|---|---|---|---|---|---|---|
| swegym_verifier_32b | 0.671 [0.583, 0.761] | 0.650 [0.500, 0.791] | 0.029 | 0.021 [-0.113, 0.159] | 0.2402 | 0.1805 | 0.722 |
| telemetry_lr | 0.598 [0.498, 0.698] | 0.550 [0.404, 0.707] | 0.264 | 0.048 [-0.083, 0.173] | 0.4841 | 0.5068 | 0.671 |
| telemetry_gbm | 0.557 [0.463, 0.649] | 0.550 [0.395, 0.696] | 0.264 | 0.007 [-0.095, 0.119] | 0.4965 | 0.5005 | 0.671 |
| length_only | 0.615 [0.534, 0.692] | 0.450 [0.300, 0.614] | 0.736 | 0.165 [0.024, 0.297] | 0.3490 | 0.3441 | 0.620 |
| empty_patch | 0.509 [0.500, 0.529] | 0.512 [0.500, 0.541] | n/a | -0.004 [-0.013, 0.000] | 0.4898 | 0.5118 | 0.652 |
| prompt_only | 0.472 [0.386, 0.565] | 0.500 [0.500, 0.500] | n/a | -0.028 [-0.114, 0.065] | 0.5702 | 0.5833 | 0.646 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### verifier_v19: 79 runs, base rate 0.190, 0 mixed tasks / 0 same-task pairs

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE |
|---|---|---|---|---|---|---|
| swegym_verifier_32b | 0.680 [0.542, 0.807] | n/a | n/a | n/a | 0.4577 | 0.5409 |
| telemetry_lr | 0.588 [0.423, 0.756] | n/a | n/a | n/a | 0.1583 | 0.1459 |
| telemetry_gbm | 0.653 [0.500, 0.807] | n/a | n/a | n/a | 0.1580 | 0.1327 |
| length_only | 0.549 [0.387, 0.706] | n/a | n/a | n/a | 0.1552 | 0.1626 |
| empty_patch | 0.508 [0.500, 0.524] | n/a | n/a | n/a | 0.1566 | 0.1456 |
| prompt_only | 0.634 [0.479, 0.769] | n/a | n/a | n/a | 0.1674 | 0.1586 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### verifier_untruncated: 167 runs, base rate 0.539, 47 mixed tasks / 82 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 57 tasks, Random@k 0.544, Pass@k (oracle) 0.854

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 |
|---|---|---|---|---|---|---|---|
| swegym_verifier_32b | 0.700 [0.616, 0.777] | 0.659 [0.548, 0.762] | 0.006 | 0.041 [-0.031, 0.121] | 0.2996 | 0.2900 | 0.643 |
| telemetry_lr | 0.566 [0.477, 0.652] | 0.537 [0.405, 0.671] | 0.171 | 0.029 [-0.070, 0.122] | 0.3910 | 0.3765 | 0.596 |
| telemetry_gbm | 0.569 [0.491, 0.660] | 0.671 [0.550, 0.783] | 0.002 | -0.102 [-0.183, -0.022] | 0.4046 | 0.3804 | 0.661 |
| length_only | 0.744 [0.666, 0.805] | 0.720 [0.615, 0.819] | < 0.001 | 0.025 [-0.055, 0.098] | 0.2907 | 0.2687 | 0.684 |
| empty_patch | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | n/a | 0.000 [0.000, 0.000] | 0.4120 | 0.4101 | 0.544 |
| prompt_only | 0.504 [0.437, 0.578] | 0.494 [0.480, 0.500] | n/a | 0.010 [-0.058, 0.085] | 0.4768 | 0.4781 | 0.541 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### verifier_truncated: 36 runs, base rate 0.389, 5 mixed tasks / 6 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 9 tasks, Random@k 0.370, Pass@k (oracle) 0.630

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 |
|---|---|---|---|---|---|---|---|
| swegym_verifier_32b | 0.675 [0.495, 0.850] | 0.833 [0.600, 1.000] | 0.062 | -0.158 [-0.446, 0.076] | 0.3217 | 0.3218 | 0.593 |
| telemetry_lr | 0.633 [0.417, 0.845] | 0.667 [0.284, 1.000] | 0.312 | -0.034 [-0.404, 0.282] | 0.3484 | 0.3685 | 0.481 |
| telemetry_gbm | 0.555 [0.336, 0.790] | 0.667 [0.299, 1.000] | 0.312 | -0.111 [-0.517, 0.238] | 0.3358 | 0.3524 | 0.481 |
| length_only | 0.429 [0.221, 0.679] | 0.500 [0.000, 1.000] | 0.500 | -0.071 [-0.587, 0.278] | 0.2849 | 0.3180 | 0.444 |
| empty_patch | 0.523 [0.500, 0.579] | 0.500 [0.500, 0.500] | n/a | 0.023 [0.000, 0.079] | 0.3018 | 0.3553 | 0.370 |
| prompt_only | 0.649 [0.471, 0.819] | 0.500 [0.500, 0.500] | n/a | 0.149 [-0.029, 0.319] | 0.3376 | 0.3550 | 0.370 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

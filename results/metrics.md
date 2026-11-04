Point estimates are the mean over seeds [42, 43, 44] (± std across seeds, shown only where non-zero). Brackets: 95% task-level bootstrap interval (1000 resamples of tasks). gap = pooled minus within-task AUROC.

### dev: 4907 runs, base rate 0.082, 150 mixed tasks / 1199 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 1453 tasks, Random@k 0.070, Pass@k (oracle) 0.106; k=4: 192 tasks, Random@k 0.107, Pass@k (oracle) 0.259

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 | Best@4 |
|---|---|---|---|---|---|---|---|---|
| length_only | 0.703 [0.671, 0.731] | 0.818 [0.768, 0.862] | < 0.001 | -0.115 [-0.165, -0.069] | 0.0729 | 0.0166 | 0.089 | 0.189 |
| empty_patch | 0.688 [0.659, 0.719] | 0.787 [0.748, 0.822] | < 0.001 | -0.098 [-0.139, -0.055] | 0.0710 | 0.0237 | 0.087 | 0.174 |
| prompt_only | 0.638 [0.590, 0.684] | 0.500 [0.500, 0.500] | n/a | 0.138 [0.090, 0.184] | 0.0740 | 0.0140 | 0.070 | 0.107 |
| telemetry_lr | 0.844 [0.821, 0.865] | 0.865 [0.824, 0.901] | < 0.001 | -0.021 [-0.055, 0.018] | 0.0654 | 0.0094 | 0.093 | 0.206 |
| telemetry_gbm | 0.842 [0.822, 0.861] | 0.847 [0.793, 0.892] | < 0.001 | -0.005 [-0.046, 0.040] | 0.0662 | 0.0216 | 0.094 | 0.200 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### test: 1148 runs, base rate 0.078, 38 mixed tasks / 195 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 362 tasks, Random@k 0.079, Pass@k (oracle) 0.120; k=4: 38 tasks, Random@k 0.066, Pass@k (oracle) 0.185

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 | Best@4 |
|---|---|---|---|---|---|---|---|---|
| length_only | 0.695 [0.643, 0.743] | 0.733 [0.584, 0.862] | 0.006 | -0.039 [-0.163, 0.105] | 0.0703 | 0.0368 | 0.095 | 0.116 |
| empty_patch | 0.713 [0.694, 0.732] | 0.703 [0.653, 0.769] | < 0.001 | 0.011 [-0.049, 0.059] | 0.0676 | 0.0291 | 0.100 | 0.097 |
| prompt_only | 0.657 [0.576, 0.730] | 0.500 [0.500, 0.500] | n/a | 0.157 [0.076, 0.230] | 0.0699 | 0.0294 | 0.079 | 0.066 |
| telemetry_lr | 0.812 [0.773, 0.853] | 0.764 [0.626, 0.859] | < 0.001 | 0.048 [-0.044, 0.189] | 0.0657 | 0.0190 | 0.109 | 0.114 |
| telemetry_gbm | 0.835 [0.800, 0.871] | 0.749 [0.609, 0.855] | < 0.001 | 0.087 [-0.004, 0.217] | 0.0661 | 0.0238 | 0.109 | 0.113 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### test_gpt4o_only: 1110 runs, base rate 0.069, 31 mixed tasks / 100 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 362 tasks, Random@k 0.076, Pass@k (oracle) 0.115; k=4: 38 tasks, Random@k 0.038, Pass@k (oracle) 0.097

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 | Best@4 |
|---|---|---|---|---|---|---|---|---|
| length_only | 0.684 [0.630, 0.736] | 0.720 [0.533, 0.864] | 0.020 | -0.036 [-0.185, 0.142] | 0.0642 | 0.0344 | 0.091 | 0.058 |
| empty_patch | 0.718 [0.698, 0.738] | 0.700 [0.650, 0.767] | < 0.001 | 0.018 [-0.051, 0.070] | 0.0615 | 0.0238 | 0.096 | 0.050 |
| prompt_only | 0.661 [0.574, 0.736] | 0.500 [0.500, 0.500] | n/a | 0.161 [0.074, 0.236] | 0.0632 | 0.0297 | 0.076 | 0.038 |
| telemetry_lr | 0.811 [0.766, 0.854] | 0.830 [0.684, 0.922] | < 0.001 | -0.019 [-0.108, 0.132] | 0.0603 | 0.0237 | 0.105 | 0.067 |
| telemetry_gbm | 0.836 [0.798, 0.875] | 0.800 [0.723, 0.897] | < 0.001 | 0.036 [-0.050, 0.116] | 0.0599 | 0.0241 | 0.105 | 0.059 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### test_nonempty_patch: 696 runs, base rate 0.128, 24 mixed tasks / 116 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 162 tasks, Random@k 0.142, Pass@k (oracle) 0.198; k=4: 26 tasks, Random@k 0.117, Pass@k (oracle) 0.289

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 | Best@4 |
|---|---|---|---|---|---|---|---|---|
| length_only | 0.533 [0.467, 0.594] | 0.629 [0.463, 0.777] | 0.327 | -0.097 [-0.246, 0.068] | 0.1138 | 0.0667 | 0.146 | 0.152 |
| empty_patch | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | n/a | 0.000 [0.000, 0.000] | 0.1116 | 0.0632 | 0.142 | 0.117 |
| prompt_only | 0.657 [0.574, 0.732] | 0.500 [0.500, 0.500] | n/a | 0.157 [0.074, 0.232] | 0.1108 | 0.0604 | 0.142 | 0.117 |
| telemetry_lr | 0.672 [0.611, 0.734] | 0.603 [0.394, 0.767] | 0.120 | 0.068 [-0.097, 0.301] | 0.1084 | 0.0469 | 0.163 | 0.158 |
| telemetry_gbm | 0.713 [0.655, 0.772] | 0.578 [0.388, 0.735] | 0.027 | 0.135 [-0.006, 0.319] | 0.1091 | 0.0440 | 0.169 | 0.148 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

### judge_sample: 50 runs, base rate 0.500, 25 mixed tasks / 25 same-task pairs

Best@k reference points (tasks with >= k runs): k=2: 25 tasks, Random@k 0.500, Pass@k (oracle) 1.000

| method | pooled AUROC | within-task AUROC | Wilcoxon p | gap | Brier | ECE | Best@2 |
|---|---|---|---|---|---|---|---|
| length_only | 0.744 [0.582, 0.886] | 0.680 [0.480, 0.840] | 0.036 | 0.064 [-0.037, 0.162] | 0.3946 | 0.4196 | 0.680 |
| empty_patch | 0.720 [0.620, 0.820] | 0.720 [0.620, 0.820] | < 0.001 | 0.000 [-0.000, 0.000] | 0.3795 | 0.4490 | 0.720 |
| prompt_only | 0.500 [0.500, 0.500] | 0.500 [0.500, 0.500] | n/a | 0.000 [-0.000, 0.000] | 0.4195 | 0.4090 | 0.500 |
| telemetry_lr | 0.782 [0.626, 0.912] | 0.800 [0.640, 0.920] | 0.001 | -0.018 [-0.104, 0.056] | 0.3751 | 0.3988 | 0.800 |
| telemetry_gbm | 0.834 [0.718, 0.926] | 0.760 [0.600, 0.920] | 0.005 | 0.074 [-0.021, 0.168] | 0.3468 | 0.3818 | 0.760 |
| llm_judge | 0.750 [0.624, 0.861] | 0.740 [0.620, 0.860] | 0.001 | 0.010 [-0.033, 0.045] | 0.2594 | 0.2410 | 0.740 |

Wilcoxon p: one-sided signed-rank test that per-task AUROC > 0.5 over mixed tasks.

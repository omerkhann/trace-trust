### dev: 4907 runs, base rate 0.082, 150 mixed tasks / 1199 same-task pairs

| method | pooled AUROC | within-task AUROC | Brier | ECE |
|---|---|---|---|---|
| length_only | 0.703 | 0.818 | 0.0729 | 0.0166 |
| prompt_only | 0.638 | 0.500 | 0.0740 | 0.0140 |
| telemetry_lr | 0.844 | 0.865 | 0.0654 | 0.0094 |
| telemetry_gbm | 0.842 | 0.847 | 0.0662 | 0.0216 |

### test: 1148 runs, base rate 0.078, 38 mixed tasks / 195 same-task pairs

| method | pooled AUROC | within-task AUROC | Brier | ECE |
|---|---|---|---|---|
| length_only | 0.695 | 0.733 | 0.0703 | 0.0368 |
| prompt_only | 0.657 | 0.500 | 0.0699 | 0.0294 |
| telemetry_lr | 0.812 | 0.764 | 0.0657 | 0.0190 |
| telemetry_gbm | 0.835 | 0.749 | 0.0661 | 0.0238 |

### judge_sample: 50 runs, base rate 0.500, 25 mixed tasks / 25 same-task pairs

| method | pooled AUROC | within-task AUROC | Brier | ECE |
|---|---|---|---|---|
| length_only | 0.744 | 0.680 | 0.3946 | 0.4196 |
| prompt_only | 0.500 | 0.500 | 0.4195 | 0.4090 |
| telemetry_lr | 0.782 | 0.800 | 0.3751 | 0.3988 |
| telemetry_gbm | 0.834 | 0.760 | 0.3468 | 0.3818 |
| llm_judge | 0.750 | 0.740 | 0.2594 | 0.2410 |

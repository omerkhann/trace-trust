237 verifier-scored SWE-bench Lite runs, 158 same-task pairs. Telemetry fit on SWE-Gym dev. Hybrid: LR on logits, 5-fold cross-fit by task, mean over seeds [42, 43, 44]. Verifier 75.7 s/run.

### Standalone

| method | pooled | within-task | share of verifier's within-task | ECE | Best@2 |
|---|---|---|---|---|---|
| verifier | 0.734 | 0.741 |  | 0.290 | 0.654 |
| telemetry20_lr | 0.624 | 0.627 | 85% | 0.327 | 0.578 |
| telemetry_lr | 0.604 | 0.589 | 79% | 0.361 | 0.553 |
| telemetry_gbm | 0.609 | 0.665 | 90% | 0.363 | 0.603 |
| length_only | 0.691 | 0.696 | 94% | 0.241 | 0.624 |

### Hybrids

| model | pooled | within-task | ECE | Best@2 | pooled minus verifier | within-task minus verifier |
|---|---|---|---|---|---|---|
| verifier + telemetry20_lr (primary) | 0.666 | 0.677 ± 0.006 | 0.084 | 0.612 | -0.050 [-0.115, +0.013] | -0.063 [-0.158, +0.019] |
| verifier + telemetry_lr | 0.668 | 0.694 ± 0.016 | 0.098 | 0.623 | -0.045 [-0.102, +0.010] | -0.032 [-0.108, +0.044] |
| verifier + telemetry_gbm | 0.655 | 0.686 ± 0.016 | 0.102 | 0.617 | -0.061 [-0.116, -0.005] | -0.038 [-0.108, +0.044] |
| verifier + length_only | 0.737 | 0.770 ± 0.004 | 0.095 | 0.674 | +0.012 [-0.047, +0.068] | +0.032 [-0.051, +0.108] |

Normalised minus raw on MoatlessTools. Paired task-level bootstrap, 1,000 resamples. Models fit on SWE-Gym dev.

| method | set | pooled AUROC diff | within-task AUROC diff | ECE normalised | ECE raw |
|---|---|---|---|---|---|
| telemetry_lr | 32b_8runs | +0.066 [+0.031, +0.098] | +0.042 [+0.005, +0.085] | 0.134 | 0.566 |
| telemetry_lr | 32b_zero | +0.042 [-0.017, +0.101] | n/a | 0.087 | 0.450 |
| telemetry_lr | 7b_zero | +0.016 [+0.001, +0.030] | n/a | 0.038 | 0.287 |
| telemetry_gbm | 32b_8runs | +0.009 [-0.015, +0.035] | +0.035 [-0.005, +0.078] | 0.225 | 0.173 |
| telemetry_gbm | 32b_zero | -0.024 [-0.069, +0.018] | n/a | 0.189 | 0.125 |
| telemetry_gbm | 7b_zero | -0.011 [-0.033, +0.013] | n/a | 0.038 | 0.104 |
| telemetry20_lr | 32b_8runs | +0.028 [+0.005, +0.052] | +0.037 [+0.001, +0.077] | 0.072 | 0.081 |
| telemetry20_lr | 32b_zero | +0.011 [-0.036, +0.055] | n/a | 0.098 | 0.105 |
| telemetry20_lr | 7b_zero | +0.007 [-0.009, +0.021] | n/a | 0.039 | 0.031 |
| telemetry20_gbm | 32b_8runs | +0.046 [+0.021, +0.076] | +0.035 [-0.028, +0.098] | 0.127 | 0.177 |
| telemetry20_gbm | 32b_zero | -0.030 [-0.076, +0.016] | n/a | 0.147 | 0.180 |
| telemetry20_gbm | 7b_zero | +0.003 [-0.038, +0.038] | n/a | 0.065 | 0.065 |

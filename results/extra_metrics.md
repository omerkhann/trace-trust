AUPRC (chance level = base rate) and macro-F1 at the threshold that maximises it on dev out-of-fold predictions. Models fit on SWE-Gym dev.

### SWE-Gym test: 1148 runs, base rate 0.078

| method | AUPRC | macro-F1 | F1 (pass class) |
|---|---|---|---|
| length_only | 0.125 | 0.544 | 0.183 |
| empty_patch | 0.128 | 0.480 | 0.000 |
| prompt_only | 0.143 | 0.569 | 0.212 |
| telemetry_lr | 0.210 | 0.583 | 0.282 |
| telemetry_gbm | 0.239 | 0.599 | 0.278 |
| telemetry20_lr | 0.255 | 0.626 | 0.335 |
| telemetry20_gbm | 0.217 | 0.613 | 0.320 |

### SWE-Gym test, non-empty: 696 runs, base rate 0.128

| method | AUPRC | macro-F1 | F1 (pass class) |
|---|---|---|---|
| length_only | 0.138 | 0.520 | 0.196 |
| empty_patch | 0.128 | 0.466 | 0.000 |
| prompt_only | 0.215 | 0.578 | 0.256 |
| telemetry_lr | 0.210 | 0.534 | 0.282 |
| telemetry_gbm | 0.239 | 0.568 | 0.278 |
| telemetry20_lr | 0.255 | 0.594 | 0.335 |
| telemetry20_gbm | 0.217 | 0.574 | 0.320 |

### OpenHands Lite: 1199 runs, base rate 0.344

| method | AUPRC | macro-F1 | F1 (pass class) |
|---|---|---|---|
| length_only | 0.429 | 0.528 | 0.480 |
| empty_patch | 0.355 | 0.396 | 0.000 |
| prompt_only | 0.317 | 0.394 | 0.000 |
| telemetry_lr | 0.444 | 0.584 | 0.439 |
| telemetry_gbm | 0.437 | 0.538 | 0.318 |
| telemetry20_lr | 0.453 | 0.586 | 0.430 |
| telemetry20_gbm | 0.470 | 0.602 | 0.470 |

### Moatless 32b_8runs: 2398 runs, base rate 0.190

| method | AUPRC | macro-F1 | F1 (pass class) |
|---|---|---|---|
| length_only | 0.137 | 0.160 | 0.320 |
| empty_patch | 0.209 | 0.447 | 0.000 |
| prompt_only | 0.143 | 0.447 | 0.000 |
| telemetry_lr | 0.313 | 0.384 | 0.370 |
| telemetry_gbm | 0.355 | 0.524 | 0.413 |
| telemetry20_lr | 0.304 | 0.533 | 0.197 |
| telemetry20_gbm | 0.328 | 0.571 | 0.273 |

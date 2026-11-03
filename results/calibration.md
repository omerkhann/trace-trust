Calibrators are fit only on data the evaluation never sees (see experiments/calibration.py). Few-shot rows: mean ± std over 3 random draws of the labelled tasks.

### A. SWE-Gym test: 1148 runs, base rate 0.078

| predictor | calibrator | ECE | Brier | NLL | pooled AUROC | within-task AUROC |
|---|---|---|---|---|---|---|
| telemetry_lr | none | 0.019 | 0.066 | 0.219 | 0.812 | 0.764 |
| telemetry_lr | platt | 0.018 | 0.066 | 0.219 | 0.812 | 0.764 |
| telemetry_lr | isotonic | 0.022 | 0.067 | 0.222 | 0.811 | 0.767 |
| telemetry_gbm | none | 0.024 | 0.066 | 0.219 | 0.835 | 0.749 |
| telemetry_gbm | platt | 0.019 | 0.064 | 0.214 | 0.835 | 0.749 |
| telemetry_gbm | isotonic | 0.031 | 0.065 | 0.215 | 0.829 | 0.728 |
| length_only | none | 0.037 | 0.070 | 0.257 | 0.695 | 0.733 |
| length_only | platt | 0.037 | 0.070 | 0.257 | 0.695 | 0.733 |
| length_only | isotonic | 0.027 | 0.070 | 0.251 | 0.694 | 0.744 |

### B. Lite zero-shot: 1199 runs, base rate 0.344

| predictor | calibrator | ECE | Brier | NLL | pooled AUROC | within-task AUROC |
|---|---|---|---|---|---|---|
| telemetry_lr | none (SWE-Gym dev) | 0.218 | 0.261 | 0.812 | 0.655 | 0.696 |
| telemetry_lr | platt (SWE-Gym dev) | 0.218 | 0.261 | 0.808 | 0.655 | 0.696 |
| telemetry_lr | isotonic (SWE-Gym dev) | 0.218 | 0.262 | 0.900 | 0.655 | 0.694 |
| telemetry_gbm | none (SWE-Gym dev) | 0.240 | 0.275 | 0.923 | 0.636 | 0.720 |
| telemetry_gbm | platt (SWE-Gym dev) | 0.218 | 0.266 | 0.823 | 0.636 | 0.720 |
| telemetry_gbm | isotonic (SWE-Gym dev) | 0.221 | 0.269 | 0.824 | 0.635 | 0.712 |
| length_only | none (SWE-Gym dev) | 0.131 | 0.242 | 0.711 | 0.591 | 0.666 |
| length_only | platt (SWE-Gym dev) | 0.131 | 0.242 | 0.711 | 0.591 | 0.666 |
| length_only | isotonic (SWE-Gym dev) | 0.167 | 0.244 | 0.709 | 0.588 | 0.675 |
| telemetry_lr | none (SWE-Gym dev) | 0.218 | 0.261 | 0.812 | 0.655 | 0.696 |
| telemetry_lr | platt (SWE-Gym dev) | 0.218 | 0.261 | 0.808 | 0.655 | 0.696 |
| telemetry_lr | isotonic (SWE-Gym dev) | 0.218 | 0.262 | 0.900 | 0.655 | 0.694 |
| telemetry_gbm | none (SWE-Gym dev) | 0.240 | 0.275 | 0.923 | 0.636 | 0.720 |
| telemetry_gbm | platt (SWE-Gym dev) | 0.218 | 0.266 | 0.823 | 0.636 | 0.720 |
| telemetry_gbm | isotonic (SWE-Gym dev) | 0.221 | 0.269 | 0.824 | 0.635 | 0.712 |
| length_only | none (SWE-Gym dev) | 0.131 | 0.242 | 0.711 | 0.591 | 0.666 |
| length_only | platt (SWE-Gym dev) | 0.131 | 0.242 | 0.711 | 0.591 | 0.666 |
| length_only | isotonic (SWE-Gym dev) | 0.167 | 0.244 | 0.709 | 0.588 | 0.675 |

### B. Lite few-shot (K=5 tasks): 1179 runs, base rate 0.341

| predictor | calibrator | ECE | Brier | NLL | pooled AUROC | within-task AUROC |
|---|---|---|---|---|---|---|
| telemetry_lr | none | 0.215 ± 0.001 | 0.260 ± 0.001 | 0.807 ± 0.003 | 0.654 | 0.692 |
| telemetry_lr | prior (dev rate -> K-task rate) | 0.192 ± 0.032 | 0.256 ± 0.014 | 0.714 ± 0.036 | 0.654 | 0.692 |
| telemetry_lr | platt (dev) + prior | 0.193 ± 0.033 | 0.256 ± 0.015 | 0.713 ± 0.036 | 0.654 | 0.692 |
| telemetry_lr | platt (refit on K tasks) | 0.175 ± 0.089 | 0.250 ± 0.042 | 0.705 ± 0.115 | 0.654 | 0.692 |
| telemetry_gbm | none | 0.237 ± 0.003 | 0.273 ± 0.001 | 0.915 ± 0.005 | 0.636 | 0.718 |
| telemetry_gbm | prior (dev rate -> K-task rate) | 0.165 ± 0.014 | 0.259 ± 0.008 | 0.754 ± 0.023 | 0.636 | 0.718 |
| telemetry_gbm | platt (dev) + prior | 0.183 ± 0.041 | 0.262 ± 0.014 | 0.737 ± 0.037 | 0.636 | 0.718 |
| telemetry_gbm | platt (refit on K tasks) | 0.165 ± 0.075 | 0.247 ± 0.027 | 0.693 ± 0.065 | 0.636 | 0.718 |
| length_only | none | 0.133 ± 0.001 | 0.241 ± 0.000 | 0.710 ± 0.001 | 0.588 | 0.662 |
| length_only | prior (dev rate -> K-task rate) | 0.367 ± 0.037 | 0.363 ± 0.025 | 1.028 ± 0.074 | 0.588 | 0.662 |
| length_only | platt (dev) + prior | 0.367 ± 0.037 | 0.363 ± 0.025 | 1.027 ± 0.074 | 0.588 | 0.662 |
| length_only | platt (refit on K tasks) | 0.164 ± 0.035 | 0.256 ± 0.013 | 0.727 ± 0.033 | 0.588 | 0.662 |

### B. Lite few-shot (K=20 tasks): 1119 runs, base rate 0.347

| predictor | calibrator | ECE | Brier | NLL | pooled AUROC | within-task AUROC |
|---|---|---|---|---|---|---|
| telemetry_lr | none | 0.221 ± 0.007 | 0.262 ± 0.004 | 0.811 ± 0.015 | 0.663 | 0.698 |
| telemetry_lr | prior (dev rate -> K-task rate) | 0.105 ± 0.014 | 0.222 ± 0.003 | 0.641 ± 0.014 | 0.663 | 0.698 |
| telemetry_lr | platt (dev) + prior | 0.104 ± 0.015 | 0.221 ± 0.003 | 0.638 ± 0.014 | 0.663 | 0.698 |
| telemetry_lr | platt (refit on K tasks) | 0.115 ± 0.048 | 0.225 ± 0.019 | 0.644 ± 0.046 | 0.663 | 0.698 |
| telemetry_gbm | none | 0.239 ± 0.001 | 0.276 ± 0.003 | 0.925 ± 0.010 | 0.639 | 0.719 |
| telemetry_gbm | prior (dev rate -> K-task rate) | 0.156 ± 0.013 | 0.247 ± 0.003 | 0.731 ± 0.022 | 0.639 | 0.719 |
| telemetry_gbm | platt (dev) + prior | 0.105 ± 0.015 | 0.232 ± 0.004 | 0.663 ± 0.011 | 0.639 | 0.719 |
| telemetry_gbm | platt (refit on K tasks) | 0.093 ± 0.066 | 0.228 ± 0.019 | 0.649 ± 0.049 | 0.639 | 0.719 |
| length_only | none | 0.136 ± 0.006 | 0.244 ± 0.003 | 0.716 ± 0.009 | 0.591 | 0.670 |
| length_only | prior (dev rate -> K-task rate) | 0.201 ± 0.066 | 0.275 ± 0.027 | 0.782 ± 0.073 | 0.591 | 0.670 |
| length_only | platt (dev) + prior | 0.201 ± 0.066 | 0.275 ± 0.027 | 0.782 ± 0.073 | 0.591 | 0.670 |
| length_only | platt (refit on K tasks) | 0.087 ± 0.052 | 0.230 ± 0.013 | 0.659 ± 0.041 | 0.591 | 0.670 |

### C. verifier runs (5-fold by task): 237 runs, base rate 0.494

| predictor | calibrator | ECE | Brier | NLL | pooled AUROC | within-task AUROC |
|---|---|---|---|---|---|---|
| swegym_verifier_32b | none | 0.290 | 0.313 | 0.893 | 0.734 | 0.741 |
| swegym_verifier_32b | prior | 0.287 | 0.312 | 0.890 | 0.691 | 0.741 |
| swegym_verifier_32b | platt | 0.100 | 0.230 | 0.669 | 0.667 | 0.741 |
| swegym_verifier_32b | isotonic | 0.131 | 0.216 | 0.677 | 0.696 | 0.725 |
| telemetry_lr | none | 0.361 | 0.376 | 1.123 | 0.604 | 0.589 |
| telemetry_lr | prior | 0.142 | 0.257 | 0.722 | 0.597 | 0.589 |
| telemetry_lr | platt | 0.094 | 0.243 | 0.679 | 0.584 | 0.589 |
| telemetry_lr | isotonic | 0.086 | 0.245 | 0.730 | 0.580 | 0.614 |

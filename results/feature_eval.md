Dev only, out-of-fold, 4907 runs. Diff CIs: extended minus base, paired task bootstrap.

### A. Full-run models, extended vs base

| model | subset | runs | pairs | pooled_base | pooled_ext | pooled_diff_ci | within_base | within_ext | within_diff_ci |
|---|---|---|---|---|---|---|---|---|---|
| lr | all | 4907 | 1199 | 0.844 | 0.852 | +0.008 [+0.001, +0.015] | 0.865 | 0.868 | +0.003 [-0.011, +0.017] |
| lr | non_empty_patch | 2979 | 511 | 0.727 | 0.741 | +0.014 [+0.002, +0.026] | 0.683 | 0.691 | +0.008 [-0.025, +0.043] |
| lr | gpt4o_only | 4716 | 763 | 0.850 | 0.855 | +0.005 [-0.002, +0.013] | 0.870 | 0.874 | +0.004 [-0.014, +0.023] |
| gbm | all | 4907 | 1199 | 0.842 | 0.854 | +0.013 [+0.004, +0.022] | 0.847 | 0.869 | +0.023 [-0.008, +0.052] |
| gbm | non_empty_patch | 2979 | 511 | 0.724 | 0.746 | +0.022 [+0.008, +0.037] | 0.640 | 0.693 | +0.053 [-0.011, +0.126] |
| gbm | gpt4o_only | 4716 | 763 | 0.853 | 0.861 | +0.008 [-0.001, +0.016] | 0.858 | 0.864 | +0.005 [-0.020, +0.031] |

### B. Checkpoint models, extended vs base

| after_actions | model | runs | pairs | pooled_base | pooled_ext | pooled_diff_ci | within_base | within_ext | recall_base | recall_ext |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | lr | 3313 | 650 | 0.641 | 0.635 | -0.005 [-0.019, +0.007] | 0.670 | 0.679 | 0.243 | 0.242 |
| 5 | gbm | 3313 | 650 | 0.619 | 0.622 | +0.003 [-0.012, +0.018] | 0.622 | 0.644 | 0.242 | 0.251 |
| 10 | lr | 3022 | 540 | 0.609 | 0.607 | -0.003 [-0.020, +0.013] | 0.624 | 0.626 | 0.174 | 0.179 |
| 10 | gbm | 3022 | 540 | 0.597 | 0.595 | -0.002 [-0.026, +0.019] | 0.581 | 0.606 | 0.189 | 0.191 |
| 15 | lr | 2399 | 304 | 0.537 | 0.530 | -0.007 [-0.037, +0.024] | 0.562 | 0.562 | 0.121 | 0.089 |
| 15 | gbm | 2399 | 304 | 0.554 | 0.577 | +0.023 [-0.015, +0.059] | 0.553 | 0.576 | 0.115 | 0.170 |
| 20 | lr | 1871 | 196 | 0.535 | 0.549 | +0.014 [-0.031, +0.054] | 0.587 | 0.684 | 0.178 | 0.164 |
| 20 | gbm | 1871 | 196 | 0.546 | 0.597 | +0.051 [+0.007, +0.094] | 0.510 | 0.566 | 0.152 | 0.142 |
| 30 | lr | 1006 | 22 | 0.469 | 0.501 | +0.031 [-0.025, +0.089] | 0.364 | 0.364 | 0.065 | 0.101 |
| 30 | gbm | 1006 | 22 | 0.604 | 0.631 | +0.027 [-0.036, +0.087] | 0.318 | 0.409 | 0.186 | 0.274 |
| 40 | lr | 755 | 11 | 0.353 | 0.331 | -0.022 [-0.121, +0.071] | 0.273 | 0.273 | 0.071 | 0.046 |
| 40 | gbm | 755 | 11 | 0.579 | 0.447 | -0.132 [-0.230, -0.032] | 0.727 | 0.818 | 0.192 | 0.143 |

### C. Family ablation

| model | dropped | pooled | within | pooled_non_empty | within_non_empty |
|---|---|---|---|---|---|
| lr | none (extended) | 0.852 | 0.868 | 0.741 | 0.691 |
| lr | verification | 0.847 | 0.862 | 0.732 | 0.677 |
| lr | stuck | 0.852 | 0.872 | 0.742 | 0.701 |
| lr | pacing | 0.853 | 0.866 | 0.742 | 0.685 |
| lr | cognitive | 0.852 | 0.869 | 0.741 | 0.693 |
| gbm | none (extended) | 0.854 | 0.869 | 0.746 | 0.693 |
| gbm | verification | 0.852 | 0.861 | 0.742 | 0.673 |
| gbm | stuck | 0.854 | 0.875 | 0.745 | 0.706 |
| gbm | pacing | 0.849 | 0.851 | 0.736 | 0.650 |
| gbm | cognitive | 0.854 | 0.865 | 0.744 | 0.683 |

### D. Single features

| feature | share_non_default | mean_pass | mean_fail | auroc_all | auroc_non_empty |
|---|---|---|---|---|---|
| last_check_passed | 0.411 | 0.878 | 0.369 | 0.755 | 0.622 |
| checked_after_last_edit | 0.499 | 0.943 | 0.460 | 0.741 | 0.581 |
| fail_then_pass | 0.253 | 0.438 | 0.237 | 0.600 | 0.516 |
| max_identical_streak | 0.607 | 1.450 | 2.519 | 0.280 | 0.405 |
| max_error_streak | 0.578 | 1.211 | 1.213 | 0.539 | 0.393 |
| error_responsiveness | 0.074 | 0.988 | 0.983 | 0.510 | 0.529 |
| first_edit_step | 0.674 | 7.015 | 7.204 | 0.535 | 0.434 |
| phase_entropy | 0.680 | 1.977 | 1.075 | 0.803 | 0.662 |
| thought_action_mismatch | 0.307 | 0.225 | 0.124 | 0.576 | 0.485 |

### E. Adoption rule

| model | rises_on_hard_subset | holds_on_all_runs | use_extended |
|---|---|---|---|
| end of run | True | True | True |
| after 5 actions | False | False | False |
| after 10 actions | False | True | False |
| after 15 actions | False | False | False |
| after 20 actions | True | True | True |
| after 30 actions | True | True | True |
| after 40 actions | False | False | False |

Runs with any thought-action mismatch: 0.307; by model: claude-3-5-sonnet-20241022 0.277, gpt-4o-2024-08-06 0.308

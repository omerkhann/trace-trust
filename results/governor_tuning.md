Dev only (4907 runs, base rate 0.082), out-of-fold scores.

### A. Hint patterns

| pattern | runs_fired | pass_rate_fired | pass_rate_not_fired | failing_runs_covered | passing_runs_hit | median_first_step | hint_worthy |
|---|---|---|---|---|---|---|---|
| finish_without_edit | 0.042 | 0.000 | 0.085 | 0.045 | 0.000 | 6.000 | True |
| finish_unverified | 0.291 | 0.187 | 0.039 | 0.258 | 0.664 | 18.000 | False |
| repeat_loop | 0.529 | 0.032 | 0.138 | 0.558 | 0.204 | 4.000 | True |
| error_streak | 0.116 | 0.079 | 0.082 | 0.116 | 0.112 | 15.000 | False |
| no_edit_yet | 0.063 | 0.061 | 0.083 | 0.065 | 0.047 | 15.000 | False |
| blind_edit | 0.009 | 0.067 | 0.082 | 0.009 | 0.007 | 19.000 | False |

### A2. Loop definition

| window | repeats | runs_fired | pass_rate_fired | pass_rate_not_fired | failing_runs_covered | passing_runs_hit |
|---|---|---|---|---|---|---|
| 8 | 3 | 0.756 | 0.046 | 0.195 | 0.786 | 0.420 |
| 8 | 4 | 0.529 | 0.032 | 0.138 | 0.558 | 0.204 |
| 8 | 5 | 0.065 | 0.038 | 0.085 | 0.068 | 0.030 |
| 12 | 4 | 0.590 | 0.039 | 0.144 | 0.618 | 0.279 |
| 12 | 5 | 0.196 | 0.057 | 0.088 | 0.202 | 0.137 |
| 12 | 6 | 0.107 | 0.042 | 0.087 | 0.112 | 0.055 |
| 16 | 6 | 0.161 | 0.047 | 0.089 | 0.168 | 0.092 |

### B. Abort budget

| far_per_checkpoint | checkpoints | runs_aborted | failures_stopped | passing_runs_lost | passes_lost_n | tokens_saved |
|---|---|---|---|---|---|---|
| 0.005 | [5, 10] | 93 | 0.020 | 0.012 | 5 | 0.014 |
| 0.010 | [5, 10] | 152 | 0.032 | 0.022 | 9 | 0.023 |
| 0.025 | [5, 10] | 322 | 0.068 | 0.040 | 16 | 0.056 |
| 0.050 | [5, 10] | 557 | 0.117 | 0.070 | 28 | 0.105 |
| 0.100 | [5, 10] | 935 | 0.194 | 0.149 | 60 | 0.190 |

### C. Review budget

| far | tau | runs_flagged | failures_flagged | passing_flagged |
|---|---|---|---|---|
| 0.050 | 0.034 | 0.515 | 0.556 | 0.052 |
| 0.100 | 0.080 | 0.632 | 0.680 | 0.102 |
| 0.200 | 0.118 | 0.706 | 0.751 | 0.201 |

### D. Loop escape

| group | runs | pass_rate | empty_patch | median_more_actions |
|---|---|---|---|---|
| escaped (>= 5 more actions) | 1060 | 0.059 | 0.078 | 17.000 |
| stayed stuck | 1535 | 0.012 | 0.879 | 0.000 |

### E. REJECT vs the empty-patch rule

| policy | runs_stopped | failures_stopped | passing_lost | reviewed | pass_rate_in_review |
|---|---|---|---|---|---|
| agent (ABORT + REJECT) | 0.666 | 0.712 | 0.149 | 0.334 | 0.209 |
| rule: reject if patch empty | 0.393 | 0.428 | 0.000 | 0.607 | 0.135 |
| share of agent's stops with an empty patch | 0.590 | nan | nan | nan | nan |
| stops among non-empty patches only | 0.450 | 0.497 | 0.149 | nan | nan |

### F. Reroute, mean and std over seeds

| policy | solved_mean | solved_std | tokens_per_task_mean | tokens_per_task_std | reviews_per_task_mean | reviews_per_task_std | solved_per_million_tokens_mean | solved_per_million_tokens_std | solved_per_review_mean | solved_per_review_std | subset | tasks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random_attempt | 0.069 | 0.002 | 14522.859 | 103.894 | 1.000 | 0.000 | 4.779 | 0.111 | 0.069 | 0.002 | all_tasks | 1950 |
| review_all | 0.118 | 0.000 | 34292.007 | 0.000 | 2.250 | 0.009 | 3.454 | 0.000 | 0.053 | 0.000 | all_tasks | 1950 |
| governor_single | 0.063 | 0.001 | 12819.806 | 83.924 | 0.347 | 0.004 | 4.894 | 0.091 | 0.181 | 0.004 | all_tasks | 1950 |
| governor_reroute | 0.092 | 0.002 | 21429.485 | 175.640 | 0.523 | 0.000 | 4.291 | 0.064 | 0.176 | 0.004 | all_tasks | 1950 |
| random_attempt | 0.067 | 0.002 | 12961.561 | 139.430 | 1.000 | 0.000 | 5.168 | 0.166 | 0.067 | 0.002 | multi_attempt_tasks | 1453 |
| review_all | 0.133 | 0.000 | 39492.764 | 0.000 | 2.677 | 0.012 | 3.363 | 0.000 | 0.050 | 0.000 | multi_attempt_tasks | 1453 |
| governor_single | 0.060 | 0.002 | 11531.768 | 112.630 | 0.340 | 0.005 | 5.212 | 0.136 | 0.177 | 0.006 | multi_attempt_tasks | 1453 |
| governor_reroute | 0.099 | 0.003 | 23086.395 | 235.718 | 0.576 | 0.000 | 4.302 | 0.080 | 0.172 | 0.005 | multi_attempt_tasks | 1453 |

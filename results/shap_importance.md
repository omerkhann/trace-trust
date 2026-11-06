Mean |SHAP| in log-odds, telemetry20_lr fit on SWE-Gym dev (exact linear SHAP). Rank 1 = most important of 20. coef: standardised coefficient; positive means higher values favour success.

| feature | coef | test | rank: test | rank: test non-empty | rank: Lite | rank: Moatless |
|---|---|---|---|---|---|---|
| patch_empty | -2.23 | 2.183 | 1 | 1 | 1 | 1 |
| patch_lines | -0.77 | 0.716 | 2 | 2 | 2 | 4 |
| phase_entropy | +0.65 | 0.608 | 3 | 3 | 3 | 10 |
| n_steps | -0.61 | 0.547 | 4 | 4 | 5 | 2 |
| n_distinct_tools | +0.56 | 0.505 | 5 | 5 | 4 | 3 |
| last_check_passed | +0.33 | 0.331 | 6 | 6 | 6 | 5 |
| checked_after_last_edit | +0.22 | 0.221 | 7 | 7 | 8 | 8 |
| max_identical_streak (H3) | -0.22 | 0.201 | 8 | 8 | 7 | 7 |
| patch_files | +0.21 | 0.178 | 9 | 10 | 9 | 15 |
| n_errors | +0.20 | 0.164 | 10 | 9 | 10 | 9 |
| total_tokens | -0.11 | 0.096 | 11 | 12 | 12 | 11 |
| n_repeated_commands (H3) | -0.13 | 0.090 | 12 | 11 | 11 | 6 |
| finished | -0.06 | 0.057 | 13 | 13 | 14 | 13 |
| first_edit_step | -0.07 | 0.050 | 14 | 14 | 13 | 16 |
| fail_then_pass | +0.04 | 0.036 | 15 | 15 | 15 | 17 |
| max_error_streak | -0.05 | 0.036 | 16 | 16 | 17 | 14 |
| thought_action_mismatch (H3) | +0.03 | 0.020 | 17 | 18 | 19 | 18 |
| error_responsiveness (H3) | +0.05 | 0.020 | 18 | 17 | 18 | 19 |
| ran_tests | +0.02 | 0.018 | 19 | 19 | 16 | 20 |
| n_edits_without_view | +0.03 | 0.004 | 20 | 20 | 20 | 12 |

### Share of total |SHAP| by family

| family | dev | test | test_non_empty | lite | moatless_8runs |
|---|---|---|---|---|---|
| volume_pacing | 11.4% | 11.4% | 10.8% | 9.7% | 11.7% |
| action_phase | 17.9% | 18.3% | 16.9% | 16.3% | 11.3% |
| error_loop | 8.7% | 8.4% | 10.3% | 10.0% | 16.4% |
| verification | 10.9% | 10.9% | 13.4% | 12.2% | 12.4% |
| cognitive | 0.4% | 0.4% | 0.5% | 0.6% | 1.7% |
| patch | 50.7% | 50.6% | 48.0% | 51.2% | 46.6% |

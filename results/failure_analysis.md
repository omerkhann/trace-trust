Wrong decisions of the locked agent, from the saved E3 outcomes. Lite: seed 42 draw. Categories defined in experiments/failure_analysis.py.

### test: 1148 runs

Correct: 729 failures stopped, 78 passes sent to review. Wrong: 330 missed failures, 11 lost passes (4 aborted, 7 rejected). Pass rate of all reviewed runs: 19.1 %.

| category | missed failures | share | reviewed runs of this kind | their pass rate | median actions | median patch lines |
|---|---|---|---|---|---|---|
| M1 looks verified | 285 | 86.4% | 361 | 21.1% | 18.0 | 28.0 |
| M2 check not green | 31 | 9.4% | 31 | 0.0% | 17.0 | 28.0 |
| M3 not verified | 14 | 4.2% | 16 | 12.5% | 13.0 | 14.0 |

Rejected passes, feature that lowered the score most: {'patch_lines': 5, 'n_steps': 2}. Aborted passes, by checkpoint: {5.0: 4}.
Hint false alarms: 23 passing runs hinted, of 578 hinted; by pattern {'repeat_loop': 23}.

### lite: 1119 runs

Correct: 187 failures stopped, 357 passes sent to review. Wrong: 544 missed failures, 31 lost passes (0 aborted, 31 rejected). Pass rate of all reviewed runs: 39.6 %.

| category | missed failures | share | reviewed runs of this kind | their pass rate | median actions | median patch lines |
|---|---|---|---|---|---|---|
| M1 looks verified | 433 | 79.6% | 757 | 42.8% | 18.0 | 53.0 |
| M2 check not green | 50 | 9.2% | 68 | 26.5% | 30.0 | 68.5 |
| M3 not verified | 61 | 11.2% | 76 | 19.7% | 30.0 | 67.0 |

Rejected passes, feature that lowered the score most: {'patch_lines': 29, 'max_identical_streak': 2}. Aborted passes, by checkpoint: {}.
Hint false alarms: 134 passing runs hinted, of 431 hinted; by pattern {'repeat_loop': 134}.

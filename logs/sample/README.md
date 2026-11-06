# Sample agent logs

One JSON line per agent action (what it observed, tool calls with inputs and outputs, phase change, decision, reason), then the run's outcome. Runs were chosen by the fixed rule in experiments/failure_analysis.py. test_* are SWE-Gym held-out runs, lite_* are SWE-bench Lite runs. All logs are produced by `python -m experiments.governor_eval` into logs/e3_*/ (not committed; about 30 MB). results/failure_cases.md prints the test runs readably.

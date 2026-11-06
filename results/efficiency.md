Timed on Windows-10-10.0.26300-SP0, CPU only, over 1148 SWE-Gym test runs (mean 19 actions), 3 passes, mean ± std.

| step | time per run |
|---|---|
| feature extraction | 0.30 ± 0.01 ms |
| end-of-run score | 5.15 ± 0.46 ms |
| full agent replay | 22.04 ± 0.14 ms |
| 7B judge (RTX 4050, 50 runs) | 10.0 s |
| 32B verifier (Kaggle 2 x T4, 237 runs) | 75.7 s |

Full agent replay is 0.029 % of the verifier's time per run (H3 target: under 1 %), on different hardware. Per action: 1.16 ms. Model fit at startup: 2.2 s. Peak traced Python memory (fit + replay of 200 runs): 6 MB. Tool calls per run: 16.0. The agent uses no LLM tokens, GPU or paid API. Failed tool calls in E3: 0.

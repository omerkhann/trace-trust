telemetry20_lr end-of-run model, 7.8% of passing labels flipped, seeds [42, 43, 44], mean ± std.

| setting | pooled AUROC | within-task AUROC | ECE | failures rejected | passes rejected | reviewed | pass rate in review |
|---|---|---|---|---|---|---|---|
| train clean, test as released | 0.836 ± 0.000 | 0.779 ± 0.000 | 0.013 | 65.8% | 11.2% | 38.4% | 17.9% |
| train noisy, test as released | 0.836 ± 0.001 | 0.773 ± 0.003 | 0.012 | 63.8% | 10.5% | 40.3% | 17.2% |
| train noisy, test noisy | 0.839 ± 0.007 | 0.765 ± 0.002 | 0.013 | 63.5% | 9.8% | 40.3% | 16.0% |

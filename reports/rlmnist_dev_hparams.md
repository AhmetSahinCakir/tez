# Hyper-parameter search: results/rlmnist_dev

Metric: `auc_norm` (higher is better), mean ± std over seeds.

| base | variant | auc_norm (mean ± std) | n | seeds | chosen |
|---|---|---|---|---|---|
| baseline | baseline/lr=0.001 | 0.2994 ± 0.0000 | 1 | 0 | * |
| baseline | baseline/lr=0.03 | 0.2069 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.003 | 0.1927 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.01 | 0.1837 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.001/gamma=2.5 | 0.3068 ± 0.0000 | 1 | 0 | * |
| sin | sin/lr=0.001/gamma=1.5 | 0.2879 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.003/gamma=1.5 | 0.2752 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.003/gamma=2.5 | 0.2739 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.03/gamma=1.5 | 0.2011 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01/gamma=1.5 | 0.1965 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01/gamma=2.5 | 0.1936 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.03/gamma=2.5 | 0.1749 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.001/gamma=1.5 | 0.2895 ± 0.0000 | 1 | 0 | * |
| tanh | tanh/lr=0.001/gamma=2.5 | 0.2890 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.003/gamma=1.5 | 0.2761 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.003/gamma=2.5 | 0.2716 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01/gamma=1.5 | 0.2254 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01/gamma=2.5 | 0.2079 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.03/gamma=2.5 | 0.1798 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.03/gamma=1.5 | 0.1759 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.03/gamma=1.0 | 0.2896 ± 0.0000 | 1 | 0 | * |
| tri | tri/lr=0.001/gamma=1.5 | 0.2816 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.03/gamma=1.5 | 0.2495 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.003/gamma=1.5 | 0.2368 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.001/gamma=1.0 | 0.2279 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.01/gamma=1.0 | 0.2239 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.003/gamma=1.0 | 0.2113 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.01/gamma=1.5 | 0.1866 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.001/kappa=2.0 | 0.3073 ± 0.0000 | 1 | 0 | * |
| weight_clipping | weight_clipping/lr=0.03/kappa=1.0 | 0.2839 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.003/kappa=2.0 | 0.2629 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.03/kappa=2.0 | 0.2602 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.001/kappa=1.0 | 0.2418 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.01/kappa=1.0 | 0.2169 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.003/kappa=1.0 | 0.2133 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.01/kappa=2.0 | 0.1977 ± 0.0000 | 1 | 0 |  |

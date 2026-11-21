# Hyper-parameter search: results/rlmnist_ctrl_dev

Metric: `auc_norm` (higher is better), mean ± std over seeds.

| base | variant | auc_norm (mean ± std) | n | seeds | chosen |
|---|---|---|---|---|---|
| l2_init | l2_init/lr=0.01/lam=0.01 | 0.9513 ± 0.0000 | 1 | 0 | * |
| l2_init | l2_init/lr=0.03/lam=0.001 | 0.9365 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.03/lam=0.01 | 0.9016 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.003/lam=0.01 | 0.8793 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.01/lam=0.001 | 0.7087 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.001/lam=0.01 | 0.6349 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.003/lam=0.001 | 0.4331 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.001/lam=0.001 | 0.3824 ± 0.0000 | 1 | 0 |  |
| stdin_baseline | stdin_baseline/lr=0.001 | 0.2424 ± 0.0000 | 1 | 0 | * |
| stdin_baseline | stdin_baseline/lr=0.003 | 0.2056 ± 0.0000 | 1 | 0 |  |
| stdin_baseline | stdin_baseline/lr=0.01 | 0.1801 ± 0.0000 | 1 | 0 |  |
| stdin_baseline | stdin_baseline/lr=0.03 | 0.1748 ± 0.0000 | 1 | 0 |  |
| stdin_tri | stdin_tri/lr=0.03/gamma=1.0 | 0.6305 ± 0.0000 | 1 | 0 | * |
| stdin_tri | stdin_tri/lr=0.001/gamma=1.0 | 0.5354 ± 0.0000 | 1 | 0 |  |
| stdin_tri | stdin_tri/lr=0.003/gamma=1.0 | 0.3006 ± 0.0000 | 1 | 0 |  |
| stdin_tri | stdin_tri/lr=0.001/gamma=1.5 | 0.2786 ± 0.0000 | 1 | 0 |  |
| stdin_tri | stdin_tri/lr=0.03/gamma=1.5 | 0.2521 ± 0.0000 | 1 | 0 |  |
| stdin_tri | stdin_tri/lr=0.01/gamma=1.0 | 0.2507 ± 0.0000 | 1 | 0 |  |
| stdin_tri | stdin_tri/lr=0.003/gamma=1.5 | 0.2200 ± 0.0000 | 1 | 0 |  |
| stdin_tri | stdin_tri/lr=0.01/gamma=1.5 | 0.2085 ± 0.0000 | 1 | 0 |  |

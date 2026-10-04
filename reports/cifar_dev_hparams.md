# Hyper-parameter search: results/cifar_dev

Metric: `auc_norm` (higher is better), mean ± std over seeds.

| base | variant | auc_norm (mean ± std) | n | seeds | chosen |
|---|---|---|---|---|---|
| baseline | baseline/lr=0.003 | 0.7463 ± 0.0000 | 1 | 0 | * |
| baseline | baseline/lr=0.001 | 0.7357 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.01 | 0.7197 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.03 | 0.5241 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.1 | 0.5000 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.003/replacement_rate=0.0001 | 0.7534 ± 0.0000 | 1 | 0 | * |
| continual_backprop | continual_backprop/lr=0.003/replacement_rate=0.001 | 0.7488 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.01/replacement_rate=0.0001 | 0.7359 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.01/replacement_rate=0.001 | 0.7218 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.03/replacement_rate=0.0001 | 0.5095 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.03/replacement_rate=0.001 | 0.5011 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.1/replacement_rate=0.0001 | 0.5000 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.1/replacement_rate=0.001 | 0.5000 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.003/lam=0.01 | 0.7527 ± 0.0000 | 1 | 0 | * |
| l2_init | l2_init/lr=0.003/lam=0.001 | 0.7520 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.01/lam=0.001 | 0.7480 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.01/lam=0.01 | 0.7429 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.03/lam=0.01 | 0.6634 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.03/lam=0.001 | 0.6518 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.1/lam=0.001 | 0.5000 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.1/lam=0.01 | 0.5000 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.003/weight_decay=0.001 | 0.7420 ± 0.0000 | 1 | 0 | * |
| ln_wd | ln_wd/lr=0.01/weight_decay=0.001 | 0.7416 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.01/weight_decay=0.0001 | 0.7359 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.003/weight_decay=0.0001 | 0.7351 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.03/weight_decay=0.0001 | 0.7244 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.03/weight_decay=0.001 | 0.6937 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.1/weight_decay=0.001 | 0.5612 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.1/weight_decay=0.0001 | 0.5466 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.03 | 0.7545 ± 0.0000 | 1 | 0 | * |
| nap | nap/lr=0.01 | 0.7499 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.1 | 0.7435 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.003 | 0.7412 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.3 | 0.7072 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.003/gamma=5.0 | 0.7481 ± 0.0000 | 1 | 0 | * |
| sin | sin/lr=0.003/gamma=2.5 | 0.7477 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.003 | 0.7412 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.001 | 0.7311 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01/gamma=5.0 | 0.7245 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01/gamma=2.5 | 0.7227 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01 | 0.7047 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.03 | 0.5524 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.1 | 0.5010 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.003/gamma=5.0 | 0.7471 ± 0.0000 | 1 | 0 | * |
| tanh | tanh/lr=0.003/gamma=2.5 | 0.7456 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.003 | 0.7414 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01/gamma=2.5 | 0.7393 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01 | 0.7319 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.001 | 0.7290 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01/gamma=5.0 | 0.7235 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.03 | 0.6736 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.1 | 0.5088 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.003/kappa=1.0 | 0.7502 ± 0.0000 | 1 | 0 | * |
| weight_clipping | weight_clipping/lr=0.003/kappa=2.0 | 0.7487 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.01/kappa=1.0 | 0.7360 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.01/kappa=2.0 | 0.7196 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.03/kappa=1.0 | 0.6214 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.03/kappa=2.0 | 0.5154 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.1/kappa=2.0 | 0.5005 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.1/kappa=1.0 | 0.5003 ± 0.0000 | 1 | 0 |  |

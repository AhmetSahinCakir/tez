# Hyper-parameter search: results/pmnist_dev

Metric: `auc_norm` (higher is better), mean ± std over seeds.

| base | variant | auc_norm (mean ± std) | n | seeds | chosen |
|---|---|---|---|---|---|
| baseline | baseline/lr=0.003 | 0.7954 ± 0.0000 | 1 | 0 | * |
| baseline | baseline/lr=0.001 | 0.7864 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.01 | 0.7502 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.03 | 0.4489 ± 0.0000 | 1 | 0 |  |
| baseline | baseline/lr=0.1 | 0.1038 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.003/replacement_rate=0.0001 | 0.8217 ± 0.0000 | 1 | 0 | * |
| continual_backprop | continual_backprop/lr=0.003/replacement_rate=0.001 | 0.8198 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.01/replacement_rate=0.0001 | 0.8140 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.01/replacement_rate=0.001 | 0.8099 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.03/replacement_rate=0.0001 | 0.7562 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.03/replacement_rate=0.001 | 0.7536 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.1/replacement_rate=0.001 | 0.1032 ± 0.0000 | 1 | 0 |  |
| continual_backprop | continual_backprop/lr=0.1/replacement_rate=0.0001 | 0.1031 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.003/lam=0.01 | 0.8317 ± 0.0000 | 1 | 0 | * |
| l2_init | l2_init/lr=0.003/lam=0.001 | 0.8233 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.01/lam=0.001 | 0.8156 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.01/lam=0.01 | 0.8136 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.03/lam=0.001 | 0.7574 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.03/lam=0.01 | 0.7238 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.1/lam=0.01 | 0.3542 ± 0.0000 | 1 | 0 |  |
| l2_init | l2_init/lr=0.1/lam=0.001 | 0.1014 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.003/weight_decay=0.001 | 0.8217 ± 0.0000 | 1 | 0 | * |
| ln_wd | ln_wd/lr=0.01/weight_decay=0.001 | 0.8119 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.003/weight_decay=0.0001 | 0.8106 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.01/weight_decay=0.0001 | 0.8029 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.03/weight_decay=0.0001 | 0.7909 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.03/weight_decay=0.001 | 0.7905 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.1/weight_decay=0.0001 | 0.6233 ± 0.0000 | 1 | 0 |  |
| ln_wd | ln_wd/lr=0.1/weight_decay=0.001 | 0.3869 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.01 | 0.8322 ± 0.0000 | 1 | 0 | * |
| nap | nap/lr=0.003 | 0.8300 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.03 | 0.8272 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.1 | 0.7836 ± 0.0000 | 1 | 0 |  |
| nap | nap/lr=0.3 | 0.6645 ± 0.0000 | 1 | 0 |  |
| parseval | parseval/lr=0.003/beta=0.01 | 0.8419 ± 0.0000 | 1 | 0 | * |
| parseval | parseval/lr=0.003/beta=0.001 | 0.8404 ± 0.0000 | 1 | 0 |  |
| parseval | parseval/lr=0.01/beta=0.001 | 0.8268 ± 0.0000 | 1 | 0 |  |
| parseval | parseval/lr=0.01/beta=0.01 | 0.8095 ± 0.0000 | 1 | 0 |  |
| shrink_perturb | shrink_perturb/lr=0.01/shrink=0.0001 | 0.8102 ± 0.0000 | 1 | 0 | * |
| shrink_perturb | shrink_perturb/lr=0.003/shrink=0.0001 | 0.7743 ± 0.0000 | 1 | 0 |  |
| shrink_perturb | shrink_perturb/lr=0.01/shrink=1e-05 | 0.7379 ± 0.0000 | 1 | 0 |  |
| shrink_perturb | shrink_perturb/lr=0.003/shrink=1e-05 | 0.7066 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.003 | 0.8011 ± 0.0000 | 1 | 0 | * |
| sin | sin/lr=0.003/gamma=2.5 | 0.8009 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.003/gamma=5.0 | 0.7970 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01 | 0.7862 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.001 | 0.7809 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01/gamma=2.5 | 0.7712 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.01/gamma=5.0 | 0.7520 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.03 | 0.6963 ± 0.0000 | 1 | 0 |  |
| sin | sin/lr=0.1 | 0.3750 ± 0.0000 | 1 | 0 |  |
| sin_act | sin_act/lr=0.003 | 0.8174 ± 0.0000 | 1 | 0 | * |
| sin_act | sin_act/lr=0.001 | 0.7951 ± 0.0000 | 1 | 0 |  |
| sin_act | sin_act/lr=0.01 | 0.7676 ± 0.0000 | 1 | 0 |  |
| sin_act | sin_act/lr=0.03 | 0.1107 ± 0.0000 | 1 | 0 |  |
| sin_act | sin_act/lr=0.1 | 0.1003 ± 0.0000 | 1 | 0 |  |
| sin_floor | sin_floor/lr=0.003/gamma=1.5 | 0.8034 ± 0.0000 | 1 | 0 | * |
| sin_floor | sin_floor/lr=0.01/gamma=1.5 | 0.7967 ± 0.0000 | 1 | 0 |  |
| sin_refl | sin_refl/lr=0.01/gamma=1.0 | 0.8289 ± 0.0000 | 1 | 0 | * |
| sin_refl | sin_refl/lr=0.003/gamma=1.0 | 0.8286 ± 0.0000 | 1 | 0 |  |
| sin_refl | sin_refl/lr=0.01/gamma=1.5 | 0.8129 ± 0.0000 | 1 | 0 |  |
| sin_refl | sin_refl/lr=0.003/gamma=1.5 | 0.8115 ± 0.0000 | 1 | 0 |  |
| smooth_leaky | smooth_leaky/lr=0.03 | 0.8110 ± 0.0000 | 1 | 0 | * |
| smooth_leaky | smooth_leaky/lr=0.01 | 0.8054 ± 0.0000 | 1 | 0 |  |
| smooth_leaky | smooth_leaky/lr=0.003 | 0.7702 ± 0.0000 | 1 | 0 |  |
| smooth_leaky | smooth_leaky/lr=0.001 | 0.7133 ± 0.0000 | 1 | 0 |  |
| smooth_leaky | smooth_leaky/lr=0.1 | 0.1879 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.003 | 0.8038 ± 0.0000 | 1 | 0 | * |
| tanh | tanh/lr=0.003/gamma=2.5 | 0.8007 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.003/gamma=5.0 | 0.7978 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01 | 0.7942 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01/gamma=2.5 | 0.7768 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.001 | 0.7734 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.03 | 0.7685 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.01/gamma=5.0 | 0.7625 ± 0.0000 | 1 | 0 |  |
| tanh | tanh/lr=0.1 | 0.7422 ± 0.0000 | 1 | 0 |  |
| tanh_refl | tanh_refl/lr=0.01/gamma=1.0 | 0.8110 ± 0.0000 | 1 | 0 | * |
| tanh_refl | tanh_refl/lr=0.003/gamma=1.5 | 0.8071 ± 0.0000 | 1 | 0 |  |
| tanh_refl | tanh_refl/lr=0.003/gamma=1.0 | 0.8066 ± 0.0000 | 1 | 0 |  |
| tanh_refl | tanh_refl/lr=0.01/gamma=1.5 | 0.7984 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.003/gamma=1.0 | 0.8353 ± 0.0000 | 1 | 0 | * |
| tri | tri/lr=0.003/gamma=1.5 | 0.8216 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.01/gamma=1.5 | 0.7998 ± 0.0000 | 1 | 0 |  |
| tri | tri/lr=0.01/gamma=1.0 | 0.7768 ± 0.0000 | 1 | 0 |  |
| upgd | upgd/lr=0.01/sigma=0.0001 | 0.8112 ± 0.0000 | 1 | 0 | * |
| upgd | upgd/lr=0.003/sigma=0.001 | 0.8105 ± 0.0000 | 1 | 0 |  |
| upgd | upgd/lr=0.003/sigma=0.0001 | 0.8100 ± 0.0000 | 1 | 0 |  |
| upgd | upgd/lr=0.01/sigma=0.001 | 0.8090 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.003/kappa=1.0 | 0.8341 ± 0.0000 | 1 | 0 | * |
| weight_clipping | weight_clipping/lr=0.003/kappa=2.0 | 0.8090 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.01/kappa=2.0 | 0.7947 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.01/kappa=1.0 | 0.7866 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.03/kappa=2.0 | 0.5500 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.03/kappa=1.0 | 0.3376 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.1/kappa=2.0 | 0.1165 ± 0.0000 | 1 | 0 |  |
| weight_clipping | weight_clipping/lr=0.1/kappa=1.0 | 0.1088 ± 0.0000 | 1 | 0 |  |

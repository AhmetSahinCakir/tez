
### results/pmnist_fullstream: tri − weight_clipping, eşdeğerlik marjı Δ = 0.005

| Ölçüt | n (ortak tohum) | Fark (a − b) | %95 bootstrap GA | %90 t-GA (TOST aralığı) | p_TOST | Karar |
|---|---:|---|---|---|---|---|
| auc_norm | 10 | +0.0003 | [-0.0000, +0.0006] | [+0.0000, +0.0006] | 0.000 | eşdeğer (Δ içinde) |
| final_window_mean | 10 | -0.0001 | [-0.0006, +0.0005] | [-0.0006, +0.0005] | 0.000 | eşdeğer (Δ içinde) |
| retention_ratio | 10 | -0.0003 | [-0.0015, +0.0008] | [-0.0015, +0.0008] | 0.000 | eşdeğer (Δ içinde) |
| fresh_gap_final | 10 | -0.0012 | [-0.0029, +0.0005] | [-0.0028, +0.0005] | 0.001 | eşdeğer (Δ içinde) |

### results/pmnist_fullstream: tri − weight_clipping, eşdeğerlik marjı Δ = 0.01

| Ölçüt | n (ortak tohum) | Fark (a − b) | %95 bootstrap GA | %90 t-GA (TOST aralığı) | p_TOST | Karar |
|---|---:|---|---|---|---|---|
| auc_norm | 10 | +0.0003 | [-0.0000, +0.0006] | [+0.0000, +0.0006] | 0.000 | eşdeğer (Δ içinde) |
| final_window_mean | 10 | -0.0001 | [-0.0006, +0.0005] | [-0.0006, +0.0005] | 0.000 | eşdeğer (Δ içinde) |
| retention_ratio | 10 | -0.0003 | [-0.0015, +0.0008] | [-0.0015, +0.0008] | 0.000 | eşdeğer (Δ içinde) |
| fresh_gap_final | 10 | -0.0012 | [-0.0029, +0.0005] | [-0.0028, +0.0005] | 0.000 | eşdeğer (Δ içinde) |

### results/pmnist_fullstream: tanh − baseline, eşdeğerlik marjı Δ = 0.005

| Ölçüt | n (ortak tohum) | Fark (a − b) | %95 bootstrap GA | %90 t-GA (TOST aralığı) | p_TOST | Karar |
|---|---:|---|---|---|---|---|
| auc_norm | 10 | +0.0026 | [+0.0015, +0.0038] | [+0.0015, +0.0038] | 0.002 | eşdeğer (Δ içinde) |
| final_window_mean | 10 | -0.0050 | [-0.0075, -0.0026] | [-0.0074, -0.0025] | 0.487 | karar verilemez |
| retention_ratio | 10 | -0.0123 | [-0.0148, -0.0099] | [-0.0147, -0.0098] | 1.000 | farklı (GA Δ dışında) |
| fresh_gap_final | 10 | +0.0008 | [-0.0036, +0.0051] | [-0.0035, +0.0051] | 0.053 | karar verilemez |

### results/pmnist_fullstream: tanh − baseline, eşdeğerlik marjı Δ = 0.01

| Ölçüt | n (ortak tohum) | Fark (a − b) | %95 bootstrap GA | %90 t-GA (TOST aralığı) | p_TOST | Karar |
|---|---:|---|---|---|---|---|
| auc_norm | 10 | +0.0026 | [+0.0015, +0.0038] | [+0.0015, +0.0038] | 0.000 | eşdeğer (Δ içinde) |
| final_window_mean | 10 | -0.0050 | [-0.0075, -0.0026] | [-0.0074, -0.0025] | 0.002 | eşdeğer (Δ içinde) |
| retention_ratio | 10 | -0.0123 | [-0.0148, -0.0099] | [-0.0147, -0.0098] | 0.939 | karar verilemez |
| fresh_gap_final | 10 | +0.0008 | [-0.0036, +0.0051] | [-0.0035, +0.0051] | 0.002 | eşdeğer (Δ içinde) |

### results/pmnist_fullstream: sin_refl − ln_wd, eşdeğerlik marjı Δ = 0.005

| Ölçüt | n (ortak tohum) | Fark (a − b) | %95 bootstrap GA | %90 t-GA (TOST aralığı) | p_TOST | Karar |
|---|---:|---|---|---|---|---|
| auc_norm | 3 | -0.0014 | [-0.0018, -0.0011] | [-0.0020, -0.0007] | 0.002 | eşdeğer (Δ içinde) |
| final_window_mean | 3 | -0.0025 | [-0.0033, -0.0019] | [-0.0037, -0.0013] | 0.012 | eşdeğer (Δ içinde) |
| retention_ratio | 3 | +0.0026 | [+0.0022, +0.0033] | [+0.0016, +0.0037] | 0.011 | eşdeğer (Δ içinde) |
| fresh_gap_final | 3 | -0.0047 | [-0.0062, -0.0035] | [-0.0070, -0.0024] | 0.380 | karar verilemez |

### results/pmnist_fullstream: sin_refl − ln_wd, eşdeğerlik marjı Δ = 0.01

| Ölçüt | n (ortak tohum) | Fark (a − b) | %95 bootstrap GA | %90 t-GA (TOST aralığı) | p_TOST | Karar |
|---|---:|---|---|---|---|---|
| auc_norm | 3 | -0.0014 | [-0.0018, -0.0011] | [-0.0020, -0.0007] | 0.000 | eşdeğer (Δ içinde) |
| final_window_mean | 3 | -0.0025 | [-0.0033, -0.0019] | [-0.0037, -0.0013] | 0.001 | eşdeğer (Δ içinde) |
| retention_ratio | 3 | +0.0026 | [+0.0022, +0.0033] | [+0.0016, +0.0037] | 0.001 | eşdeğer (Δ içinde) |
| fresh_gap_final | 3 | -0.0047 | [-0.0062, -0.0035] | [-0.0070, -0.0024] | 0.011 | eşdeğer (Δ içinde) |

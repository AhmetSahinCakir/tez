
### results/pmnist_fullstream: koruma ölçütü (koruma ≥ 0.99, eğim/100 ≥ -0.002, taze model farkı ≤ +0.01)

| Model | n | Koruma oranı | Eğim / 100 görev | Taze model farkı (son) | Ölçüt (1)(2)(3) geçen tohum | Karar |
|---|---:|---|---|---|---|---|
| baseline | 10 | 0.924 ± 0.002 | -0.0083 | 0.084 ± 0.008 | 0/0/0 (0/10 tümü) | **kaybediyor** |
| ln_wd | 10 | 0.992 ± 0.001 | -0.0008 | 0.008 ± 0.004 | 9/10/8 (8/10 tümü) | **kısmen** |
| sin | 10 | 0.792 ± 0.018 | -0.0237 | 0.212 ± 0.020 | 0/0/0 (0/10 tümü) | **kaybediyor** |
| sin_refl | 3 | 0.994 ± 0.000 | -0.0010 | 0.005 ± 0.004 | 3/3/3 (3/3 tümü) | **korunmuş** |
| tanh | 10 | 0.912 ± 0.004 | -0.0101 | 0.085 ± 0.004 | 0/0/0 (0/10 tümü) | **kaybediyor** |
| tri | 10 | 1.005 ± 0.001 | +0.0004 | -0.023 ± 0.002 | 10/10/10 (10/10 tümü) | **korunmuş** |
| weight_clipping | 10 | 1.006 ± 0.001 | +0.0005 | -0.022 ± 0.003 | 10/10/10 (10/10 tümü) | **korunmuş** |

# results/pmnist_inputnorm: adam_std (n=5), adam_sin (n=5), adam_tanh (n=5), adam_tri (n=5), adam_weight_clipping (n=5), sgd_std (n=5), sgd_tri (n=5), sgd_weight_clipping (n=5)

## Summary metrics, mean ± sd over seeds
| label | n | auc_norm | early_window_mean | final_window_mean | retention_ratio | fresh_gap_final | final_dead_frac/all | final_effective_rank/last | final_w_abs_mean/all | final_step_dw_norm | final_sat_frac/all | final_jac_sq_mean/all | final_w_over_A_mean/all |
| adam_std | 5 | 0.728 ± 0.012 | 0.772 ± 0.009 | 0.696 ± 0.021 | 0.902 ± 0.030 | 0.125 ± 0.031 | 0.580 ± 0.025 | 3.822 ± 0.426 | 0.552 ± 0.013 | 0.053 ± 0.001 | — | — | — |
| adam_sin | 5 | 0.822 ± 0.001 | 0.816 ± 0.001 | 0.823 ± 0.001 | 1.009 ± 0.002 | -0.016 ± 0.008 | 0.318 ± 0.016 | 29.083 ± 0.620 | 0.125 ± 0.000 | 0.036 ± 0.001 | 0.342 ± 0.002 | 0.369 ± 0.002 | 0.729 ± 0.001 |
| adam_tanh | 5 | 0.785 ± 0.004 | 0.810 ± 0.002 | 0.775 ± 0.013 | 0.956 ± 0.015 | 0.043 ± 0.028 | 0.414 ± 0.016 | 20.370 ± 2.033 | 0.141 ± 0.001 | 0.020 ± 0.000 | 0.636 ± 0.005 | 0.155 ± 0.003 | 0.846 ± 0.003 |
| adam_tri | 5 | 0.814 ± 0.001 | 0.806 ± 0.001 | 0.816 ± 0.002 | 1.013 ± 0.004 | -0.021 ± 0.003 | 0.352 ± 0.024 | 21.248 ± 0.951 | 0.062 ± 0.000 | 0.055 ± 0.001 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.523 ± 0.001 |
| adam_weight_clipping | 5 | 0.815 ± 0.002 | 0.807 ± 0.001 | 0.817 ± 0.003 | 1.013 ± 0.003 | -0.019 ± 0.009 | 0.381 ± 0.005 | 20.232 ± 1.056 | 0.066 ± 0.000 | 0.055 ± 0.001 | — | — | — |
| sgd_std | 5 | 0.781 ± 0.001 | 0.828 ± 0.001 | 0.752 ± 0.002 | 0.908 ± 0.002 | 0.081 ± 0.018 | 0.169 ± 0.014 | 22.228 ± 2.055 | 0.111 ± 0.000 | 0.032 ± 0.001 | — | — | — |
| sgd_tri | 5 | 0.836 ± 0.001 | 0.836 ± 0.002 | 0.835 ± 0.001 | 0.999 ± 0.002 | -0.016 ± 0.007 | 0.044 ± 0.010 | 44.052 ± 1.992 | 0.059 ± 0.000 | 0.030 ± 0.002 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.490 ± 0.001 |
| sgd_weight_clipping | 5 | 0.836 ± 0.000 | 0.837 ± 0.001 | 0.834 ± 0.001 | 0.997 ± 0.002 | -0.007 ± 0.004 | 0.047 ± 0.012 | 43.272 ± 1.206 | 0.060 ± 0.000 | 0.029 ± 0.001 | — | — | — |

## online_accuracy in blocks of 25 tasks, mean over seeds (sd in brackets)
- adam_std: 0.766 (0.010) → 0.734 (0.008) → 0.733 (0.006) → 0.736 (0.009) → 0.729 (0.012) → 0.718 (0.026) → 0.713 (0.025) → 0.697 (0.022)
- adam_sin: 0.817 (0.001) → 0.823 (0.002) → 0.823 (0.002) → 0.824 (0.002) → 0.822 (0.003) → 0.823 (0.002) → 0.825 (0.003) → 0.823 (0.001)
- adam_tanh: 0.809 (0.001) → 0.797 (0.002) → 0.786 (0.002) → 0.786 (0.003) → 0.773 (0.006) → 0.779 (0.005) → 0.773 (0.012) → 0.774 (0.012)
- adam_tri: 0.807 (0.002) → 0.812 (0.002) → 0.813 (0.001) → 0.815 (0.001) → 0.814 (0.001) → 0.815 (0.001) → 0.816 (0.002) → 0.816 (0.002)
- adam_weight_clipping: 0.808 (0.001) → 0.814 (0.003) → 0.816 (0.003) → 0.816 (0.002) → 0.814 (0.002) → 0.817 (0.001) → 0.818 (0.002) → 0.817 (0.003)
- sgd_std: 0.827 (0.000) → 0.806 (0.001) → 0.790 (0.003) → 0.782 (0.002) → 0.771 (0.002) → 0.765 (0.002) → 0.758 (0.004) → 0.753 (0.003)
- sgd_tri: 0.836 (0.002) → 0.837 (0.001) → 0.836 (0.002) → 0.836 (0.002) → 0.835 (0.002) → 0.836 (0.002) → 0.837 (0.002) → 0.835 (0.001)
- sgd_weight_clipping: 0.837 (0.001) → 0.835 (0.001) → 0.836 (0.002) → 0.837 (0.002) → 0.835 (0.002) → 0.836 (0.001) → 0.836 (0.001) → 0.834 (0.000)

## Mechanism trajectories at tasks [0, 10, 50, 100, 199], mean over seeds
- online_accuracy:
    adam_std         0.803 → 0.757 → 0.716 → 0.705 → 0.683
    adam_sin         0.807 → 0.813 → 0.827 → 0.814 → 0.823
    adam_tanh        0.810 → 0.807 → 0.791 → 0.781 → 0.768
    adam_tri         0.792 → 0.803 → 0.814 → 0.811 → 0.814
    adam_weight_clipping 0.793 → 0.805 → 0.813 → 0.808 → 0.813
    sgd_std          0.823 → 0.824 → 0.798 → 0.769 → 0.743
    sgd_tri          0.822 → 0.837 → 0.839 → 0.831 → 0.837
    sgd_weight_clipping 0.823 → 0.835 → 0.839 → 0.833 → 0.829
- sat_frac/all:
    adam_sin         0.003 → 0.290 → 0.326 → 0.334 → 0.342
    adam_tanh        0.000 → 0.285 → 0.561 → 0.605 → 0.637
    adam_tri         0.000 → 0.000 → 0.000 → 0.000 → 0.000
    sgd_tri          0.000 → 0.000 → 0.000 → 0.000 → 0.000
- sat_frac/L0:
    adam_sin         0.004 → 0.361 → 0.391 → 0.392 → 0.397
    adam_tanh        0.001 → 0.358 → 0.687 → 0.724 → 0.749
    adam_tri         0.000 → 0.000 → 0.000 → 0.000 → 0.000
    sgd_tri          0.000 → 0.000 → 0.000 → 0.000 → 0.000
- jac_sq_mean/all:
    adam_sin         0.789 → 0.435 → 0.390 → 0.378 → 0.369
    adam_tanh        0.670 → 0.318 → 0.193 → 0.170 → 0.155
    adam_tri         1.000 → 1.000 → 1.000 → 1.000 → 1.000
    sgd_tri          1.000 → 1.000 → 1.000 → 1.000 → 1.000
- w_over_A_mean/all:
    adam_sin         0.389 → 0.675 → 0.712 → 0.722 → 0.729
    adam_tanh        0.389 → 0.687 → 0.809 → 0.831 → 0.846
    adam_tri         0.506 → 0.518 → 0.519 → 0.522 → 0.520
    sgd_tri          0.501 → 0.498 → 0.491 → 0.490 → 0.490
- w_abs_mean/all:
    adam_std         0.068 → 0.154 → 0.307 → 0.415 → 0.564
    adam_sin         0.068 → 0.109 → 0.119 → 0.123 → 0.126
    adam_tanh        0.068 → 0.110 → 0.132 → 0.137 → 0.141
    adam_tri         0.060 → 0.062 → 0.062 → 0.062 → 0.062
    adam_weight_clipping 0.062 → 0.064 → 0.065 → 0.066 → 0.066
    sgd_std          0.061 → 0.066 → 0.083 → 0.096 → 0.112
    sgd_tri          0.060 → 0.060 → 0.059 → 0.059 → 0.059
    sgd_weight_clipping 0.060 → 0.060 → 0.060 → 0.060 → 0.060
- dead_frac/all:
    adam_std         0.106 → 0.332 → 0.508 → 0.553 → 0.599
    adam_sin         0.094 → 0.203 → 0.266 → 0.287 → 0.338
    adam_tanh        0.087 → 0.181 → 0.285 → 0.353 → 0.423
    adam_tri         0.119 → 0.215 → 0.313 → 0.335 → 0.349
    adam_weight_clipping 0.120 → 0.223 → 0.321 → 0.359 → 0.385
    sgd_std          0.002 → 0.005 → 0.068 → 0.100 → 0.180
    sgd_tri          0.001 → 0.004 → 0.013 → 0.027 → 0.041
    sgd_weight_clipping 0.001 → 0.004 → 0.015 → 0.027 → 0.045
- effective_rank/last:
    adam_std         40.832 → 17.035 → 5.999 → 5.264 → 3.786
    adam_sin         42.075 → 29.003 → 27.824 → 28.831 → 28.468
    adam_tanh        43.258 → 28.698 → 25.534 → 22.171 → 20.249
    adam_tri         38.288 → 30.164 → 25.405 → 22.001 → 20.772
    adam_weight_clipping 39.528 → 28.799 → 22.841 → 20.402 → 20.235
    sgd_std          49.506 → 49.642 → 34.458 → 27.247 → 22.077
    sgd_tri          48.349 → 49.508 → 46.396 → 44.697 → 42.498
    sgd_weight_clipping 48.262 → 49.056 → 46.407 → 44.069 → 42.972
- step_dw_norm:
    adam_std         0.052 → 0.055 → 0.057 → 0.058 → 0.052
    adam_sin         0.048 → 0.037 → 0.035 → 0.037 → 0.036
    adam_tanh        0.046 → 0.031 → 0.023 → 0.021 → 0.020
    adam_tri         0.053 → 0.057 → 0.054 → 0.057 → 0.056
    adam_weight_clipping 0.052 → 0.055 → 0.055 → 0.057 → 0.053
    sgd_std          0.034 → 0.029 → 0.028 → 0.031 → 0.032
    sgd_tri          0.035 → 0.027 → 0.031 → 0.028 → 0.029
    sgd_weight_clipping 0.034 → 0.028 → 0.027 → 0.031 → 0.027

## Paired differences (a − b, same seeds): mean [95% bootstrap CI], t-test p, Cohen d_z
- adam_sin − adam_weight_clipping | auc_norm                   n=5 | +0.0075 [+0.0060, +0.0090] p_t=0.001 d_z=+3.87 | per seed: +0.0052, +0.0084, +0.0095, +0.0057, +0.0087
- adam_sin − adam_weight_clipping | early_window_mean          n=5 | +0.0092 [+0.0086, +0.0098] p_t=0.000 d_z=+10.93 | per seed: +0.0091, +0.0085, +0.0094, +0.0083, +0.0105
- adam_sin − adam_weight_clipping | final_window_mean          n=5 | +0.0060 [+0.0029, +0.0082] p_t=0.020 d_z=+1.68 | per seed: -0.0000, +0.0075, +0.0089, +0.0056, +0.0079
- adam_sin − adam_weight_clipping | retention_ratio            n=5 | -0.0041 [-0.0078, -0.0015] p_t=0.101 d_z=-0.95 | per seed: -0.0114, -0.0014, -0.0008, -0.0035, -0.0033
- adam_sin − adam_weight_clipping | fresh_gap_final            n=5 | +0.0026 [-0.0042, +0.0082] p_t=0.522 d_z=+0.31 | per seed: +0.0016, +0.0020, +0.0070, -0.0100, +0.0124
- adam_sin − adam_weight_clipping | final_dead_frac/all        n=5 | -0.0632 [-0.0727, -0.0539] p_t=0.000 d_z=-5.22 | per seed: -0.0793, -0.0533, -0.0600, -0.0720, -0.0513
- adam_sin − adam_weight_clipping | final_effective_rank/last  n=5 | +8.8511 [+8.3599, +9.3498] p_t=0.000 d_z=+13.70 | per seed: +9.4406, +9.6550, +8.3435, +8.2958, +8.5206
- adam_sin − adam_tri | auc_norm                   n=5 | +0.0088 [+0.0080, +0.0095] p_t=0.000 d_z=+8.88 | per seed: +0.0073, +0.0097, +0.0097, +0.0086, +0.0088
- adam_sin − adam_tri | early_window_mean          n=5 | +0.0095 [+0.0088, +0.0102] p_t=0.000 d_z=+10.53 | per seed: +0.0095, +0.0083, +0.0100, +0.0090, +0.0107
- adam_sin − adam_tri | final_window_mean          n=5 | +0.0067 [+0.0048, +0.0081] p_t=0.003 d_z=+3.00 | per seed: +0.0029, +0.0086, +0.0066, +0.0076, +0.0079
- adam_sin − adam_tri | retention_ratio            n=5 | -0.0036 [-0.0062, -0.0014] p_t=0.064 d_z=-1.13 | per seed: -0.0083, +0.0002, -0.0043, -0.0019, -0.0037
- adam_sin − adam_tri | fresh_gap_final            n=5 | +0.0052 [-0.0000, +0.0105] p_t=0.162 d_z=+0.77 | per seed: -0.0024, +0.0068, +0.0138, -0.0010, +0.0090
- adam_sin − adam_tri | final_dead_frac/all        n=5 | -0.0337 [-0.0623, -0.0061] p_t=0.104 d_z=-0.94 | per seed: -0.0800, -0.0213, +0.0040, -0.0627, -0.0087
- adam_sin − adam_tri | final_effective_rank/last  n=5 | +7.8348 [+7.0662, +8.7924] p_t=0.000 d_z=+7.08 | per seed: +9.5856, +7.0493, +6.9493, +7.3338, +8.2560
- adam_tri − adam_weight_clipping | auc_norm                   n=5 | -0.0013 [-0.0023, -0.0004] p_t=0.071 d_z=-1.09 | per seed: -0.0021, -0.0012, -0.0002, -0.0029, -0.0001
- adam_tri − adam_weight_clipping | early_window_mean          n=5 | -0.0004 [-0.0006, -0.0001] p_t=0.084 d_z=-1.02 | per seed: -0.0004, +0.0002, -0.0006, -0.0007, -0.0003
- adam_tri − adam_weight_clipping | final_window_mean          n=5 | -0.0008 [-0.0022, +0.0009] p_t=0.441 d_z=-0.38 | per seed: -0.0030, -0.0011, +0.0022, -0.0020, +0.0000
- adam_tri − adam_weight_clipping | retention_ratio            n=5 | -0.0005 [-0.0022, +0.0015] p_t=0.685 d_z=-0.19 | per seed: -0.0032, -0.0015, +0.0035, -0.0016, +0.0004
- adam_tri − adam_weight_clipping | fresh_gap_final            n=5 | -0.0026 [-0.0073, +0.0020] p_t=0.379 d_z=-0.44 | per seed: +0.0040, -0.0048, -0.0068, -0.0090, +0.0034
- adam_tri − adam_weight_clipping | final_dead_frac/all        n=5 | -0.0295 [-0.0491, -0.0099] p_t=0.064 d_z=-1.14 | per seed: +0.0007, -0.0320, -0.0640, -0.0093, -0.0427
- adam_tri − adam_weight_clipping | final_effective_rank/last  n=5 | +1.0163 [+0.2403, +1.8952] p_t=0.101 d_z=+0.95 | per seed: -0.1450, +2.6057, +1.3942, +0.9620, +0.2646
- adam_sin − adam_std | auc_norm                   n=5 | +0.0941 [+0.0848, +0.1043] p_t=0.000 d_z=+7.66 | per seed: +0.1047, +0.1098, +0.0877, +0.0826, +0.0856
- adam_sin − adam_std | early_window_mean          n=5 | +0.0438 [+0.0375, +0.0516] p_t=0.000 d_z=+4.96 | per seed: +0.0344, +0.0580, +0.0451, +0.0423, +0.0393
- adam_sin − adam_std | final_window_mean          n=5 | +0.1272 [+0.1098, +0.1410] p_t=0.000 d_z=+6.34 | per seed: +0.1465, +0.1400, +0.0946, +0.1296, +0.1254
- adam_sin − adam_std | retention_ratio            n=5 | +0.1074 [+0.0828, +0.1299] p_t=0.001 d_z=+3.79 | per seed: +0.1431, +0.1076, +0.0636, +0.1126, +0.1103
- adam_sin − adam_std | fresh_gap_final            n=5 | -0.1408 [-0.1658, -0.1158] p_t=0.001 d_z=-4.38 | per seed: -0.1820, -0.1134, -0.1630, -0.1064, -0.1392
- adam_sin − adam_std | final_dead_frac/all        n=5 | -0.2619 [-0.2857, -0.2332] p_t=0.000 d_z=-7.46 | per seed: -0.2953, -0.2053, -0.2700, -0.2840, -0.2547
- adam_sin − adam_std | final_effective_rank/last  n=5 | +25.2609 [+24.9077, +25.6157] p_t=0.000 d_z=+53.61 | per seed: +25.7751, +25.0015, +24.9936, +24.7749, +25.7595
- adam_tri − adam_std | auc_norm                   n=5 | +0.0853 [+0.0759, +0.0949] p_t=0.000 d_z=+6.85 | per seed: +0.0974, +0.1001, +0.0779, +0.0740, +0.0768
- adam_tri − adam_std | early_window_mean          n=5 | +0.0343 [+0.0277, +0.0425] p_t=0.001 d_z=+3.63 | per seed: +0.0249, +0.0496, +0.0351, +0.0333, +0.0286
- adam_tri − adam_std | final_window_mean          n=5 | +0.1205 [+0.1025, +0.1350] p_t=0.000 d_z=+5.80 | per seed: +0.1436, +0.1314, +0.0879, +0.1220, +0.1175
- adam_tri − adam_std | retention_ratio            n=5 | +0.1110 [+0.0864, +0.1352] p_t=0.001 d_z=+3.74 | per seed: +0.1513, +0.1074, +0.0679, +0.1145, +0.1139
- adam_tri − adam_std | fresh_gap_final            n=5 | -0.1460 [-0.1722, -0.1199] p_t=0.001 d_z=-4.41 | per seed: -0.1796, -0.1202, -0.1768, -0.1054, -0.1482
- adam_tri − adam_std | final_dead_frac/all        n=5 | -0.2281 [-0.2567, -0.2027] p_t=0.000 d_z=-6.74 | per seed: -0.2153, -0.1840, -0.2740, -0.2213, -0.2460
- adam_tri − adam_std | final_effective_rank/last  n=5 | +17.4261 [+16.7924, +17.8993] p_t=0.000 d_z=+23.53 | per seed: +16.1895, +17.9522, +18.0443, +17.4411, +17.5035
- adam_weight_clipping − adam_std | auc_norm                   n=5 | +0.0866 [+0.0772, +0.0963] p_t=0.000 d_z=+6.83 | per seed: +0.0996, +0.1013, +0.0782, +0.0769, +0.0770
- adam_weight_clipping − adam_std | early_window_mean          n=5 | +0.0346 [+0.0281, +0.0426] p_t=0.001 d_z=+3.75 | per seed: +0.0253, +0.0494, +0.0357, +0.0339, +0.0289
- adam_weight_clipping − adam_std | final_window_mean          n=5 | +0.1213 [+0.1014, +0.1376] p_t=0.000 d_z=+5.35 | per seed: +0.1466, +0.1325, +0.0857, +0.1240, +0.1175
- adam_weight_clipping − adam_std | retention_ratio            n=5 | +0.1115 [+0.0846, +0.1377] p_t=0.001 d_z=+3.48 | per seed: +0.1545, +0.1090, +0.0644, +0.1161, +0.1136
- adam_weight_clipping − adam_std | fresh_gap_final            n=5 | -0.1434 [-0.1718, -0.1149] p_t=0.001 d_z=-3.91 | per seed: -0.1836, -0.1154, -0.1700, -0.0964, -0.1516
- adam_weight_clipping − adam_std | final_dead_frac/all        n=5 | -0.1987 [-0.2132, -0.1751] p_t=0.000 d_z=-7.50 | per seed: -0.2160, -0.1520, -0.2100, -0.2120, -0.2033
- adam_weight_clipping − adam_std | final_effective_rank/last  n=5 | +16.4098 [+15.8048, +16.9061] p_t=0.000 d_z=+23.90 | per seed: +16.3345, +15.3465, +16.6501, +16.4791, +17.2389
- adam_tanh − adam_std | auc_norm                   n=5 | +0.0562 [+0.0457, +0.0668] p_t=0.001 d_z=+4.14 | per seed: +0.0663, +0.0739, +0.0535, +0.0427, +0.0447
- adam_tanh − adam_std | early_window_mean          n=5 | +0.0386 [+0.0320, +0.0463] p_t=0.001 d_z=+4.26 | per seed: +0.0297, +0.0521, +0.0429, +0.0360, +0.0322
- adam_tanh − adam_std | final_window_mean          n=5 | +0.0793 [+0.0597, +0.0989] p_t=0.002 d_z=+3.20 | per seed: +0.1044, +0.1042, +0.0504, +0.0601, +0.0775
- adam_tanh − adam_std | retention_ratio            n=5 | +0.0548 [+0.0282, +0.0800] p_t=0.020 d_z=+1.67 | per seed: +0.0970, +0.0707, +0.0119, +0.0343, +0.0600
- adam_tanh − adam_std | fresh_gap_final            n=5 | -0.0820 [-0.1176, -0.0381] p_t=0.025 d_z=-1.56 | per seed: -0.1284, -0.0764, -0.1232, +0.0030, -0.0850
- adam_tanh − adam_std | final_dead_frac/all        n=5 | -0.1661 [-0.1883, -0.1324] p_t=0.001 d_z=-4.30 | per seed: -0.1860, -0.0987, -0.1947, -0.1713, -0.1800
- adam_tanh − adam_std | final_effective_rank/last  n=5 | +16.5479 [+15.0407, +18.0551] p_t=0.000 d_z=+8.55 | per seed: +17.7731, +16.8274, +18.9510, +14.6509, +14.5373
- sgd_tri − sgd_weight_clipping | auc_norm                   n=5 | +0.0003 [-0.0002, +0.0008] p_t=0.414 d_z=+0.41 | per seed: +0.0003, +0.0007, +0.0010, -0.0003, -0.0005
- sgd_tri − sgd_weight_clipping | early_window_mean          n=5 | -0.0010 [-0.0018, -0.0002] p_t=0.097 d_z=-0.97 | per seed: -0.0001, -0.0001, -0.0024, -0.0018, -0.0007
- sgd_tri − sgd_weight_clipping | final_window_mean          n=5 | +0.0010 [-0.0003, +0.0024] p_t=0.270 d_z=+0.57 | per seed: +0.0035, +0.0006, +0.0017, +0.0006, -0.0013
- sgd_tri − sgd_weight_clipping | retention_ratio            n=5 | +0.0024 [+0.0006, +0.0043] p_t=0.086 d_z=+1.01 | per seed: +0.0043, +0.0008, +0.0049, +0.0029, -0.0008
- sgd_tri − sgd_weight_clipping | fresh_gap_final            n=5 | -0.0088 [-0.0148, -0.0028] p_t=0.062 d_z=-1.15 | per seed: -0.0134, -0.0086, -0.0062, +0.0022, -0.0182
- sgd_tri − sgd_weight_clipping | final_dead_frac/all        n=5 | -0.0029 [-0.0101, +0.0013] p_t=0.464 d_z=-0.36 | per seed: +0.0013, -0.0007, +0.0000, -0.0173, +0.0020
- sgd_tri − sgd_weight_clipping | final_effective_rank/last  n=5 | +0.7793 [-0.0587, +1.5365] p_t=0.153 d_z=+0.79 | per seed: -0.7156, +0.6120, +1.9790, +1.2415, +0.7797
- sgd_tri − sgd_std | auc_norm                   n=5 | +0.0545 [+0.0535, +0.0555] p_t=0.000 d_z=+42.86 | per seed: +0.0533, +0.0546, +0.0559, +0.0531, +0.0556
- sgd_tri − sgd_std | early_window_mean          n=5 | +0.0076 [+0.0070, +0.0081] p_t=0.000 d_z=+10.76 | per seed: +0.0081, +0.0084, +0.0069, +0.0068, +0.0077
- sgd_tri − sgd_std | final_window_mean          n=5 | +0.0827 [+0.0813, +0.0838] p_t=0.000 d_z=+54.25 | per seed: +0.0820, +0.0804, +0.0835, +0.0835, +0.0841
- sgd_tri − sgd_std | retention_ratio            n=5 | +0.0907 [+0.0885, +0.0926] p_t=0.000 d_z=+34.69 | per seed: +0.0890, +0.0869, +0.0926, +0.0928, +0.0922
- sgd_tri − sgd_std | fresh_gap_final            n=5 | -0.0970 [-0.1090, -0.0832] p_t=0.000 d_z=-5.94 | per seed: -0.0850, -0.0752, -0.1052, -0.1150, -0.1046
- sgd_tri − sgd_std | final_dead_frac/all        n=5 | -0.1245 [-0.1344, -0.1169] p_t=0.000 d_z=-10.72 | per seed: -0.1167, -0.1147, -0.1233, -0.1240, -0.1440
- sgd_tri − sgd_std | final_effective_rank/last  n=5 | +21.8236 [+20.9860, +22.7154] p_t=0.000 d_z=+20.05 | per seed: +20.3854, +21.7803, +21.5318, +23.4216, +21.9991
- sgd_weight_clipping − sgd_std | auc_norm                   n=5 | +0.0542 [+0.0533, +0.0553] p_t=0.000 d_z=+44.56 | per seed: +0.0530, +0.0539, +0.0549, +0.0534, +0.0560
- sgd_weight_clipping − sgd_std | early_window_mean          n=5 | +0.0086 [+0.0083, +0.0090] p_t=0.000 d_z=+20.30 | per seed: +0.0082, +0.0084, +0.0093, +0.0086, +0.0084
- sgd_weight_clipping − sgd_std | final_window_mean          n=5 | +0.0817 [+0.0796, +0.0838] p_t=0.000 d_z=+30.00 | per seed: +0.0785, +0.0798, +0.0819, +0.0830, +0.0854
- sgd_weight_clipping − sgd_std | retention_ratio            n=5 | +0.0883 [+0.0859, +0.0910] p_t=0.000 d_z=+27.00 | per seed: +0.0847, +0.0861, +0.0877, +0.0899, +0.0930
- sgd_weight_clipping − sgd_std | fresh_gap_final            n=5 | -0.0882 [-0.1044, -0.0726] p_t=0.001 d_z=-4.27 | per seed: -0.0716, -0.0666, -0.0990, -0.1172, -0.0864
- sgd_weight_clipping − sgd_std | final_dead_frac/all        n=5 | -0.1216 [-0.1340, -0.1119] p_t=0.000 d_z=-8.14 | per seed: -0.1180, -0.1140, -0.1233, -0.1067, -0.1460
- sgd_weight_clipping − sgd_std | final_effective_rank/last  n=5 | +21.0443 [+20.1990, +21.7722] p_t=0.000 d_z=+22.29 | per seed: +21.1010, +21.1683, +19.5527, +22.1801, +21.2194

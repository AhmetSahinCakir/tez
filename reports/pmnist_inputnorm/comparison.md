## Referans: adam_std

_eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_

| Ölçüt                              | Yöntem               | Referans | Eşleşme sayısı | Ortalama (yöntem) | Ortalama (referans) | Fark (yöntem − referans) [%95 GA] | p (permütasyon) | p (Holm) | Cohen d_z |
|------------------------------------|----------------------|----------|---------------:|-------------------|---------------------|-----------------------------------|-----------------|----------|-----------|
| Normalize AUC                      | adam_sin             | adam_std |              5 | 0.822             | 0.728               | 0.094 [0.085, 0.104]              | 0.0625          | 0.4375   | 7.66      |
| Normalize AUC                      | adam_tanh            | adam_std |              5 | 0.785             | 0.728               | 0.056 [0.046, 0.067]              | 0.0625          | 0.4375   | 4.14      |
| Normalize AUC                      | sgd_std              | adam_std |              5 | 0.781             | 0.728               | 0.053 [0.044, 0.063]              | 0.0625          | 0.4375   | 4.26      |
| Normalize AUC                      | adam_tri             | adam_std |              5 | 0.814             | 0.728               | 0.085 [0.076, 0.095]              | 0.0625          | 0.4375   | 6.85      |
| Normalize AUC                      | adam_weight_clipping | adam_std |              5 | 0.815             | 0.728               | 0.087 [0.077, 0.096]              | 0.0625          | 0.4375   | 6.83      |
| Normalize AUC                      | sgd_tri              | adam_std |              5 | 0.836             | 0.728               | 0.108 [0.098, 0.118]              | 0.0625          | 0.4375   | 8.90      |
| Normalize AUC                      | sgd_weight_clipping  | adam_std |              5 | 0.836             | 0.728               | 0.107 [0.099, 0.117]              | 0.0625          | 0.4375   | 9.10      |
| Son pencere ort.                   | adam_sin             | adam_std |              5 | 0.823             | 0.696               | 0.127 [0.110, 0.141]              | 0.0625          | 0.4375   | 6.34      |
| Son pencere ort.                   | adam_tanh            | adam_std |              5 | 0.775             | 0.696               | 0.079 [0.060, 0.099]              | 0.0625          | 0.4375   | 3.20      |
| Son pencere ort.                   | sgd_std              | adam_std |              5 | 0.752             | 0.696               | 0.057 [0.038, 0.073]              | 0.0625          | 0.4375   | 2.54      |
| Son pencere ort.                   | adam_tri             | adam_std |              5 | 0.816             | 0.696               | 0.121 [0.103, 0.135]              | 0.0625          | 0.4375   | 5.80      |
| Son pencere ort.                   | adam_weight_clipping | adam_std |              5 | 0.817             | 0.696               | 0.121 [0.101, 0.138]              | 0.0625          | 0.4375   | 5.35      |
| Son pencere ort.                   | sgd_tri              | adam_std |              5 | 0.835             | 0.696               | 0.139 [0.121, 0.154]              | 0.0625          | 0.4375   | 6.52      |
| Son pencere ort.                   | sgd_weight_clipping  | adam_std |              5 | 0.834             | 0.696               | 0.138 [0.120, 0.153]              | 0.0625          | 0.4375   | 6.58      |
| Koruma oranı                       | adam_sin             | adam_std |              5 | 1.009             | 0.902               | 0.107 [0.083, 0.130]              | 0.0625          | 0.4375   | 3.79      |
| Koruma oranı                       | adam_tanh            | adam_std |              5 | 0.956             | 0.902               | 0.055 [0.028, 0.080]              | 0.0625          | 0.4375   | 1.67      |
| Koruma oranı                       | sgd_std              | adam_std |              5 | 0.908             | 0.902               | 0.007 [-0.020, 0.031]             | 0.5625          | 0.5625   | 0.22      |
| Koruma oranı                       | adam_tri             | adam_std |              5 | 1.013             | 0.902               | 0.111 [0.086, 0.135]              | 0.0625          | 0.4375   | 3.74      |
| Koruma oranı                       | adam_weight_clipping | adam_std |              5 | 1.013             | 0.902               | 0.112 [0.085, 0.138]              | 0.0625          | 0.4375   | 3.48      |
| Koruma oranı                       | sgd_tri              | adam_std |              5 | 0.999             | 0.902               | 0.098 [0.072, 0.121]              | 0.0625          | 0.4375   | 3.28      |
| Koruma oranı                       | sgd_weight_clipping  | adam_std |              5 | 0.997             | 0.902               | 0.095 [0.069, 0.118]              | 0.0625          | 0.4375   | 3.15      |
| Taze model farkı (son)             | adam_sin             | adam_std |              5 | -0.016            | 0.125               | -0.141 [-0.166, -0.116]           | 0.0625          | 0.4375   | -4.38     |
| Taze model farkı (son)             | adam_tanh            | adam_std |              5 | 0.043             | 0.125               | -0.082 [-0.118, -0.038]           | 0.1250          | 0.4375   | -1.56     |
| Taze model farkı (son)             | sgd_std              | adam_std |              5 | 0.081             | 0.125               | -0.044 [-0.072, -0.010]           | 0.1250          | 0.4375   | -1.06     |
| Taze model farkı (son)             | adam_tri             | adam_std |              5 | -0.021            | 0.125               | -0.146 [-0.172, -0.120]           | 0.0625          | 0.4375   | -4.41     |
| Taze model farkı (son)             | adam_weight_clipping | adam_std |              5 | -0.019            | 0.125               | -0.143 [-0.172, -0.115]           | 0.0625          | 0.4375   | -3.91     |
| Taze model farkı (son)             | sgd_tri              | adam_std |              5 | -0.016            | 0.125               | -0.141 [-0.169, -0.112]           | 0.0625          | 0.4375   | -3.89     |
| Taze model farkı (son)             | sgd_weight_clipping  | adam_std |              5 | -0.007            | 0.125               | -0.132 [-0.158, -0.106]           | 0.0625          | 0.4375   | -4.07     |
| Son: Ölü birim oranı               | adam_sin             | adam_std |              5 | 0.318             | 0.580               | -0.262 [-0.286, -0.233]           | 0.0625          | 0.4375   | -7.46     |
| Son: Ölü birim oranı               | adam_tanh            | adam_std |              5 | 0.414             | 0.580               | -0.166 [-0.188, -0.132]           | 0.0625          | 0.4375   | -4.30     |
| Son: Ölü birim oranı               | sgd_std              | adam_std |              5 | 0.169             | 0.580               | -0.411 [-0.432, -0.386]           | 0.0625          | 0.4375   | -13.79    |
| Son: Ölü birim oranı               | adam_tri             | adam_std |              5 | 0.352             | 0.580               | -0.228 [-0.257, -0.203]           | 0.0625          | 0.4375   | -6.74     |
| Son: Ölü birim oranı               | adam_weight_clipping | adam_std |              5 | 0.381             | 0.580               | -0.199 [-0.213, -0.175]           | 0.0625          | 0.4375   | -7.50     |
| Son: Ölü birim oranı               | sgd_tri              | adam_std |              5 | 0.044             | 0.580               | -0.536 [-0.555, -0.506]           | 0.0625          | 0.4375   | -16.48    |
| Son: Ölü birim oranı               | sgd_weight_clipping  | adam_std |              5 | 0.047             | 0.580               | -0.533 [-0.553, -0.505]           | 0.0625          | 0.4375   | -16.33    |
| Son: Etkin rank (son gizli katman) | adam_sin             | adam_std |              5 | 29.083            | 3.822               | 25.261 [24.908, 25.616]           | 0.0625          | 0.4375   | 53.61     |
| Son: Etkin rank (son gizli katman) | adam_tanh            | adam_std |              5 | 20.370            | 3.822               | 16.548 [15.041, 18.055]           | 0.0625          | 0.4375   | 8.55      |
| Son: Etkin rank (son gizli katman) | sgd_std              | adam_std |              5 | 22.228            | 3.822               | 18.406 [17.267, 19.990]           | 0.0625          | 0.4375   | 10.26     |
| Son: Etkin rank (son gizli katman) | adam_tri             | adam_std |              5 | 21.248            | 3.822               | 17.426 [16.792, 17.899]           | 0.0625          | 0.4375   | 23.53     |
| Son: Etkin rank (son gizli katman) | adam_weight_clipping | adam_std |              5 | 20.232            | 3.822               | 16.410 [15.805, 16.906]           | 0.0625          | 0.4375   | 23.90     |
| Son: Etkin rank (son gizli katman) | sgd_tri              | adam_std |              5 | 44.052            | 3.822               | 40.230 [39.101, 41.734]           | 0.0625          | 0.4375   | 23.96     |
| Son: Etkin rank (son gizli katman) | sgd_weight_clipping  | adam_std |              5 | 43.272            | 3.822               | 39.451 [38.838, 40.301]           | 0.0625          | 0.4375   | 42.33     |
| Son: Ortalama \|w\|                | adam_sin             | adam_std |              5 | 0.125             | 0.552               | -0.427 [-0.437, -0.418]           | 0.0625          | 0.4375   | -34.45    |
| Son: Ortalama \|w\|                | adam_tanh            | adam_std |              5 | 0.141             | 0.552               | -0.411 [-0.422, -0.402]           | 0.0625          | 0.4375   | -32.50    |
| Son: Ortalama \|w\|                | sgd_std              | adam_std |              5 | 0.111             | 0.552               | -0.441 [-0.452, -0.433]           | 0.0625          | 0.4375   | -35.48    |
| Son: Ortalama \|w\|                | adam_tri             | adam_std |              5 | 0.062             | 0.552               | -0.490 [-0.500, -0.481]           | 0.0625          | 0.4375   | -39.27    |
| Son: Ortalama \|w\|                | adam_weight_clipping | adam_std |              5 | 0.066             | 0.552               | -0.486 [-0.497, -0.478]           | 0.0625          | 0.4375   | -38.53    |
| Son: Ortalama \|w\|                | sgd_tri              | adam_std |              5 | 0.059             | 0.552               | -0.493 [-0.503, -0.484]           | 0.0625          | 0.4375   | -39.44    |
| Son: Ortalama \|w\|                | sgd_weight_clipping  | adam_std |              5 | 0.060             | 0.552               | -0.492 [-0.503, -0.484]           | 0.0625          | 0.4375   | -39.08    |

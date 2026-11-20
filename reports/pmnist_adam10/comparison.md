## Referans: adam_std

_eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_

| Ölçüt                              | Yöntem               | Referans | Eşleşme sayısı | Ortalama (yöntem) | Ortalama (referans) | Fark (yöntem − referans) [%95 GA] | p (permütasyon) | p (Holm)  | Cohen d_z |
|------------------------------------|----------------------|----------|---------------:|-------------------|---------------------|-----------------------------------|-----------------|-----------|-----------|
| Normalize AUC                      | adam_sin             | adam_std |             10 | 0.826             | 0.741               | 0.084 [0.081, 0.088]              | 0.0020          | 0.0078 ** | 14.35     |
| Normalize AUC                      | adam_tanh            | adam_std |             10 | 0.811             | 0.741               | 0.070 [0.065, 0.074]              | 0.0020          | 0.0078 ** | 9.35      |
| Normalize AUC                      | adam_tri             | adam_std |             10 | 0.792             | 0.741               | 0.050 [0.046, 0.055]              | 0.0020          | 0.0078 ** | 6.76      |
| Normalize AUC                      | adam_weight_clipping | adam_std |             10 | 0.791             | 0.741               | 0.050 [0.046, 0.055]              | 0.0020          | 0.0078 ** | 6.59      |
| Son pencere ort.                   | adam_sin             | adam_std |             10 | 0.826             | 0.706               | 0.121 [0.116, 0.125]              | 0.0020          | 0.0078 ** | 16.38     |
| Son pencere ort.                   | adam_tanh            | adam_std |             10 | 0.805             | 0.706               | 0.099 [0.095, 0.103]              | 0.0020          | 0.0078 ** | 13.15     |
| Son pencere ort.                   | adam_tri             | adam_std |             10 | 0.786             | 0.706               | 0.080 [0.072, 0.088]              | 0.0020          | 0.0078 ** | 6.05      |
| Son pencere ort.                   | adam_weight_clipping | adam_std |             10 | 0.784             | 0.706               | 0.078 [0.066, 0.088]              | 0.0020          | 0.0078 ** | 4.03      |
| Koruma oranı                       | adam_sin             | adam_std |             10 | 0.997             | 0.875               | 0.122 [0.115, 0.130]              | 0.0020          | 0.0078 ** | 9.73      |
| Koruma oranı                       | adam_tanh            | adam_std |             10 | 0.971             | 0.875               | 0.096 [0.090, 0.102]              | 0.0020          | 0.0078 ** | 8.93      |
| Koruma oranı                       | adam_tri             | adam_std |             10 | 0.975             | 0.875               | 0.100 [0.089, 0.111]              | 0.0020          | 0.0078 ** | 5.23      |
| Koruma oranı                       | adam_weight_clipping | adam_std |             10 | 0.971             | 0.875               | 0.095 [0.080, 0.108]              | 0.0020          | 0.0078 ** | 3.92      |
| Taze model farkı (son)             | adam_sin             | adam_std |             10 | -0.002            | 0.122               | -0.124 [-0.137, -0.112]           | 0.0020          | 0.0078 ** | -5.58     |
| Taze model farkı (son)             | adam_tanh            | adam_std |             10 | 0.027             | 0.122               | -0.094 [-0.104, -0.085]           | 0.0020          | 0.0078 ** | -5.81     |
| Taze model farkı (son)             | adam_tri             | adam_std |             10 | 0.028             | 0.122               | -0.093 [-0.113, -0.065]           | 0.0039          | 0.0078 ** | -2.22     |
| Taze model farkı (son)             | adam_weight_clipping | adam_std |             10 | 0.038             | 0.122               | -0.083 [-0.111, -0.050]           | 0.0078          | 0.0078 ** | -1.62     |
| Son: Ölü birim oranı               | adam_sin             | adam_std |             10 | 0.350             | 0.686               | -0.336 [-0.361, -0.309]           | 0.0020          | 0.0078 ** | -7.75     |
| Son: Ölü birim oranı               | adam_tanh            | adam_std |             10 | 0.262             | 0.686               | -0.425 [-0.447, -0.403]           | 0.0020          | 0.0078 ** | -11.40    |
| Son: Ölü birim oranı               | adam_tri             | adam_std |             10 | 0.342             | 0.686               | -0.344 [-0.365, -0.322]           | 0.0020          | 0.0078 ** | -9.67     |
| Son: Ölü birim oranı               | adam_weight_clipping | adam_std |             10 | 0.346             | 0.686               | -0.340 [-0.363, -0.318]           | 0.0020          | 0.0078 ** | -8.91     |
| Son: Etkin rank (son gizli katman) | adam_sin             | adam_std |             10 | 26.055            | 6.115               | 19.940 [18.675, 21.261]           | 0.0020          | 0.0078 ** | 8.99      |
| Son: Etkin rank (son gizli katman) | adam_tanh            | adam_std |             10 | 24.363            | 6.115               | 18.248 [16.827, 19.537]           | 0.0020          | 0.0078 ** | 7.96      |
| Son: Etkin rank (son gizli katman) | adam_tri             | adam_std |             10 | 25.225            | 6.115               | 19.110 [17.947, 20.217]           | 0.0020          | 0.0078 ** | 9.80      |
| Son: Etkin rank (son gizli katman) | adam_weight_clipping | adam_std |             10 | 25.309            | 6.115               | 19.194 [17.126, 20.821]           | 0.0020          | 0.0078 ** | 5.98      |
| Son: Ortalama \|w\|                | adam_sin             | adam_std |             10 | 0.120             | 0.351               | -0.231 [-0.237, -0.224]           | 0.0020          | 0.0078 ** | -21.15    |
| Son: Ortalama \|w\|                | adam_tanh            | adam_std |             10 | 0.140             | 0.351               | -0.212 [-0.217, -0.205]           | 0.0020          | 0.0078 ** | -19.33    |
| Son: Ortalama \|w\|                | adam_tri             | adam_std |             10 | 0.066             | 0.351               | -0.285 [-0.291, -0.278]           | 0.0020          | 0.0078 ** | -26.05    |
| Son: Ortalama \|w\|                | adam_weight_clipping | adam_std |             10 | 0.071             | 0.351               | -0.281 [-0.287, -0.274]           | 0.0020          | 0.0078 ** | -25.45    |

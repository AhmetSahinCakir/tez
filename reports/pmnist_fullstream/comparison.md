## Referans: baseline

_eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_

| Ölçüt                              | Yöntem          | Referans | Eşleşme sayısı | Ortalama (yöntem) | Ortalama (referans) | Fark (yöntem − referans) [%95 GA] | p (permütasyon) | p (Holm) | Cohen d_z |
|------------------------------------|-----------------|----------|---------------:|-------------------|---------------------|-----------------------------------|-----------------|----------|-----------|
| Normalize AUC                      | sin             | baseline |             10 | 0.818             | 0.856               | -0.039 [-0.040, -0.037]           | 0.0020          | 0.0117 * | -12.79    |
| Normalize AUC                      | sin_refl        | baseline |              3 | 0.924             | 0.857               | 0.066 [0.066, 0.067]              | 0.2500          | 0.2500   | 122.21    |
| Normalize AUC                      | tanh            | baseline |             10 | 0.859             | 0.856               | 0.003 [0.002, 0.004]              | 0.0059          | 0.0117 * | 1.35      |
| Normalize AUC                      | ln_wd           | baseline |             10 | 0.925             | 0.856               | 0.068 [0.068, 0.069]              | 0.0020          | 0.0117 * | 51.93     |
| Normalize AUC                      | tri             | baseline |             10 | 0.939             | 0.856               | 0.082 [0.082, 0.083]              | 0.0020          | 0.0117 * | 60.81     |
| Normalize AUC                      | weight_clipping | baseline |             10 | 0.939             | 0.856               | 0.082 [0.081, 0.083]              | 0.0020          | 0.0117 * | 67.90     |
| Son pencere ort.                   | sin             | baseline |             10 | 0.720             | 0.835               | -0.115 [-0.126, -0.106]           | 0.0020          | 0.0117 * | -6.54     |
| Son pencere ort.                   | sin_refl        | baseline |              3 | 0.920             | 0.835               | 0.085 [0.082, 0.088]              | 0.2500          | 0.2500   | 27.72     |
| Son pencere ort.                   | tanh            | baseline |             10 | 0.831             | 0.835               | -0.005 [-0.007, -0.003]           | 0.0020          | 0.0117 * | -1.19     |
| Son pencere ort.                   | ln_wd           | baseline |             10 | 0.923             | 0.835               | 0.087 [0.086, 0.089]              | 0.0020          | 0.0117 * | 42.21     |
| Son pencere ort.                   | tri             | baseline |             10 | 0.940             | 0.835               | 0.104 [0.103, 0.105]              | 0.0020          | 0.0117 * | 52.41     |
| Son pencere ort.                   | weight_clipping | baseline |             10 | 0.940             | 0.835               | 0.104 [0.103, 0.105]              | 0.0020          | 0.0117 * | 56.19     |
| Koruma oranı                       | sin             | baseline |             10 | 0.792             | 0.924               | -0.131 [-0.143, -0.121]           | 0.0020          | 0.0117 * | -6.97     |
| Koruma oranı                       | sin_refl        | baseline |              3 | 0.994             | 0.923               | 0.071 [0.069, 0.075]              | 0.2500          | 0.2500   | 22.27     |
| Koruma oranı                       | tanh            | baseline |             10 | 0.912             | 0.924               | -0.012 [-0.015, -0.010]           | 0.0020          | 0.0117 * | -2.93     |
| Koruma oranı                       | ln_wd           | baseline |             10 | 0.992             | 0.924               | 0.068 [0.067, 0.069]              | 0.0020          | 0.0117 * | 29.17     |
| Koruma oranı                       | tri             | baseline |             10 | 1.005             | 0.924               | 0.082 [0.081, 0.083]              | 0.0020          | 0.0117 * | 43.83     |
| Koruma oranı                       | weight_clipping | baseline |             10 | 1.006             | 0.924               | 0.082 [0.081, 0.083]              | 0.0020          | 0.0117 * | 43.09     |
| Taze model farkı (son)             | sin             | baseline |             10 | 0.212             | 0.084               | 0.127 [0.112, 0.143]              | 0.0020          | 0.0117 * | 4.74      |
| Taze model farkı (son)             | sin_refl        | baseline |              3 | 0.005             | 0.086               | -0.082 [-0.096, -0.063]           | 0.2500          | 0.5000   | -4.84     |
| Taze model farkı (son)             | tanh            | baseline |             10 | 0.085             | 0.084               | 0.001 [-0.004, 0.005]             | 0.7383          | 0.7383   | 0.11      |
| Taze model farkı (son)             | ln_wd           | baseline |             10 | 0.008             | 0.084               | -0.076 [-0.082, -0.070]           | 0.0020          | 0.0117 * | -7.34     |
| Taze model farkı (son)             | tri             | baseline |             10 | -0.023            | 0.084               | -0.108 [-0.113, -0.102]           | 0.0020          | 0.0117 * | -12.23    |
| Taze model farkı (son)             | weight_clipping | baseline |             10 | -0.022            | 0.084               | -0.107 [-0.112, -0.101]           | 0.0020          | 0.0117 * | -11.11    |
| Son: Ölü birim oranı               | sin             | baseline |             10 | 0.425             | 0.523               | -0.098 [-0.116, -0.079]           | 0.0020          | 0.0117 * | -3.10     |
| Son: Ölü birim oranı               | sin_refl        | baseline |              3 | 0.268             | 0.525               | -0.257 [-0.296, -0.227]           | 0.2500          | 0.2500   | -7.27     |
| Son: Ölü birim oranı               | tanh            | baseline |             10 | 0.406             | 0.523               | -0.117 [-0.146, -0.084]           | 0.0020          | 0.0117 * | -2.22     |
| Son: Ölü birim oranı               | ln_wd           | baseline |             10 | 0.361             | 0.523               | -0.162 [-0.184, -0.141]           | 0.0020          | 0.0117 * | -4.37     |
| Son: Ölü birim oranı               | tri             | baseline |             10 | 0.177             | 0.523               | -0.346 [-0.369, -0.321]           | 0.0020          | 0.0117 * | -8.61     |
| Son: Ölü birim oranı               | weight_clipping | baseline |             10 | 0.177             | 0.523               | -0.346 [-0.366, -0.325]           | 0.0020          | 0.0117 * | -9.72     |
| Son: Etkin rank (son gizli katman) | sin             | baseline |             10 | 6.932             | 15.928              | -8.996 [-9.927, -8.116]           | 0.0020          | 0.0117 * | -5.78     |
| Son: Etkin rank (son gizli katman) | sin_refl        | baseline |              3 | 58.661            | 15.594              | 43.067 [42.554, 43.783]           | 0.2500          | 0.2500   | 67.36     |
| Son: Etkin rank (son gizli katman) | tanh            | baseline |             10 | 9.470             | 15.928              | -6.458 [-7.029, -5.881]           | 0.0020          | 0.0117 * | -6.58     |
| Son: Etkin rank (son gizli katman) | ln_wd           | baseline |             10 | 42.410            | 15.928              | 26.482 [25.765, 27.184]           | 0.0020          | 0.0117 * | 21.90     |
| Son: Etkin rank (son gizli katman) | tri             | baseline |             10 | 54.786            | 15.928              | 38.858 [38.153, 39.561]           | 0.0020          | 0.0117 * | 32.53     |
| Son: Etkin rank (son gizli katman) | weight_clipping | baseline |             10 | 54.777            | 15.928              | 38.849 [38.180, 39.530]           | 0.0020          | 0.0117 * | 33.99     |
| Son: Ortalama \|w\|                | sin             | baseline |             10 | 0.153             | 0.232               | -0.079 [-0.080, -0.077]           | 0.0020          | 0.0117 * | -38.69    |
| Son: Ortalama \|w\|                | sin_refl        | baseline |              3 | 0.084             | 0.231               | -0.147 [-0.148, -0.146]           | 0.2500          | 0.2500   | -141.41   |
| Son: Ortalama \|w\|                | tanh            | baseline |             10 | 0.145             | 0.232               | -0.086 [-0.087, -0.086]           | 0.0020          | 0.0117 * | -58.70    |
| Son: Ortalama \|w\|                | ln_wd           | baseline |             10 | 0.021             | 0.232               | -0.211 [-0.212, -0.210]           | 0.0020          | 0.0117 * | -130.81   |
| Son: Ortalama \|w\|                | tri             | baseline |             10 | 0.071             | 0.232               | -0.160 [-0.161, -0.159]           | 0.0020          | 0.0117 * | -96.38    |
| Son: Ortalama \|w\|                | weight_clipping | baseline |             10 | 0.071             | 0.232               | -0.160 [-0.161, -0.159]           | 0.0020          | 0.0117 * | -107.35   |

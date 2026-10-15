## Referans: baseline

_eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_

| Ölçüt                              | Yöntem          | Referans | Eşleşme sayısı | Ortalama (yöntem) | Ortalama (referans) | Fark (yöntem − referans) [%95 GA] | p (permütasyon) | p (Holm) | Cohen d_z |
|------------------------------------|-----------------|----------|---------------:|-------------------|---------------------|-----------------------------------|-----------------|----------|-----------|
| Normalize AUC                      | sin             | baseline |              3 | 0.818             | 0.857               | -0.040 [-0.043, -0.036]           | 0.2500          | 1.0000   | -11.67    |
| Normalize AUC                      | tanh            | baseline |              3 | 0.858             | 0.857               | 0.000 [-0.001, 0.001]             | 0.7500          | 1.0000   | 0.44      |
| Normalize AUC                      | tri             | baseline |              3 | 0.939             | 0.857               | 0.082 [0.081, 0.082]              | 0.2500          | 1.0000   | 282.97    |
| Normalize AUC                      | weight_clipping | baseline |              3 | 0.939             | 0.857               | 0.081 [0.081, 0.082]              | 0.2500          | 1.0000   | 137.05    |
| Son pencere ort.                   | sin             | baseline |              3 | 0.716             | 0.835               | -0.119 [-0.141, -0.101]           | 0.2500          | 1.0000   | -5.92     |
| Son pencere ort.                   | tanh            | baseline |              3 | 0.828             | 0.835               | -0.007 [-0.012, -0.001]           | 0.2500          | 1.0000   | -1.33     |
| Son pencere ort.                   | tri             | baseline |              3 | 0.939             | 0.835               | 0.104 [0.101, 0.108]              | 0.2500          | 1.0000   | 29.85     |
| Son pencere ort.                   | weight_clipping | baseline |              3 | 0.940             | 0.835               | 0.105 [0.101, 0.108]              | 0.2500          | 1.0000   | 29.78     |
| Koruma oranı                       | sin             | baseline |              3 | 0.788             | 0.923               | -0.135 [-0.158, -0.116]           | 0.2500          | 1.0000   | -6.32     |
| Koruma oranı                       | tanh            | baseline |              3 | 0.908             | 0.923               | -0.014 [-0.019, -0.008]           | 0.2500          | 1.0000   | -2.43     |
| Koruma oranı                       | tri             | baseline |              3 | 1.005             | 0.923               | 0.082 [0.080, 0.085]              | 0.2500          | 1.0000   | 32.72     |
| Koruma oranı                       | weight_clipping | baseline |              3 | 1.005             | 0.923               | 0.083 [0.081, 0.086]              | 0.2500          | 1.0000   | 28.36     |
| Taze model farkı (son)             | sin             | baseline |              3 | 0.214             | 0.086               | 0.128 [0.097, 0.172]              | 0.2500          | 1.0000   | 3.26      |
| Taze model farkı (son)             | tanh            | baseline |              3 | 0.085             | 0.086               | -0.001 [-0.009, 0.010]            | 1.0000          | 1.0000   | -0.15     |
| Taze model farkı (son)             | tri             | baseline |              3 | -0.022            | 0.086               | -0.108 [-0.119, -0.091]           | 0.2500          | 1.0000   | -7.10     |
| Taze model farkı (son)             | weight_clipping | baseline |              3 | -0.019            | 0.086               | -0.105 [-0.113, -0.089]           | 0.2500          | 1.0000   | -7.36     |
| Son: Ölü birim oranı               | sin             | baseline |              3 | 0.432             | 0.525               | -0.093 [-0.119, -0.055]           | 0.2500          | 1.0000   | -2.73     |
| Son: Ölü birim oranı               | tanh            | baseline |              3 | 0.437             | 0.525               | -0.088 [-0.136, -0.057]           | 0.2500          | 1.0000   | -2.09     |
| Son: Ölü birim oranı               | tri             | baseline |              3 | 0.177             | 0.525               | -0.348 [-0.393, -0.311]           | 0.2500          | 1.0000   | -8.37     |
| Son: Ölü birim oranı               | weight_clipping | baseline |              3 | 0.167             | 0.525               | -0.358 [-0.372, -0.339]           | 0.2500          | 1.0000   | -20.46    |
| Son: Etkin rank (son gizli katman) | sin             | baseline |              3 | 6.486             | 15.594              | -9.108 [-10.167, -8.536]          | 0.2500          | 1.0000   | -9.92     |
| Son: Etkin rank (son gizli katman) | tanh            | baseline |              3 | 9.467             | 15.594              | -6.128 [-6.840, -5.482]           | 0.2500          | 1.0000   | -9.00     |
| Son: Etkin rank (son gizli katman) | tri             | baseline |              3 | 54.719            | 15.594              | 39.124 [38.659, 39.445]           | 0.2500          | 1.0000   | 94.88     |
| Son: Etkin rank (son gizli katman) | weight_clipping | baseline |              3 | 54.621            | 15.594              | 39.027 [38.548, 39.689]           | 0.2500          | 1.0000   | 65.95     |
| Son: Ortalama \|w\|                | sin             | baseline |              3 | 0.153             | 0.231               | -0.078 [-0.079, -0.076]           | 0.2500          | 1.0000   | -44.06    |
| Son: Ortalama \|w\|                | tanh            | baseline |              3 | 0.145             | 0.231               | -0.086 [-0.088, -0.085]           | 0.2500          | 1.0000   | -61.26    |
| Son: Ortalama \|w\|                | tri             | baseline |              3 | 0.071             | 0.231               | -0.160 [-0.161, -0.159]           | 0.2500          | 1.0000   | -150.64   |
| Son: Ortalama \|w\|                | weight_clipping | baseline |              3 | 0.071             | 0.231               | -0.160 [-0.162, -0.158]           | 0.2500          | 1.0000   | -98.15    |

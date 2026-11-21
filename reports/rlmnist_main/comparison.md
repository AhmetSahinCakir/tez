## Referans: baseline

_eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_

| Ölçüt                              | Yöntem          | Referans | Eşleşme sayısı | Ortalama (yöntem) | Ortalama (referans) | Fark (yöntem − referans) [%95 GA] | p (permütasyon) | p (Holm)  | Cohen d_z |
|------------------------------------|-----------------|----------|---------------:|-------------------|---------------------|-----------------------------------|-----------------|-----------|-----------|
| Normalize AUC                      | sin             | baseline |             10 | 0.286             | 0.253               | 0.033 [0.001, 0.064]              | 0.0840          | 0.1211    | 0.61      |
| Normalize AUC                      | tanh            | baseline |             10 | 0.292             | 0.253               | 0.039 [0.005, 0.069]              | 0.0605          | 0.1211    | 0.70      |
| Normalize AUC                      | tri             | baseline |             10 | 0.367             | 0.253               | 0.113 [0.077, 0.145]              | 0.0020          | 0.0078 ** | 1.93      |
| Normalize AUC                      | weight_clipping | baseline |             10 | 0.307             | 0.253               | 0.054 [0.017, 0.086]              | 0.0293          | 0.0879    | 0.90      |
| Son pencere ort.                   | sin             | baseline |             10 | 0.161             | 0.151               | 0.010 [-0.020, 0.038]             | 0.5703          | 0.7891    | 0.20      |
| Son pencere ort.                   | tanh            | baseline |             10 | 0.168             | 0.151               | 0.016 [-0.020, 0.048]             | 0.3945          | 0.7891    | 0.28      |
| Son pencere ort.                   | tri             | baseline |             10 | 0.122             | 0.151               | -0.029 [-0.054, -0.008]           | 0.0566          | 0.2266    | -0.72     |
| Son pencere ort.                   | weight_clipping | baseline |             10 | 0.174             | 0.151               | 0.022 [-0.008, 0.052]             | 0.1973          | 0.5918    | 0.43      |
| Koruma oranı                       | sin             | baseline |             10 | 0.264             | 0.244               | 0.020 [-0.019, 0.057]             | 0.3340          | 0.3340    | 0.32      |
| Koruma oranı                       | tanh            | baseline |             10 | 0.330             | 0.244               | 0.086 [0.030, 0.136]              | 0.0215          | 0.0645    | 0.94      |
| Koruma oranı                       | tri             | baseline |             10 | 0.170             | 0.244               | -0.074 [-0.110, -0.041]           | 0.0020          | 0.0078 ** | -1.24     |
| Koruma oranı                       | weight_clipping | baseline |             10 | 0.282             | 0.244               | 0.038 [-0.006, 0.082]             | 0.1484          | 0.2969    | 0.50      |
| Taze model farkı (son)             | sin             | baseline |             10 | 0.217             | 0.246               | -0.030 [-0.059, 0.003]            | 0.1113          | 0.2227    | -0.56     |
| Taze model farkı (son)             | tanh            | baseline |             10 | 0.144             | 0.246               | -0.102 [-0.136, -0.063]           | 0.0039          | 0.0117 *  | -1.62     |
| Taze model farkı (son)             | tri             | baseline |             10 | 0.819             | 0.246               | 0.572 [0.547, 0.601]              | 0.0020          | 0.0078 ** | 12.40     |
| Taze model farkı (son)             | weight_clipping | baseline |             10 | 0.224             | 0.246               | -0.023 [-0.054, 0.009]            | 0.2012          | 0.2227    | -0.42     |
| Son: Ölü birim oranı               | sin             | baseline |             10 | 0.497             | 0.433               | 0.064 [0.012, 0.113]              | 0.0488          | 0.1465    | 0.73      |
| Son: Ölü birim oranı               | tanh            | baseline |             10 | 0.441             | 0.433               | 0.008 [-0.034, 0.049]             | 0.7363          | 0.7363    | 0.11      |
| Son: Ölü birim oranı               | tri             | baseline |             10 | 0.826             | 0.433               | 0.393 [0.255, 0.517]              | 0.0039          | 0.0156 *  | 1.76      |
| Son: Ölü birim oranı               | weight_clipping | baseline |             10 | 0.486             | 0.433               | 0.053 [-0.013, 0.115]             | 0.1504          | 0.3008    | 0.48      |
| Son: Etkin rank (son gizli katman) | sin             | baseline |             10 | 5.611             | 3.490               | 2.121 [0.344, 3.897]              | 0.0586          | 0.1172    | 0.70      |
| Son: Etkin rank (son gizli katman) | tanh            | baseline |             10 | 7.209             | 3.490               | 3.719 [1.691, 5.409]              | 0.0137          | 0.0547    | 1.16      |
| Son: Etkin rank (son gizli katman) | tri             | baseline |             10 | 2.324             | 3.490               | -1.166 [-2.911, 0.388]            | 0.2480          | 0.2480    | -0.41     |
| Son: Etkin rank (son gizli katman) | weight_clipping | baseline |             10 | 6.915             | 3.490               | 3.424 [1.096, 5.455]              | 0.0254          | 0.0762    | 0.90      |
| Son: Ortalama \|w\|                | sin             | baseline |             10 | 0.081             | 0.081               | -0.000 [-0.003, 0.002]            | 0.8223          | 1.0000    | -0.07     |
| Son: Ortalama \|w\|                | tanh            | baseline |             10 | 0.075             | 0.081               | -0.006 [-0.008, -0.004]           | 0.0020          | 0.0078 ** | -1.74     |
| Son: Ortalama \|w\|                | tri             | baseline |             10 | 0.079             | 0.081               | -0.002 [-0.004, -0.000]           | 0.1250          | 0.3750    | -0.59     |
| Son: Ortalama \|w\|                | weight_clipping | baseline |             10 | 0.081             | 0.081               | 0.001 [-0.002, 0.003]             | 0.6973          | 1.0000    | 0.12      |

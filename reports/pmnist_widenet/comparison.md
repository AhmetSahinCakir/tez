## Referans: baseline

_eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_

| Ölçüt                              | Yöntem | Referans | Eşleşme sayısı | Ortalama (yöntem) | Ortalama (referans) | Fark (yöntem − referans) [%95 GA] | p (permütasyon) | p (Holm) | Cohen d_z |
|------------------------------------|--------|----------|---------------:|-------------------|---------------------|-----------------------------------|-----------------|----------|-----------|
| Normalize AUC                      | sin    | baseline |              3 | 0.865             | 0.866               | -0.002 [-0.002, -0.001]           | 0.2500          | 0.7500   | -7.11     |
| Normalize AUC                      | tanh   | baseline |              3 | 0.863             | 0.866               | -0.004 [-0.004, -0.004]           | 0.2500          | 0.7500   | -18.74    |
| Normalize AUC                      | tri    | baseline |              3 | 0.860             | 0.866               | -0.006 [-0.006, -0.006]           | 0.2500          | 0.7500   | -79.71    |
| Son pencere ort.                   | sin    | baseline |              3 | 0.864             | 0.863               | 0.001 [-0.001, 0.002]             | 0.7500          | 0.7500   | 0.30      |
| Son pencere ort.                   | tanh   | baseline |              3 | 0.864             | 0.863               | 0.000 [0.000, 0.001]              | 0.2500          | 0.7500   | 1.53      |
| Son pencere ort.                   | tri    | baseline |              3 | 0.862             | 0.863               | -0.001 [-0.002, -0.000]           | 0.2500          | 0.7500   | -1.08     |
| Koruma oranı                       | sin    | baseline |              3 | 1.008             | 1.003               | 0.004 [0.002, 0.006]              | 0.2500          | 0.7500   | 1.81      |
| Koruma oranı                       | tanh   | baseline |              3 | 1.012             | 1.003               | 0.008 [0.007, 0.009]              | 0.2500          | 0.7500   | 6.07      |
| Koruma oranı                       | tri    | baseline |              3 | 1.013             | 1.003               | 0.009 [0.008, 0.010]              | 0.2500          | 0.7500   | 7.76      |
| Taze model farkı (son)             | sin    | baseline |              3 | -0.029            | -0.028              | -0.001 [-0.004, 0.002]            | 0.5000          | 0.7500   | -0.46     |
| Taze model farkı (son)             | tanh   | baseline |              3 | -0.035            | -0.028              | -0.007 [-0.011, -0.003]           | 0.2500          | 0.7500   | -1.67     |
| Taze model farkı (son)             | tri    | baseline |              3 | -0.030            | -0.028              | -0.002 [-0.002, -0.002]           | 0.2500          | 0.7500   | -9.81     |
| Son: Ölü birim oranı               | sin    | baseline |              3 | 0.027             | 0.045               | -0.019 [-0.020, -0.016]           | 0.2500          | 0.7500   | -7.61     |
| Son: Ölü birim oranı               | tanh   | baseline |              3 | 0.020             | 0.045               | -0.026 [-0.028, -0.022]           | 0.2500          | 0.7500   | -8.73     |
| Son: Ölü birim oranı               | tri    | baseline |              3 | 0.039             | 0.045               | -0.006 [-0.009, -0.002]           | 0.2500          | 0.7500   | -1.55     |
| Son: Etkin rank (son gizli katman) | sin    | baseline |              3 | 328.680           | 352.007             | -23.327 [-37.123, -9.837]         | 0.2500          | 0.7500   | -1.71     |
| Son: Etkin rank (son gizli katman) | tanh   | baseline |              3 | 347.763           | 352.007             | -4.243 [-18.153, 5.762]           | 0.7500          | 0.7500   | -0.34     |
| Son: Etkin rank (son gizli katman) | tri    | baseline |              3 | 300.788           | 352.007             | -51.219 [-63.032, -41.440]        | 0.2500          | 0.7500   | -4.68     |
| Son: Ortalama \|w\|                | sin    | baseline |              3 | 0.032             | 0.032               | -0.000 [-0.000, -0.000]           | 0.2500          | 0.7500   | -31.41    |
| Son: Ortalama \|w\|                | tanh   | baseline |              3 | 0.032             | 0.032               | -0.000 [-0.000, -0.000]           | 0.2500          | 0.7500   | -26.65    |
| Son: Ortalama \|w\|                | tri    | baseline |              3 | 0.030             | 0.032               | -0.002 [-0.002, -0.002]           | 0.2500          | 0.7500   | -202.06   |

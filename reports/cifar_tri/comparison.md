## Referans: tri

_eşleştirilmiş fark (yöntem − referans), %95 bootstrap GA, işaret-çevirme permütasyon testi, Holm düzeltmesi; * p<0.05, ** p<0.01, *** p<0.001 (Holm)_

| Ölçüt                              | Yöntem       | Referans | Eşleşme sayısı | Ortalama (yöntem) | Ortalama (referans) | Fark (yöntem − referans) [%95 GA] | p (permütasyon) | p (Holm) | Cohen d_z |
|------------------------------------|--------------|----------|---------------:|-------------------|---------------------|-----------------------------------|-----------------|----------|-----------|
| Normalize AUC                      | tri_gamma1.5 | tri      |              5 | 0.747             | 0.751               | -0.003 [-0.005, -0.001]           | 0.1250          | 0.1250   | -1.22     |
| Son pencere ort.                   | tri_gamma1.5 | tri      |              5 | 0.746             | 0.752               | -0.005 [-0.008, -0.003]           | 0.0625          | 0.0625   | -1.58     |
| Koruma oranı                       | tri_gamma1.5 | tri      |              5 | 0.980             | 0.986               | -0.007 [-0.009, -0.004]           | 0.0625          | 0.0625   | -2.02     |
| Taze model farkı (son)             | tri_gamma1.5 | tri      |              5 | 0.020             | 0.010               | 0.010 [-0.019, 0.037]             | 0.6250          | 0.6250   | 0.26      |
| Son: Ölü birim oranı               | tri_gamma1.5 | tri      |              5 | 0.200             | 0.202               | -0.002 [-0.023, 0.023]            | 0.8750          | 0.8750   | -0.08     |
| Son: Etkin rank (son gizli katman) | tri_gamma1.5 | tri      |              5 | 12.298            | 11.986              | 0.312 [-0.623, 1.438]             | 0.7500          | 0.7500   | 0.22      |
| Son: Ortalama \|w\|                | tri_gamma1.5 | tri      |              5 | 0.046             | 0.040               | 0.007 [0.006, 0.007]              | 0.0625          | 0.0625   | 30.95     |

# RUNBOOK — uzun süreli deney zincirinin işletimi

Bu dosya, ölçek büyütme paketlerini (`scripts/run_scaleup_chain.sh`) sonuna kadar götürmek için gereken
işletim adımlarını tek yerde toplar. Hedef okuyucu, bu depoda çalışan (ana ya da yedek) Claude Code
oturumudur. Bilimsel içerik için `docs/RAPOR.md`'ye bakın.

## 1. Ne çalışıyor?

| Parça | Betik | Bitiş işareti (`results/scaleup_chain.log`) |
|---|---|---|
| (B) Zincir: `pmnist_adam_methods` → `pmnist_fullstream` (15 koşu, 800 görev × 60 000 örnek) → `pmnist_widenet` (12 koşu, 3×2000 ağ); her paketten sonra `analyze.py` | `scripts/run_scaleup_chain.sh` | `SCALEUP CHAIN FINISHED` |
| (C) Ardıl: zincir bitince üç paketi yeniden tarar (eksik koşu varsa tamamlar) | `scripts/run_after_scaleup.sh` | `FOLLOW-ON FINISHED` |
| Otomatik kayıt: 30 dk'da bir `results/ reports/ suites/` commit + push; saatte bir küçük-ağ checkpoint'lerini `checkpoint.sync.pt` olarak git'e kopyalar | `scripts/autocommit_results.sh` | — |

Koşular her görev sonunda `checkpoint.pt` yazar (`configs/base.yaml: checkpoint.every_n_tasks: 1`) ve yeniden
başlatılınca kaldığı görevden birebir devam eder (model, optimizer, yöntem izleri, tüm RNG'ler).

Durum tek komutla: `python3 scripts/suite_status.py` (paket başına tamam/toplam; bitmemiş koşuların görev sayısı,
checkpoint yaşı, kilit). `--check-done` ile çıkış kodu 0 = her şey bitti.

## 2. Konteyner yaşam döngüsü ve kendini onaran parçalar

Bulut konteyneri oturum boşta kalınca geri alınır: arka plan süreçleri ölür, disk (results/, checkpoint.pt,
running.lock) kalır, PID'ler sıfırlanır. Buna karşı:

1. **SessionStart kancası** (`.claude/hooks/session-start.sh` → `scripts/ensure_chain.sh`): her oturum
   başlangıcında eksik parçaları (zincir, ardıl, autocommit) başlatır; idempotenttir.
2. **Keep-alive beklemesi**: oturum canlı kaldıkça konteyner de canlı kalır. Her uyanışta
   `bash scripts/keepalive_wait.sh 110` komutunu Bash `run_in_background=true`, `timeout=7200000` ile başlat;
   zincir logu değişince, bir koşu başarısız olunca, zincir bitince ya da 110 dk sonra döner ve oturumu uyandırır.
   **Her dönüşte yenisini başlat.**
3. **Oturum içi rutinler** (bu oturuma bağlı): günlük `trig_0159ThCt6E5YQHbDnLMCsQNa`
   (`CRON_TZ=Europe/Istanbul 52 8 * * *`) ve 3 saatte bir `trig_01Fg4a7dHThaLJnVwNM1edu9` (`17 */3 * * *`).
   Oturum sağlıklıysa konteyneri uyandırır ve kontrol döngüsünü çalıştırır.
4. **Yedek bekçi (taze oturum)** `trig_01Miz3V1Xy1dhVmE3pEi5mtY` (`47 1-23/3 * * *`, her seferinde yeni
   oturum/konteyner): ana oturum takılı kalırsa (2026-10-09→14 arasında 4,5 gün böyle kaybedildi) devreye
   girer. `scripts/fallback_takeover.sh`:
   * `ALL DONE` — her koşunun `summary.json`'u var → hiçbir şey yapma.
   * `NOT NEEDED` — uzak daldaki son autosave commit'i 150 dk'dan genç ve **başka bir diskin** etiketini
     taşıyor (`[disk xxxxxxxx]`) → o konteyner yaşıyor, hiçbir şey yapma.
   * `TAKEOVER` — kimse yaşamıyor → ortamı kurar (pip, veri), `checkpoint.sync.pt` dosyalarını `checkpoint.pt`
     olarak geri yükler, zinciri burada başlatır. Bu andan itibaren o oturum ana oturumdur: §3'teki döngüyü uygular.

   Aynı mekanizma `ensure_chain.sh`'nin içinde de vardır: başka bir konteyner aktifken (uzak autosave genç ve
   etiketi farklı) hiçbir şey başlatmaz, `results/ensure_chain.log`'a `skipped: another container is active`
   yazar. Eski ana oturum dirilirse bunu görür ve zinciri **başlatmaz**; yalnızca durumu bildirir.

   Disk kimliği `results/.disk_id` (izlenmez) dosyasındadır; `scripts/disk_id.sh` üretir/okur.
   `checkpoint.sync.pt` yalnızca küçük-ağ paketleri için tutulur (~0,4 MB/koşu); geniş ağ (~40 MB) için
   tutulmaz — oradaki yarım koşular devralmada baştan başlar.

## 3. Her uyanışta yapılacak kontrol döngüsü

```
ps -eo pid,etime,args | grep -E "run_suite|run_scaleup|run_after|autocommit|keepalive" | grep -v grep
tail -n 5 results/scaleup_chain.log; tail -n 3 results/ensure_chain.log
python3 scripts/suite_status.py
grep -l -E "failed \(exit code|Traceback" results/suite_*.log     # başarısız koşu var mı?
```
* Süreç yoksa ve `ensure_chain.log`'da "skipped: another container" **yoksa**: `bash scripts/ensure_chain.sh`.
* Başarısız koşu varsa `results/<paket>/<koşu>/stdout.log`'a bak, düzelt, zinciri yeniden başlat
  (sırayla: zincir bash'ine `kill -TERM`, sonra `run_suite.py`'a `kill -TERM` — çocuklarını kendisi kapatır;
  `pkill/pgrep -f` kullanma).
* `git add -A results reports && git commit && git push -u origin claude/tez-projesi-tamamla-m2x560`.
* Keep-alive beklemesini yeniden başlat (§2.2).
* CPU 4 çekirdek: başka ağır iş başlatma; testleri kısa tut.

## 4. Bir paket bitince

Zincir `reports/<paket>/` altına `analyze.py` çıktısını üretir (yoksa:
`python3 scripts/analyze.py --results results/<paket> --out reports/<paket> --reference baseline --smooth 5`).
`docs/RAPOR.md`'deki yer tutucuları doldur:

* **§14.2 `pmnist_fullstream`** — 800 görevlik uzun ufukta baseline/sin/tanh/weight_clipping/tri × 3 tohum:
  model başına ortalama ± %95 GA (AUC, erken pencere, son pencere, koruma oranı = son/erken, taze model farkı),
  eşleştirilmiş farklar (aynı tohum) baseline'a göre ve tri − weight_clipping; 100 görevlik bloklarda eğri şekli;
  mekanizma ölçütleri (doygunluk payı, |w|/A, cos², ölü birim, etkin kerte, güncelleme normu); §6 ve §13'teki
  200 görevlik sonuçlarla karşılaştırma; `reports/pmnist_fullstream/*.pdf` şekillerine atıf.
* **§14.3 `pmnist_widenet`** — 3×2000 ağda baseline/sin/tanh/tri × 3 tohum, aynı yapı; §6 ile karşılaştırma
  (geniş ağda plastisite kaybı daha hafif; farklar daralıyor mu?).
* README'deki bulgular tablosunu gerekirse güncelle. Commit + push.

İstatistik yardımcıları: `plasticity/analysis/stats.py` (eşleştirilmiş t-testi + bootstrap GA); örnek kullanım
§14.1 için `reports/pmnist_adam_methods/` üretilirken kullanılan `scripts/analyze.py`.

## 5. Her şey bitince

`SCALEUP CHAIN FINISHED` ve `FOLLOW-ON FINISHED` yazılmış, `python3 scripts/suite_status.py --check-done` 0
dönüyor, §14.2 ve §14.3 dolu: üç rutini de `update_trigger(enabled=false)` ile kapat
(`trig_0159ThCt6E5YQHbDnLMCsQNa`, `trig_01Fg4a7dHThaLJnVwNM1edu9`, `trig_01Miz3V1Xy1dhVmE3pEi5mtY`;
araç yoksa kullanıcıdan claude.ai rutinler sayfasından kapatmasını iste), son commit + push, kullanıcıya
Türkçe 5–8 satırlık özet.

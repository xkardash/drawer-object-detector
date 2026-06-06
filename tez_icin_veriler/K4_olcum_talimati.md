# K4 — Cihaz-üstü gecikme + termal ölçüm talimatı (Redmi Note 11)

Amaç: gerçek-zaman iddiasını sağlamlaştırmak — **p50/p95/p99 gecikme** + **zaman-içinde-FPS (termal
throttling)** eğrisi. Uygulamaya kayıt mekanizması eklendi (`lib/latency_logger.dart` + REC butonu).

## Adımlar (telefon)
1. **Derle/kur:** `flutter build apk --release` → APK'yı Redmi Note 11'e kur (veya `flutter run --release`).
2. **Auto'yu KAPAT**, sabit bir boyut seç (örn. **320**). Termal eğri tek modele ait olsun. (Birden çok
   boyut istiyorsan her birini ayrı koşuyla ölç: 320, 512, 640.)
3. Üst bardaki **"⦿ Kayıt"** çipine dokun → kırmızı **"● REC"** olur, süre + kare sayısı sayar.
4. Telefonu **elde tut**, çekmece sahnesine doğrult, **~5-10 dk** normal kullan (uzun koşu = net termal
   sinyal; elde tutmak gerçekçi ısınma verir).
5. Tekrar dokun → durur, **snackbar CSV yolunu** gösterir
   (`/storage/emulated/0/Android/data/<paket>/files/latency_<tarih>.csv`).

## CSV'yi PC'ye al (en güvenilir: adb)
```
adb pull /storage/emulated/0/Android/data/com.example.cekmece_detector/files/  ./pull
```
(Tam yol snackbar'da yazıyor.) Ya da USB → dosya yöneticisi → `Android/data/<paket>/files/`.
İndirdiğin `latency_*.csv` dosyasını **`flutter_app/tez_icin_veriler/`** içine koy.

## Analiz (ben yaparım)
```
python analyze_latency.py            # en yeni latency_*.csv'yi işler
```
Üretir: p50/p95/p99 gecikme, FPS (p50 + en kötü p05), **termal** (ilk-üçte-bir vs son-üçte-bir FPS düşüşü),
ve `thesis_plots/k4_latency_thermal.png` (FPS & gecikme vs zaman). Sonucu PERFORMANCE.md'ye yazarım.

## İpuçları
- En değerli koşu: **640** (en yavaş, throttling en olası) + **320** (dağıtım sweet-spot). 800px'in
  "ölçüm artefaktı"nı da netleştirmek için 800'ü de bir koş.
- CSV küçük (~birkaç bin satır); birden çok koşu olursa hepsini gönder, ayrı ayrı işlerim.

# Performans Optimizasyon Notları

Bu doküman, Flutter + YOLOv8 çekmece içi nesne tespit uygulamasının 1 FPS'ten
kullanılabilir tempoya getirilmesi sürecindeki bulguları, çözümleri ve
başarısız denemeleri kayıt altına alır.

---

## Başlangıç Durumu

- **FPS**: 1 (Redmi Note 11, debug mode'da bile aynı, release çok az daha iyi)
- **Model**: YOLOv8 fp32 TFLite, 320 ve 800 boyut
- **Pipeline**: `yuv420ToImage` → `copyRotate` → `letterbox` → `imageToFloat32Input` → `Interpreter.run` → nested-list output parse

---

## Tespit Edilen Darboğazlar

Etki sırasına göre:

### 1. `Float32ListReshape` extension (en büyük)
Her karede `Float32List → 4D nested Dart List` çevirisi yapıyordu.
- 320 modelinde: **307K** boxed `Double` allocate
- 800 modelinde: **1.92M** boxed `Double` allocate
- Tek başına yüzlerce ms

### 2. `imageToFloat32Input` per-pixel `getPixel(x, y)`
`image` paketi 4.x'te `Pixel` object alloc per pixel. 800×800 = **640K** çağrı.

### 3. `yuv420ToImage` + `copyRotate` + `letterbox`
Aynı veri üzerinde **4 ayrı tam-image pass**, hepsi saf Dart'ta.
- `yuv420ToImage`: pixel-by-pixel `setPixelRgb` (slow)
- `copyRotate`: tam image rotation copy
- `letterbox`: resize + fill + composite (3 ayrı pass)

### 4. Output buffer nested `List.generate`
Her karede ~94K boxed `Double` allocate (800 modelinde).

### 5. Her şey UI isolate'ında
Inference UI thread'ini bloke ediyor, kamera preview takılıyor.

### 6. Manuel frame skip (1/5)
`_frameCount % 5 != 0 return` — inference yavaşken görünür FPS'i 5× daha düşürüyordu.

### 7. GPU/NNAPI delegate yok
Sadece XNNPACK CPU, donanım hızlandırması kullanılmıyor.

---

## Uygulanan Optimizasyonlar

### A. Fused YUV→Input single-pass preprocessing
`image_utils.dart` tamamen yeniden yazıldı. `image` paketi by-pass edildi.

**YUV420 düzlemlerinden doğrudan Float32 buffer'a**, tek döngüde:
- 90° rotation (sampling sırasında, ayrı pass yok)
- Letterbox (padX/padY hesaplanır, gri padding yazılır)
- BT.601 YUV→RGB (integer aritmetiği, fixed-point Q16)
- Normalize [0, 1]

Sonuç: 4 tam pass + ara allocation → **tek pass**, ~%80-90 daha hızlı.

### B. Pre-allocated `Float32List` buffer'lar
Detector load time'da bir kere `Float32List` allocate edilir, her karede içine yazılır.
Reshape extension'ı tamamen silindi.

### C. ByteBuffer üzerinden TFLite I/O
```dart
interpreter.runForMultipleInputs(
  [_inputBuffer!.buffer],     // ByteBuffer fast-path
  {0: _outputBuffer!.buffer},
);
```

`tflite_flutter` 0.12'nin `Tensor.copyTo(Float32List)` Float32List'i `List<double>` olarak görüp shape kontrolünde patlıyor (kritik bug — bkz. aşağı). `ByteBuffer` raw byte fast-path'i kullanmak hem doğru çalışır hem daha hızlı.

### D. Output flat-indexing
Nested list yerine `Float32List` üzerinden `out[c * anchors + a]` ile direkt erişim. Row-major layout, sıfır allocation.

### E. Frame skip kaldırıldı
`_processing` flag yeterli — yeni kare geldiğinde önceki bitmemişse atlanır.
Bu doğal back-pressure, manuel `% 5` skip'inden çok daha iyi.

### F. GPU Delegate (Adreno OpenCL)
`GpuDelegateV2` + `isPrecisionLossAllowed: true` ile fp16 GPU inference:
```dart
GpuDelegateOptionsV2(
  isPrecisionLossAllowed: true,
  inferencePreference: 1, // SUSTAINED_SPEED
  inferencePriority1: 2,  // MIN_LATENCY
)
```
Init başarısızsa **otomatik CPU XNNPACK + 4 thread fallback**.

---

## Faz 2 Optimizasyonları (UI akıcılığı + off-thread inference)

Faz 1'den sonra inference 30-80ms süresince UI thread'ini bloke ediyordu;
`setState` her frame'de tüm widget ağacını rebuild ediyordu. Faz 2 bu iki
problemi hedefler.

### G. IsolateInterpreter (background isolate inference)
`tflite_flutter` 0.12'nin `IsolateInterpreter`'ı interpreter'ı sarmalayarak
inference'i ayrı bir isolate'te koşturur. Native interpreter address
paylaşıldığı için tensors aynı yerde kalır; sadece input/output ByteBuffer'ları
isolate sınırını geçer.

```dart
_interpreter = await Interpreter.fromAsset(asset, options: opts);
_isolateInterpreter = await IsolateInterpreter.create(
  address: _interpreter!.address,
);
// detect içinde:
await _isolateInterpreter!.runForMultipleInputs(
  [_inputBuffer!.buffer], {0: _outputBuffer!.buffer},
);
```

UI thread artık inference süresince serbest — kamera frame teslimi ve render
loop'u bloklanmıyor. Preprocessing ve parse hâlâ UI isolate'inde (Faz 3'te
tek worker isolate'e taşınabilir).

### H. ValueNotifier + RepaintBoundary (overlay izolasyonu)
Frame başına değişen state (`detections`, `imageSize`, `fps`) `ValueNotifier`
arkasında. `ValueListenableBuilder` ile sadece bbox overlay ve fps chip
rebuild ediliyor; `CameraPreview` ve diğer chrome dokunulmuyor.

```dart
final _overlay = ValueNotifier<_OverlayFrame>(...);
final _fps = ValueNotifier<double>(0.0);

// _onFrame içinde:
_overlay.value = _OverlayFrame(detections, imageSize);
_fps.value = ema;

// build içinde:
Positioned.fill(
  child: RepaintBoundary(
    child: ValueListenableBuilder<_OverlayFrame>(
      valueListenable: _overlay,
      builder: (_, f, __) => CustomPaint(painter: DetectionPainter(...)),
    ),
  ),
),
```

`RepaintBoundary` paint katmanını izole ediyor — bbox repaint'i parent
Stack'in compositing'ini tetiklemiyor.

---

## Tespit Edilen Bug'lar (Önemli)

### B1. `Tensor.copyTo(Float32List)` sessiz exception
**Sorun**: `tflite_flutter` 0.12'de `Tensor.copyTo` Float32List'i `List<double>` olarak yorumlar, nested-list'e çevirir, sonra `_duplicateList` shape kontrolünde
`[length]` vs `[1, channels, anchors]` uyuşmazlığı → `ArgumentError`.
`_onFrame`'deki `try/catch (_) {}` bu hatayı sessizce yutuyordu → kullanıcı için **kare akıcı ama hiç detection yok**.

**Çözüm**: `Float32List.buffer` (ByteBuffer) geçir. ByteBuffer için raw-bytes fast-path var. Float32List `.buffer` ile aynı belleği paylaşır, yazımlar görünür.

**Ders**: Detection hatalarını sessizce yutma. `debugPrint(e, st)` ile yüzeyleştir.

### B2. YOLOv8 output normalized [0..1], pixel değil
**Sorun**: Ultralytics TFLite export bbox koordinatlarını `cx/cy/w/h` olarak **normalized [0..1]** çıkarır. Orijinal kod bunları pixel olarak yorumluyordu — `(cx - padX=40)` matematiğinde `0.5 - 40 = -39.5` → clamp → bbox `(0, 0, 0, 1)` → bbox **görünmüyor**.

**Çözüm**: Parser'da `cx, cy, w, h` değerlerini `inputSize` (örn. 320) ile çarp:
```dart
final cx = out[a] * inSize;
final cy = out[anchors + a] * inSize;
final w  = out[2 * anchors + a] * inSize;
final h  = out[3 * anchors + a] * inSize;
```

### B3. INT8 quantization + YOLOv8 DFL = bozuk model
**Sorun**: Ultralytics'in INT8 TFLite export'unda **cx ve cy değerleri her zaman 0**.
`w` ve `h` çalışıyor ama bbox merkez koordinatı kaybolduğu için detection'lar
sol-üst köşeye sıkışıyor.

**Doğrulama**: Bu Flutter sorunu **değil**. Aynı INT8 modelini TensorFlow'un
Python interpreter'ında çalıştırdık, aynen `cx=0, cy=0` aldık.

**Kök neden**: YOLOv8 bbox prediction'ı klasik regression değil. Her koordinat
için **16-bin distribution** tahmin eder (reg_max=16), sonra softmax + weighted sum
ile gerçek değeri çıkarır (DFL — Distribution Focal Loss).
INT8 quantization'da:
1. 16-bin softmax precision'ını dramatik kaybeder
2. Weighted sum, quantization hatasını biriktirir
3. cx/cy değerleri sıfıra çöker

**Çözüm**: Karar — fp32'de kal, GPU delegate ile hız al. INT8 modelleri silindi.

**Bilgi**: Bu bilinen bir Ultralytics issue. Açık GitHub issue'ları var.

**GÜNCELLEME (Faz 3A, 2026-05-29) — nms=True bypass TEST EDİLDİ, BAŞARISIZ:**
Hipotez (`int8=True nms=True` DFL'i bypass eder) iki ayrı şekilde çürütüldü:
1. **int8 + nms=True**: ultralytics 8.3.167 + onnx2tf 2.4.0 ile export başarılı (output
   `[1, 300, 6]` decoded), AMA üretilen gömülü-NMS grafiği `SELECT` op'unu hem `tf.lite`
   hem `ai-edge-litert` runtime'ında `allocate_tensors`'ta hazırlayamıyor
   (`select.cc: has_low_rank_input_condition was not true`). 4 varyant da (fp32/fp16/
   int8/full-int8) aynı sebepten allocate olmuyor. `tflite_flutter` cihazda aynı runtime
   ailesini kullandığı için telefonda da çalışmaz. → Op uyumsuzluğu, mimari engel.
2. **Standart int8 (nms yok), güncel toolchain**: cx/cy hâlâ tam `0.0000`'a çöküyor
   (w/h ve class skorları çalışıyor). onnx2tf 2.4.0 DFL int8 bug'ını DÜZELTMEMİŞ.

**Sonuç**: Bu model + toolchain ile INT8 yolu tamamen kapalı. Hız için fp16 GPU delegate'te
kal; başka kaldıraçlara bak (512 model / pipeline restructure / native plugin).
Test script'leri: `export_int8_nms.py`, `test_int8_nms_model.py`, `diagnose_nms_models.py`,
`test_int8_standard.py`.

---

## Performans Sonuçları (Redmi Note 11, release mode)

| Mod | Önce | Faz 1 CPU | Faz 1 GPU | Faz 2 GPU + Isolate | Faz 3 ölçülen (2026-05-29) |
|-----|------|-----------|-----------|---------------------|----------------------------|
| 320 | ~1 FPS | ~8 FPS | 12 FPS | — | — |
| 512 | n/a | — | — | — | **6 FPS** (yeni orta yol) |
| 640 | n/a | ~2 FPS | 4.5 FPS | — | **4 FPS** (pre=11 inf=224 parse=0 pure=224) |
| 800 | ~1 FPS | ~1 FPS | **1–2 FPS** | — | 1–2 FPS (2026-06-05) |

**640 kırılımı (per-stage timing, Redmi Note 11, release):** inference compute %95,
preprocess %5, parse ~0, IsolateInterpreter kopya yükü ~0 (inf≈pure). → **inference-bound.**
Pipeline/worker-isolate restructure faydasız (kopya zaten 0). Tek kaldıraç input boyutu.
512 = 224×(512/640)²+11 ≈ 154ms tahmini → ölçülen 6 FPS, tahmin tuttu.

**Faz 1 net kazanç**: 320 modunda **12×**, 640 modunda yenisi sayılır.
**Faz 2 beklenen**: UI takılması bitmiş + %30-50 ek FPS (inference UI thread'inden çıktığı için frame teslimi kuyruğa girmiyor).

---

## Çekirdek Dosyalar

| Dosya | Sorumluluk |
|-------|-----------|
| `lib/yolo_detector.dart` | Model load, GPU delegate, inference, NMS, output parse |
| `lib/image_utils.dart` | Fused YUV→Float32 single-pass preprocessing |
| `lib/home_screen.dart` | UI, camera stream, model selector |
| `lib/detection_painter.dart` | Bbox overlay (Canvas) |

---

## Açık Konular / Gelecek İyileştirmeler

### O1. Küçük nesne tespiti (uzaktan)
**Durum**: 320 modeli ile çok yaklaşmak gerekiyor. 640 ile menzil iyileşti
ama 4.5 FPS biraz düşük. 800 daha geniş menzil ama hız ödün.

**Olası çözümler**:
- Kamera çözünürlüğünü `ResolutionPreset.low` → `medium`/`high`'a çek.
  Aynı 320 model input'una **daha temiz/detaylı** downsampling sağlar.
  FPS etkisi minimal, ~%20-30 menzil kazancı bekliyorum.
- Confidence threshold'u 0.20-0.25'e düşür — düşük-skorlu uzak detection'ları
  kaybetmemek için. False positive riski artar.

### O2. INT8 ile tekrar deneme (DFL bypass) — ❌ KAPALI (Faz 3A'da test edildi)
`int8=True nms=True` denendi: gömülü-NMS grafiği `SELECT` op'u yüzünden hiçbir TFLite
runtime'ında allocate olmuyor (cihazda da çalışmaz). Standart int8 ise hâlâ cx/cy=0.
Tam analiz B3'te. **Bu yolu bir daha önerme.**

### O3. Native Kotlin plugin
`tflite_flutter` yerine custom Flutter plugin yaz, Kotlin'de:
- CameraX `ImageAnalysis` → hardware-accelerated YUV→RGB
- `org.tensorflow:tensorflow-lite-gpu` AAR direkt dependency
- Zero-copy frame buffer paylaşımı
**Tahmini kazanç**: %20-40 ek hız (preprocessing native). iOS yeniden yazılır.

### O4. Inference'i isolate'a taşı
Dart `compute()` veya custom Isolate ile inference UI thread'inden çıkar.
GPU delegate ile UI bloke süresi zaten kısa (~30-80ms), bu öncelikli değil.

### O5. fp16 model dene
Ultralytics `half=True` ile fp16 weights export. CPU'da fp32 ile aynı hız
(fp16 op'ları CPU'da pratikte yok) ama GPU delegate'te 2× hızlanma olabilir.
Mevcut `isPrecisionLossAllowed: true` ayarı zaten fp16 GPU inference yapıyor,
weights'ten zaten dequantize ediyor. Test edilmesi gerek.

---

## Sürüm Geçmişi (özet)

- **Initial**: 1 FPS, bbox yok (sessiz hata + normalized coord bug'ı).
- **+Fused preprocessing**: ~5 FPS, hala bbox yok (Tensor.copyTo bug).
- **+ByteBuffer I/O**: ~7-8 FPS, bbox `(0,0,0,1)` (normalize bug).
- **+inputSize çarpımı**: ~8 FPS, bbox çıkıyor. **İlk çalışan sürüm.**
- **INT8 quantization denemesi**: Hız aynı/azaldı, cx/cy=0 bug, geri alındı.
- **GPU delegate + fp32**: 320 = **12 FPS**, 640 = 4.5 FPS.
- **+IsolateInterpreter (Faz 2)**: inference background isolate'inde — UI bloke yok.
- **+ValueNotifier + RepaintBoundary (Faz 2)**: per-frame `setState` kaldırıldı, sadece overlay + fps chip repaint. Beklenen: 320 = 15-18 FPS, 640 = 6-7 FPS (cihazda ölçülecek).
- **Faz 3A — INT8 + nms=True denemesi (2026-05-29)**: Karar kapısı Python'da çalıştırıldı.
  Her iki int8 yolu da çürütüldü (nms=True → SELECT op allocate fail; standart int8 → cx/cy=0).
  Flutter'a hiç dokunulmadı (~30 dk). INT8 yönü kapatıldı. Detay: B3 + O2.

---

## Faz 4 — fp16 model + adaptif "Auto" model seçimi (2026-06-05)

Öncelik: **denge** (kullanılabilir FPS + makul menzil), kapsam: **sadece Flutter** (native O3 ertelendi).

### I. fp16 model swap (O5 hayata geçirildi)
`export_fp16.py` (kök) ile `best.pt`'den 320/512/640/800 fp16 TFLite üretildi
(`YOLO(...).export(format="tflite", half=True, imgsz=N)` → `best_float16.tflite` kopyalandı).
- **Boyut**: fp32 11.9 MB → **fp16 6.0 MB** (tam yarısı). Daha hızlı yükleme, daha az bellek.
- **I/O tensörleri float32 kalır** → `Float32List` buffer kodu DEĞİŞMEDEN çalışıyor. `loadModel`'e
  tensör-tipi float32 doğrulaması eklendi (B1 dersi: hata sessizce yutulmasın → tip yanlışsa `StateError`).
- **Beklenti (dürüst)**: GPU delegate zaten `isPrecisionLossAllowed` ile fp16 hesaplıyordu, bu yüzden
  steady-state GPU FPS kazancı GARANTİ DEĞİL. Kesin kazanç: yükleme/bellek. FPS etkisi **cihazda ölçülecek**.
- fp32 dosyaları silinmedi; `pubspec.yaml`'de hâlâ listeli → A/B için `ModelSize.assetPath`'i flip'le.

### II. Adaptif "Auto" model seçimi (`home_screen.dart`)
Doküman cost∝size² ilişkisini doğrulamıştı; bunu kullanan **çalışma anında oynamayan** seçim:
- Açılışta **320 ile probe** → `pureInferenceMs` ölçülür → `ModelSize.pickForBudget()` ile FPS bütçesine
  (`_autoBudgetMs=70`, ~12 FPS) sığan **en büyük** boyut seçilir, gerekiyorsa bir kez yeniden yüklenir.
- **Tek seferlik düzeltme**: ~40 kare sonra ölçülen FPS bant dışıysa (`<9` veya `>18`) bir kademe in/çık,
  sonra sabitle. **Sürekli reaktif switch YOK** — model reload GPU kernel'lerini yeniden derliyor (~1–3s,
  tflite_flutter'da cache API'si yok), oscillation bilinçli olarak engellendi.
- 320 probe'tan büyük boyutlara ekstrapolasyon hafif **muhafazakâr** (sabit GPU overhead size² değil) →
  akıcılığa yaslar; tek-seferlik düzeltme headroom'u geri alır.
- UI: model seçicisine **"Oto"** chip'i eklendi (varsayılan açık). Sabit boyuta tıklama Auto'yu kapatır.
- `_switching` guard'ı: model yüklenirken kareler düşürülür (kapanan interpreter'a detect çalışmaz).

### III. Hijyen
- Kullanılmayan `image: ^4.3.0` bağımlılığı kaldırıldı (lib'de hiç import yok; 5 transitive paket düştü).
- Kare-başı `debugPrint` (DET / PERF) `kDebugMode` arkasına alındı → release'te logcat spam'i yok.

### Bilinçli ATLANANLAR (dürüstlük)
- **90° önişleme fast-path (Faz 3)**: önişleme 640'ta toplamın ~%5'i, uygulama inference-bound →
  doğru/sıcak döngüye karmaşıklık eklemek FPS'i kıpırdatmaz. Yapılmadı.
- **NNAPI delegate / GPU kernel-cache**: `tflite_flutter` Dart API'sinde **yok** (paket içinde
  `nnapi_delegate.dart` ve `serializationDir` bulunmuyor). Android'de gerçek delegate seçeneği
  yalnızca GPU vs CPU-XNNPACK — ki doküman zaten ölçtü (GPU kazanıyor). Native plugin (O3) gerekir.

### Cihazda yapılacak doğrulama (A/B) — Redmi Note 11
1. `flutter build apk --release` → kur, ekrandaki perf chip'i (`pre·inf·parse·pure`) ve FPS chip'ini izle.
2. **fp16 vs fp32**: her boyutta `ModelSize.assetPath`'i `fp16`↔`fp32` flip'le, rebuild, `pure` ms'i karşılaştır.
   Tablodaki "Faz 4 fp16" sütununu doldur.
3. **Auto mod**: aç → ~10–15 FPS bandında bir boyuta oturuyor mu, oscillation var mı doğrula.
   Üst durum chip'i Auto'nun seçtiği boyutu (`Npx`) gösterir.
4. **Doğruluk regresyonu**: fp16 ile bbox'lar fp32 ile aynı mı (birkaç sahne yan yana) — küçük nesne kaybı yok.

### fp32 baseline — cihaz ölçümü (tez bulgusu)

Redmi Note 11, GPU delegate (Adreno, fp16 OpenCL), release mode. Giriş boyutuna göre
ölçülen kare hızı (kullanıcı ölçümü, 2026-06-05):

| Model girişi | fp32 FPS (ölçülen) | ≈ kare süresi | fp16 FPS (ölçülecek) |
|--------------|--------------------|---------------|----------------------|
| 320 px | **12 FPS** | ~83 ms | TBD |
| 512 px | **6 FPS**  | ~167 ms | TBD |
| 640 px | **4 FPS**  | ~250 ms | TBD |
| 800 px | **1–2 FPS** | ~500–1000 ms | TBD |

#### Per-stage timing kırılımı (fp32, ms — kullanıcı ölçümü 2026-06-05)

Ekrandaki perf chip'ten (`pre · inf · parse · pure`) okunan aşama süreleri. `inf` = inference +
IsolateInterpreter kopya (canlı, EMA); `pure` = saf inference (yükleme anı benchmark, ana-isolate,
kopya hariç); `kopya ≈ inf − pure`.

| Boyut | pre | inf (+kopya) | parse | pure | kopya (inf−pure) | FPS |
|-------|-----|--------------|-------|------|------------------|-----|
| 320 | 3  | 70    | 0 | 61  | ~9     | 12 |
| 512 | 7  | 150   | 0 | 129 | ~21    | 6 |
| 640 | 12 | 230   | 0 | 221 | ~9     | 4 |
| 800 | 18 | 185 ⚠ | 0 | 371 | −186 ⚠ | 1–2 |

**Yorum (tez için — güçlü teknik bulgu):**
- **parse = 0 ms her boyutta** ve **pre çok küçük (3–18 ms)** → toplam sürenin neredeyse tamamı
  inference. Uygulama **kesin biçimde inference-bound**; preprocessing/parse optimizasyonu
  faydasız olurdu (ölçümle kanıtlandı, mühendislik kararı gerekçelendirildi).
- **Çapraz doğrulama:** ölçülen FPS ≈ 1000/(pre+inf+parse) — 320: 73ms→13.7≈12, 512: 157ms→6.4≈6,
  640: 242ms→4.1≈4. Aşama toplamı gözlenen kare hızını açıklıyor (artık zaman = kamera kare teslimi).
- **`kopya = inf − pure` küçük (~9–21 ms)** (320/512/640) → `IsolateInterpreter` kare-başı buffer
  kopya maliyeti ihmal edilebilir; inference'i background isolate'e taşımak UI'yi serbest bırakıp
  throughput'u neredeyse hiç bozmuyor. Pipeline/worker-isolate yeniden yapılanması gereksiz.
- **`pure` inference temiz biçimde `boyut²` ile ölçekleniyor**: `pure/boyut²` → 320:5.96 · 512:4.92 ·
  640:5.40 · 800:5.80 (×10⁻⁴ ms/px², ortalama ~5.5, dar bant). Maliyet modelini **4 boyutta birden**
  nicel doğrular ve Faz 4 "Auto" seçicinin boyut tahmininin temelini oluşturur.
- **800px ölçüm tutarsızlığı (dürüst not):** Canlı `inf` (185 ms) hem `pure` (371 ms) hem de gözlenen
  FPS (1–2 ⇒ 500–1000 ms/kare) ile çelişiyor. `pure` (371 ms) hem FPS'le hem boyut² eğilimiyle
  tutarlı → 800'de gerçek inference ≈ **371 ms**. Düşük kare hızında canlı `inf` EMA'sı güvenilir
  oturmuyor (ölçüm artefaktı). 800 yine de inference-bound; sadece per-frame `inf` okuması yanıltıcı.
  **Yeniden ölçüm önerilir** (telefonu daha uzun sabit tut). Not: bu, boyut² eğrisinin 800'de de
  geçerli olduğunu gösterir — "kaynak tavanı / kopuş" yok.
- **Pratik çıkarım:** bu cihazda gerçek-zamanlı **denge noktası 320–512**; 800px (≈371 ms) uygun
  değil. Faz 4 "Auto" seçici bu boyut² eğilimini kullanarak cihaza göre boyutu otomatik seçer.

#### Hız/Doğruluk ödünleşimi — export imgsz (fp32, tez merkez bulgusu)

Model **imgsz=800'de eğitildi**; tek ağırlık seti dört giriş çözünürlüğüne export edildi (export
yalnızca giriş tensörünü yeniden boyutlandırır, ağırlık aynı — dört .tflite de ~11.9 MB). Doğruluk
`yolo val` ile `best.pt` üzerinde her imgsz'de ölçüldü. Ana sonuç **held-out TEST seti** (40 görüntü,
eğitimde hiç kullanılmadı); val seti (40 görüntü) karşılaştırma için. `mAP` = COCO metriği. Bu sayılar
fp32 TFLite dağıtımının doğruluğuna karşılık gelir. Script: `val_imgsz.py`.

**TEST seti (ana sonuç):**

| imgsz | FPS (fp32) | pure inf (ms) | mAP@0.5 | mAP@0.5:0.95 | P | R |
|-------|-----------|---------------|---------|--------------|------|------|
| 320 | 12 | 61 | 0.276 | 0.145 | 0.474 | **0.265** |
| 512 | 6 | 129 | 0.582 | 0.363 | 0.678 | 0.532 |
| 640 | 4 | 221 | 0.636 | 0.420 | 0.846 | 0.509 |
| 800 | 1–2 | 371 | 0.690 | 0.460 | 0.811 | 0.604 |

**VAL seti (karşılaştırma):**

| imgsz | mAP@0.5 | mAP@0.5:0.95 | P | R |
|-------|---------|--------------|------|------|
| 320 | 0.317 | 0.163 | 0.492 | 0.292 |
| 512 | 0.579 | 0.362 | 0.679 | 0.514 |
| 640 | 0.688 | 0.446 | 0.801 | 0.564 |
| 800 | 0.736 | 0.501 | 0.827 | 0.636 |

**Yorum (tez için):**
- **Monotonik ödünleşim** (test): imgsz büyüdükçe doğruluk artar, hız düşer — klasik speed-accuracy
  trade-off. 320→800: mAP@0.5 **0.28→0.69 (~2.5×)** ama FPS **12→1–2 (~8–12× yavaş)**.
- **Recall çarpıcı**: 320'de **R=0.26** — nesnelerin ~%74'ü kaçıyor. Küçük nesneler (anahtar, kalem,
  tırnak makası) 320×320 downscale'de yok oluyor; 800'de R=0.60'a çıkıyor. "Küçük nesne × düşük giriş
  çözünürlüğü" probleminin **doğrudan nicel kanıtı**.
- **Lokalizasyon (mAP@0.5:0.95)** çözünürlükle daha hızlı iyileşir: 0.15→0.46 (**~3×**).
- **Test ≈ Val**: iki split çok yakın (örn. 512 mAP@0.5: test 0.582 / val 0.579) → model **iyi
  genelleşiyor**, overfit yok. Test hafifçe düşük (beklenen, held-out).
- **"Ücretsiz hız" yanılgısı**: 320 sadece FPS'e bakınca cazip ama R=0.26 ile pratikte kullanılamaz.
  **Gerçek denge 512** (test mAP@0.5=0.58, R=0.53, 6 FPS): kabul edilebilir doğruluk + akıcılık. 640
  doğruluk için biraz daha iyi (mAP=0.64) ama 4 FPS. → Faz 4 "Auto" budget'ı 512'ye yaslanacak şekilde
  ayarlanabilir (şu an salt-hız budget'ı 320 seçebilir).
- **Metodoloji notu**: model 800'de eğitildiğinden 800 = native çözünürlük = doğruluk tavanı; küçük
  imgsz doğruluktan ödün vererek hız satın alır. Setler küçük (40'ar görüntü) → mutlak mAP'lerde
  varyans olabilir (örn. test'te 640 P>800 P — 40-görüntü gürültüsü), ama mAP eğilimi net monotonik.

#### Bilgisayar (.pt) vs Telefon export (.tflite) — gerçek export etkisi

Yukarıdaki `.pt` (PyTorch, PC) doğruluğu bir **vekildir**; burada **gerçek `.tflite` export dosyaları**
PC'de (CPU, TFLite runtime) validate edildi — telefonun çalıştırdığı ağın aynısı. Script: `val_tflite_imgsz.py`.

**mAP@0.5 — .pt (bilgisayar) vs fp32 .tflite (telefon export):**

| imgsz | TEST .pt | TEST .tflite | Δ | VAL .pt | VAL .tflite | Δ |
|-------|----------|--------------|------|---------|-------------|------|
| 320 | 0.276 | 0.246 | −0.030 | 0.317 | 0.295 | −0.022 |
| 512 | 0.582 | 0.578 | −0.004 | 0.579 | 0.579 | ~0 |
| 640 | 0.636 | 0.613 | −0.023 | 0.688 | 0.696 | +0.008 |
| 800 | 0.690 | 0.677 | −0.013 | 0.736 | 0.716 | −0.020 |

**mAP@0.5:0.95 — .pt (bilgisayar) vs fp32 .tflite (telefon export):**

| imgsz | TEST .pt | TEST .tflite | Δ | VAL .pt | VAL .tflite | Δ |
|-------|----------|--------------|------|---------|-------------|------|
| 320 | 0.145 | 0.132 | −0.013 | 0.163 | 0.146 | −0.016 |
| 512 | 0.363 | 0.361 | −0.001 | 0.362 | 0.356 | −0.005 |
| 640 | 0.420 | 0.408 | −0.012 | 0.446 | 0.445 | −0.001 |
| 800 | 0.460 | 0.460 | ~0     | 0.501 | 0.480 | −0.022 |

**Bulgu:** TFLite (fp32) export doğruluğu **çok az** etkiliyor — her iki metrikte de kayıp genelde
≤0.02, en fazla 320'de (mAP@0.5 −0.03). Export sadık; küçük çözünürlükte kayıp biraz daha belirgin
(yeniden boyutlandırma + op dönüşümü 320'de daha hassas). 40-görüntü gürültüsü nedeniyle val'de bazı
hücreler ufak artı bile veriyor. → Tezde: "fp32 TFLite'a export, hem mAP@0.5 hem mAP@0.5:0.95'i
ihmal edilebilir düzeyde (≤~3 puan) etkiler — model dağıtıma sadık biçimde taşınmıştır."

> Referans için P (Precision) ve R (Recall) tam değerleri de `val_imgsz.py` / `val_tflite_imgsz.py`
> çıktısında mevcut (ödünleşim tablolarında listelendi); gerekirse aynı .pt-vs-.tflite formatına dökülebilir.

**fp16 .tflite — PC CPU'da ÇALIŞMADI (önemli bulgu):**
Sekiz fp16 validation'ın hepsi `CONV_2D ... input_type float16 was not true` ile patladı. Ultralytics'in
`half=True` TFLite export'u **GPU-hedefli** graf üretir (aktivasyonlar float16); plain CPU referans
kernel'i float16 conv girişini desteklemez. Telefonda **GPU delegate** ile çalışır (uygulamada böyle), ama:
- fp16 doğruluğu bu yolla PC CPU'da ölçülemedi. float16 nicemlemesi mAP'yi pratikte ~0 değiştirir →
  beklenti **fp16 ≈ fp32**. (Kesin ölçüm için TFLite GPU-delegate runtime gerekir.)
- **Uygulama riski (Faz 4'ten):** telefon fp16'ya geçti; GPU init başarısız olursa kod CPU XNNPACK'e
  düşüyor. fp16'nın CPU'da çalışacağı garanti değil (referans kernel patlıyor; XNNPACK fp16'yı
  kurtarabilir ama doğrulanmadı). **Öneri:** GPU başarısızsa CPU fallback'inde **fp32 modele düş**
  (kesin CPU-uyumlu). → açık iş (O6).

#### Native-çözünürlük eğitimi vs 800px-downscale (fp32, tez merkez bulgusu — 2026-06-06)

Yukarıdaki tüm tablolar **tek** ağırlık setinden (imgsz=800'de eğitilmiş model, 320/512/640'a
export) gelir. Bu bölüm **tamamlayıcı deneyi** ekler: çözünürlüğünde **bizzat eğitilmiş** modeller
(`runs/cekmece_v4_141epoch_{320,512,640}imgsz`, eşleştirilmiş: aynı v4, 141 epoch, yalnızca eğitim
imgsz'i değişir). Soru: **dağıtım çözünürlüğünde eğitmek mi, 800'de eğitip girişi küçültmek mi daha
doğru?** Her model SADECE kendi boyutunda test edildi (320@320, 512@512, 640@640) → tek değişken =
eğitim çözünürlüğü. Script'ler: `export_native_fp32.py`, `val_native.py`. Native fp32 .tflite'lar
PC CPU'da sorunsuz allocate oldu (float32 I/O — fp16'nın aksine).

**Tablo A — PC (`.pt`): native-N vs 800-eğitilmiş@N**

| split | imgsz | native mAP50 | 800@N mAP50 | Δ mAP50 | native mAP50-95 | 800@N mAP50-95 | Δ | native R | 800@N R |
|-------|------|------|------|------|------|------|------|------|------|
| TEST | 320 | **0.334** | 0.276 | **+0.058** | 0.193 | 0.145 | +0.048 | 0.326 | 0.265 |
| TEST | 512 | 0.526 | 0.582 | −0.056 | 0.318 | 0.363 | −0.045 | 0.479 | 0.532 |
| TEST | 640 | 0.583 | 0.636 | −0.053 | 0.369 | 0.420 | −0.051 | 0.508 | 0.509 |
| VAL  | 320 | **0.466** | 0.317 | **+0.149** | 0.250 | 0.163 | +0.087 | 0.436 | 0.292 |
| VAL  | 512 | 0.604 | 0.579 | +0.025 | 0.358 | 0.362 | −0.004 | 0.557 | 0.514 |
| VAL  | 640 | 0.684 | 0.688 | −0.004 | 0.454 | 0.446 | +0.008 | 0.593 | 0.564 |

**Tablo B — telefon (fp32 `.tflite`): native-N vs 800-eğitilmiş@N**

| split | imgsz | native mAP50 | 800@N mAP50 | Δ mAP50 | native mAP50-95 | 800@N mAP50-95 | Δ |
|-------|------|------|------|------|------|------|------|
| TEST | 320 | **0.337** | 0.246 | **+0.091** | 0.193 | 0.132 | +0.061 |
| TEST | 512 | 0.521 | 0.578 | −0.057 | 0.319 | 0.361 | −0.042 |
| TEST | 640 | 0.590 | 0.613 | −0.023 | 0.363 | 0.408 | −0.045 |
| VAL  | 320 | **0.466** | 0.295 | **+0.171** | 0.244 | 0.146 | +0.098 |
| VAL  | 512 | 0.597 | 0.579 | +0.018 | 0.355 | 0.356 | −0.001 |
| VAL  | 640 | 0.682 | 0.696 | −0.014 | 0.454 | 0.445 | +0.009 |

**Tablo C — export sadakati: native `.pt` → native fp32 `.tflite`** (her ikisi de kendi boyutunda)

| split | imgsz | .pt mAP50 | .tflite mAP50 | Δ | .pt mAP50-95 | .tflite mAP50-95 | Δ |
|-------|------|------|------|------|------|------|------|
| TEST | 320 | 0.334 | 0.337 | +0.003 | 0.193 | 0.193 | ~0 |
| TEST | 512 | 0.526 | 0.521 | −0.005 | 0.318 | 0.319 | +0.001 |
| TEST | 640 | 0.583 | 0.590 | +0.006 | 0.369 | 0.363 | −0.006 |
| VAL  | 320 | 0.466 | 0.466 | ~0 | 0.250 | 0.244 | −0.005 |
| VAL  | 512 | 0.604 | 0.597 | −0.007 | 0.358 | 0.355 | −0.003 |
| VAL  | 640 | 0.684 | 0.682 | −0.002 | 0.454 | 0.454 | ~0 |

**Yorum (tez için — güçlü, sezgiye aykırı bulgu):**
- **"Train high, downscale" yalnızca dağıtım çözünürlüğü eğitime yakınken işe yarar.** En agresif
  downscale'de (**320**) çözünürlüğünde **bizzat eğitmek belirgin biçimde daha iyi**: `.pt` TEST
  **+0.058** mAP50 / +0.048 mAP50-95, VAL **+0.149** / +0.087. Telefon export'unda fark daha da büyük
  (TEST +0.091, VAL +0.171). **Recall** da 320'de native lehine (TEST +0.061, VAL +0.144) — küçük
  nesneleri (anahtar/kalem/tırnak makası) çok daha iyi yakalıyor.
- **512/640'ta iki yaklaşım denk.** TEST'te 800-eğitilmiş ~0.05 önde, VAL'de native eşit/önde —
  yön değişimi 40-görüntü split gürültüsüyle tutarlı. Net, tutarlı sinyal yalnızca 320'de (her iki
  split, her iki metrik, hem `.pt` hem `.tflite` aynı yönde).
- **Mekanizma:** 800-eğitilmiş model özelliklerini ~800px ölçeğinde öğrenir; girdi 320'ye
  küçültülünce öğrenilen ölçekler eşleşmez ve küçük-nesne recall'ı çöker. 320'de eğitilen model
  doğrudan o ölçekte öğrenir. Çözünürlük açığı küçüldükçe (512/640 → 800) bu uyumsuzluk kaybolur.
- **Export sadakati (Tablo C):** native `.pt` → fp32 `.tflite` kaybı **ihmal edilebilir** (|Δ|≤0.007),
  hatta 800-modelin küçültülmüş export'undan **daha sadık** (orada 320'de −0.03 vardı). Sebep: native
  yolda model aynı boyutta eğitilir + export edilir + validate edilir → ekstra yeniden-boyutlandırma
  uyumsuzluğu yok.
- **Pratik çıkarım:** hedef dağıtım **320** ise → **320'de eğit** (tek 800 modelini küçültmek doğruluk
  bırakır). Hedef **512/640** ise → tek 800 modeli yeterli; per-çözünürlük eğitim ek doğruluk vermez,
  bakım/depolama açısından tek model tercih edilir.
- **Metodoloji notu:** setler küçük (40'ar görüntü); mutlak mAP'lerde ve özellikle 512/640
  test-val yön farkında varyans var. 320'deki kazanç büyüklüğü ve tüm hücrelerdeki tutarlılığı bu
  gürültünün üzerinde — bulgu sağlam.

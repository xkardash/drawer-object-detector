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

## Flutter Uygulama Mimarisi

Tek-ekranlı, gerçek-zamanlı çekmece-içi nesne tespit uygulaması. Tasarım hedefi: **düşük-uçlu bir telefonda
(Redmi Note 11 / Snapdragon 680) kamera akışını akıcı tutarken YOLOv8n TFLite çıkarımı koşturmak.** Tüm kod
**saf Dart** (native plugin yok → tek kod tabanı, kolay bakım; native hızlandırma O3 olarak ertelendi).

### Teknoloji yığını
- **Flutter / Dart** (sdk ^3.12), Material UI.
- **tflite_flutter ^0.12** — TFLite çıkarımı + `GpuDelegateV2` + `IsolateInterpreter`.
- **camera ^0.11** — YUV420 kare akışı (`startImageStream`).
- **permission_handler ^11.3** — kamera izni.
- **path_provider ^2.1** — K4 cihaz-gecikme CSV'sini app harici dizine yazmak için.
- `image` paketi **bilinçli kaldırıldı** — önişleme elle yazıldı (aşağıdaki fused preprocessing; 5 transitive
  paket de düştü).

### Kare-başı veri akışı (pipeline)
`CameraImage (YUV420)` → **[1]** fused önişleme (`image_utils`) → önceden ayrılmış `Float32List` (NHWC, [0,1])
→ **[2]** `IsolateInterpreter` (arka-plan isolate, ByteBuffer fast-path) → ham çıktı `[1, 4+nc, anchors]` →
**[3]** parse (`yolo_detector._parseOutput`: flat-index + normalized→pixel + letterbox geri-al + NMS) →
`Detection` listesi (döndürülmüş-görüntü koordinatlarında) → **[4]** `BoxTracker.update` → **[5]** `Ticker`
her vsync'te `predict()` → `ValueNotifier` → **[6]** `RepaintBoundary` + `CustomPaint` overlay.

### İş parçacığı (threading) modeli
- **UI isolate:** kamera kare teslimi, önişleme, parse, render.
- **Arka-plan isolate (`IsolateInterpreter`):** yalnız native çıkarım. Interpreter'ın **native adresi**
  paylaşılır (tensörler native tarafta kalır); kare başına yalnız giriş/çıkış **ByteBuffer**'ları isolate
  sınırını geçer → 30–400 ms'lik çıkarım UI thread'ini bloklamaz.
- **Doğal back-pressure:** `_processing` bayrağı — önceki kare bitmeden yeni kare gelirse atlanır (manuel
  `%5` frame-skip YOK; throughput'a göre kendiliğinden ayarlanır).

### Dosya sorumlulukları
| Dosya | Sorumluluk |
|-------|-----------|
| `lib/main.dart` | Giriş noktası → `HomeScreen` |
| `lib/home_screen.dart` | UI; kamera akışı + kare döngüsü (`_onFrame`); model seçici + **"Oto"**; konfidans slider; perf chip; **REC** kaydı; `Ticker`-tabanlı tracking overlay sürücüsü; yaşam döngüsü |
| `lib/yolo_detector.dart` | `ModelSize` enum + `pickForBudget` (Auto maliyet modeli); `loadModel` (GPU delegate + CPU XNNPACK fallback, float32 I/O doğrulaması, warmup+benchmark, `IsolateInterpreter` sarma); `detect` (önişleme→çıkarım→parse + aşama zamanlaması); `_parseOutput` (flat-index, normalized→pixel); `_nms` |
| `lib/image_utils.dart` | `fillFloat32InputBufferFromCameraImage`: tek-pass YUV420 → 90° döndür → letterbox → BT.601 YUV→RGB (Q16 fixed-point) → [0,1] normalize → NHWC float32; `PreprocessResult` (scale/pad — ters dönüşüm için) |
| `lib/tracker.dart` | `BoxTracker`: predict-then-match (centroid gate, hız EMA, `maxHorizon=160ms`, `maxMisses=2`) — çıkarım hızını ekran hızından ayırır |
| `lib/detection.dart` | `Detection` modeli (bbox/classId/className/confidence) + sınıf renkleri |
| `lib/detection_painter.dart` | `CustomPainter`: bbox'ları görüntü→ekran ölçekler, yuvarlatılmış kutu + etiket çizer |
| `lib/latency_logger.dart` | K4: kare-başı gecikme/termal CSV kaydı (REC butonu) |

### Çekirdek mühendislik özellikleri
1. **Fused tek-pass önişleme** (`image_utils`): `image` paketi by-pass; YUV420 düzlemlerinden **tek döngüde**
   90° döndürme + letterbox (114/255 gri padding) + BT.601 YUV→RGB (tamsayı **Q16** aritmetiği) + [0,1]
   normalize. 4 ayrı tam-image pass yerine 1 → ~%80-90 daha hızlı (Faz 1 darboğazı).
2. **Sıfır-allocation sıcak döngü:** giriş/çıkış `Float32List`'leri load anında bir kez ayrılır, her karede
   üzerine yazılır. TFLite I/O **ByteBuffer fast-path** (`runForMultipleInputs([buf.buffer], {0: out.buffer})`)
   — nested-list yürüyüşü yok. Çıktı **flat-index** ile okunur (`out[c*anchors+a]`), boxed-`Double` yok.
3. **GPU delegate + otomatik CPU fallback:** Android'de `GpuDelegateV2` (Adreno OpenCL,
   `isPrecisionLossAllowed:true` → fp16 hesap, `SUSTAINED_SPEED`/`MIN_LATENCY`); init başarısızsa **CPU
   XNNPACK + 4 thread**'e otomatik düşer.
4. **Off-thread çıkarım** (`IsolateInterpreter`): yukarıdaki threading modeli — UI akıcı kalır.
5. **Overlay izolasyonu:** kare-başı state (`detections`/`imageSize`/`fps`) `ValueNotifier` arkasında;
   `ValueListenableBuilder` + `RepaintBoundary` ile **yalnız** bbox overlay + FPS chip repaint olur —
   `CameraPreview` ve chrome her karede rebuild edilmez (Faz 2).
6. **BoxTracker** (`tracker.dart`): ~6 FPS çıkarım ile ~60 FPS ekran arasını köprüler; her vsync'te track'ler
   tahmini hızla kaydırılır → telefon gezdirilirken bbox **akıcı** (zıplamaz), büyük modelin doğruluğu korunur.
7. **Adaptif "Oto" model seçimi:** 320 probe → `cost∝size²` → ~70 ms (≈12 FPS) bütçesi → en büyük boyut +
   tek-seferlik FPS düzeltmesi (cihaza-uyarlı; tam kriter: Faz 4 II).
8. **Ölçüm altyapısı (tez için):** load-anı saf-inference benchmark (`pureInferenceMs`); `detect`'te per-stage
   Stopwatch + EMA (pre/inf/parse); ekranda perf chip; **K4 REC** → CSV (p50/p95/p99 + termal).

### Sağlamlaştırma (kodda gömülü bug dersleri)
- **B1 — sessiz hata yok:** float32 I/O doğrulaması (`StateError` — model tipi değişirse buffer sessizce
  bozulmasın); detection hataları `debugPrint(e, st)` ile yüzeye çıkar.
- **B2 — koordinat:** YOLOv8 export bbox'u **normalized [0..1]** verir → parse'ta `×inputSize` (yoksa bbox
  sol-üst köşeye çöker).
- **B3 — sayı formatı:** INT8 (DFL → cx/cy=0) ve fp16 (cihazda yüklenmiyor) **ölü** → **yalnız fp32 native**
  dağıtılır (320/512/640/800 + ablation 320s).

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

### I. fp16 model swap — ❌ GERİ ALINDI (negatif sonuç, tez için geçerli bulgu)

> **GÜNCELLEME (2026-06-06): fp16 dağıtımı BAŞARISIZ → native fp32'ye dönüldü.** fp16 TFLite
> **iki platformda da yüklenemedi**: (1) PC CPU referans kernel'i float16 conv girişini desteklemiyor
> (`CONV_2D ... input_type float16 was not true` — bkz. aşağı, "fp16 PC CPU'da ÇALIŞMADI"); (2)
> **telefonda da model yüklenmedi** (interpreter init/allocate başarısız, uygulama takıldı — kullanıcı
> teyidi). Sonuç: **dağıtılan model = native fp32** (`pubspec.yaml` + `yolo_detector.dart` yalnız
> `cekmece_v4_native{320,512,640,800}_fp32.tflite` referanslıyor). Aşağıdaki bu bölümdeki tüm fp16 A/B
> adımları **geçersiz**. **Tez çıkarımı:** Ultralytics `half=True` TFLite export'u GPU-hedefli graf
> üretir; `tflite_flutter` GPU delegate ile bu cihazda güvenilir yüklenmedi → **fp32 native tek
> dağıtılabilir yol**. (Bellek/yükleme avantajı ne olursa olsun çalışmayan model işe yaramaz.)

**[TARİHSEL — geri alınan deneme]** `export_fp16.py` (kök) ile `best.pt`'den 320/512/640/800 fp16 TFLite üretildi
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

**"Oto" pikseli NEYE GÖRE seçer — adım adım (kod: `ModelSize.pickForBudget` + `_maybeAutoCorrect`):**
1. **Probe:** açılışta en küçük modeli (**320**) yükler, yükleme anında **saf inference süresini** ölçer
   (`pureInferenceMs`, ana-isolate, kopya hariç) → bu, *o cihazın* gerçek hızı.
2. **Ekstrapolasyon:** doğrulanmış **cost∝boyut²** yasasıyla her boyut N için süreyi tahmin eder:
   `tahmin(N) = pureInferenceMs × (N² / 320²)`.
3. **Seçim:** tahmini süre **bütçeye** (`_autoBudgetMs = 70 ms`, ≈12 FPS hedefi) sığan **en büyük** boyutu
   seçer. (Yalnız Auto-uygun native boyutlar 320/512/640/800; YOLOv8s "320s" `autoSelectable:false` → hariç.)
4. **Tek-seferlik düzeltme:** ~40 kare sonra **ölçülen** FPS EMA'sına bakar — **<9 FPS** ise bir kademe
   küçültür, **>18 FPS** ise bir kademe büyütür, sonra **kilitler** (oscillation yok; reload GPU kernel'ini
   ~1–3 s yeniden derler).
5. **Sonuç = cihaza-uyarlı:** yavaş cihaz → küçük boyut, hızlı cihaz → büyük boyut. Ekstrapolasyon
   muhafazakâr (sabit GPU overhead size² değil) → akıcılığa yaslar; tek düzeltme headroom'u geri alır.

**Redmi Note 11 örneği (somut):** 320 probe'ta saf inference ≈ **61 ms**. tahmin(512)=61×(512/320)²≈**156 ms**,
tahmin(640)≈**244 ms** — ikisi de 70 ms bütçesini aşar; yalnız 320 (61 ms) sığar → **Oto = 320**. Ölçülen
FPS ≈12, [9,18] bandında → düzeltme yok, 320'de kalır. (Daha güçlü bir telefonda, örn. probe ≈20 ms olsaydı:
tahmin(512)≈51 ms sığar, tahmin(640)≈80 ms aşar → **Oto = 512** seçerdi.)

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
  **✅ ÇÖZÜLDÜ (K4, cihaz-üstü dağılım ölçümü):** doğrudan per-frame total = **404 ms** (p50, 456 kare),
  FPS = **2.4** → gerçek değer ~400 ms doğrulandı (185 ms artefakt). Boyut² eğrisi 800'de de geçerli —
  "kaynak tavanı / kopuş" yok (bkz. K4 bölümü).
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
- **Uygulama riski (Faz 4) — ✅ ÇÖZÜLDÜ (geri dönüş):** fp16 telefonda yüklenemedi (yalnız PC CPU değil,
  cihazda da init/allocate başarısız — kullanıcı teyidi). "GPU başarısızsa fp32'ye düş" önerisi yerine
  **tüm dağıtım native fp32'ye döndürüldü** (kesin platform-uyumlu, hem GPU delegate hem CPU XNNPACK
  fallback'te çalışır). O6 kapandı: fp16 yolu tamamen terk edildi. Detay: Faz 4 bölümü başındaki güncelleme.

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
  test-val yön farkında varyans var. 320'deki kazanç tüm hücrelerde tutarlı yönde — ama büyüklüğünün
  gürültünün üzerinde olup olmadığı aşağıda **bootstrap CI** (görüntü-örnekleme) ve **E2 çok-tohumlu eğitim**
  (eğitim-rastgeleliği) ile nicelleştirildi. **Net sonuç:** 320 native üstünlüğü **seed-robust** (E2: 3/3 seed
  ayrık, Δ=+0.083); görüntü-CI'de "sınırda" görünmesi yalnız küçük test setinin örnekleme gürültüsüdür. →
  "net kazanıyor" iddiası **seed bazında doğru**, ama farkın kesin büyüklüğü 40-görüntüde belirsiz.

#### Bootstrap güven aralıkları — 40-görüntü belirsizliğini nicelleştirme (tez sağlamlık — 2026-06-06)

Yukarıdaki tüm doğruluk kıyaslarında "40-görüntü gürültüsü" uyarısı **elle** veriliyordu; bu bölüm onu
**ölçer**. Script: `bootstrap_map_ci.py`. **Yöntem:** per-image `(tp, conf, pred_cls, target_cls)`
doğrudan Ultralytics `DetectionValidator`'dan yakalanır (AP motoru `model.val()` ile birebir — 40
görüntünün tamamı havuzlandığında `val.map50`'i **aynen** verir, doğrulandı), sonra 40 test görüntüsü
**B=2000× yeniden örneklenir** (image-level bootstrap, mAP CI için standart yöntem), her örnekte mAP
yeniden hesaplanır → ortalama + %95 yüzdelik CI. **Eşleştirilmiş (paired)** kıyaslarda native-N ve
800@N için **aynı** yeniden-örneklenen görüntü kümeleri kullanılır → Δ paired istatistik. Protokol:
kanonik val (`rect=True`), `batch=1` (tekrarlanabilirlik). Nokta tahminleri batched-val tablolarıyla
**≤0.011** örtüşür (kalan fark = batch gruplama, CI'dan çok küçük).

**Per-config gözlenen mAP + %95 CI (TEST, 40 görüntü, B=2000):**

| config | mAP@0.5 [%95 CI] | mAP@0.5:0.95 [%95 CI] |
|--------|------------------|-----------------------|
| 800@320 | 0.285 [0.213, 0.367] | 0.152 [0.106, 0.212] |
| 800@512 | 0.580 [0.495, 0.654] | 0.361 [0.304, 0.418] |
| 800@640 | 0.635 [0.539, 0.716] | 0.416 [0.353, 0.482] |
| 800@800 | 0.689 [0.609, 0.753] | 0.459 [0.395, 0.521] |
| native320 | 0.334 [0.258, 0.424] | 0.192 [0.141, 0.259] |
| native512 | 0.528 [0.446, 0.599] | 0.320 [0.266, 0.376] |
| native640 | 0.594 [0.502, 0.684] | 0.373 [0.306, 0.452] |

**Eşleştirilmiş bootstrap — Δ = native-N − 800@N (aynı yeniden-örneklenen görüntüler):** Δ'nın %95 CI'sı
0'ı dışlıyorsa fark istatistiksel olarak gerçek (split gürültüsü değil).

| kıyas | metrik | Δ gözlenen | %95 CI (Δ) | anlamlı? |
|-------|--------|-----------|------------|----------|
| native320 vs 800@320 | mAP@0.5 | **+0.049** | [−0.002, +0.100] | **hayır (sınırda)** |
| native320 vs 800@320 | mAP@0.5:0.95 | **+0.039** | [+0.013, +0.065] | **EVET (native önde)** |
| native512 vs 800@512 | mAP@0.5 | −0.053 | [−0.097, −0.012] | EVET (800 önde) |
| native512 vs 800@512 | mAP@0.5:0.95 | −0.040 | [−0.075, −0.012] | EVET (800 önde) |
| native640 vs 800@640 | mAP@0.5 | −0.041 | [−0.075, −0.001] | EVET (800 önde) |
| native640 vs 800@640 | mAP@0.5:0.95 | −0.043 | [−0.071, −0.012] | EVET (800 önde) |

**Yorum (tez için — bulguları sağlamlaştırır ve İNCELTİR):**
- **mAP@0.5 CI yarı-genişliği ±0.07–0.09** (320:±0.077 · 512:±0.080 · 640:±0.088 · 800:±0.072; native
  benzer). → Tek bir 40-görüntü mAP@0.5 değerinin doğal örnekleme belirsizliği **~±0.08**. "40-görüntü
  gürültüsü" uyarısı **nicel kanıtlandı**: |Δ|<0.05 olan tek-split farkları tek başına yorumlanamaz.
- **native-320 avantajı incelendi:** **lokalizasyonda anlamlı** (mAP@0.5:0.95 Δ=+0.039, CI
  [+0.013,+0.065] → 0'ı dışlar) ama **mAP@0.5'te sınırda** (Δ=+0.049, CI [−0.002,+0.100] → 0'a değiyor,
  p≈0.05–0.06). VAL split'i de yönü native lehine doğruluyor. → Doğru ifade: "320'de native eğitim
  **ölçülü ama gerçek** bir kazanç sağlar; etki özellikle **lokalizasyon** (sıkı IoU) ve küçük-nesne
  recall'ında — kaba mAP@0.5'te etki sınırda." ("Net kazanıyor" yerine bu.)
- **512/640 'denk' iddiası düzeltildi:** TEST içinde 800-trained **her iki metrikte de anlamlı** önde
  (Δ CI 0'ı dışlıyor). Dokümanın "denk" sonucu **yalnız** TEST↔VAL yön farkından geliyor (VAL'de native
  eşit/önde) — yani split-içi CI çakışmasından değil, **split seçimi belirsizliğinden**. Daha doğru:
  "512/640'ta üstünlük **split-bağımlı**: tek split içinde fark anlamlı, fakat yön iki 40-görüntü split
  arasında değişiyor → kesin sıralama için daha çok veri / çok-tohum gerekir."
- **Metodolojik kazanım (tez Yöntem/Tartışma için):** image-level bootstrap **split-içi** örnekleme
  belirsizliğini verir; TEST↔VAL yön farkı ise **AYRI** bir belirsizlik kaynağıdır (hangi 40 görüntü).
  Küçük-veri rejiminde tek bir mAP farkına güvenmek yerine **CI + çoklu-split + çok-tohumlu eğitim**
  birlikte gerekir (çok-tohum aşağıda E2'de yapıldı). Çıktı: `tez_icin_veriler/bootstrap_ci.log`.

#### E2 — Çok-tohumlu eğitim: native-vs-downscale seed-robustluğu (K3 kapatma, tez merkez bulgu — 2026-06-06)

Bootstrap **görüntü-örnekleme** belirsizliğini ölçtü ama her config tek eğitim run'ına (seed=42) dayanıyordu →
**eğitim-rastgeleliği** ölçülmemişti (K3). Bu bölüm onu kapatır: native {320,512,640} ve 800-model **3 seed**
(42/43/44) ile eğitildi (`train_multiseed.py`, config `args.yaml` ile birebir, tek değişken seed), TEST'te
deterministik val edildi (`eval_multiseed.py`). seed=42 sayıları yukarıdaki tablolarla **birebir** (doğrulama).
Çıktı: `tez_icin_veriler/multiseed_{train,eval}.log`.

**TEST mAP, 3 seed üzerinden ortalama ± std:**

| config | mAP@0.5 (ort ± std) | mAP@0.5:0.95 (ort ± std) |
|--------|---------------------|--------------------------|
| 800@320 | 0.271 ± 0.012 | 0.139 ± 0.012 |
| 800@512 | 0.566 ± 0.018 | 0.347 ± 0.017 |
| 800@640 | 0.638 ± 0.005 | 0.413 ± 0.003 |
| 800@800 | 0.690 ± 0.002 | 0.454 ± 0.005 |
| native320 | **0.354 ± 0.019** | **0.197 ± 0.005** |
| native512 | 0.533 ± 0.013 | 0.330 ± 0.010 |
| native640 | 0.608 ± 0.012 | 0.385 ± 0.012 |

**Verdict — native-N vs 800@N (fark seed-rastgeleliğine dayanıklı mı?):** "ayrık" = native'in 3 seed'i
800'ün 3 seed'ini hiç çakışmadan geçiyor (en güçlü sonuç).

| kıyas | metrik | Δort | sonuç |
|-------|--------|------|-------|
| native320 vs 800@320 | mAP@0.5 | **+0.083** | **ROBUST — native (aralıklar ayrık)** |
| native320 vs 800@320 | mAP@0.5:0.95 | **+0.058** | **ROBUST — native (ayrık)** |
| native512 vs 800@512 | mAP@0.5 | −0.033 | muhtemel — 800 (|Δ|>birleşik std) |
| native512 vs 800@512 | mAP@0.5:0.95 | −0.017 | seed gürültüsü içinde (denk) |
| native640 vs 800@640 | mAP@0.5 | −0.030 | ROBUST — 800 (ayrık) |
| native640 vs 800@640 | mAP@0.5:0.95 | −0.029 | ROBUST — 800 (ayrık) |

**Yorum (tez merkez bulgu):**
- **native-320 üstünlüğü SEED-ROBUST — headline kurtuldu.** 3 native seed'in HEPSİ (mAP@0.5:
  0.334/0.355/0.372) 3 800-model seed'in HEPSİNİ (0.285/0.267/0.261) geçer → aralıklar **ayrık**,
  Δort=+0.083 ≫ birleşik std 0.023; lokalizasyonda da ayrık (+0.058). Tek-run **fluke değil**, etki
  tekrarlanabilir.
- **İki belirsizlik kaynağını ayır (kilit metodoloji):** bootstrap mAP@0.5'te native-320'yi "sınırda"
  bulmuştu (görüntü-CI [−0.002,+0.100]); E2 "robust" diyor. **Çelişki değil — farklı sorular.**
  *Görüntü-CI:* "başka 40 görüntü olsa fark kaybolur mu?" → küçük test seti yüzünden geniş.
  *Seed:* "başka eğitim olsa kaybolur mu?" → hayır, 3/3 aynı yönde, ayrık. **Sentez:** etki **gerçek ve
  eğitim-rastgeleliğine dayanıklı**; yalnız 40-görüntülük test seti farkın kesin **büyüklüğünü** belirsiz
  bırakır. (Tezde tam da böyle ifade edilmeli.)
- **Ters yön de seed-robust (512/640):** native640 vs 800@640 → **800 ROBUST önde** (her iki metrik ayrık);
  native512 → 800 mAP@0.5'te muhtemel önde, lokalizasyonda denk. → "800-downscale 512/640'ta üstün" tarafı
  da doğrulandı (özellikle 640 net).
- **Nihai, monotonik, seed-doğrulanmış tablo:** dağıtım çözünürlüğü eğitim çözünürlüğüne (800) yaklaştıkça
  "800'de eğit + küçült" iyileşir ve native'i geçer; en agresif küçültmede (320) native eğitim **belirgin**
  kazanır. **Geçiş ~512 civarında.** Seed std'leri küçük (0.002–0.019) → eğitim varyansı görüntü-örnekleme
  CI'sinin (±0.08) çok altında. **K3 kapandı:** bulgu artık 3 bağımsız eğitimle desteklenmiştir.

#### Sınıf-bazlı mAP — çözünürlük × nesne-boyutu etkileşimi (TEST, .pt, 2026-06-06)

Yukarıdaki aggregate mAP, sınıf-bazında kırıldığında *neden* öyle davrandığını gösteriyor. Held-out
TEST, `.pt` (fp32 .tflite sadakati zaten ≤0.007 → temsilî). Script: `val_perclass_plots.py`.
Her config için confusion matrix + PR/F1/P/R eğrileri: `tez_icin_veriler/thesis_plots/<config>/` (tez figürleri).

**Sınıf-bazlı mAP@0.5:**

| config | anahtar | çakmak | kalem | makas | tırnak makası |
|--------|------|------|------|------|------|
| 800-trained @320 | 0.211 | 0.198 | 0.316 | 0.514 | 0.141 |
| 800-trained @512 | 0.691 | 0.460 | 0.571 | 0.776 | 0.410 |
| 800-trained @640 | 0.727 | 0.539 | 0.624 | 0.822 | 0.466 |
| 800-trained @800 | 0.764 | 0.628 | 0.638 | 0.878 | 0.539 |
| native 320 | 0.244 | 0.304 | 0.405 | 0.604 | 0.113 |
| native 512 | 0.594 | 0.441 | 0.549 | 0.774 | 0.271 |
| native 640 | 0.731 | 0.506 | 0.625 | 0.725 | 0.328 |

**Sınıf-bazlı mAP@0.5:0.95:**

| config | anahtar | çakmak | kalem | makas | tırnak makası |
|--------|------|------|------|------|------|
| 800-trained @320 | 0.073 | 0.097 | 0.140 | 0.324 | 0.091 |
| 800-trained @512 | 0.349 | 0.302 | 0.330 | 0.578 | 0.253 |
| 800-trained @640 | 0.407 | 0.376 | 0.389 | 0.603 | 0.324 |
| 800-trained @800 | 0.431 | 0.432 | 0.429 | 0.649 | 0.361 |
| native 320 | 0.076 | 0.182 | 0.237 | 0.397 | 0.071 |
| native 512 | 0.276 | 0.276 | 0.327 | 0.526 | 0.184 |
| native 640 | 0.391 | 0.340 | 0.401 | 0.503 | 0.210 |

**Yorum (tez için — sınıf düzeyinde mekanizma):**
- **Çözünürlük hassasiyeti nesne boyutuyla ters orantılı.** 800-trained 320→800 giriş, mAP@0.5
  artışı: **tırnak makası ×3.8** (0.141→0.539), anahtar ×3.6, çakmak ×3.2, kalem ×2.0,
  **makas ×1.7** (0.514→0.878). Makas en büyük/belirgin nesne → 320'de bile 0.51, çözünürlükten en
  az etkilenen. Tırnak makası en küçük/ince → en çok etkilenen. **"Küçük nesne × düşük giriş
  çözünürlüğü" problemi sınıf bazında nicel kanıtlandı.**
- **320'de native eğitim 5 sınıfın 4'ünde kazanıyor** (mAP@0.5, native−800): çakmak **+0.106**,
  makas +0.090, kalem +0.089, anahtar +0.033. Tek istisna **tırnak makası −0.028** — ama ikisi de
  taban civarı (~0.11–0.14): en küçük nesne 320×320'de eğitim çözünürlüğünden bağımsız olarak
  zaten kayıp; native eğitim onu kurtaramıyor. Lokalizasyonda da (mAP@0.5:0.95) aynı: çakmak +0.085,
  kalem +0.097, makas +0.073.
- **512/640'ta 800-trained'in üstünlüğü en zor (en küçük) sınıflarda yoğunlaşıyor.** Tırnak makası:
  512'de native 0.271 vs 800 **0.410** (−0.139), 640'ta 0.328 vs **0.466** (−0.138). Anahtar 512'de
  −0.097. Kolay/büyük sınıflar (makas, kalem) ise denk. Mekanizma: yeterli giriş çözünürlüğü olunca,
  800'de eğitilmiş modelin **eğitim sırasında ince detay görmüşlüğü** en küçük nesnelerde fark yaratır;
  oysa yalnız 512/640'ta eğitilen model o detayı hiç görmemiştir.
- **Tırnak makası = darboğaz sınıf.** Native 800'de bile yalnız 0.539 (mAP@0.5) / 0.361 (mAP@0.5:0.95).
  Pratik tez çıkarımı: bu sınıf için çözüm ek çözünürlük *değil* — daha fazla/çeşitli örnek, daha büyük
  model, veya yakın-çekim kullanım kılavuzu. Diğer dört sınıf 512–640'ta zaten kullanılabilir.
- **Sentez:** aggregate "320'de native kazanır, 512/640 denk" doğru ama eksik; sınıf-bazında **native'in
  320 avantajı orta-boy sınıfları toparlamaktan gelir, en küçük nesneyi kurtaramaz**, ve **800-trained'in
  512/640 avantajı tam da en küçük nesnededir**. Çözünürlük seçimi sınıf-bağımlı bir karar. **(UYARI:
  aşağıdaki per-class bootstrap CI bu sınıf-bazlı iddiaları nicelleştirip İNCELTİR — özellikle "×3.8"
  gibi oranlar ve "native 4/5 sınıfta kazanır" deseni büyük ölçüde sınıf-içi gürültü çıkıyor.)**

##### Per-class bootstrap CI — sınıf-bazlı iddiaların belirsizliği (tez sağlamlık — 2026-06-06)

Yukarıdaki per-class tablosu **çok küçük sınıf örneklerinde** ince iddialar üretiyor (tırnak makası TEST'te
yalnız **27 instance**). Bu bölüm onlara %95 CI takar. Script: `bootstrap_perclass_ci.py` (E1'in validator-
capture motorunu yeniden kullanır → AP `model.val()` ile birebir). 40 test görüntüsü B=2000× yeniden
örneklenir, **sınıf-bazlı AP@0.5** bootstrap'lenir; iki başlık iddiası için **eşleştirilmiş** Δ.

**Per-class AP@0.5 gözlenen [%95 CI] (TEST) — CI GENİŞLİĞİNE dikkat:**

| config | anahtar | çakmak | kalem | makas | tırnak makası |
|--------|---------|--------|-------|-------|---------------|
| 800@320 | 0.21 [0.07, 0.31] | 0.21 [0.11, 0.34] | 0.32 [0.21, 0.46] | 0.52 [0.36, 0.67] | 0.16 [0.04, 0.34] |
| 800@800 | 0.76 [0.47, 0.93] | 0.63 [0.48, 0.75] | 0.64 [0.52, 0.76] | 0.88 [0.79, 0.94] | 0.54 [0.34, 0.72] |
| native320 | 0.24 [0.10, 0.42] | 0.30 [0.19, 0.43] | 0.40 [0.26, 0.56] | 0.60 [0.43, 0.76] | 0.12 [0.04, 0.29] |

**İddia A — "native-320 4/5 sınıfta kazanır" (paired Δ = native320 − 800@320):**

| sınıf | Δ | %95 CI | anlamlı? |
|-------|---|--------|----------|
| anahtar | +0.034 | [−0.078, +0.190] | hayır |
| çakmak | +0.091 | [−0.002, +0.185] | hayır (sınırda) |
| **kalem** | **+0.080** | **[+0.009, +0.139]** | **EVET** |
| makas | +0.085 | [−0.025, +0.194] | hayır |
| tırnak makası | −0.045 | [−0.156, +0.072] | hayır |

**İddia B — çözünürlük etkisi (paired Δ = 800@800 − 800@320):** 5 sınıfın **hepsi** anlamlı pozitif
(anahtar +0.555 [+0.363,+0.710], çakmak +0.417 [+0.292,+0.524], kalem +0.315 [+0.242,+0.385],
makas +0.360 [+0.228,+0.501], tırnak makası +0.377 [+0.215,+0.516]).

**Yorum (tez için — per-class iddiaları İNCELTİR):**
- **İddia A büyük ölçüde gürültü:** "native 4/5 sınıfta kazanır" desenindeki 5 farktan **yalnız kalem
  istatistiksel anlamlı** (+0.080, CI 0'ı dışlar). Diğer dört sınıfın CI'sı 0'ı kapsıyor → per-class
  örnek (≤60 instance/sınıf) bu incelikteki bir kıyas için **çok küçük**. Doğru ifade: "320'de native
  eğitim **kalem**'de anlamlı, diğer sınıflarda eğilim native lehine ama sınıf-içi gürültünün içinde."
- **İddia B'nin YÖNÜ kaya gibi sağlam, MAGNİTÜDÜ değil:** yüksek çözünürlük **her sınıfa** anlamlı yarar
  sağlıyor (5/5 pozitif). AMA "tırnak makası ×3.8" gibi **oran** ifadeleri yanıltıcı: tırnak @320 = 0.16
  [0.04, 0.34], @800 = 0.54 [0.34, 0.72] → gözlenen oran 3.34× ama CI uçlarıyla oran **1.0×–16.9×**
  arasında. Payda (@320) o kadar belirsiz ki oran **anlamsız** bir kesinlikte. **Mutlak Δ** ile sıralama
  da değişiyor: en büyük kazanç **anahtar (+0.555)**, tırnak makası değil. → "çözünürlük hassasiyeti
  boyutla ters orantılı, tırnak en hassas" ifadesi **küçük-taban oran artefaktı**; sağlam olan tek şey
  "yüksek çözünürlük tüm sınıflara anlamlı yarar sağlar."
- **Metodolojik ders (tez Tartışma/Kısıtlar için):** aggregate mAP'de güvenilir bulgular, sınıf-bazına
  inilince (≤60, hatta 27 instance) **çoğunlukla anlamlılığını yitirir**; per-class kıyaslarda **oran
  (×) değil, anlamlı mutlak Δ** raporlanmalı. Çıktı: `tez_icin_veriler/bootstrap_perclass_ci.log`.

#### Dataset istatistikleri, model verimliliği, eğitim yakınsaması (tez destek — 2026-06-06)

Script: `thesis_stats.py` (read-only). Üç tablo: (1) dataset'in per-class bulguyu *neden* öyle
yaptığını açıklayan istatistik, (2) mobil verimlilik, (3) eğitimin sağlıklı yakınsadığının kanıtı.

**1) Dataset (5 sınıf, train 400 / val 40 / test 40 görüntü):**

| sınıf | train adet | %inst | mean w% | mean h% | mean alan% | test adet |
|------|------|------|------|------|------|------|
| anahtar | 533 | 21.3 | 5.35 | 9.85 | **0.557** | 46 |
| çakmak | 425 | 17.0 | 8.01 | 16.26 | 1.339 | 51 |
| kalem | **874** | 34.9 | 10.66 | 25.59 | 2.504 | 175 |
| makas | 396 | 15.8 | 16.34 | 31.22 | **5.251** | 60 |
| tırnak makası | **274** | 11.0 | 7.52 | 13.04 | 1.024 | **27** |

train: 2502 instance, 6.25 nesne/görüntü (val/test ~8.9 — test sahneleri daha kalabalık).

**Bulgu (per-class sonucu veriye bağlar):**
- **Darboğaz açıklandı:** tırnak makası hem **en az örneğe** (274 train, sadece 27 test) hem küçük
  boyuta (alan %1.02) sahip → "az veri × küçük nesne" çifte dezavantajı. Native 800'de bile 0.539'da
  takılmasının nedeni bu (ek çözünürlük değil, ek/çeşitli veri gerekir).
- **bbox boyutu ↔ çözünürlük hassasiyeti güçlü ters korelasyon:** en küçük 3 sınıf (anahtar 0.56%,
  tırnak makası 1.02%, çakmak 1.34%) 320→800'de **×3.2–3.8** kazanır; en büyük 2 (kalem 2.50%,
  makas 5.25%) yalnız **×1.7–2.0**. Makas (en büyük) 320'de bile 0.51 → büyük nesne downscale'e dayanır.
  "Küçük nesne × düşük çözünürlük" problemi **5 sınıf üzerinde nicel doğrulandı.**
- Orta düzey sınıf dengesizliği (kalem %34.9 vs tırnak makası %11.0).

**2) Model verimliliği (native modeller; mimari = YOLOv8n, 3.01M param sabit):**

| imgsz | params | GFLOPs | .tflite MB | cihaz FPS | test mAP@0.5 | test mAP@0.5:0.95 |
|------|------|------|------|------|------|------|
| 320 | 3.01M | 2.05 | 11.6 | 12 | 0.334 | 0.193 |
| 512 | 3.01M | 5.25 | 11.7 | 6 | 0.526 | 0.318 |
| 640 | 3.01M | 8.20 | 11.8 | 4 | 0.583 | 0.369 |
| 800 | 3.01M | 12.81 | 12.5 | 1–2 | 0.690 | 0.460 |

**Bulgu:** GFLOPs **tam kuadratik** — GFLOPs/piksel² = sabit **2.0×10⁻⁵** (2.05/320² = 5.25/512² =
8.20/640² = 12.81/800²). Bu, cihaz per-stage timing'inden ampirik türetilen `pure ∝ boyut²` yasasının
(yukarıda `pure/boyut² ≈ 5.5×10⁻⁴ ms/px²`) **teorik karşılığıdır** → maliyet modeli iki bağımsız yoldan
doğrulandı (FLOP sayımı + cihaz ölçümü). FPS ≈ 1/GFLOPs eğilimi (512/640'ta ~32 GFLOP·FPS sabiti; 320'de
sabit kernel overhead, 800'de düşük-FPS ölçüm artefaktı bant dışı). Param sabit → boyutlar arası fark
yalnızca giriş çözünürlüğünden (ağırlık seti aynı mimari).

#### Model kapasitesi ablation: YOLOv8n vs YOLOv8s — "neden en küçük model?" (2026-06-06)

Tez baştan sona YOLOv8n (en küçük varyant) kullanır. Bu seçimi gerekçelendirmek için YOLOv8s, **aynı
konfigürasyonla** (native, seed 42, 141 epoch) 320 ve 640'ta eğitilip TEST'te kıyaslandı. Script'ler:
`train_yolov8s.py`, `eval_yolov8s.py`. Cihaz FPS tahmini doğrulanmış maliyet modeliyle (gecikme ∝ GFLOPs)
yapıldı (anchor: YOLOv8n native cihaz FPS — 320:12, 640:4).

| model | params | GFLOPs | TEST mAP@0.5 | TEST mAP@0.5:0.95 | cihaz FPS (ölçülen/tahmin) |
|-------|--------|--------|--------------|-------------------|----------------------------|
| YOLOv8n @320 | 3.0M | 2.0 | 0.334 | 0.192 | 12 (ölçülen) |
| YOLOv8n @640 | 3.0M | 8.1 | 0.594 | 0.373 | 4 (ölçülen) |
| YOLOv8s @320 | 11.1M | 7.1 | 0.485 | 0.281 | ~3.4 (tahmin) |
| YOLOv8s @640 | 11.1M | 28.4 | 0.719 | 0.478 | ~1.1 (tahmin) |

**Yorum (tez için — model seçimi gerekçesi):**
- **YOLOv8s belirgin daha doğru, ama gerçek-zaman dışı.** Aynı boyutta s, n'i mAP@0.5'te **+0.124–0.152**
  geçiyor (yok sayılamaz). Bedeli: **3.5× GFLOPs** (11.1M vs 3.0M param) → cihazda ~1/3 FPS (s@320 ≈ 3.4,
  s@640 ≈ 1.1 FPS). Gerçek-zaman hedefi (≥~10 FPS) için s **uygun değil**.
- **Sabit compute bütçesinde çözünürlük > kapasite (güçlü, sezgiye-aykırı bulgu).** ~Eşit FLOP'ta
  (n@640 = 8.1 vs s@320 = 7.1 GFLOPs) **n@640 (0.594) > s@320 (0.485)** — **+0.109 mAP@0.5**, üstelik n@640
  daha hızlı (4 vs ~3.4 FPS). Yani bu görevde verilen bir FLOP bütçesini **giriş çözünürlüğüne** harcamak,
  **model kapasitesine** harcamaktan daha verimli. Bu, dokümanın çözünürlük-merkezli bulgularını pekiştirir:
  asıl kaldıraç giriş boyutu.
- **Gerekçe (nihai):** YOLOv8n "yeterince iyi" olduğu için değil, **gerçek-zaman mobil hedef için FLOP-optimal
  aile** olduğu için seçildi; doğruluk gerektiğinde **kapasiteyi büyütmek yerine çözünürlüğü** büyütmek daha
  verimli. (s yalnız hız hedefi gevşerse, örn. tek-kare/çevrimdışı analizde, mantıklı.) Çıktı:
  `tez_icin_veriler/yolov8s_eval.log`.
- **s@320 — .pt vs fp32 tflite + test/val (export sadakati model boyutundan bağımsız):** YOLOv8s@320 da
  fp32 TFLite'a export edildi (44.6 MB) ve doğrulandı. **Export kaybı ihmal edilebilir** (n ile aynı):
  TEST mAP@0.5 .pt 0.481 → tflite 0.481 (Δ≈0), VAL 0.526 → 0.526 (Δ=0); mAP@0.5:0.95'te en fazla −0.006.
  s−n farkı her iki split'te de native lehine s: **TEST +0.147, VAL +0.060** mAP@0.5. → fp32 TFLite export
  sadakati **YOLOv8s'te de** geçerli (Tablo C bulgusu kapasiteye genellenir). Bu s@320 fp32 tflite app'e
  **"320s"** olarak eklendi (cihaz FPS ölçümü için). Çıktı: `tez_icin_veriler/s320_vs_n320.log`.
- **s@320 CİHAZ FPS — ölçüldü (Redmi Note 11):** 81 kare / 15 s → per-frame gecikme **p50=180 ms**
  (p95=188, p99=194), **FPS p50=5.3** (en kötü 4.8). Karşılaştırma: n@320 = 75 ms / 12 FPS. **Tahmin vs
  gerçek (cost model nüansı):** naif tahmin (toplam∝GFLOPs) s@320 ≈ 3.4 FPS demişti; **ölçülen 5.3 FPS**
  daha hızlı — çünkü kare süresi = **sabit yük** (kamera/önişleme/GPU-dispatch/kopya) + inference; yalnız
  inference 3.5× ölçeklenir → toplam 180/75 = **2.4×** (3.5× değil). → cost∝boyut²/GFLOPs **saf inference**
  için geçerli; **toplam FPS**'te sabit overhead olduğundan kapasite artışı FPS'i tahminden az düşürür
  (dürüst rafine). **"Çözünürlük > kapasite" eşit-FPS'te yine geçerli:** n@512 (6 FPS, mAP 0.528) **hem daha
  hızlı hem daha doğru** s@320'den (5.3 FPS, 0.481). **Kararlılık:** 44 MB'lık s modeli düşük-uçlu Adreno
  610'da GPU delegate yüklemede zaman zaman **native çöktü** (uygulama kapandı) → pratik güvenilirlik de
  YOLOv8n lehine. Çıktı: `tez_icin_veriler/latency_2026-06-06T21-27-24.csv`.

**3) Eğitim yakınsaması (her run 141 epoch, `results.csv`):**

| run | best epoch | best mAP50-95 (val) | final mAP50 | final mAP50-95 | final val loss box/cls/dfl |
|------|------|------|------|------|------|
| 320 | 127 | 0.2514 | 0.4582 | 0.2435 | 1.702 / 1.417 / 1.165 |
| 512 | 81 | 0.3574 | 0.5911 | 0.3537 | 1.434 / 1.326 / 1.192 |
| 640 | 121 | 0.4603 | 0.6970 | 0.4507 | 1.259 / 1.216 / 1.145 |
| 800 | 121 | 0.4980 | 0.7101 | 0.4737 | **1.249** / 1.234 / 1.186 |

**Bulgu:** (a) best epoch geç (81–127/141) ve **best ≈ final** → modeller düzgün yakınsadı, ağır overfit
yok. (b) best mAP50-95 imgsz ile monotonik artar (0.25→0.50); val box-loss düşer (1.70→1.25) → yüksek
çözünürlük eğitimi daha iyi lokalize eder. (c) **Çapraz doğrulama:** training-val mAP50-95
(0.251/0.357/0.460) bizim bağımsız `val_native.py` VAL ölçümümüzle (0.250/0.358/0.454) neredeyse birebir
→ tüm değerlendirme zinciri tutarlı. Hazır eğitim figürleri: `runs/cekmece_v4_141epoch_{N}imgsz/`
(results.png eğitim eğrileri, labels.jpg sınıf/boyut dağılımı, confusion_matrix.png, val_batch*_pred.jpg).

#### Görsel tespit örnekleri — 320 vs 800 giriş (tez figürü — 2026-06-06)

Script: `detect_examples.py`. Aynı ağırlık (800-trained), aynı görüntü, yalnız giriş çözünürlüğü
320↔800 (conf=0.25) → saf çözünürlük etkisi izole. Çıktı: `tez_icin_veriler/thesis_examples/in{320,800}/`.
Küçük/zor sınıf (anahtar, tırnak makası) içeren 4 test görüntüsü seçildi.

| görüntü | GT nesne | tespit @320 | tespit @800 |
|------|------|------|------|
| 0029 (kalabalık) | 27 | 25 | 31 |
| 0068 | 8 | 7 | 6 |
| **0087** | **8** | **2** | **9** |
| 0094 | 2 | 2 | 1 |

**Önerilen figür — `0087`:** dolu çekmece sahnesi. **@320:** yalnız `kalem` + `cakmak (0.67)` bulunur;
makas ve anahtarlar **kaçar**. **@800:** `makas 0.91`, `makas 0.93`, **`anahtar 0.83/0.60/0.48`** (ince
anahtarlar artık görünür), `kalem`, ve `cakmak` güveni **0.67→0.93**. Küçük/ince nesnelerin yalnız
yüksek giriş çözünürlüğünde ortaya çıktığının doğrudan görsel kanıtı — per-class ve dataset
bulgularını (anahtar/tırnak makası en hassas) sahne düzeyinde somutlaştırır.

**Dürüst uyarı:** ham kutu sayısı IoU-eşleşmeli *değil* (bir @800 kutusu FP olabilir; 0094'te @800 daha
az sayıyor — gürültü). Nicel doğruluk ölçütü yukarıdaki mAP tablolarıdır; bu görüntüler **niteliksel
illüstrasyon**dur. 0087 en temiz gösterimi sunar; 0029 yoğun sahnede @800'ün daha çok nesne yakaladığını
(bir miktar fazla-sayım ile) gösterir.

**Sunum için yan-yana kıyas (40 test görüntüsü):** `compare_320_800.py` her test görüntüsü için
etiketli **320 px | 800 px** yan-yana kompozit üretir → `tez_icin_veriler/thesis_examples/compare/<name>.jpg`,
ve görüntüleri "800 − 320 tespit farkı"na göre sıralar (`tez_icin_veriler/compare_320_800.log`). En güçlü
sunum slaytları (gözle doğrulandı):
- **`IMG_..._130316`** (0 → 6): 320'de **hiçbir tespit yok**, 800'de anahtar 0.76 + 2× makas + 2× kalem
  + çakmak — en çarpıcı "hiçbir şey → her şey" slaytı (küçük anahtar dahil).
- **`0087`** (2 → 9): 320'de yalnız kalem+çakmak; 800'de 2× makas + 3× anahtar eklenir.
- **`0125`** (4 → 15), **`IMG_..._130559`** (4 → 12): orta-yoğun sahneler, güçlü fark.
- ⚠️ `0282` (5 → 19): en büyük sayısal fark ama sahne çok kalabalık → görsel okunaksız, sunum için
  önerilmez (yoğun sahnede fazla-sayım/FP riski).

#### Eğitim konfigürasyonu (tekrarlanabilirlik, K5) + fp16-compute doğruluk kontrolü (K1) — 2026-06-06

**Eğitim hiperparametreleri (tüm v4/141-epoch run'ları; kaynak: `runs/.../args.yaml`, `train.py`).** Tek
değişken **imgsz**'dir (320/512/640/800); diğer her şey sabit → kıyaslar tek-değişkenli.

| ayar | değer | ayar | değer |
|------|-------|------|-------|
| mimari | YOLOv8n (pretrained COCO), 3.01M | optimizer | AdamW |
| lr0 / lrf | 0.001 / 0.01 | momentum / weight_decay | 0.937 / 0.0005 |
| warmup epochs | 3 (mom 0.8, bias_lr 0.1) | cos_lr | false |
| epochs / patience | 141 / 100 | batch / nbs | 32 / 64 |
| seed | **42** | deterministic / amp | true / true (fp16 eğitim) |
| loss gains (box/cls/dfl) | 7.5 / 0.5 / 1.5 | val NMS (iou/max_det/conf) | 0.7 / 300 / 0.001 |
| augment | mosaic 1.0 (son 10 ep kapalı), fliplr 0.5, flipud 0, hsv(0.015/0.7/0.4), translate 0.1, scale 0.5, degrees 0, auto_augment randaugment, erasing 0.4, mixup 0 | | |

> **Not (E2 ile ilgili):** tüm modeller **seed=42** ile eğitildi; `deterministic:true` → aynı seed aynı
> sonucu verir. E2 "çok-tohum" = seed 43/44... ile yeniden eğitip strateji farkının seed'e dayanıklılığını ölçmek.

**fp16-compute doğruluk kontrolü (K1 proxy).** Telefon fp32 ağırlıkları GPU delegate ile **fp16 hesaplar**;
bunu doğrudan ölçemedik (cihaz). Proxy: aynı `.pt`'yi PyTorch'ta **fp16 (`half=True`) vs fp32** CUDA'da val
ettik (fp16 = IEEE half, GPU üreticisinden bağımsız → baskın etki yarım-hassasiyet numeriği). Script:
`fp16_proxy_val.py`. Çıktı: `tez_icin_veriler/fp16_proxy.log`.

| config | fp32 mAP50 | fp16 mAP50 | Δ50 | fp32 mAP | fp16 mAP | Δ |
|--------|-----------|-----------|------|----------|----------|------|
| 800@320 | 0.2846 | 0.2846 | −0.0000 | 0.1524 | 0.1510 | −0.0013 |
| 800@640 | 0.6352 | 0.6348 | −0.0004 | 0.4162 | 0.4141 | −0.0021 |
| 800@800 | 0.6893 | 0.6900 | +0.0007 | 0.4593 | 0.4608 | +0.0016 |
| native320 | 0.3337 | 0.3337 | +0.0000 | 0.1918 | 0.1916 | −0.0001 |
| native640 | 0.5944 | 0.5958 | +0.0014 | 0.3729 | 0.3734 | +0.0005 |

**Bulgu:** fp16-compute **doğruluk-nötr** — tüm config/metriklerde **max |Δ| = 0.0021**, bootstrap CI
(±0.08)'in **~40× altında**. → "cihaz-üstü fp16 ≈ PC fp32" varsayımı kanıtla destekleniyor. *Kalan boşluk
(küçük):* proxy PyTorch-CUDA half'tir, Adreno/TFLite delegate op-kernel'lerinin **birebir** aynısı değil;
ama farkın kaynağı olan yarım-hassasiyet numeriğinin etkisiz olduğu gösterildi.

---

## Faz 5 — saf CPU fp32 (GPU delegate kapalı) gecikme çalışması (2026-06-07)

**Araştırma sorusu:** Önceki tüm cihaz-üstü FPS sayıları **GPU delegate** ile alındı; delegate
`isPrecisionLossAllowed:true` ile içeride **fp16** hesaplıyordu (yani "fp32 model" deniyor olsa da compute
fp16'ydı). Bu faz **gerçek fp32 compute maliyetini** GPU olmadan ölçer: delegate kapatılıp **saf CPU
(XNNPACK, 4 thread)** ile koşuldu. Amaç: (1) fp16-GPU yolunun sağladığı hızlanmayı niceliklemek, (2)
"delegate olmadan bu cihaz fp32'yi gerçek-zamanlı koşturabilir mi?" sorusunu yanıtlamak.

**Kurulum:** `YoloDetector._useGpuDelegate=false` (kod anahtarı; `true`=GPU A/B için saklı). Native fp32
.tflite (320/640/800), I/O float32 (fp16'nın aksine sorunsuz allocate olur). Redmi Note 11 (SD680),
release. Ölçüm: K4 on-device logger — per-frame `total_ms=pre+inf+parse`, `dt_ms` kareler-arası gecikme,
`pure_ms` yükleme-anı saf-inference benchmark. Ham veri: `tez_icin_veriler/cpu_fp32_csv/cpu_fp32_{640,320}.csv`.

### Ölçülen (CPU fp32, Redmi Note 11, release)

| Giriş | pure inf (ms) | total p50 | total p95 | total p99 | dt p50 (ms) | FPS (1000/dt p50) | kare / süre |
|-------|---------------|-----------|-----------|-----------|-------------|-------------------|-------------|
| 320 | 112.0 | 154.6 | 242.7 | 263.6 | 176 | **5.68** | 190 / 35 s |
| 640 | 416.7 | 545.4 | 637.3 | 665.2 | 542 | **1.85** | 77 / 43 s |
| 800 | ~670 (ekstrapole) | ~850 (ekstrapole) | — | — | — | **~1.1–1.3** | ⚠ ölçülemedi |

> ⚠ **800 ölçümü kayboldu (dürüst not):** 800 kaydı ilk build ile alınmıştı; switch-crash fixi için
> yeniden kurulum (`flutter install` "Uninstalling old version...") Android'in `/Android/data/<pkg>/files/`
> dizinini sildiğinden CSV silindi. 320/640 PC'ye çekilmişti → güvende. 800 satırı **size² yasasından
> ekstrapolasyon** (pure 416.7×(800/640)²≈651, ortalama-katsayıyla ≈676 → ~670 ms; total ≈545×1.5625≈850 ms).
> İstenirse tek build ile (varsayılan 800) yeniden ölçülebilir; bir sonraki kayıttan **önce** reinstall
> yapılmamalı (CSV'yi önce çek).

### CPU fp32 vs GPU fp16 — ana bulgu

GPU delegate sayıları = yukarıdaki "fp32 baseline — cihaz ölçümü" tablosu (Adreno OpenCL, fp16 compute).
Aynı cihaz, aynı native modeller, tek değişken = backend:

| Giriş | GPU fp16 pure | CPU fp32 pure | CPU/GPU gecikme | GPU FPS | CPU FPS | GPU/CPU throughput |
|-------|---------------|---------------|-----------------|---------|---------|--------------------|
| 320 | 61 | 112.0 | **1.84×** | 12 | 5.68 | 2.11× |
| 640 | 221 | 416.7 | **1.89×** | 4 | 1.85 | 2.16× |
| 800 | 371 | ~670 (est) | ~1.81× | 2.4 (K4) | ~1.2 (est) | ~2.0× |

**Yorum (tez için):**
- **fp16-GPU yolu, saf fp32-CPU'ya kıyasla ~1.85× daha düşük gecikme** ve ~2× daha yüksek throughput
  sağlıyor (üç boyutta da tutarlı). "GPU delegate kullan" mühendislik kararını **niceliksel** doğrular:
  kazanç hem fp16 aritmetiğinden hem inference'i CPU'dan boşaltmaktan gelir.
- **size² yasası CPU'da da geçerli:** pure/px² = 320:1.09 · 640:1.02 (×10⁻³ ms/px²), dar bant —
  GPU'daki (~5.5×10⁻⁴) eğilimin ~1.9× kaymış hali. Maliyet modeli **backend'den bağımsız** doğrulanmış oldu.
- **Gerçek-zamanlılık:** saf CPU fp32 bu cihazda **hiçbir boyutta akıcı değil** — en hızlı 320 bile yalnız
  ~5.7 FPS, 640 ~1.85 FPS, 800 ~1 FPS. Kullanılabilir tempo (≥~10 FPS) için **GPU delegate zorunlu**;
  fp16 hassasiyet-kaybı doğruluğu ihmal edilebilir etkiler (fp16-proxy bölümü) ama hızı ~2× artırır.
- **Termal:** koşular kısa (35–43 s) → throttle gözlenmedi (dt median drift hafif **negatif**: warmup/EMA
  oturması). Sürdürülen termal eğri için ≥10 dk koşu gerekir; bu faz **steady-state latency** ölçümüdür.

### B4. Model-switch use-after-free crash (bu fazda bulundu + giderildi)
**Sorun:** CPU modunda model değiştirince uygulama çöküyordu. **Kök neden:** `_onFrame` bir kareyi
`detect()`'e sokmuşken (CPU 640/800'de inference ~0.4–0.9 s → uçuşta uzun süre), kullanıcı boyut
değiştirince `_reloadModel → _detector.close()` çağrılıyor; `close()` native interpreter'ı **inference
arka-plan isolate'inde hâlâ çalışırken** serbest bırakıyor → use-after-free → SIGSEGV. GPU'da pencere küçük
(61 ms) olduğundan nadirdi; CPU'nun büyük penceresinde **kesin** crash.
**Çözüm:** `_reloadModel`/`_enableAuto` içinde stream durdurulduktan sonra `_drainInFlight()` ile uçuştaki
`detect()` bitene kadar beklenip sonra `close()` çağrılıyor. `_switching` bayrağı zaten **yeni** kareleri
engelliyordu; eksik olan tek uçuştaki kareyi drain etmekti.
**Ders:** Native kaynağı serbest bırakmadan önce o kaynağı kullanan async iş tamamlanmalı — bayrak "yeni iş
başlamasın" için yeterli, "mevcut iş bitsin" için değil.

---

## Kısıtlar ve Geçerlilik Tehditleri (Limitations & Threats to Validity) — 2026-06-06

Bulguların dürüst sınırları. Tez "Tartışma/Kısıtlar" bölümünün temeli; jüri zaten bunları arar. Birçoğu
deneyle değil **açık beyanla** ele alınır; kapatılabilir olanlar "gelecek iş" olarak işaretlendi.

**K1. Cihaz-üstü doğruluk — fp16-proxy ile büyük ölçüde giderildi; tam delegate kernel'i açık.**
Tüm mAP sayıları **PC, CPU, fp32**. Telefon ise fp32 ağırlıkları **GPU delegate ile fp16 hesaplar**
(`isPrecisionLossAllowed:true`, Adreno OpenCL). Bu fark **proxy ile ölçüldü** (yukarıdaki fp16-compute
kontrolü): PyTorch `half=True` vs fp32 → **max |Δ| = 0.002** (CI'nin ~40× altında) → fp16 numeriği
doğruluk-nötr, "cihaz ≈ PC fp32" desteklendi. Ek olarak fp32 `.tflite` export sadakati de ölçüldü
(≤0.007–0.03). *Kalan (küçük) boşluk:* Adreno/TFLite delegate op-kernel'lerinin birebir cihaz ölçümü
yapılmadı (yalnız yarım-hassasiyet etkisi izole edildi).

**K2. Küçük, tek-bölünmeli veri seti; çapraz doğrulama yok.**
400 train / 40 val / 40 test, **tek sabit split**. Tüm sayılar bu bölünmeye koşulludur ve "best epoch"
seçimi de **40-görüntülük gürültülü val** üzerinde yapıldı. Bootstrap (E1/#4) yalnız **test-içi örnekleme**
belirsizliğini ölçer (mAP@0.5 CI yarı-genişliği **±0.08**); **hangi split** belirsizliğini değil —
**TEST↔VAL yön farkları tam da bu kısıtın yansımasıdır.** Sınıf-bazında durum daha ağır (≤60, tırnak makası
**27** instance) → ince per-class iddialar zayıf (bkz. per-class CI bölümü). *Kapatma yolu:* k-fold CV veya
daha büyük held-out set.

**K3. Çok-tohumlu eğitim — ✅ giderildi (E2).**
Eskiden her config tek seed'e (42) dayanıyordu; eğitim-rastgeleliği ölçülmemişti. **E2 ile kapandı:** native
{320,512,640} + 800-model ×3 seed (42/43/44) eğitilip TEST'te değerlendirildi (yukarıdaki E2 bölümü). Sonuç:
native-320 üstünlüğü **seed-robust** (3/3 seed ayrık, Δ=+0.083 ≫ std), 640'ta 800 robust önde — imza bulgu
artık 3 bağımsız eğitimle desteklenmiştir. Seed std'leri küçük (0.002–0.019). *Kalan (opsiyonel):* daha çok
seed (n=5+) güç artırır ama yön zaten net.

**K4. Cihaz gecikme dağılımı + termal — ✅ giderildi (320 + 800 ölçüldü, 800px artefaktı çözüldü).**
Cihaz-üstü kaydedici eklendi (`lib/latency_logger.dart` + REC butonu); **iki uç boyut** Redmi Note 11'de
(release) ölçüldü (`analyze_latency.py`). Per-frame compute gecikme dağılımı + anlık FPS + termal:

| boyut | kare/süre | gecikme p50 · p95 · p99 (max) | FPS p50 (en kötü p05) | termal (ilk→son ⅓) |
|-------|-----------|-------------------------------|------------------------|--------------------|
| 320px | 2197 / 193 s | 75 · 78 · 83 ms (94) | 11.9 (9.9) | 12.4→11.9 (−4%) **stabil** |
| 640px | 116 / 30 s | 241 · 267 · 273 ms (274) | 4.0 (3.4) | (koşu kısa ~30 s) |
| 800px | 456 / 190 s | 404 · 410 · 447 ms (492) | 2.4 (2.3) | 2.4→2.4 (−1%) **stabil** |

- **Dağılım dar (düşük jitter):** p99, p50'den yalnız %10–13 fazla → gerçek-zaman için öngörülebilir.
  **Üç native boyut da cihazda dağılımla doğrulandı:** 320 = **12 FPS** (75 ms) · 640 = **4.0 FPS** (241 ms)
  · 800 = **2.4 FPS** (404 ms) — dokümanın FPS iddialarıyla birebir (artık EMA tahmini değil).
- **800px "ölçüm artefaktı" ÇÖZÜLDÜ:** doğrudan per-frame total = **404 ms** (p50) → gerçek değer ~400 ms
  (≈ pure 371 + pre/parse), eski yanıltıcı canlı `inf`=185 ms bir EMA-artefaktıydı. FPS 2.4 = 1000/404
  (öz-tutarlı). **3-nokta total gecikme** 75→241→404 ms (320→640→800); oranlar 3.2× / 5.4×
  (boyut²: 4.0× / 6.25×; total daha düşük çünkü pre/parse ~sabit) → **cost∝boyut² cihazda dağılım düzeyinde
  de doğrulandı.**
- **Termal:** her iki yükte de throttling yok (320: −4%, 800: −1%, ~3 dk). *Kalan (küçük):* tek cihaz
  (Redmi Note 11), ve daha uzun (≥10 dk) koşu daha güçlü termal kanıt verir. Çıktı:
  `k4_latency_summary.log`, `thesis_plots/k4_latency_thermal.png`.

**K5. Tekrarlanabilirlik — ✅ giderildi.**
Tam eğitim hiperparametre tablosu (optimizer, lr0/lrf, momentum, weight_decay, warmup, loss gains,
augmentation politikası, batch, **seed=42**, amp, val NMS) yukarıdaki "Eğitim konfigürasyonu" bölümünde
belgelendi (kaynak `runs/.../args.yaml` + `train.py`). Kalan tek not: 141 epoch sayısı ampirik seçim
(eğitim yakınsama bölümü best≈final, ağır overfit yok diye doğruladı).

**K6. Dış geçerlilik / kapsam.**
5 belirli sınıf (anahtar/çakmak/kalem/makas/tırnak makası), çekmece-içi sahneler, tek kaynak (Roboflow).
Aydınlatma/arka plan/kamera çeşitliliği sınırlı; bu ortam dışına genelleme doğrulanmadı.

**Özet:** İstatistiksel olarak en sağlam bulgular — (a) hız ∝ giriş², (b) yüksek çözünürlük tüm sınıflara
**anlamlı** doğruluk yararı, (c) fp32 TFLite export sadakati, (d) fp16-compute doğruluk-nötr (max|Δ|=0.002).
(e) **native-320 > 800-downscale@320 üstünlüğü seed-robust** (E2: 3/3 seed ayrık) — ve simetrik olarak 640'ta
800-downscale robust önde (geçiş ~512). **İnceltilmiş/koşullu** bulgular — per-class sıralamalar ve ×oranlar
(sınıf-içi gürültü → mutlak Δ kullanılmalı); native-vs-800 farkının kesin **büyüklüğü** (küçük test seti →
görüntü-CI geniş, ama yön seed-robust). (f) **cihaz-üstü gecikme dağılımı + termal stabilite** (K4: 320 ve 800'de p50/p95/p99 + throttling yok;
800px artefaktı çözüldü). **Belgelenmiş kısıtlar** — tek split/CV yok (K2), **tek SoC** (K4 ölçüldü ama
yalnız Redmi Note 11), dış geçerlilik (K6). Bu ayrım tezde net yapılırsa çalışma **dürüst ve savunulabilir**.

---

## Tez Metodoloji Notları — kullanıcı beyanlarıyla güncel taslak (2026-06-06)

Bu bölüm, mevcut deney sonuçlarını **değiştirmeden** tez yazımı için gerekli yöntem/kapsam bilgisini bir araya
getirir. Tez dili İngilizce olacak; aşağıdaki notlar IEEE şablonuna aktarılırken İngilizce akademik dile
dönüştürülecek ham kaynak olarak kullanılacaktır.

### Tez kimliği ve ana kapsam

- Tez türü: **Bilgisayar Mühendisliği lisans bitirme tezi**.
- Tez başlığı: **YOLOv8 Tabanlı Mobil Çekmece İçi Nesne Tespit Uygulaması**.
- Uygulama adı: **Çekmece İçi Nesne Tespit Uygulaması**.
- Ana katkı: YOLOv8 tabanlı bir nesne tespit modelinin eğitilmesi ve bu modelin Flutter/TensorFlow Lite
  ile mobil cihazda gerçek zamanlı çalışacak şekilde **uçtan uca entegre edilmesi**.
- Çalışma ticari ürün değil, akademik/prototip amaçlıdır. Amaç yalnız en yüksek doğruluk değil; mobil cihazda
  kullanılabilir hızda çalışan ve kabul edilebilir doğruluk sunan uçtan uca bir sistem geliştirmektir.

### Motivasyon ve problem seçimi

Çalışmanın temel motivasyonu, YOLOv8'in mobil entegrasyonunun pratikte nasıl çalıştığını incelemektir. Çekmece
ortamı, veri toplamanın kolay olması ve evde yaygın bulunan küçük nesneleri içermesi nedeniyle seçilmiştir.
Hedef sınıflar ev ortamında sık karşılaşılan nesnelerden oluşturulmuştur:

- `anahtar`
- `cakmak`
- `kalem`
- `makas`
- `tirnak_makasi`

Uygulama daha çok akademik gösterim/prototip amacı taşır. Hedef, telefon kamerası çekmeceye tutulduğunda bu beş
nesneyi gerçek zamanlı olarak bounding box ve sınıf etiketiyle gösterebilmektir.

### Veri toplama ortamı

Gerçek görseller ev ortamında hazırlanmıştır. Evde yaygın olarak bulunan nesnelerle karışık çekmece sahneleri
oluşturulmuş ve fotoğraflar farklı çekmecelerde çekilmiştir. Sahneler kullanıcı tarafından özellikle düzenlenmiş,
ancak gerçek kullanım koşulunu temsil edecek şekilde dağınık bırakılmıştır. Hedef sınıflar dışında ilaç kutusu,
kablo ve benzeri ev içi nesneler de sahnede bulunmuş; bu nesneler arka plan/distractor olarak kullanılmıştır.

Çekimler **Redmi Note 11** telefon kamerasıyla, varsayılan kamera ayarları kullanılarak yapılmıştır. Görüntüler
hem gün ışığında hem oda ışığında çekilmiş; üstten, çapraz, yakın ve farklı açılar denenmiştir. Orijinal fotoğraf
çözünürlüğü ayrıca sabitlenmemiş, telefonun varsayılan kamera çözünürlüğü kullanılmıştır. Cihaz donanım özellikleri
tezde yazılırken ayrıca kaynaklandırılacaktır.

### Etiketleme süreci

Etiketleme Roboflow üzerinde bounding box yöntemiyle **manuel** olarak yapılmıştır. Herhangi bir otomatik etiketleme
veya AI destekli etiketleme aracı kullanılmamıştır. Kutular mümkün olduğunca nesne sınırlarına sıkı çizilmeye
çalışılmıştır. Bununla birlikte, özellikle çapraz duran veya yan yana gelen ince/uzun nesnelerde (örneğin kalemler)
bounding box alanı doğal olarak komşu nesneleri veya arka planın bir kısmını içerebilmiştir. Bu durum çekmece içi
nesne tespitinin gerçekçi zorluklarından biri olarak değerlendirilecektir.

Etiketler YOLO formatında dışa aktarılmıştır. Hedef sınıflar dışında kalan ilaç kutusu, kablo vb. nesneler
etiketlenmemiştir. Etiketleme tamamlandıktan sonra eksik kutu, yanlış sınıf ve bariz etiket hataları için manuel
kontrol yapılmıştır. Sentetik görseller de gerçek görsellerle aynı şekilde Roboflow üzerinde manuel bounding box
yöntemiyle etiketlenmiştir.

### Sentetik veri üretimi

İlk aşamada yaklaşık **300 gerçek görselden oluşan** bir veri seti hazırlanmıştır. Sentetik veri üretiminde bu gerçek
çekimler referans alınmış ve DALL-E'den benzer çekmece ortamları üretmesi istenmiştir. Üretimde ışık koşulu, kamera
açısı ve sahne karmaşıklığı gibi değişkenlerin değiştirilmesi özellikle talep edilmiştir. Toplam yaklaşık **200
sentetik görsel** üretilmiş, bunların içinden kalite, gerçek çekmece ortamına benzerlik, hedef nesnelerin
ayırt edilebilirliği ve etiketlenebilirlik açısından daha güçlü bulunan **80 görsel** seçilmiştir.

Kullanılan prompt mantığı, mevcut gerçek veri setine benzer ama daha zorlayıcı çekmece sahneleri üretmeye
odaklanmıştır. Promptlarda; çekmece içinin karmakarışık olması, loş ışık koşulları, nesnelerin üst üste binmesi,
çekmecede en az bir hedef sınıfın bulunması ve beş hedef sınıf için kendi içinde çeşitlilik sağlanması istenmiştir.
Örneğin farklı türde kalemler, farklı türde makaslar, farklı anahtar/çakmak/tırnak makası görünümleri, değişen
ışık, açı ve sahne karmaşıklığı özellikle vurgulanmıştır. Bu yaklaşım, sentetik görsellerin yalnızca temiz ürün
fotoğrafı gibi değil, gerçek çekmece karmaşıklığına daha yakın sahneler olarak üretilmesini hedeflemiştir.

Son veri seti toplam **480 görselden** oluşur:

| kaynak | adet | not |
|--------|------|-----|
| gerçek çekim | 400 | Ev ortamında Redmi Note 11 ile çekildi |
| DALL-E sentetik | 80 | 200 üretim içinden seçildi |
| toplam | 480 | Eğitim/doğrulama/test için kullanıldı |

Roboflow üzerinde ek augmentation uygulanmamıştır. Sentetik görseller yalnızca eğitim kümesine dahil edilmiştir;
validation ve test kümelerinde sentetik görsel kullanılmamıştır.

### Train/validation/test bölmesi

Veri seti GPT destekli olarak bölünmüştür:

| split | toplam görsel | gerçek görsel | sentetik görsel |
|-------|---------------|---------------|-----------------|
| train | 400 | 320 | 80 |
| validation | 40 | 40 | 0 |
| test | 40 | 40 | 0 |

Final değerlendirmelerde **test seti ana referans**, validation seti ise eğitim/doğrulama takibi için yardımcı
referans olarak ele alınacaktır. Bölme sürecinde validation/test setlerinin sentetik veri içermemesine dikkat
edilmiştir. Ancak sahne benzerliği, sınıf bazlı tam dengeleme veya ışık/açı dağılımı için ayrıntılı stratifikasyon
garanti edilmemiştir; bu durum kısıtlar/geçerlilik tehditleri bölümünde açıkça belirtilecektir.

### Eğitim ortamı ve eğitim süreci

Model eğitimleri Windows işletim sistemine sahip, **NVIDIA RTX 4060 GPU** bulunan kişisel bir dizüstü bilgisayarda
gerçekleştirilmiştir. Eğitim için Ultralytics YOLOv8 kullanılmıştır. Eğitimler proje kökündeki `train.py` betiği
üzerinden, gerekli parametreler manuel verilerek `python train.py` komutuyla başlatılmıştır.

Çalışmanın ana model ailesi **YOLOv8n** olarak belirlenmiştir. YOLOv8 daha önce staj sürecinde kullanıldığı ve
nesne tespitinde güncel/yaygın bir mimari olduğu için tercih edilmiştir. Eğitimlerdeki temel ayarlar mevcut
`train.py` ve `runs/.../args.yaml` dosyalarıyla uyumludur:

- model: YOLOv8n (`yolov8n.pt`)
- optimizer: AdamW
- lr0: 0.001
- batch: 32
- seed: 42
- epoch: ana deneylerde 141
- input size (`imgsz`): deneysel olarak 320, 512, 640 ve 800

`imgsz`, modelin eğitim ve çıkarım sırasında görüntüyü ölçeklediği giriş boyutudur. Daha küçük `imgsz` değerleri
telefonda daha yüksek FPS sağlarken, daha büyük `imgsz` değerleri özellikle küçük nesnelerde doğruluğu artırır.

### Yazılım ve donanım ortamı notları

Tezde yeniden üretilebilirlik için aşağıdaki ortam tablosu kullanılabilir. Eğitim paketi sürümleri, kullanıcının
eğitim ortamında çalıştırdığı doğrulama komutuyla ayrıca teyit edilmiştir.

| bileşen | değer / not |
|---------|-------------|
| Eğitim işletim sistemi | Windows |
| Eğitim donanımı | NVIDIA RTX 4060 GPU'lu dizüstü bilgisayar |
| Mevcut terminal Python | Python 3.12.10 |
| Eğitim kütüphanesi | Ultralytics YOLOv8 8.3.167 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| Eğitim GPU adı | NVIDIA GeForce RTX 4060 Laptop GPU |
| Flutter | 3.44.0 stable |
| Dart SDK | 3.12.0 |
| DevTools | 2.57.0 |
| `tflite_flutter` | 0.12.1 |
| `camera` | 0.11.4 |
| `path_provider` | 2.1.5 |
| `permission_handler` | 11.4.0 |
| Android minSdk | 24 |
| Android build | Release APK |
| Android Gradle Plugin | 9.0.1 |
| Kotlin plugin | 2.3.20 |
| Gradle wrapper | 9.1.0 |
| JVM hedefi | Java/JVM 17 |
| Mobil test cihazı | Redmi Note 11 |

### Model dışa aktarma ve mobil entegrasyon

Eğitilen `.pt` modeller mobil uygulamada kullanılmak üzere TensorFlow Lite `.tflite` formatına çevrilmiştir.
Dönüştürme işlemi Ultralytics export komutlarıyla yapılmış ve final mobil uygulamada fp32 TFLite modelleri
tercih edilmiştir. INT8 ve fp16 yolları ayrıca denenmiş, ancak mevcut araç zinciri ve cihaz/runtime uyumluluğu
nedeniyle final dağıtımda kullanılmamıştır; bu negatif sonuçlar tezde mühendislik bulgusu olarak anlatılacaktır.

Mobil uygulama Flutter ile sıfırdan geliştirilmiştir. Flutter, tek kod tabanı ile Android ve iOS hedefleyebildiği
için seçilmiştir. Bu çalışma kapsamında deneysel testler yalnızca Android tabanlı Redmi Note 11 üzerinde yapılmıştır;
iOS desteği Flutter altyapısı nedeniyle mümkün olsa da doğrulanmamıştır.

Uygulama internet bağlantısı gerektirmeden, tamamen cihaz üzerinde çalışır. Kamera görüntüsü herhangi bir sunucuya
gönderilmez ve cihazda kalıcı olarak saklanmaz; tespitler gerçek zamanlı olarak ekranda gösterilir.

Uygulamanın temel özellikleri:

- canlı kamera görüntüsü,
- tespit edilen nesneler için bounding box ve sınıf etiketi,
- confidence eşik ayarı,
- model giriş boyutu seçimi,
- FPS göstergesi,
- cihaz üstü gecikme ölçümü için latency kayıt butonu.

Genel sistem akışı: kamera görüntüsü alınır → görüntü model giriş boyutuna hazırlanır → TFLite model çıkarım yapar
→ sonuçlar filtrelenir ve NMS uygulanır → bounding box ve sınıf etiketleri kamera görüntüsü üzerine çizilir.

### Değerlendirme yöntemi

Model doğruluğu şu metriklerle değerlendirilmiştir:

- mAP@0.5,
- mAP@0.5:0.95,
- precision,
- recall.

Validation ve test klasörleri değerlendirme için kullanılmıştır; tezde test seti ana referans kabul edilecektir.
Mobil performans ölçümleri Redmi Note 11 üzerinde, **release APK** ile alınmıştır. FPS hem uygulama arayüzündeki
FPS göstergesiyle gözlenmiş hem de latency kayıt butonuyla CSV olarak kaydedilmiştir. Telefon testlerinde cihaz
gerçek kullanıma benzer şekilde çekmece üzerinde hareket ettirilmiştir.

Gerçek zamanlılık yorumu için yaklaşık **10 FPS ve üzeri** kullanılabilir gerçek zamanlı deneyim, **2 FPS civarı**
ise gerçek zamanlı kullanım için yetersiz olarak yorumlanacaktır. Bu eşik kamera uygulamalarındaki 30 FPS standardı
değil, nesne kutularının kullanıcı hareketine yeterince hızlı tepki vermesi açısından pratik bir eşiktir.

### Ana deney ve yan deney kurgusu

Tezde ana deney, **800 pikselde eğitilmiş `cekmece_v4` YOLOv8n modelinin 320/512/640 export/değerlendirmeleri ile
native çözünürlükte eğitilmiş YOLOv8n modellerin karşılaştırılmasıdır**. 800px sonucu doğruluk üst sınırı/accuracy
ceiling olarak ayrıca gösterilebilir.

Ana deneyin cevapladığı soru:

> Bir YOLOv8n modeli yüksek çözünürlükte eğitilip daha düşük çözünürlükte çalıştırıldığında mı daha iyi sonuç verir,
> yoksa doğrudan hedef mobil çözünürlüğünde eğitmek mi daha avantajlıdır?

Deneysel motivasyon, PC ortamında 800px modelin doğruluk değerlerinin iyi görünmesine rağmen telefonda yaklaşık
2 FPS seviyesine düşmesi ve bu nedenle mobil kullanımda FPS/doğruluk dengesinin ayrıca incelenmesi gereğidir.
Kullanıcı gözlemine göre 640px daha iyi bir denge noktası olarak değerlendirilmiştir; 320/512/640/800 seçenekleri
ise hız-doğruluk ödünleşimini göstermek için deneysel olarak eklenmiştir. Auto mode, cihazın kaldırabileceği uygun
model boyutunu seçebilmesi fikriyle eklenmiştir.

Tezde yan deney olarak **YOLOv8n@320 vs YOLOv8s@320** karşılaştırması planlanmıştır. Bu deneyin amacı, düşük
çözünürlükte daha büyük model kullanmanın doğruluk kazancını ve mobil maliyetini incelemektir. Var olan ek
YOLOv8s sonuçları destekleyici/opsiyonel bilgi olarak tutulabilir; tez odağı 320n-320s kapasite kıyasıdır.

Final kullanım önerisi şu çizgide verilecektir:

- gerçek zamanlı kullanım öncelikliyse 320 veya 512,
- daha yüksek doğruluk isteniyor ve düşük FPS kabul edilebiliyorsa 640 veya 800,
- kullanıcı/cihaz dengesini otomatik seçmek için Auto mode.

### Kapsam dışı bırakılanlar ve sınırlılıklar

Uygulama yalnızca beş hedef sınıfı tanır. Sesli uyarı, belirli nesne arama modu ("kalemi bul" vb.) ve kullanıcıyı
nesneye yönlendirme gibi özellikler kapsam dışı bırakılmıştır. Çalışma genel olarak çekmece ortamına odaklanır;
çekmece dışı ortamlarda genelleme ayrıca test edilmemiştir. Veri seti tek ev ortamındaki gerçek çekimler ve bu
çekimleri referans alan sentetik görsellerden oluşur. Mobil testler yalnızca Redmi Note 11 üzerinde yapılmıştır.

Bu nedenle tezde açıkça şu sınırlar belirtilecektir:

- sınırlı veri seti,
- tek sabit train/validation/test split,
- tek cihaz üzerinde mobil performans ölçümü,
- iOS üzerinde doğrulama yapılmaması,
- çekmece dışı ortamlara genellemenin ölçülmemiş olması.

### Yazım ve sunum kararları

Tez IEEE şablonu baz alınarak İngilizce yazılacaktır. Odak; deney sonuçları, çözünürlük/doğruluk/FPS dengesi ve
mobil optimizasyonlar olacaktır. Uzun kod blokları yerine sistem akış diyagramı, eğitim parametre tablosu, sonuç
tabloları ve görseller kullanılacaktır. Tezde kullanılması önerilen görseller:

- dataset örnekleri,
- sistem mimarisi/pipeline diyagramı,
- confusion matrix,
- PR/F1/precision/recall eğrileri,
- 320/512/640/800 doğruluk-FPS karşılaştırması,
- K4 latency/thermal grafiği,
- uygulama ekran görüntüsü.

Kaynakça kısmında YOLOv8/Ultralytics, TensorFlow Lite, Flutter, nesne tespit metrikleri (precision, recall, mAP)
ve mobil/edge AI konularında akademik veya teknik kaynaklar kullanılacaktır.

### Kaynakça adayları — IEEE için gözden geçirilmiş liste

Bu liste final tez kaynakçası için adaydır. IEEE biçimine son tez yazımı sırasında dönüştürülecektir.

| kod | kaynak | tezde kullanım amacı |
|-----|--------|----------------------|
| R1 | J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," CVPR, 2016. https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html | YOLO yaklaşımının gerçek zamanlı nesne tespiti temeli |
| R2 | Ultralytics, "YOLOv8," official documentation. https://docs.ultralytics.com/models/yolov8/ | YOLOv8 model ailesi ve mimari/uygulama bağlamı |
| R3 | Ultralytics, "Model Export with Ultralytics YOLO," official documentation. https://docs.ultralytics.com/modes/export/ | `.pt` modellerin TFLite gibi dağıtım formatlarına export edilmesi |
| R4 | TensorFlow, "TensorFlow Lite guide." https://www.tensorflow.org/lite/guide | Mobil/edge cihazlarda cihaz üstü ML çıkarımı |
| R5 | TensorFlow Lite, "GPU delegate." https://android.googlesource.com/platform/external/tensorflow/+/refs/heads/master/tensorflow/lite/g3doc/performance/gpu.md | Android/iOS GPU delegate ve mobil hızlandırma |
| R6 | Flutter, "Flutter architectural overview." https://docs.flutter.dev/resources/architectural-overview | Flutter'ın çapraz platform mimarisi ve platform entegrasyonu |
| R7 | `tflite_flutter` package, pub.dev. https://pub.dev/packages/tflite_flutter | Flutter içinde TFLite interpreter ve delegate/isolate desteği |
| R8 | `camera` package, pub.dev. https://pub.dev/packages/camera | Flutter kamera akışı ve image stream kullanımı |
| R9 | Roboflow, "Introduction to Roboflow Annotate." https://docs.roboflow.com/annotate | Bounding box tabanlı manuel etiketleme aracı |
| R10 | T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV, 2014. https://arxiv.org/abs/1405.0312 | Nesne tespitinde AP/mAP değerlendirme geleneği ve COCO bağlamı |
| R11 | J. Betker et al., "Improving Image Generation with Better Captions," OpenAI, 2023. https://cdn.openai.com/papers/dall-e-3.pdf | DALL-E 3 / text-to-image sentetik görsel üretimi bağlamı |
| R12 | A. Ge et al., "DALL-E for Detection: Language-driven Compositional Image Synthesis for Object Detection," arXiv, 2022. https://arxiv.org/abs/2206.09592 | Text-to-image üretimin nesne tespiti eğitim verisi için kullanımı |
| R13 | A. Ignatov et al., "On-Device Neural Net Inference with Mobile GPUs," arXiv, 2019. https://arxiv.org/abs/1907.01989 | Mobil GPU üzerinde cihaz üstü çıkarım ve gecikme/performans bağlamı |

Final kaynakçada gereksiz kalabalığı önlemek için R1-R10 temel kaynaklar olarak yeterlidir; R11-R13 ise sentetik
veri ve mobil GPU tartışması tezde ne kadar yer kaplayacağına göre eklenebilir.

### Tez görsel seçimi — native `cekmece_v4_141epoch_640imgsz` odaklı liste

Kullanıcı kararı: confusion matrix ve PR curve için **native `cekmece_v4_141epoch_640imgsz`** sonuçları kullanılacak.
Seçili dosyalar sabit isimlerle `flutter_app/tez_icin_veriler/thesis_selected_figures/` klasörüne kopyalandı.

| tez öğesi | seçilen dosya / kaynak | kullanım notu |
|-----------|------------------------|---------------|
| Dataset örneği | `thesis_selected_figures/fig_dataset_native640_labels.jpg` | Karışık çekmece ortamını ve manuel bounding box yoğunluğunu göstermek için. |
| Uygulama ekran görüntüsü | Kullanıcının gönderdiği telefon fotoğrafı; yerel dosya olarak ayrıca kaydedilmeli | Canlı Flutter uygulamasında bounding box, sınıf etiketi, FPS ve model seçimi görünür. |
| Ana deney tablo/grafik | PERFORMANCE.md ana deney tabloları: native-N vs 800-trained@N (`Tablo A/B/C`) | Tezde tablo olarak verilmesi önerilir; 320/512/640 native-vs-downscale kıyasının ana kanıtı. |
| Confusion matrix | `thesis_selected_figures/fig_native640_confusion_matrix_normalized.png` | Native640 sınıf karışıklıklarını oranla gösterir. |
| PR curve | `thesis_selected_figures/fig_native640_pr_curve.png` | Native640 mAP@0.5 ve sınıf bazlı PR davranışını gösterir. |
| 320 vs 800 karşılaştırma | `thesis_selected_figures/fig_320_vs_800_comparison_0087.jpg` | Çözünürlük artışının tespit sayısına etkisini görsel olarak gösterir (320px: 2, 800px: 9 tespit). |
| Latency grafiği | `thesis_selected_figures/fig_latency_thermal.png` | Redmi Note 11 cihazında latency/FPS/termal davranışı için. |
| INT8 negatif sonuç figürü | `thesis_selected_figures/fig_native640_int8_diagnostic_0087.jpg` | Native640 INT8 export denemesinde `cx/cy` koordinatlarının 0'a çökmesini ve tespit üretilememesini gösterir. |

**Uygulama ekran görüntüsü notu:** Sohbete gönderilen fotoğraf tez için uygundur; ancak DOCX üretiminde kullanmak
için aynı görselin proje klasörüne dosya olarak kaydedilmesi gerekir. Önerilen yol:
`flutter_app/tez_icin_veriler/thesis_selected_figures/fig_app_screenshot_native640.jpg`.

### Native640 INT8 diagnostic — tez için negatif dağıtım kanıtı

Kullanıcı isteğiyle native `cekmece_v4_141epoch_640imgsz` modelinin INT8 TFLite export'u ayrıca test edilmiştir.
Script: `export_test_native640_int8.py`. Test görseli: `dataset/images/test/0087_jpg.rf.VbQnnnDVMfxstb64jImx.jpg`.

Çıktılar:

- INT8 model: `flutter_app/tez_icin_veriler/native_exports/cekmece_v4_native640_int8.tflite` (~3.0 MB)
- Log: `flutter_app/tez_icin_veriler/native_int8_diagnostic.log`
- Figür: `flutter_app/tez_icin_veriler/thesis_selected_figures/fig_native640_int8_diagnostic_0087.jpg`

Diagnostic sonucu:

| kanal | min | max | yorum |
|-------|-----|-----|-------|
| `cx` | 0.00000 | 0.00000 | tüm anchor'larda merkez x çöktü |
| `cy` | 0.00000 | 0.00000 | tüm anchor'larda merkez y çöktü |
| `w` | 0.00781 | 0.78125 | genişlik kanalı sıfıra çökmedi |
| `h` | 0.00781 | 0.53125 | yükseklik kanalı sıfıra çökmedi |
| class scores | 0.00000 | 0.90625 | sınıf skorları üretildi |

Sonuç: `DFL/cx-cy collapsed: True`; NMS sonrası tespit sayısı `0`. Bu, INT8 sorununun yalnız 800-trained eski
export'a özgü olmadığını; native640 modelde de YOLOv8 TFLite INT8 quantization sonrası bbox merkez koordinatlarının
çöktüğünü gösterir. Bu nedenle INT8 yolu final mobil uygulama için kullanılmamış, final dağıtım fp32 TFLite olarak
korunmuştur.

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
| 800 | ~1 FPS | ~1 FPS | TBD | — | — |

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

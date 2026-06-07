# Görev: fp16 TFLite yükleme hatasını tez için kanıtlama

**Hedef:** fp16 modelin cihazda/uygulamada yüklenememesini tez-kalitesinde, yeniden
üretilebilir kanıta dönüştürmek (negatif bulgu → fp32 native deploy gerekçesi).

**Karar (kullanıcı):** PC repro + tez metni + **gerçek cihaz kanıtı** (logcat + ekran görüntüsü).

## Kesinleşen olgular (bu oturum)
- fp16 `.tflite` I/O'su **float16** (in & out). Doküman "float32 I/O kalır" demişti → YANLIŞ, düzeltilecek.
- Hata (LiteRT, allocate anında, cihazla aynı runtime ailesi):
  `conv.cc input_type == kTfLiteFloat32 ... was not true. Node 1 (CONV_2D) failed to prepare.`
- fp32 native sorunsuz allocate.
- Çıkarım: `half=True` GPU-hedefli graf üretir; CONV_2D prepare float16 girişi reddeder
  → hem PC CPU hem cihazda yüklenmez. Delegate fp16 *compute* (fp32 ağırlık) sorun değil.

## Adımlar
- [x] 1. PC repro script: `tez_icin_veriler/reproduce_fp16_load_failure.py`
- [x] 2. PC log yakala: `tez_icin_veriler/fp16_load_failure.log` (3 fp16 düşer, 3 fp32 OK)
- [x] 3. Cihaz repro: pubspec'e fp16_640 + `ModelSize.s640`→fp16 + `_useGpuDelegate=true`
- [x] 4. `flutter build apk --release` → install → grant camera → launch
- [x] 5. Yakala: `fp16_device_failure_excerpt.txt` + `fp16_device_failure_logcat_full.txt` + `fp16_device_failure.png`
- [x] 6. `git checkout -- lib/yolo_detector.dart pubspec.yaml` + fp32 APK reinstall (telefon temiz)
- [x] 7. PERFORMANCE.md: I/O float16 düzeltmesi + "Cihaz-üstü yeniden üretim" notu (artık "kullanıcı teyidi" değil)
- [x] 8. Tez EN/IEEE alt-bölümü: `tez_icin_veriler/thesis_fp16_negative_result_EN.md`

## Review

**Sonuç:** fp16 yükleme hatası tez-kalitesinde, üç katmanlı kanıta dönüştürüldü ve negatif-bulgu olarak
belgelendi. Çalışan fp32 dağıtımı bozulmadı; çalışma ağacındaki tek değişiklik istenen dokümantasyon +
artifact'ler.

**Kanıt zinciri (kullanıcının gördüğü → kök neden → çapraz doğrulama):**
1. **Ekran görüntüsü** (`fp16_device_failure.png`): app `'Loading model...'`'da kalıcı takılı = kullanıcının raporu.
2. **Cihaz logcat** (`fp16_device_failure_excerpt.txt`): tek koşuda **GPU delegate + CPU fallback ikisi de**
   `CONV_2D failed to prepare` → `Bad state: failed precondition` (stack: allocateTensors→loadModel→_initialize).
3. **PC repro** (`reproduce_fp16_load_failure.py`/`.log`): aynı LiteRT runtime ailesi, deterministik;
   fp16 I/O=`float16` → 3 boyut düşer, fp32 I/O=`float32` → 3 kontrol OK.

**Kök neden:** `half=True` export graf-içi float16 aktivasyon üretir → CONV_2D prepare float16'yı reddeder
(delegate-bağımsız). Önceki "float32 I/O kalır" iddiası ölçümle çürütüldü (PERFORMANCE.md düzeltildi).

**Zarif tez çıkarımı:** fp16 *export formatı* = ölü; fp16 *delegate compute* (fp32 graf +
`isPrecisionLossAllowed`) = benimsenen kazanç. fp16-proxy (max|Δ|mAP=0.002) doğruluk-nötr olduğunu gösterir.

**Doğrulama:** `git status` → `lib/`+`pubspec.yaml` clean; fp32 APK (98.8MB) reinstall "Success";
build exit 0; cihaz çalışır.

**Açık (opsiyonel):** istenirse bu değişiklikler commit edilebilir (kullanıcı onayı bekliyor).

---

# Görev 2: native320 int8 başarısızlık kanıtı (tam int8 hikâyesi)

**Karar (kullanıcı):** İkisi de — (A) standart int8 cx/cy=0 çöküşü + (B) int8+NMS SELECT yükleme çökmesi; PC + cihaz.

## Olgular
- native320 `best.pt` + `dataset/data.yaml` mevcut → int8 export (kalibrasyonlu) yapılabilir.
- int8 iki ayrı ölüm modu (PERFORMANCE.md Faz 3A):
  - **Standart int8 (nms yok):** yüklenir AMA cx/cy → `0.0000` (onnx2tf DFL bug). Decode hatası.
  - **int8 + nms=True:** allocate'te `SELECT` op (`has_low_rank_input_condition was not true`) → yükleme çökmesi (fp16 gibi).

## Adımlar
- [x] 1. `export_test_native320_int8.py`: standart + nms int8 export, cx/cy çöküşü + SELECT crash
- [x] 2. Çalıştır → `int8_native320_failure.log` + `fig_native320_int8_diagnostic.jpg` + iki tflite
- [x] 3. Cihaz repro (int8+nms): s640→int8_nms + pubspec → build → 'Loading model' takıldı + logcat/screenshot
- [x] 4. Kodu geri al (`git checkout`) + temp asset sil + fp32 APK reinstall (telefon temiz)
- [x] 5. PERFORMANCE.md int8 bölümü: native320 cihaz/PC kanıtı + artifact'ler
- [x] 6. Tez EN metni: `thesis_fp16_negative_result_EN.md` int8 bölümüyle genişletildi (başlık → FP16+INT8)

## Review

**Sonuç:** int8 native320 için **iki bağımsız ölüm modu** tez-kalitesinde kanıtlandı; fp32 dağıtımı bozulmadı.

1. **Standart int8 (NMS yok) — yüklenir ama bozuk:** `allocate OK`, ama `cx=cy=0.00000` (DFL int8 bug);
   w/h + sınıf skorları sağlam → geçerli kutu yok. Kanıt: `int8_native320_failure.log` + figür.
2. **int8 + NMS — yüklenme çökmesi (fp16 gibi):** PC **ve** cihazda `select.cc ... SELECT failed to prepare`
   → `Bad state` → app 'Loading model' takıldı. Kanıt: `int8_nms_device_failure_excerpt.txt` + `.png`.

**Kanıt türü farkı (önemli):** fp16 = saf yükleme çökmesi; int8 = (A) sessiz decode hatası + (B) yükleme
çökmesi. Üçü birlikte → **tek dağıtılabilir yol native fp32**.

**Doğrulama:** `git status` lib/+pubspec clean; temp `assets/models/*int8_nms*` silindi; fp32 APK (98.8MB)
reinstall "Success"; build exit 0; telefon çalışır.

**Açık (opsiyonel):** tüm fp16+int8 dokümantasyon/artifact değişiklikleri commit edilebilir (kullanıcı onayı bekliyor).

---

# Görev 3: Telefon vs PC TFLite mAP batch-değerlendirme (deployment doğruluğu)

**Hedef:** Aynı 5 fp32 TFLite modelin mAP'ını PC vs telefon (GPU delegate AÇIK/KAPALI) olarak
40 görüntülük test setinde karşılaştır. Telefon yalnız prediction üretir; mAP PC'de eşleşmiş
pipeline (conf=0.001, iou=0.7) ile hesaplanır → Δ = saf deployment etkisi (GPU fp16 vs CPU fp32).

Modeller: native320/640/800, 800@320 (`cekmece_v4_fp32_320`), 800@640 (`cekmece_v4_fp32_640`).

## Flutter
- [x] `dataset/images/test/*` → `assets/eval_images/` (40 jpg)
- [ ] pubspec.yaml: `image` dep + assets (eval_images + 2 ek fp32 tflite)
- [ ] `flutter pub get`
- [ ] image_utils.dart: `fillFloat32InputBufferFromRgb` (letterbox, rotasyon yok)
- [ ] yolo_detector.dart: `loadModelFromAsset(path,size,{useGpu})` + `detectDecoded(rgb,w,h)`
- [ ] eval_runner.dart (yeni): EvalModel listesi + `runEval(...)` → JSON yaz
- [ ] eval_screen.dart (yeni): debug UI, kendi detector'ı, GPU aç/kapa, çalıştır
- [ ] home_screen.dart: TEST chip → EvalScreen (önce kamerayı durdur + detector kapat)
- [ ] `flutter analyze` temiz

## PC
- [x] eval_phone_vs_pc.py: Dart-eşleşmiş letterbox+NMS, run_tflite, load_gt, compute_map, tablo
- [x] TFLite interpreter: ai_edge_litert 2.1.5 (LiteRT, telefonla aynı runtime ailesi)
- [x] PC mAP üretildi: native320=0.336 / 640=0.599 / 800=0.678 / 800@320=0.244 / 800@640=0.630 (mAP50)
- [x] Harness doğrulandı: custom vs Ultralytics 800@640 → mAP50-95 0.4060 vs 0.4082 (Δ0.002)

## Flutter (devam)
- [x] pubspec, image_utils, yolo_detector, eval_runner, eval_screen, home_screen
- [x] flutter analyze temiz (sadece pre-existing info lint); flutter build apk OK (217MB)
- [x] adb install + camera grant + launch

## Verify
- [x] Telefonda 5 × iki backend → 10 JSON; GPU koşularında gpu_active=true ✓
- [x] adb pull eval_out → flutter_app/tez_icin_veriler/phone_eval (git-tracked dir)
- [x] python eval_phone_vs_pc.py → tablo (md+csv)
- [x] Sanity: custom mAP ≈ Ultralytics (800@640 mAP50-95 0.4060 vs 0.4082, Δ0.002)
- [x] PERFORMANCE.md Faz 6 + K1 kapanışı yazıldı

## Review

**Sonuç:** Cihaz-üstü dağıtım doğruluğu tez-kalitesinde ölçüldü. Telefon (GPU-fp16 + CPU-fp32) mAP'ı
PC TFLite fp32 ile **|Δ| ≤ 0.008 mAP50, ≤ 0.0044 mAP50-95** → gürültü düzeyinde eşit. **K1 kapandı**
(önceden yalnız fp16-proxy ile dolaylı ölçülmüştü; artık gerçek Adreno delegate'inde doğrudan).

**Ana bulgular:**
- GPU delegate fp16 **doğruluk-nötr** (max sapma −0.0050) ve **~1.8× hızlı** (1.66–2.09×) → bedava hızlanma.
- GPU ≈ CPU doğrulukta → hız için GPU seçmek doğruluktan ödün değil.
- Model sıralaması cihazda korunur; native320 (0.336) > 800@320 (0.244) → native-çözünürlük tezi cihazda da geçerli.

**Metodoloji (özet):** telefon yalnız prediction üretir; mAP PC'de GT'ye karşı hesaplanır. İki tarafta
**birebir eşleşmiş** letterbox + NMS + conf=0.001/iou=0.7/max_det=300 → Δ = saf runtime/dağıtım etkisi.
mAP harness'i Ultralytics'e karşı doğrulandı.

**Doğrulama:** flutter analyze temiz, build+install Success, 10/10 JSON üretildi, harness Ultralytics ile
±0.002 uyumlu, artifact'ler git-tracked `flutter_app/tez_icin_veriler/`'a konsolide edildi.

**Açık (opsiyonel):** kod + doküman + artifact değişiklikleri commit edilebilir (kullanıcı onayı bekliyor).

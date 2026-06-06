import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'detection.dart';
import 'image_utils.dart';

class DetectionFrame {
  final List<Detection> detections;
  final int frameWidth;
  final int frameHeight;
  const DetectionFrame(this.detections, this.frameWidth, this.frameHeight);
}

enum ModelSize {
  // Native fp32: each size is the model TRAINED at that resolution
  // (cekmece_v4_141epoch_{N}imgsz), not the 800-model downscaled. PERFORMANCE.md
  // shows native training wins clearly at 320 (small-object recall) and ties at
  // 512/640. I/O is float32 → Float32List buffer path unchanged; the GPU delegate
  // still runs them as fp16 internally (isPrecisionLossAllowed).
  // NB: the earlier fp16 .tflite exports were broken — float16 I/O, no DEQUANTIZE —
  // so they failed to allocate on both GPU and CPU (app hung on load). See PERFORMANCE.md.
  s320(320, 'assets/models/cekmece_v4_native320_fp32.tflite', '320'),
  s640(640, 'assets/models/cekmece_v4_native640_fp32.tflite', '640'),
  s800(800, 'assets/models/cekmece_v4_native800_fp32.tflite', '800');

  final int pixels;
  final String assetPath;
  final String label;
  const ModelSize(this.pixels, this.assetPath, this.label);

  /// Largest size whose predicted pure-inference time stays within [budgetMs],
  /// extrapolating a single measured [refMs] at [refSize] by the cost∝size²
  /// law validated in PERFORMANCE.md. Falls back to the smallest size if even
  /// that exceeds the budget.
  static ModelSize pickForBudget({
    required double refMs,
    required int refSize,
    double budgetMs = 70,
  }) {
    var best = ModelSize.s320; // values are ascending in pixels
    for (final s in ModelSize.values) {
      final pred = refMs * (s.pixels * s.pixels) / (refSize * refSize);
      if (pred <= budgetMs) best = s;
    }
    return best;
  }
}

class YoloDetector {
  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
  GpuDelegateV2? _gpuDelegate;
  List<String> _labels = [];
  int _inputSize = 320;
  String _currentModel = '';
  bool _gpuActive = false;

  Float32List? _inputBuffer;
  Float32List? _outputBuffer;
  int _outChannels = 0;
  int _outAnchors = 0;

  // --- Faz 3A teşhis: per-stage timing (EMA, ms) ---
  // Amaç: 640'taki ~250ms/kare'nin nereye gittiğini kanıtla. inference baskınsa
  // → 512 model; preprocess+copy baskınsa → pipeline/worker-isolate.
  double _emaPre = 0, _emaInf = 0, _emaParse = 0;
  // Ana-isolate'te ölçülen saf inference (kopya yükü HARİÇ). _emaInf ile farkı
  // = IsolateInterpreter'ın kare başına buffer-kopya maliyeti.
  double pureInferenceMs = 0;
  int _perfFrame = 0;

  static const String labelsPath = 'assets/models/labels.txt';

  int get inputSize => _inputSize;
  String get currentModel => _currentModel;
  bool get gpuActive => _gpuActive;
  List<String> get labels => _labels;
  bool get isReady => _isolateInterpreter != null;
  double get preMs => _emaPre;
  double get infMs => _emaInf;
  double get parseMs => _emaParse;

  Future<void> loadModel(ModelSize size) async {
    await close();

    _inputSize = size.pixels;
    _currentModel = size.label;

    // Try GPU delegate first (Adreno on Android via OpenCL).
    // Fall back to multi-threaded CPU + XNNPACK if GPU init fails.
    // Constants from TFLite C API (not re-exported by tflite_flutter):
    //   inferencePreference: 1 = SUSTAINED_SPEED (vs FAST_SINGLE_ANSWER=0)
    //   inferencePriority1:  2 = MIN_LATENCY    (vs MAX_PRECISION=1)
    Interpreter? interp;
    if (Platform.isAndroid) {
      try {
        final gpu = GpuDelegateV2(
          options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: true, // fp16 inference on GPU = faster
            inferencePreference: 1,
            inferencePriority1: 2,
          ),
        );
        final opts = InterpreterOptions()..addDelegate(gpu);
        interp = await Interpreter.fromAsset(size.assetPath, options: opts);
        _gpuDelegate = gpu;
        _gpuActive = true;
        debugPrint('YoloDetector: GPU delegate active');
      } catch (e) {
        debugPrint('YoloDetector: GPU delegate failed ($e) — falling back to CPU');
        _gpuDelegate = null;
        _gpuActive = false;
      }
    }
    if (interp == null) {
      final opts = InterpreterOptions()..threads = 4;
      interp = await Interpreter.fromAsset(size.assetPath, options: opts);
    }
    _interpreter = interp;

    final inTensor = _interpreter!.getInputTensor(0);
    final outTensor = _interpreter!.getOutputTensor(0);
    // fp16 export must still expose float32 I/O — otherwise the Float32List
    // buffer packing silently corrupts (B1: never let detection fail quietly).
    if (inTensor.type != TensorType.float32 ||
        outTensor.type != TensorType.float32) {
      await close();
      throw StateError(
        'YoloDetector: float32 I/O bekleniyordu, '
        'in=${inTensor.type} out=${outTensor.type} — '
        'model export tensör tipini değiştirdiyse buffer kodu güncellenmeli.',
      );
    }
    final inShape = inTensor.shape; // [1,H,W,3]
    final outShape = outTensor.shape; // [1, channels, anchors]
    _outChannels = outShape[1];
    _outAnchors = outShape[2];

    _inputBuffer = Float32List(inShape[0] * inShape[1] * inShape[2] * inShape[3]);
    _outputBuffer = Float32List(outShape[0] * outShape[1] * outShape[2]);

    // Warmup + saf-inference benchmark (ana-isolate, kamera akışı başlamadan,
    // isolate'e sarmadan önce). İlk GPU run kernel'leri derler (çok yavaş) —
    // warmup bunu yutar. Buradaki ölçüm kopya yükü İÇERMEZ; detect()'teki
    // inf+copy ile farkı IsolateInterpreter'ın kare başına kopya maliyetidir.
    try {
      for (int i = 0; i < 3; i++) {
        _interpreter!
            .runForMultipleInputs([_inputBuffer!.buffer], {0: _outputBuffer!.buffer});
      }
      const bench = 12;
      final sw = Stopwatch()..start();
      for (int i = 0; i < bench; i++) {
        _interpreter!
            .runForMultipleInputs([_inputBuffer!.buffer], {0: _outputBuffer!.buffer});
      }
      pureInferenceMs = sw.elapsedMicroseconds / 1000.0 / bench;
      debugPrint('YoloDetector: pure inference (ana-isolate, kopya hariç) = '
          '${pureInferenceMs.toStringAsFixed(1)} ms/kare [$_currentModel gpu=$_gpuActive]');
    } catch (e) {
      debugPrint('YoloDetector: warmup/bench atlandı: $e');
    }

    // Wrap the interpreter so inference runs in a background isolate.
    // The interpreter address is shared; tensors stay on the native side.
    // Only the input/output ByteBuffers cross the isolate boundary per frame.
    _isolateInterpreter = await IsolateInterpreter.create(
      address: _interpreter!.address,
    );

    final labelData = await rootBundle.loadString(labelsPath);
    _labels = labelData.split('\n').where((s) => s.trim().isNotEmpty).toList();
  }

  Future<void> close() async {
    await _isolateInterpreter?.close();
    _isolateInterpreter = null;
    _interpreter?.close();
    _interpreter = null;
    _gpuDelegate?.delete();
    _gpuDelegate = null;
    _gpuActive = false;
  }

  Future<DetectionFrame> detect(
    CameraImage image, {
    int rotationDeg = 90,
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) async {
    final interp = _isolateInterpreter;
    if (interp == null || _inputBuffer == null || _outputBuffer == null) {
      return const DetectionFrame([], 0, 0);
    }

    final sw = Stopwatch()..start();

    final pp = fillFloat32InputBufferFromCameraImage(
      image: image,
      targetSize: _inputSize,
      buffer: _inputBuffer!,
      rotationDeg: rotationDeg,
    );
    final preUs = sw.elapsedMicroseconds;
    sw.reset();

    // IsolateInterpreter runs inference on a background isolate; UI thread
    // stays free for camera frames and rendering during the 30-80ms call.
    // ByteBuffer fast-path: raw bytes, no nested-list walk.
    // Bu süre = isolate'e kopya + native inference + isolate'ten kopya.
    await interp.runForMultipleInputs(
      [_inputBuffer!.buffer],
      {0: _outputBuffer!.buffer},
    );
    final infUs = sw.elapsedMicroseconds;
    sw.reset();

    final detections = _parseOutput(
      _outputBuffer!,
      _outChannels,
      _outAnchors,
      pp,
      _inputSize,
      confThreshold,
      iouThreshold,
    );
    final parseUs = sw.elapsedMicroseconds;

    _recordTiming(preUs, infUs, parseUs);

    return DetectionFrame(detections, pp.rotatedWidth, pp.rotatedHeight);
  }

  void _recordTiming(int preUs, int infUs, int parseUs) {
    double ema(double prev, double v) => prev == 0 ? v : prev * 0.8 + v * 0.2;
    _emaPre = ema(_emaPre, preUs / 1000.0);
    _emaInf = ema(_emaInf, infUs / 1000.0);
    _emaParse = ema(_emaParse, parseUs / 1000.0);
    if (kDebugMode && ++_perfFrame % 30 == 0) {
      final total = _emaPre + _emaInf + _emaParse;
      final copy = _emaInf - pureInferenceMs; // kopya yükü tahmini
      debugPrint(
        'PERF[$_currentModel gpu=$_gpuActive] '
        'pre=${_emaPre.toStringAsFixed(1)} '
        'inf+copy=${_emaInf.toStringAsFixed(1)} '
        '(pure=${pureInferenceMs.toStringAsFixed(1)} copy≈${copy.toStringAsFixed(1)}) '
        'parse=${_emaParse.toStringAsFixed(1)} '
        'total=${total.toStringAsFixed(1)}ms ~${(1000 / total).toStringAsFixed(1)}fps',
      );
    }
  }

  List<Detection> _parseOutput(
    Float32List out,
    int channels,
    int anchors,
    PreprocessResult pp,
    int inputSize,
    double confThreshold,
    double iouThreshold,
  ) {
    final int numClasses = channels - 4;
    final double invScale = 1.0 / pp.scale;
    final double maxX = pp.rotatedWidth.toDouble();
    final double maxY = pp.rotatedHeight.toDouble();
    // YOLOv8 Ultralytics TFLite export emits normalized [0..1] bbox coords.
    final double inSize = inputSize.toDouble();

    final List<_Candidate> candidates = [];
    for (int a = 0; a < anchors; a++) {
      double maxScore = 0.0;
      int maxClassId = 0;
      for (int c = 0; c < numClasses; c++) {
        final s = out[(4 + c) * anchors + a];
        if (s > maxScore) {
          maxScore = s;
          maxClassId = c;
        }
      }
      if (maxScore < confThreshold) continue;

      final cx = out[a] * inSize;
      final cy = out[anchors + a] * inSize;
      final w = out[2 * anchors + a] * inSize;
      final h = out[3 * anchors + a] * inSize;

      double x1 = (cx - w * 0.5 - pp.padX) * invScale;
      double y1 = (cy - h * 0.5 - pp.padY) * invScale;
      double x2 = (cx + w * 0.5 - pp.padX) * invScale;
      double y2 = (cy + h * 0.5 - pp.padY) * invScale;

      if (x1 < 0) x1 = 0; else if (x1 > maxX) x1 = maxX;
      if (y1 < 0) y1 = 0; else if (y1 > maxY) y1 = maxY;
      if (x2 < 0) x2 = 0; else if (x2 > maxX) x2 = maxX;
      if (y2 < 0) y2 = 0; else if (y2 > maxY) y2 = maxY;

      candidates.add(_Candidate(x1, y1, x2, y2, maxScore, maxClassId));
    }

    return _nms(candidates, iouThreshold)
        .map((c) => Detection(
              bbox: Rect.fromLTRB(c.x1, c.y1, c.x2, c.y2),
              classId: c.classId,
              className: c.classId < _labels.length
                  ? _labels[c.classId]
                  : 'cls_${c.classId}',
              confidence: c.score,
            ))
        .toList();
  }

  List<_Candidate> _nms(List<_Candidate> candidates, double iouThreshold) {
    final grouped = <int, List<_Candidate>>{};
    for (final c in candidates) {
      grouped.putIfAbsent(c.classId, () => []).add(c);
    }
    final List<_Candidate> keep = [];
    grouped.forEach((_, list) {
      list.sort((a, b) => b.score.compareTo(a.score));
      final suppressed = List.filled(list.length, false);
      for (int i = 0; i < list.length; i++) {
        if (suppressed[i]) continue;
        keep.add(list[i]);
        for (int j = i + 1; j < list.length; j++) {
          if (suppressed[j]) continue;
          if (_iou(list[i], list[j]) > iouThreshold) suppressed[j] = true;
        }
      }
    });
    return keep;
  }

  double _iou(_Candidate a, _Candidate b) {
    final ix1 = math.max(a.x1, b.x1);
    final iy1 = math.max(a.y1, b.y1);
    final ix2 = math.min(a.x2, b.x2);
    final iy2 = math.min(a.y2, b.y2);
    final iw = ix2 - ix1;
    final ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0.0;
    final inter = iw * ih;
    final union = (a.x2 - a.x1) * (a.y2 - a.y1) +
        (b.x2 - b.x1) * (b.y2 - b.y1) -
        inter;
    return union <= 0 ? 0.0 : inter / union;
  }
}

class _Candidate {
  final double x1, y1, x2, y2, score;
  final int classId;
  _Candidate(this.x1, this.y1, this.x2, this.y2, this.score, this.classId);
}

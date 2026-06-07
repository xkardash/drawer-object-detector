import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'detection.dart';
import 'detection_painter.dart';
import 'latency_logger.dart';
import 'tracker.dart';
import 'yolo_detector.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _OverlayFrame {
  final List<Detection> detections;
  final Size imageSize;
  const _OverlayFrame(this.detections, this.imageSize);
}

class _HomeScreenState extends State<HomeScreen>
    with WidgetsBindingObserver, SingleTickerProviderStateMixin {
  CameraController? _camera;
  List<CameraDescription>? _cameras;
  final YoloDetector _detector = YoloDetector();

  // Inference rate'i (6 FPS) ekran rate'inden ayıran tracker + ticker.
  // Ticker her ekran karesinde track'leri tahmini hızla kaydırıp overlay'i
  // tazeler → telefon gezdirilirken bbox akıcı görünür.
  final BoxTracker _tracker = BoxTracker();
  Ticker? _ticker;
  Size _trackImageSize = const Size(640, 640);
  bool _lastOverlayEmpty = true;

  // Per-frame state lives in notifiers — only the overlay + fps chip rebuild
  // on each detection, not the entire Stack (which would also reconcile
  // CameraPreview every frame).
  final ValueNotifier<_OverlayFrame> _overlay = ValueNotifier(
    const _OverlayFrame([], Size(640, 640)),
  );
  final ValueNotifier<double> _fps = ValueNotifier(0.0);
  // Faz 3A teşhis: per-stage timing okuması (ekranda, logcat gerekmeden).
  final ValueNotifier<String> _perf = ValueNotifier('');

  // K4: cihaz-üstü gecikme + termal/sürdürülen FPS kaydı (CSV → app harici dizin).
  final LatencyLogger _logger = LatencyLogger();
  final ValueNotifier<String> _logStatus = ValueNotifier('');

  bool _processing = false;
  bool _switching = false; // model reload in progress → drop frames
  ModelSize _modelSize = ModelSize.s640; // boot model (manual) + auto fallback
  double _confThreshold = 0.30;
  DateTime _lastFrameTs = DateTime.now();
  String _status = 'Initializing...';

  // Adaptive "Auto" model selection (balance: usable FPS + reasonable range).
  // A single load-time benchmark (pureInferenceMs) drives a cost∝size² estimate
  // (PERFORMANCE.md) to pick the largest size within the FPS budget; one runtime
  // correction claims/returns headroom. No continuous switching — a model reload
  // recompiles GPU kernels (~1–3s, no cache API), so oscillation is avoided.
  // Default OFF for the CPU-only fp32 latency study: the 70ms budget is
  // GPU-calibrated (would always pick 320 on CPU) and a mid-run switch would
  // taint a per-size recording. Boot into a fixed size; the "Oto" chip still
  // re-enables it on demand.
  bool _auto = false;
  bool _autoSettled = false;
  int _autoFrames = 0;
  static const double _autoBudgetMs = 70; // ~12 FPS pure-inference budget
  static const double _autoLowFps = 9;
  static const double _autoHighFps = 18;
  static const int _autoEvalFrames = 40;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    // Ekran-rate overlay sürücüsü: her vsync'te tahmini pozisyonları çiz.
    _ticker = createTicker(_onTick)..start();
    _initialize();
  }

  void _onTick(Duration _) {
    if (!mounted) return;
    final preds = _tracker.predict(DateTime.now());
    // Hiçbir şey yokken boşa repaint etme (boş→boş geçişi atla).
    if (preds.isEmpty && _lastOverlayEmpty) return;
    _lastOverlayEmpty = preds.isEmpty;
    _overlay.value = _OverlayFrame(preds, _trackImageSize);
  }

  Future<void> _initialize() async {
    setState(() => _status = 'Requesting camera permission...');
    final granted = await Permission.camera.request();
    if (!granted.isGranted) {
      setState(() => _status = 'Camera permission denied');
      return;
    }

    setState(() => _status = 'Loading model...');
    if (_auto) {
      await _loadAuto();
    } else {
      await _detector.loadModel(_modelSize);
    }

    setState(() => _status = 'Starting camera...');
    _cameras = await availableCameras();
    final back = _cameras!.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras!.first,
    );

    _camera = CameraController(
      back,
      ResolutionPreset.medium, // ~720p — modele daha temiz detay, ~2× menzil
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    await _camera!.initialize();
    if (!mounted) return;

    await _camera!.startImageStream(_onFrame);
    setState(() => _status = 'Ready');
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final cam = _camera;
    if (cam == null || !cam.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      cam.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initialize();
    }
  }

  void _onFrame(CameraImage image) async {
    if (_processing || _switching) return;
    _processing = true;
    try {
      final frame = await _detector.detect(
        image,
        rotationDeg: 90,
        confThreshold: _confThreshold,
      );
      if (!mounted) return;

      final now = DateTime.now();
      final dt = now.difference(_lastFrameTs).inMilliseconds;
      _lastFrameTs = now;
      final instantFps = dt > 0 ? 1000.0 / dt : 0.0;
      final prev = _fps.value;
      _fps.value = prev == 0 ? instantFps : (prev * 0.7 + instantFps * 0.3);
      _perf.value = 'pre ${_detector.preMs.toStringAsFixed(0)} · '
          'inf ${_detector.infMs.toStringAsFixed(0)} · '
          'parse ${_detector.parseMs.toStringAsFixed(0)} ms · '
          'pure ${_detector.pureInferenceMs.toStringAsFixed(0)}';

      // K4: kayıt aktifse bu karenin gecikme metriklerini biriktir.
      if (_logger.active) {
        _logger.add(
          dtMs: dt,
          totalMs: _detector.preMs + _detector.infMs + _detector.parseMs,
          pureMs: _detector.pureInferenceMs,
          fps: _fps.value,
          model: _detector.inputSize,
        );
        _logStatus.value = '● REC ${_logger.elapsedSec}s · ${_logger.count}f';
      }

      _maybeAutoCorrect();

      if (kDebugMode && frame.detections.isNotEmpty) {
        final d = frame.detections.first;
        debugPrint(
          'DET: ${d.className} ${d.confidence.toStringAsFixed(2)} '
          'bbox=(${d.bbox.left.toStringAsFixed(0)},${d.bbox.top.toStringAsFixed(0)},'
          '${d.bbox.right.toStringAsFixed(0)},${d.bbox.bottom.toStringAsFixed(0)}) '
          'frame=${frame.frameWidth}x${frame.frameHeight}',
        );
      }

      // Overlay'i doğrudan set etme — tracker'a besle; ekran-rate ticker
      // (_onTick) tahmini pozisyonlarla overlay'i sürer.
      _trackImageSize =
          Size(frame.frameWidth.toDouble(), frame.frameHeight.toDouble());
      _tracker.update(frame.detections, now);
    } catch (e, st) {
      debugPrint('Detection error: $e\n$st');
    } finally {
      _processing = false;
    }
  }

  /// Block until the in-flight detect() (if any) completes, so the interpreter
  /// is never closed mid-inference. close() frees the native interpreter while
  /// the background isolate may still be running it → use-after-free → crash.
  /// The window is large on CPU (~1–2 s/frame at 800px), so a mid-inference
  /// model switch reliably crashed; draining first fixes it.
  Future<void> _drainInFlight() async {
    while (_processing) {
      await Future<void>.delayed(const Duration(milliseconds: 20));
    }
  }

  /// Core model reload: stop stream → drain in-flight → close → load → restart.
  /// Guarded by [_switching] so no NEW frame runs detect() against a closing
  /// interpreter. Does not touch [_auto]/[_status] so callers decide semantics.
  Future<void> _reloadModel(ModelSize size) async {
    _switching = true;
    _fps.value = 0;
    _tracker.clear();
    _overlay.value = const _OverlayFrame([], Size(640, 640));
    _lastOverlayEmpty = true;
    await _camera?.stopImageStream();
    await _drainInFlight();
    await _detector.close();
    await _detector.loadModel(size);
    _modelSize = size;
    if (mounted) await _camera!.startImageStream(_onFrame);
    _switching = false;
  }

  /// Probe with the cheapest model to get a reliable per-size cost, then load
  /// the largest size within the FPS budget. Assumes the stream is stopped and
  /// the detector closed (or fresh) — used by [_initialize] and [_enableAuto].
  Future<void> _loadAuto() async {
    await _detector.loadModel(ModelSize.s320);
    final picked = _autoPick();
    if (picked != ModelSize.s320) {
      await _detector.close();
      await _detector.loadModel(picked);
    }
    _modelSize = picked;
    _autoSettled = false;
    _autoFrames = 0;
  }

  /// Maps the load-time pure-inference benchmark to a target size. Extrapolation
  /// from the 320 probe is slightly conservative for larger sizes (fixed GPU
  /// overhead isn't size²), biasing toward smoothness; [_maybeAutoCorrect] then
  /// reclaims headroom if real FPS is high.
  ModelSize _autoPick() {
    final ref = _detector.pureInferenceMs;
    if (ref <= 0) return _modelSize; // bench unavailable → keep fallback
    return ModelSize.pickForBudget(
      refMs: ref,
      refSize: _detector.inputSize,
      budgetMs: _autoBudgetMs,
    );
  }

  /// One-shot runtime correction: after the FPS EMA stabilizes, nudge one step
  /// if measured FPS is clearly out of band, then settle (no oscillation).
  void _maybeAutoCorrect() {
    if (!_auto || _autoSettled || _switching) return;
    if (++_autoFrames < _autoEvalFrames) return;
    _autoSettled = true;
    final fps = _fps.value;
    final idx = _modelSize.index;
    ModelSize? target;
    if (fps < _autoLowFps && idx > 0) {
      target = ModelSize.values[idx - 1];
    } else if (fps > _autoHighFps && idx < ModelSize.values.length - 1) {
      target = ModelSize.values[idx + 1];
    }
    if (target != null && target != _modelSize) {
      final t = target;
      scheduleMicrotask(() {
        if (mounted && _auto) {
          _reloadModel(t).then((_) {
            if (mounted) setState(() {});
          });
        }
      });
    }
  }

  /// User tapped a fixed size → leave Auto, switch to that size.
  Future<void> _selectModel(ModelSize size) async {
    if (!_auto && size == _modelSize) return;
    setState(() {
      _auto = false;
      _status = 'Switching model...';
    });
    _autoSettled = true; // manual override stops auto-correction
    await _reloadModel(size);
    if (mounted) setState(() => _status = 'Ready');
  }

  /// User tapped "Oto" → re-enable Auto and re-benchmark.
  Future<void> _enableAuto() async {
    if (_auto) return;
    setState(() {
      _auto = true;
      _status = 'Auto: ölçülüyor...';
    });
    _switching = true;
    _fps.value = 0;
    _tracker.clear();
    _overlay.value = const _OverlayFrame([], Size(640, 640));
    _lastOverlayEmpty = true;
    await _camera?.stopImageStream();
    await _drainInFlight();
    await _detector.close();
    await _loadAuto();
    if (mounted) await _camera!.startImageStream(_onFrame);
    _switching = false;
    if (mounted) setState(() => _status = 'Ready');
  }

  /// K4: start/stop on-device latency+thermal recording. On stop, writes the CSV
  /// and shows its path. Tip: turn Auto OFF and pick a fixed size before a run so
  /// the thermal curve reflects one model (model_px is logged regardless).
  Future<void> _toggleLog() async {
    if (_logger.active) {
      final path = await _logger.stop();
      _logStatus.value = '';
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: Text('Kayıt bitti: $path'),
          duration: const Duration(seconds: 10),
        ));
      }
    } else {
      _logger.start();
      _logStatus.value = '● REC 0s · 0f';
    }
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ticker?.dispose();
    _camera?.dispose();
    _detector.close();
    _overlay.dispose();
    _fps.dispose();
    _perf.dispose();
    _logStatus.dispose();
    SystemChrome.setPreferredOrientations(DeviceOrientation.values);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _camera;
    if (cam == null || !cam.value.isInitialized) {
      return Scaffold(
        backgroundColor: const Color(0xFF0E0E12),
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const CircularProgressIndicator(color: Color(0xFF4D96FF)),
              const SizedBox(height: 16),
              Text(_status, style: const TextStyle(color: Colors.white70)),
            ],
          ),
        ),
      );
    }

    final mediaSize = MediaQuery.of(context).size;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview (fill, may crop)
          ClipRect(
            child: OverflowBox(
              alignment: Alignment.center,
              child: FittedBox(
                fit: BoxFit.cover,
                child: SizedBox(
                  width: cam.value.previewSize?.height ?? mediaSize.width,
                  height: cam.value.previewSize?.width ?? mediaSize.height,
                  child: CameraPreview(cam),
                ),
              ),
            ),
          ),

          // Bbox overlay — repaints only when detections change, isolated
          // from the rest of the Stack by RepaintBoundary so camera preview
          // and chrome don't re-layer.
          Positioned.fill(
            child: RepaintBoundary(
              child: ValueListenableBuilder<_OverlayFrame>(
                valueListenable: _overlay,
                builder: (_, frame, __) => CustomPaint(
                  painter: DetectionPainter(
                    detections: frame.detections,
                    imageSize: frame.imageSize,
                  ),
                ),
              ),
            ),
          ),

          // Top bar
          _buildTopBar(),

          // Bottom controls
          _buildBottomControls(),
        ],
      ),
    );
  }

  Widget _buildTopBar() {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            _statusChip(),
            const SizedBox(width: 8),
            _recChip(),
            const Spacer(),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                _fpsChip(),
                const SizedBox(height: 6),
                _perfChip(),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _statusChip() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.55),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white24),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: const BoxDecoration(
              color: Color(0xFF6BCB77),
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Text(
            '${_detector.currentModel}px',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _fpsChip() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.55),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white24),
      ),
      child: ValueListenableBuilder<double>(
        valueListenable: _fps,
        builder: (_, fps, __) => Text(
          '${fps.toStringAsFixed(1)} FPS',
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w600,
            fontFeatures: [FontFeature.tabularFigures()],
          ),
        ),
      ),
    );
  }

  /// K4 record toggle — tap to start/stop latency+thermal CSV logging.
  Widget _recChip() {
    final on = _logger.active;
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: _toggleLog,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.55),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: on ? const Color(0xFFFF6B6B) : Colors.white24,
          ),
        ),
        child: ValueListenableBuilder<String>(
          valueListenable: _logStatus,
          builder: (_, s, __) => Text(
            s.isEmpty ? '⦿ Kayıt' : s,
            style: TextStyle(
              color: on ? const Color(0xFFFF6B6B) : Colors.white70,
              fontSize: 11,
              fontWeight: FontWeight.w600,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ),
      ),
    );
  }

  Widget _perfChip() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.55),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white24),
      ),
      child: ValueListenableBuilder<String>(
        valueListenable: _perf,
        builder: (_, perf, __) => Text(
          perf.isEmpty ? '— ms' : perf,
          style: const TextStyle(
            color: Colors.white70,
            fontSize: 10,
            fontFeatures: [FontFeature.tabularFigures()],
          ),
        ),
      ),
    );
  }

  Widget _segChip({
    required String label,
    required bool selected,
    required VoidCallback onTap,
  }) {
    return Expanded(
      child: GestureDetector(
        behavior: HitTestBehavior.opaque,
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 160),
          padding: const EdgeInsets.symmetric(vertical: 7),
          decoration: BoxDecoration(
            color: selected ? const Color(0xFF4D96FF) : Colors.transparent,
            borderRadius: BorderRadius.circular(8),
          ),
          alignment: Alignment.center,
          child: Text(
            label,
            style: TextStyle(
              color: selected ? Colors.white : Colors.white70,
              fontSize: 12,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBottomControls() {
    return Align(
      alignment: Alignment.bottomCenter,
      child: SafeArea(
        child: Container(
          margin: const EdgeInsets.all(16),
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.65),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white24),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Detection count
              Row(
                children: [
                  const Icon(Icons.center_focus_strong, color: Color(0xFF4D96FF), size: 18),
                  const SizedBox(width: 8),
                  ValueListenableBuilder<_OverlayFrame>(
                    valueListenable: _overlay,
                    builder: (_, frame, __) => Text(
                      '${frame.detections.length} nesne tespit edildi',
                      style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              // Confidence slider
              Row(
                children: [
                  const SizedBox(
                    width: 80,
                    child: Text('Konfidans', style: TextStyle(color: Colors.white70, fontSize: 12)),
                  ),
                  Expanded(
                    child: SliderTheme(
                      data: SliderTheme.of(context).copyWith(
                        trackHeight: 3,
                        thumbColor: const Color(0xFF4D96FF),
                        activeTrackColor: const Color(0xFF4D96FF),
                        inactiveTrackColor: Colors.white24,
                        thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
                        overlayShape: const RoundSliderOverlayShape(overlayRadius: 14),
                      ),
                      child: Slider(
                        min: 0.1,
                        max: 0.9,
                        value: _confThreshold,
                        onChanged: (v) => setState(() => _confThreshold = v),
                      ),
                    ),
                  ),
                  SizedBox(
                    width: 40,
                    child: Text(
                      _confThreshold.toStringAsFixed(2),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),
              // Model size segmented selector
              Row(
                children: [
                  const SizedBox(
                    width: 80,
                    child: Text('Model', style: TextStyle(color: Colors.white70, fontSize: 12)),
                  ),
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(color: Colors.white12),
                      ),
                      padding: const EdgeInsets.all(3),
                      child: Row(
                        children: [
                          _segChip(
                            label: 'Oto',
                            selected: _auto,
                            onTap: _enableAuto,
                          ),
                          ...ModelSize.values.map((s) => _segChip(
                                label: s.label,
                                selected: !_auto && s == _modelSize,
                                onTap: () => _selectModel(s),
                              )),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'detection.dart';
import 'detection_painter.dart';
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

  bool _processing = false;
  ModelSize _modelSize = ModelSize.s640;
  double _confThreshold = 0.30;
  DateTime _lastFrameTs = DateTime.now();
  String _status = 'Initializing...';

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
    await _detector.loadModel(_modelSize);

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
    if (_processing) return;
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

      if (frame.detections.isNotEmpty) {
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

  Future<void> _selectModel(ModelSize size) async {
    if (size == _modelSize) return;
    setState(() {
      _modelSize = size;
      _status = 'Switching model...';
    });
    _fps.value = 0;
    _tracker.clear();
    _overlay.value = const _OverlayFrame([], Size(640, 640));
    _lastOverlayEmpty = true;
    await _camera?.stopImageStream();
    await _detector.close();
    await _detector.loadModel(size);
    await _camera!.startImageStream(_onFrame);
    setState(() => _status = 'Ready');
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
                        children: ModelSize.values.map((s) {
                          final selected = s == _modelSize;
                          return Expanded(
                            child: GestureDetector(
                              behavior: HitTestBehavior.opaque,
                              onTap: () => _selectModel(s),
                              child: AnimatedContainer(
                                duration: const Duration(milliseconds: 160),
                                padding: const EdgeInsets.symmetric(vertical: 7),
                                decoration: BoxDecoration(
                                  color: selected
                                      ? const Color(0xFF4D96FF)
                                      : Colors.transparent,
                                  borderRadius: BorderRadius.circular(8),
                                ),
                                alignment: Alignment.center,
                                child: Text(
                                  s.label,
                                  style: TextStyle(
                                    color: selected ? Colors.white : Colors.white70,
                                    fontSize: 12,
                                    fontWeight: selected
                                        ? FontWeight.w700
                                        : FontWeight.w500,
                                    fontFeatures: const [FontFeature.tabularFigures()],
                                  ),
                                ),
                              ),
                            ),
                          );
                        }).toList(),
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

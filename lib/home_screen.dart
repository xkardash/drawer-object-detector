import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'detection.dart';
import 'detection_painter.dart';
import 'yolo_detector.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  CameraController? _camera;
  List<CameraDescription>? _cameras;
  final YoloDetector _detector = YoloDetector();

  List<Detection> _detections = [];
  Size _imageSize = const Size(640, 640);
  bool _processing = false;
  bool _highQuality = false; // Redmi Note 11 için 320 default
  double _confThreshold = 0.30;
  double _fps = 0.0;
  DateTime _lastFrameTs = DateTime.now();
  String _status = 'Initializing...';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
    _initialize();
  }

  Future<void> _initialize() async {
    setState(() => _status = 'Requesting camera permission...');
    final granted = await Permission.camera.request();
    if (!granted.isGranted) {
      setState(() => _status = 'Camera permission denied');
      return;
    }

    setState(() => _status = 'Loading model...');
    await _detector.loadModel(highQuality: _highQuality);

    setState(() => _status = 'Starting camera...');
    _cameras = await availableCameras();
    final back = _cameras!.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras!.first,
    );

    _camera = CameraController(
      back,
      ResolutionPreset.low, // 320p — minimum CPU yuku
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

      final now = DateTime.now();
      final dt = now.difference(_lastFrameTs).inMilliseconds;
      _lastFrameTs = now;
      final instantFps = dt > 0 ? 1000.0 / dt : 0.0;
      _fps = _fps == 0 ? instantFps : (_fps * 0.7 + instantFps * 0.3);

      if (frame.detections.isNotEmpty) {
        final d = frame.detections.first;
        debugPrint(
          'DET: ${d.className} ${d.confidence.toStringAsFixed(2)} '
          'bbox=(${d.bbox.left.toStringAsFixed(0)},${d.bbox.top.toStringAsFixed(0)},'
          '${d.bbox.right.toStringAsFixed(0)},${d.bbox.bottom.toStringAsFixed(0)}) '
          'frame=${frame.frameWidth}x${frame.frameHeight}',
        );
      }

      if (mounted) {
        setState(() {
          _detections = frame.detections;
          _imageSize = Size(
            frame.frameWidth.toDouble(),
            frame.frameHeight.toDouble(),
          );
        });
      }
    } catch (e, st) {
      debugPrint('Detection error: $e\n$st');
    } finally {
      _processing = false;
    }
  }

  Future<void> _toggleQuality(bool value) async {
    setState(() {
      _highQuality = value;
      _status = 'Switching model...';
      _fps = 0;
    });
    await _camera?.stopImageStream();
    await _detector.close();
    await _detector.loadModel(highQuality: value);
    await _camera!.startImageStream(_onFrame);
    setState(() => _status = 'Ready');
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _camera?.dispose();
    _detector.close();
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

          // Bbox overlay
          Positioned.fill(
            child: CustomPaint(
              painter: DetectionPainter(
                detections: _detections,
                imageSize: _imageSize,
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
            _fpsChip(),
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
      child: Text(
        '${_fps.toStringAsFixed(1)} FPS',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.w600,
          fontFeatures: [FontFeature.tabularFigures()],
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
                  Text(
                    '${_detections.length} nesne tespit edildi',
                    style: const TextStyle(color: Colors.white, fontSize: 14, fontWeight: FontWeight.w600),
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
              const SizedBox(height: 8),
              // Quality toggle
              Row(
                children: [
                  const SizedBox(
                    width: 80,
                    child: Text('Mod', style: TextStyle(color: Colors.white70, fontSize: 12)),
                  ),
                  const Text(
                    'Hizli (320)',
                    style: TextStyle(color: Colors.white, fontSize: 12),
                  ),
                  Switch(
                    value: _highQuality,
                    activeColor: const Color(0xFFFFD93D),
                    onChanged: _toggleQuality,
                  ),
                  const Text(
                    'Kalite (800)',
                    style: TextStyle(color: Colors.white, fontSize: 12),
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

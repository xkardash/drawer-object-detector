import 'dart:math' as math;
import 'dart:typed_data';
import 'package:camera/camera.dart';
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

class YoloDetector {
  Interpreter? _interpreter;
  List<String> _labels = [];
  int _inputSize = 320;
  String _currentModel = '';

  Float32List? _inputBuffer;
  Float32List? _outputBuffer;
  int _outChannels = 0;
  int _outAnchors = 0;

  static const String model800 = 'assets/models/cekmece_v4_fp32_800.tflite';
  static const String model320 = 'assets/models/cekmece_v4_fp32_320.tflite';
  static const String labelsPath = 'assets/models/labels.txt';

  int get inputSize => _inputSize;
  String get currentModel => _currentModel;
  List<String> get labels => _labels;
  bool get isReady => _interpreter != null;

  Future<void> loadModel({required bool highQuality}) async {
    _interpreter?.close();

    final modelPath = highQuality ? model800 : model320;
    _inputSize = highQuality ? 800 : 320;
    _currentModel = highQuality ? '800' : '320';

    final options = InterpreterOptions()..threads = 4;
    _interpreter = await Interpreter.fromAsset(modelPath, options: options);

    final inShape = _interpreter!.getInputTensor(0).shape; // [1,H,W,3]
    final outShape = _interpreter!.getOutputTensor(0).shape; // [1, channels, anchors]
    _outChannels = outShape[1];
    _outAnchors = outShape[2];

    _inputBuffer = Float32List(inShape[0] * inShape[1] * inShape[2] * inShape[3]);
    _outputBuffer = Float32List(outShape[0] * outShape[1] * outShape[2]);

    final labelData = await rootBundle.loadString(labelsPath);
    _labels = labelData.split('\n').where((s) => s.trim().isNotEmpty).toList();
  }

  Future<void> close() async {
    _interpreter?.close();
    _interpreter = null;
  }

  /// Run detection on a YUV420 camera frame.
  /// Bboxes are returned in the rotated-frame coordinate system, which is
  /// also returned via [DetectionFrame] so the painter can normalize correctly.
  Future<DetectionFrame> detect(
    CameraImage image, {
    int rotationDeg = 90,
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) async {
    final interp = _interpreter;
    if (interp == null || _inputBuffer == null || _outputBuffer == null) {
      return const DetectionFrame([], 0, 0);
    }

    final pp = fillInputBufferFromCameraImage(
      image: image,
      targetSize: _inputSize,
      buffer: _inputBuffer!,
      rotationDeg: rotationDeg,
    );

    // Pass underlying ByteBuffers — tflite_flutter has a fast-path for these
    // (raw byte transfer, no nested-list walk, no shape-check throw).
    // Float32List shares memory with .buffer so reads after copyTo see updates.
    interp.runForMultipleInputs(
      [_inputBuffer!.buffer],
      {0: _outputBuffer!.buffer},
    );

    final detections = _parseOutput(
      _outputBuffer!,
      _outChannels,
      _outAnchors,
      pp,
      confThreshold,
      iouThreshold,
    );

    return DetectionFrame(detections, pp.rotatedWidth, pp.rotatedHeight);
  }

  List<Detection> _parseOutput(
    Float32List out,
    int channels,
    int anchors,
    PreprocessResult pp,
    double confThreshold,
    double iouThreshold,
  ) {
    final int numClasses = channels - 4;
    final double invScale = 1.0 / pp.scale;
    final double maxX = pp.rotatedWidth.toDouble();
    final double maxY = pp.rotatedHeight.toDouble();

    // Flat indexing: out[c * anchors + a] for channel c, anchor a.
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

      final cx = out[0 * anchors + a];
      final cy = out[1 * anchors + a];
      final w = out[2 * anchors + a];
      final h = out[3 * anchors + a];

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

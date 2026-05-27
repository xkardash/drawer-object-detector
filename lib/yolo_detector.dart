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

enum ModelSize {
  s320(320, 'assets/models/cekmece_v4_int8_320.tflite', '320'),
  s640(640, 'assets/models/cekmece_v4_int8_640.tflite', '640'),
  s800(800, 'assets/models/cekmece_v4_int8_800.tflite', '800');

  final int pixels;
  final String assetPath;
  final String label;
  const ModelSize(this.pixels, this.assetPath, this.label);
}

class YoloDetector {
  Interpreter? _interpreter;
  List<String> _labels = [];
  int _inputSize = 320;
  String _currentModel = '';

  Int8List? _inputBuffer;
  Int8List? _outputBuffer;
  Int8List? _inputLut;
  int _outChannels = 0;
  int _outAnchors = 0;

  // Output dequantization: float = (int8 - zeroPoint) * scale
  double _outScale = 1.0;
  int _outZeroPoint = 0;

  static const String labelsPath = 'assets/models/labels.txt';

  int get inputSize => _inputSize;
  String get currentModel => _currentModel;
  List<String> get labels => _labels;
  bool get isReady => _interpreter != null;

  Future<void> loadModel(ModelSize size) async {
    _interpreter?.close();

    _inputSize = size.pixels;
    _currentModel = size.label;

    final options = InterpreterOptions()..threads = 4;
    _interpreter = await Interpreter.fromAsset(size.assetPath, options: options);

    final inT = _interpreter!.getInputTensor(0);
    final outT = _interpreter!.getOutputTensor(0);
    final inShape = inT.shape; // [1,H,W,3]
    final outShape = outT.shape; // [1, channels, anchors]
    _outChannels = outShape[1];
    _outAnchors = outShape[2];

    _inputBuffer = Int8List(inShape[0] * inShape[1] * inShape[2] * inShape[3]);
    _outputBuffer = Int8List(outShape[0] * outShape[1] * outShape[2]);

    final inParams = inT.params;
    _inputLut = buildInt8Lut(inParams.scale, inParams.zeroPoint);

    final outParams = outT.params;
    _outScale = outParams.scale;
    _outZeroPoint = outParams.zeroPoint;

    final labelData = await rootBundle.loadString(labelsPath);
    _labels = labelData.split('\n').where((s) => s.trim().isNotEmpty).toList();
  }

  Future<void> close() async {
    _interpreter?.close();
    _interpreter = null;
  }

  Future<DetectionFrame> detect(
    CameraImage image, {
    int rotationDeg = 90,
    double confThreshold = 0.25,
    double iouThreshold = 0.45,
  }) async {
    final interp = _interpreter;
    if (interp == null ||
        _inputBuffer == null ||
        _outputBuffer == null ||
        _inputLut == null) {
      return const DetectionFrame([], 0, 0);
    }

    final pp = fillInt8InputBufferFromCameraImage(
      image: image,
      targetSize: _inputSize,
      buffer: _inputBuffer!,
      lut: _inputLut!,
      rotationDeg: rotationDeg,
    );

    interp.runForMultipleInputs(
      [_inputBuffer!.buffer],
      {0: _outputBuffer!.buffer},
    );

    final detections = _parseOutput(
      _outputBuffer!,
      _outChannels,
      _outAnchors,
      pp,
      _inputSize,
      confThreshold,
      iouThreshold,
    );

    return DetectionFrame(detections, pp.rotatedWidth, pp.rotatedHeight);
  }

  List<Detection> _parseOutput(
    Int8List out,
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
    final double inSize = inputSize.toDouble();
    final double oScale = _outScale;
    final int oZero = _outZeroPoint;

    // Compare scores in int8 space to skip the multiply on the hot path.
    // float >= threshold  ⇔  (int8 - zero) * scale >= threshold
    //                     ⇔  int8 >= ceil(threshold/scale) + zero
    final int int8Threshold =
        (confThreshold / oScale).ceil() + oZero;

    final List<_Candidate> candidates = [];
    for (int a = 0; a < anchors; a++) {
      int maxScoreI = -129;
      int maxClassId = 0;
      for (int c = 0; c < numClasses; c++) {
        final s = out[(4 + c) * anchors + a];
        if (s > maxScoreI) {
          maxScoreI = s;
          maxClassId = c;
        }
      }
      if (maxScoreI < int8Threshold) continue;

      // Dequantize bbox and score only for surviving candidates.
      final double cx = (out[a] - oZero) * oScale * inSize;
      final double cy = (out[anchors + a] - oZero) * oScale * inSize;
      final double w = (out[2 * anchors + a] - oZero) * oScale * inSize;
      final double h = (out[3 * anchors + a] - oZero) * oScale * inSize;
      final double maxScore = (maxScoreI - oZero) * oScale;

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

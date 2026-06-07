import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'yolo_detector.dart';

/// One model configuration to evaluate (Görev 3). The five configs cover the
/// native-resolution models plus the two 800-trained downscaled exports, so the
/// PC-vs-phone mAP table is complete. Labels avoid '@' so they are filesystem-
/// safe; the PC script maps `800at320` → "800@320" for display.
class EvalModel {
  final String label;
  final String assetPath;
  final int inputSize;
  const EvalModel(this.label, this.assetPath, this.inputSize);

  static const List<EvalModel> all = [
    EvalModel('native320', 'assets/models/cekmece_v4_native320_fp32.tflite', 320),
    EvalModel('native640', 'assets/models/cekmece_v4_native640_fp32.tflite', 640),
    EvalModel('native800', 'assets/models/cekmece_v4_native800_fp32.tflite', 800),
    EvalModel('800at320', 'assets/models/cekmece_v4_fp32_320.tflite', 320),
    EvalModel('800at640', 'assets/models/cekmece_v4_fp32_640.tflite', 640),
  ];
}

class EvalRunResult {
  final String path;
  final int images;
  final double avgInfMs;
  final bool gpuActive;
  const EvalRunResult({
    required this.path,
    required this.images,
    required this.avgInfMs,
    required this.gpuActive,
  });
}

/// Runs one (model × backend) batch evaluation over the bundled test images and
/// writes a predictions JSON to `<external>/eval_out/eval_<label>_<gpu|cpu>.json`.
///
/// Predictions are produced by [YoloDetector.detectDecoded] at conf=0.001 /
/// iou=0.7 / max_det=300 (Ultralytics `val` defaults) so they line up with the
/// PC TFLite pipeline. mAP itself is computed on the PC from this JSON + the GT
/// labels — the phone never sees the ground truth.
Future<EvalRunResult> runEval({
  required EvalModel model,
  required bool useGpu,
  required YoloDetector det,
  required void Function(int done, int total) onProgress,
}) async {
  await det.loadModelFromAsset(
    model.assetPath,
    model.inputSize,
    label: model.label,
    useGpu: useGpu,
  );

  final manifest = await AssetManifest.loadFromAssetBundle(rootBundle);
  final images = manifest
      .listAssets()
      .where((a) => a.startsWith('assets/eval_images/'))
      .toList()
    ..sort();

  final predictions = <Map<String, dynamic>>[];
  double sumInf = 0;
  int done = 0;
  final total = images.length;

  for (final asset in images) {
    final data = await rootBundle.load(asset);
    final decoded = img.decodeJpg(data.buffer.asUint8List());
    if (decoded != null) {
      final rgb = decoded.getBytes(order: img.ChannelOrder.rgb);
      final dets = await det.detectDecoded(rgb, decoded.width, decoded.height);
      sumInf += det.lastInferenceMs;
      predictions.add({
        'image': asset.split('/').last,
        'width': decoded.width,
        'height': decoded.height,
        'detections': [
          for (final d in dets)
            {
              'class_id': d.classId,
              'class_name': d.className,
              'confidence': d.confidence,
              'x1': d.bbox.left,
              'y1': d.bbox.top,
              'x2': d.bbox.right,
              'y2': d.bbox.bottom,
            },
        ],
      });
    }
    onProgress(++done, total);
    await Future<void>.delayed(Duration.zero); // yield so the UI can repaint
  }

  final gpuActive = det.gpuActive; // capture before close() resets it
  final avgInf = predictions.isEmpty ? 0.0 : sumInf / predictions.length;
  final backend = useGpu ? 'gpu' : 'cpu';

  final payload = <String, dynamic>{
    'meta': {
      'label': model.label,
      'input_size': model.inputSize,
      'backend': backend,
      'gpu_requested': useGpu,
      'gpu_active': gpuActive,
      'conf': 0.001,
      'iou': 0.7,
      'max_det': 300,
      'images': predictions.length,
      'avg_inf_ms': avgInf,
      'timestamp': DateTime.now().toIso8601String(),
    },
    'predictions': predictions,
  };

  final base = await getExternalStorageDirectory() ??
      await getApplicationDocumentsDirectory();
  final outDir = Directory('${base.path}/eval_out');
  if (!await outDir.exists()) await outDir.create(recursive: true);
  final file = File('${outDir.path}/eval_${model.label}_$backend.json');
  await file.writeAsString(const JsonEncoder.withIndent('  ').convert(payload));

  await det.close();
  return EvalRunResult(
    path: file.path,
    images: predictions.length,
    avgInfMs: avgInf,
    gpuActive: gpuActive,
  );
}

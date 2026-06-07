import 'package:flutter/material.dart';
import 'eval_runner.dart';
import 'yolo_detector.dart';

/// Debug-only batch-evaluation screen (Görev 3). Standalone: owns its own
/// [YoloDetector] and uses NO camera, so it can A/B the GPU delegate against
/// pure CPU on each of the five models without contending with the live preview.
/// Reachable from the "TEST" chip on the home screen (which stops the camera
/// and frees its interpreter before pushing this route).
class EvalScreen extends StatefulWidget {
  const EvalScreen({super.key});

  @override
  State<EvalScreen> createState() => _EvalScreenState();
}

class _EvalScreenState extends State<EvalScreen> {
  final YoloDetector _det = YoloDetector();
  bool _useGpu = true;
  bool _running = false;
  double _progress = 0;
  String _status = 'Hazır';
  final List<String> _log = [];

  @override
  void dispose() {
    _det.close();
    super.dispose();
  }

  Future<void> _runOne(EvalModel model, bool gpu) async {
    setState(() {
      _status = '${model.label} (${gpu ? "GPU" : "CPU"}) çalışıyor…';
      _progress = 0;
    });
    final res = await runEval(
      model: model,
      useGpu: gpu,
      det: _det,
      onProgress: (done, total) {
        if (mounted) setState(() => _progress = total == 0 ? 0 : done / total);
      },
    );
    if (!mounted) return;
    final backend = gpu ? (res.gpuActive ? 'GPU' : 'GPU→CPU') : 'CPU';
    setState(() {
      _log.add('✓ ${model.label} [$backend] · ${res.images} img · '
          '${res.avgInfMs.toStringAsFixed(0)} ms/img → ${res.path.split('/').last}');
    });
  }

  Future<void> _runBatch(List<(EvalModel, bool)> jobs) async {
    if (_running) return;
    setState(() => _running = true);
    final messenger = ScaffoldMessenger.of(context);
    String? outDir;
    try {
      for (final (model, gpu) in jobs) {
        await _runOne(model, gpu);
      }
      outDir = _log.isEmpty
          ? null
          : _log.last.split('→').last.trim();
    } catch (e, st) {
      if (mounted) setState(() => _log.add('✗ HATA: $e'));
      debugPrint('Eval error: $e\n$st');
    } finally {
      if (mounted) {
        setState(() {
          _running = false;
          _status = 'Bitti';
          _progress = 0;
        });
        messenger.showSnackBar(SnackBar(
          content: Text(outDir == null
              ? 'Çalışma bitti'
              : 'JSON yazıldı → eval_out/ (adb pull ile çek)'),
          duration: const Duration(seconds: 8),
        ));
      }
    }
  }

  List<(EvalModel, bool)> _allCurrent() =>
      [for (final m in EvalModel.all) (m, _useGpu)];

  List<(EvalModel, bool)> _allBoth() => [
        for (final gpu in [false, true])
          for (final m in EvalModel.all) (m, gpu),
      ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Batch Eval (mAP)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SwitchListTile(
              value: _useGpu,
              onChanged: _running ? null : (v) => setState(() => _useGpu = v),
              title: const Text('GPU delegate'),
              subtitle: Text(_useGpu
                  ? 'AÇIK — Adreno OpenCL (dahili fp16)'
                  : 'KAPALI — CPU fp32 (XNNPACK, 4 thread)'),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                FilledButton.tonal(
                  onPressed: _running ? null : () => _runBatch(_allCurrent()),
                  child: Text('Tümü (${_useGpu ? "GPU" : "CPU"})'),
                ),
                FilledButton(
                  onPressed: _running ? null : () => _runBatch(_allBoth()),
                  child: const Text('Tümü × GPU+CPU'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final m in EvalModel.all)
                  OutlinedButton(
                    onPressed:
                        _running ? null : () => _runBatch([(m, _useGpu)]),
                    child: Text(m.label),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            if (_running) ...[
              LinearProgressIndicator(value: _progress == 0 ? null : _progress),
              const SizedBox(height: 8),
            ],
            Text(_status, style: const TextStyle(fontWeight: FontWeight.w600)),
            const Divider(height: 24),
            const Text('Sonuçlar', style: TextStyle(color: Colors.white70)),
            const SizedBox(height: 8),
            Expanded(
              child: ListView(
                children: [
                  for (final line in _log.reversed)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 3),
                      child: Text(line,
                          style: const TextStyle(
                              fontSize: 12, fontFamily: 'monospace')),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

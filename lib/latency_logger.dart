import 'dart:io';
import 'package:path_provider/path_provider.dart';

/// K4 (PERFORMANCE.md) — on-device latency + sustained/thermal logging.
///
/// Records one row per processed frame to an in-memory buffer, then dumps a CSV
/// to the app's external files dir (no runtime permission needed; pull it over
/// USB from `/Android/data/<package>/files/`). Offline analysis computes p50/p95/p99
/// of per-frame compute latency and plots FPS vs elapsed time (thermal curve).
///
/// Usage: tap REC to start, point the phone at the drawer for ~10 min on a FIXED
/// model size (turn Auto off so the curve isn't confounded by model switches),
/// tap again to stop -> snackbar shows the saved path.
class _Sample {
  final int tMs; // elapsed since recording start
  final int dtMs; // inter-frame gap (throughput side)
  final double totalMs; // pre + inf + parse (per-frame compute)
  final double pureMs; // pure inference (load-time bench baseline)
  final double fps; // FPS EMA at this frame
  final int model; // input size px (so Auto switches are visible)
  const _Sample(this.tMs, this.dtMs, this.totalMs, this.pureMs, this.fps, this.model);
}

class LatencyLogger {
  final List<_Sample> _samples = [];
  DateTime? _start;

  bool get active => _start != null;
  int get count => _samples.length;
  int get elapsedSec =>
      _start == null ? 0 : DateTime.now().difference(_start!).inSeconds;

  void start() {
    _samples.clear();
    _start = DateTime.now();
  }

  void add({
    required int dtMs,
    required double totalMs,
    required double pureMs,
    required double fps,
    required int model,
  }) {
    final s = _start;
    if (s == null) return;
    _samples.add(_Sample(
      DateTime.now().difference(s).inMilliseconds,
      dtMs, totalMs, pureMs, fps, model,
    ));
  }

  /// Stops recording and writes the CSV. Returns the saved file path.
  Future<String> stop() async {
    _start = null;
    final dir =
        await getExternalStorageDirectory() ?? await getApplicationDocumentsDirectory();
    final ts = DateTime.now()
        .toIso8601String()
        .split('.')
        .first
        .replaceAll(':', '-');
    final file = File('${dir.path}/latency_$ts.csv');
    final b = StringBuffer('t_ms,dt_ms,total_ms,pure_ms,fps,model_px\n');
    for (final s in _samples) {
      b.writeln('${s.tMs},${s.dtMs},${s.totalMs.toStringAsFixed(1)},'
          '${s.pureMs.toStringAsFixed(1)},${s.fps.toStringAsFixed(2)},${s.model}');
    }
    await file.writeAsString(b.toString());
    return file.path;
  }
}

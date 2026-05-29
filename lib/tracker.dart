import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'detection.dart';

/// Inference rate'i (6 FPS) ekran rate'inden (60 FPS) ayıran hafif tracker.
///
/// Telefon gezdirilirken bbox'lar gerçek inference'ler arasında "donup zıplamak"
/// yerine her ekran karesinde tahmini hızla kaydırılır → 640'ın doğruluğu korunur,
/// hareket ~30-60 FPS akıcı hissedilir.
///
/// Yaklaşım (predict-then-match, Kalman ruhu):
///   update(): her track'i ölçüm anına tahmin et → yeni detection'larla centroid
///             eşleştir → eşleşeni güncelle + hız tahmin et (EMA).
///   predict(): track'leri "şimdi"ye, tavanla sınırlı şekilde ekstrapole et.
class BoxTracker {
  final List<_Track> _tracks = [];
  int _nextId = 0;

  /// Ekstrapolasyon tavanı: bir detection aralığından (~166ms @6FPS) fazla
  /// ileri tahmin etme — yoksa duran/yön değiştiren nesnede ghost uçar.
  final Duration maxHorizon;

  /// Bu kadar ardışık detection'da eşleşmezse track düşer (flicker toleransı).
  final int maxMisses;

  /// Hız EMA'sında yeni ölçüme verilen ağırlık (pan responsive olsun diye yüksek).
  final double velAlpha;

  BoxTracker({
    this.maxHorizon = const Duration(milliseconds: 160),
    this.maxMisses = 2,
    this.velAlpha = 0.5,
  });

  void clear() => _tracks.clear();

  /// Gerçek bir inference sonucu geldiğinde çağrılır.
  void update(List<Detection> dets, DateTime ts) {
    // 1) Track'leri ölçüm anına tahmin et (eşleştirme bunun üzerinden yapılır).
    final predicted = <_Track, Offset>{
      for (final t in _tracks) t: t.rect.center + t.vel * _dt(ts, t.lastSeen),
    };

    // 2) Aday (track, det) çiftleri: aynı sınıf, gate içinde, mesafeye göre sırala.
    final pairs = <_Pair>[];
    for (final t in _tracks) {
      final pc = predicted[t]!;
      final gate = _gate(t.rect);
      for (final d in dets) {
        if (d.classId != t.classId) continue;
        final dist = (d.bbox.center - pc).distance;
        if (dist <= gate) pairs.add(_Pair(dist, t, d));
      }
    }
    pairs.sort((a, b) => a.dist.compareTo(b.dist));

    // 3) Greedy ata.
    final assignedTracks = <_Track>{};
    final usedDets = <Detection>{};
    for (final p in pairs) {
      if (assignedTracks.contains(p.track) || usedDets.contains(p.det)) continue;
      assignedTracks.add(p.track);
      usedDets.add(p.det);
      final t = p.track;
      final dtMeas = ts.difference(t.lastSeen).inMicroseconds / 1e6;
      if (dtMeas > 1e-3) {
        final inst = (p.det.bbox.center - t.rect.center) / dtMeas;
        t.vel = t.vel * (1 - velAlpha) + inst * velAlpha;
      }
      t.rect = p.det.bbox;
      t.classId = p.det.classId;
      t.className = p.det.className;
      t.confidence = p.det.confidence;
      t.lastSeen = ts;
      t.misses = 0;
    }

    // 4) Eşleşmeyen track'leri yaşlandır / düşür.
    _tracks.removeWhere((t) {
      if (assignedTracks.contains(t)) return false;
      t.misses++;
      return t.misses > maxMisses;
    });

    // 5) Eşleşmeyen detection'lar → yeni track (hız 0, ikinci ölçümde kazanılır).
    for (final d in dets) {
      if (usedDets.contains(d)) continue;
      _tracks.add(_Track(
        id: _nextId++,
        rect: d.bbox,
        vel: Offset.zero,
        classId: d.classId,
        className: d.className,
        confidence: d.confidence,
        lastSeen: ts,
      ));
    }
  }

  /// Her ekran karesinde çağrılır: track'leri "şimdi"ye ekstrapole edilmiş
  /// pozisyonlarıyla döndürür.
  List<Detection> predict(DateTime now) {
    return [
      for (final t in _tracks)
        Detection(
          bbox: t.rect.shift(t.vel * _dt(now, t.lastSeen)),
          classId: t.classId,
          className: t.className,
          confidence: t.confidence,
        ),
    ];
  }

  double _dt(DateTime now, DateTime since) {
    final s = now.difference(since).inMicroseconds / 1e6;
    final maxS = maxHorizon.inMicroseconds / 1e6;
    if (s <= 0) return 0;
    return s > maxS ? maxS : s;
  }

  // Hızlı pan'de ilk eşleşme için cömert gate (centroid, IoU değil — boxlar
  // örtüşmese bile yakalasın).
  double _gate(Rect r) => math.max(r.width, r.height) * 1.5 + 40;
}

class _Track {
  int id;
  Rect rect; // son ölçülen rect (image koordinatları)
  Offset vel; // merkez hızı, px/saniye
  int classId;
  String className;
  double confidence;
  DateTime lastSeen;
  int misses = 0;
  _Track({
    required this.id,
    required this.rect,
    required this.vel,
    required this.classId,
    required this.className,
    required this.confidence,
    required this.lastSeen,
  });
}

class _Pair {
  final double dist;
  final _Track track;
  final Detection det;
  _Pair(this.dist, this.track, this.det);
}

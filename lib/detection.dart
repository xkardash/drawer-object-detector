import 'package:flutter/material.dart';

class Detection {
  final Rect bbox;
  final int classId;
  final String className;
  final double confidence;

  Detection({
    required this.bbox,
    required this.classId,
    required this.className,
    required this.confidence,
  });
}

const List<Color> kClassColors = [
  Color(0xFFFF6B6B),
  Color(0xFFFFD93D),
  Color(0xFF6BCB77),
  Color(0xFF4D96FF),
  Color(0xFFB983FF),
];

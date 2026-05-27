import 'package:flutter/material.dart';
import 'detection.dart';

class DetectionPainter extends CustomPainter {
  final List<Detection> detections;
  final Size imageSize;

  DetectionPainter({required this.detections, required this.imageSize});

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    final scaleX = size.width / imageSize.width;
    final scaleY = size.height / imageSize.height;

    for (final det in detections) {
      final color = kClassColors[det.classId % kClassColors.length];

      final scaledRect = Rect.fromLTRB(
        det.bbox.left * scaleX,
        det.bbox.top * scaleY,
        det.bbox.right * scaleX,
        det.bbox.bottom * scaleY,
      );

      // Bbox stroke
      final strokePaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 3.0;
      canvas.drawRRect(
        RRect.fromRectAndRadius(scaledRect, const Radius.circular(6)),
        strokePaint,
      );

      // Label background
      final label = '${det.className} ${(det.confidence * 100).toStringAsFixed(0)}%';
      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.black,
            fontSize: 12,
            fontWeight: FontWeight.w700,
            letterSpacing: 0.2,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final labelRect = Rect.fromLTWH(
        scaledRect.left,
        scaledRect.top - textPainter.height - 6,
        textPainter.width + 12,
        textPainter.height + 4,
      );

      final labelBgPaint = Paint()..color = color;
      canvas.drawRRect(
        RRect.fromRectAndRadius(labelRect, const Radius.circular(4)),
        labelBgPaint,
      );

      textPainter.paint(
        canvas,
        Offset(labelRect.left + 6, labelRect.top + 2),
      );
    }
  }

  @override
  bool shouldRepaint(covariant DetectionPainter old) {
    return old.detections != detections || old.imageSize != imageSize;
  }
}

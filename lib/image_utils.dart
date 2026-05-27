import 'dart:typed_data';
import 'package:camera/camera.dart';

/// Result of the fused preprocessing.
/// `rotatedWidth/Height` are the dimensions in the post-rotation frame —
/// detection bboxes are reported in this frame, so the painter must use
/// these as the reference image size.
class PreprocessResult {
  final int rotatedWidth;
  final int rotatedHeight;
  final double scale;
  final int padX;
  final int padY;

  const PreprocessResult({
    required this.rotatedWidth,
    required this.rotatedHeight,
    required this.scale,
    required this.padX,
    required this.padY,
  });
}

/// Fills [buffer] with int8 quantized pixels sampled directly from a YUV420
/// CameraImage, applying a 90° clockwise rotation and a letterbox-resize
/// to [targetSize] in a single pass.
///
/// [lut] is a 256-entry table mapping channel byte [0..255] to the model's
/// quantized int8 representation. Build once in [buildInt8Lut].
PreprocessResult fillInt8InputBufferFromCameraImage({
  required CameraImage image,
  required int targetSize,
  required Int8List buffer,
  required Int8List lut,
  int rotationDeg = 90,
}) {
  final srcW = image.width;
  final srcH = image.height;

  final bool swap = rotationDeg == 90 || rotationDeg == 270;
  final int rotW = swap ? srcH : srcW;
  final int rotH = swap ? srcW : srcH;

  final double scale = targetSize / (rotW > rotH ? rotW : rotH);
  final int newW = (rotW * scale).round();
  final int newH = (rotH * scale).round();
  final int padX = (targetSize - newW) >> 1;
  final int padY = (targetSize - newH) >> 1;

  final yPlane = image.planes[0];
  final uPlane = image.planes[1];
  final vPlane = image.planes[2];
  final yBytes = yPlane.bytes;
  final uBytes = uPlane.bytes;
  final vBytes = vPlane.bytes;
  final int yRowStride = yPlane.bytesPerRow;
  final int uvRowStride = uPlane.bytesPerRow;
  final int uvPixelStride = uPlane.bytesPerPixel ?? 1;

  // Quantized padding (gray 114 → quantized).
  final int padQ = lut[114];
  final double invScale = 1.0 / scale;

  int outIdx = 0;
  for (int ty = 0; ty < targetSize; ty++) {
    final int ry = ((ty - padY) * invScale).toInt();
    final bool rowInside = ry >= 0 && ry < rotH;

    for (int tx = 0; tx < targetSize; tx++) {
      if (!rowInside) {
        buffer[outIdx++] = padQ;
        buffer[outIdx++] = padQ;
        buffer[outIdx++] = padQ;
        continue;
      }

      final int rx = ((tx - padX) * invScale).toInt();
      if (rx < 0 || rx >= rotW) {
        buffer[outIdx++] = padQ;
        buffer[outIdx++] = padQ;
        buffer[outIdx++] = padQ;
        continue;
      }

      int sx, sy;
      switch (rotationDeg) {
        case 90:
          sx = ry;
          sy = srcH - 1 - rx;
          break;
        case 180:
          sx = srcW - 1 - rx;
          sy = srcH - 1 - ry;
          break;
        case 270:
          sx = srcW - 1 - ry;
          sy = rx;
          break;
        default:
          sx = rx;
          sy = ry;
      }

      final int yIdx = sy * yRowStride + sx;
      final int uvIdx = (sy >> 1) * uvRowStride + (sx >> 1) * uvPixelStride;
      if (yIdx >= yBytes.length || uvIdx >= uBytes.length) {
        buffer[outIdx++] = padQ;
        buffer[outIdx++] = padQ;
        buffer[outIdx++] = padQ;
        continue;
      }

      final int yv = yBytes[yIdx];
      final int uv = uBytes[uvIdx] - 128;
      final int vv = vBytes[uvIdx] - 128;

      int r = yv + ((91881 * vv) >> 16);
      int g = yv - ((22554 * uv + 46802 * vv) >> 16);
      int b = yv + ((116130 * uv) >> 16);

      if (r < 0) r = 0; else if (r > 255) r = 255;
      if (g < 0) g = 0; else if (g > 255) g = 255;
      if (b < 0) b = 0; else if (b > 255) b = 255;

      buffer[outIdx++] = lut[r];
      buffer[outIdx++] = lut[g];
      buffer[outIdx++] = lut[b];
    }
  }

  return PreprocessResult(
    rotatedWidth: rotW,
    rotatedHeight: rotH,
    scale: scale,
    padX: padX,
    padY: padY,
  );
}

/// Build the quantization lookup table for a single channel byte.
/// `q = round((byte/255) / scale + zeroPoint)`, clamped to int8 range.
Int8List buildInt8Lut(double scale, int zeroPoint) {
  final lut = Int8List(256);
  for (int r = 0; r < 256; r++) {
    final f = r / 255.0;
    final q = (f / scale).round() + zeroPoint;
    lut[r] = q < -128 ? -128 : (q > 127 ? 127 : q);
  }
  return lut;
}

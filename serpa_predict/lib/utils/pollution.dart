import 'package:flutter/material.dart';

String pollutionLevel(double v) {
  if (v < 0.25) return 'Baixo';
  if (v < 0.5) return 'Moderado';
  if (v < 0.75) return 'Ruim';
  return 'Péssimo';
}

Color pollutionColor(double v) {
  if (v < 0.25) return Colors.green;
  if (v < 0.5) return Colors.yellow;
  if (v < 0.75) return Colors.orange;
  return Colors.red;
}

String pollutionRangeLabel(double v) {
  if (v < 0.25) return '0.00–0.24';
  if (v < 0.5) return '0.25–0.49';
  if (v < 0.75) return '0.50–0.74';
  return '0.75–1.00';
}

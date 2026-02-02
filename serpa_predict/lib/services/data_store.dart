import 'dart:math';

import '../models/sample.dart';

class DataStore {
  final List<Sample> samples = [];
  Map<String, List<double>> stats = {};

  void addSample(Sample s) {
    samples.add(s);
    _computeStats();
  }

  void setSamples(List<Sample> s) {
    samples
      ..clear()
      ..addAll(s);
    _computeStats();
  }

  void _computeStats() {
    if (samples.isEmpty) return;
    final keys = ['pH', 'OD', 'TURB'];
    for (final k in keys) {
      final vals = samples.map((e) => e.pred).toList();
      List<double> target;
      if (k == 'pH') {
        target = vals.map((e) => e.pH).toList();
      } else if (k == 'OD') {
        target = vals.map((e) => e.od).toList();
      } else {
        target = vals.map((e) => e.turb).toList();
      }
      stats[k] = [target.reduce(min), target.reduce(max)];
    }
  }
}


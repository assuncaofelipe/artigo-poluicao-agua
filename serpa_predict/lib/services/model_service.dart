import 'dart:convert';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/prediction.dart';
import '../models/power_reading.dart';
import '../utils/normalize.dart';

class ModelService {
  bool ready = false;
  String? errorMessage;
  late Interpreter _interpreter;
  final featureCols = const ['x', 'y', 'TEMP', 'COR_A', 'TURB', 'COR_V', 'OD'];
  final targetCols = const ['pH', 'OD', 'TURB'];
  List<double> mean = [];
  List<double> std = [];
  List<double> yMean = [];
  List<double> yStd = [];
  final weights = const [0.4, 0.4, 0.2];
  final phTarget = 7.0;
  final defaults = const {
    'pH': [5.16, 6.52],
    'OD': [1.25, 2.5],
    'TURB': [0.0, 6.08],
  };

  Future<void> init() async {
    try {
      errorMessage = null;
      final normText = await _loadAssetString([
        'assets/model/norm_multi.json',
        'model/norm_multi.json',
      ]);
      final norm = jsonDecode(normText) as Map<String, dynamic>;
      mean = List<double>.from((norm['mean'] as List).map((e) => e.toDouble()));
      std = List<double>.from((norm['std'] as List).map((e) => e.toDouble()));
      yMean = List<double>.from((norm['y_mean'] as List).map((e) => e.toDouble()));
      yStd = List<double>.from((norm['y_std'] as List).map((e) => e.toDouble()));
      _interpreter = await _loadInterpreter([
        'assets/model/model.tflite',
        'model/model.tflite',
        'model.tflite',
      ]);
      ready = true;
    } catch (e) {
      ready = false;
      errorMessage = e.toString();
    }
  }

  BenchmarkResult benchmark({int runs = 5000, int warmup = 200, int chunk = 100}) {
    if (!ready) {
      throw Exception('Modelo não carregado.');
    }
    // Entrada default: média => normalização resulta em zero.
    final x = List<double>.generate(featureCols.length, (i) => 0.0);
    final inputTensor = [x];
    final output = List.filled(targetCols.length, 0.0).reshape([1, 3]);

    for (var i = 0; i < warmup; i++) {
      _interpreter.run(inputTensor, output);
    }

    final swTotal = Stopwatch()..start();
    final chunkMeans = <double>[];
    var done = 0;
    while (done < runs) {
      final n = (runs - done) < chunk ? (runs - done) : chunk;
      final sw = Stopwatch()..start();
      for (var i = 0; i < n; i++) {
        _interpreter.run(inputTensor, output);
      }
      sw.stop();
      final msPer = (sw.elapsedMicroseconds / 1000.0) / n;
      chunkMeans.add(msPer);
      done += n;
    }
    swTotal.stop();

    final meanMs = (swTotal.elapsedMicroseconds / 1000.0) / runs;
    final stdMs = _std(chunkMeans);
    final throughput = 1000.0 / meanMs;

    return BenchmarkResult(
      meanMs: meanMs,
      stdMs: stdMs,
      throughput: throughput,
      runs: runs,
    );
  }

  Future<EnergyBenchmarkResult> benchmarkWithPower({
    required Future<PowerReading?> Function() samplePower,
    int runs = 200000,
    int warmup = 200,
    int chunk = 1000,
  }) async {
    if (!ready) {
      throw Exception('Modelo não carregado.');
    }
    final x = List<double>.generate(featureCols.length, (i) => 0.0);
    final inputTensor = [x];
    final output = List.filled(targetCols.length, 0.0).reshape([1, 3]);

    for (var i = 0; i < warmup; i++) {
      _interpreter.run(inputTensor, output);
    }

    final powerSamples = <PowerReading>[];
    final swTotal = Stopwatch()..start();
    final chunkMeans = <double>[];
    var done = 0;
    while (done < runs) {
      final n = (runs - done) < chunk ? (runs - done) : chunk;
      final sw = Stopwatch()..start();
      for (var i = 0; i < n; i++) {
        _interpreter.run(inputTensor, output);
      }
      sw.stop();
      final msPer = (sw.elapsedMicroseconds / 1000.0) / n;
      chunkMeans.add(msPer);
      done += n;

      final reading = await samplePower();
      if (reading != null) {
        powerSamples.add(reading);
      }
    }
    swTotal.stop();

    final meanMs = (swTotal.elapsedMicroseconds / 1000.0) / runs;
    final stdMs = _std(chunkMeans);
    final throughput = 1000.0 / meanMs;
    final meanPowerW = powerSamples.isEmpty
        ? null
        : powerSamples.map((p) => p.powerW).reduce((a, b) => a + b) /
            powerSamples.length;

    return EnergyBenchmarkResult(
      meanMs: meanMs,
      stdMs: stdMs,
      throughput: throughput,
      runs: runs,
      powerSamples: powerSamples.length,
      meanPowerW: meanPowerW,
    );
  }

  Prediction predictSingle(Map<String, double> input,
      {Map<String, List<double>>? stats}) {
    final x = List<double>.generate(
      featureCols.length,
      (i) => (input[featureCols[i]]! - mean[i]) / std[i],
    );
    final inputTensor = [x];
    final output = List.filled(targetCols.length, 0.0).reshape([1, 3]);
    _interpreter.run(inputTensor, output);

    final out = (output.first as List)
        .map((e) => (e as double))
        .toList(growable: false);

    final pH = out[0] * yStd[0] + yMean[0];
    final od = out[1] * yStd[1] + yMean[1];
    final turb = out[2] * yStd[2] + yMean[2];

    final phStats = stats?['pH'] ?? defaults['pH']!;
    final odStats = stats?['OD'] ?? defaults['OD']!;
    final turbStats = stats?['TURB'] ?? defaults['TURB']!;

    final phDev = (pH - phTarget).abs();
    final phDevNorm = normalizeScalar(
        phDev, 0.0, max(phStats[1] - phTarget, phTarget - phStats[0]));
    final odNorm = normalizeScalar(od, odStats[0], odStats[1]);
    final turbNorm = normalizeScalar(turb, turbStats[0], turbStats[1]);
    final odInv = 1.0 - odNorm;
    final pollution =
        weights[0] * phDevNorm + weights[1] * odInv + weights[2] * turbNorm;

    return Prediction(pH: pH, od: od, turb: turb, pollution: pollution);
  }

  void dispose() {
    if (ready) {
      _interpreter.close();
    }
  }

  Future<String> _loadAssetString(List<String> paths) async {
    for (final path in paths) {
      try {
        return await rootBundle.loadString(path);
      } catch (_) {
        continue;
      }
    }
    throw Exception('Falha ao carregar norm_multi.json (asset nao encontrado)');
  }

  Future<Interpreter> _loadInterpreter(List<String> paths) async {
    for (final path in paths) {
      try {
        return await Interpreter.fromAsset(path);
      } catch (_) {
        continue;
      }
    }
    throw Exception('Falha ao carregar model.tflite (asset nao encontrado)');
  }
}

class BenchmarkResult {
  final double meanMs;
  final double stdMs;
  final double throughput;
  final int runs;

  BenchmarkResult({
    required this.meanMs,
    required this.stdMs,
    required this.throughput,
    required this.runs,
  });
}

class EnergyBenchmarkResult {
  final double meanMs;
  final double stdMs;
  final double throughput;
  final int runs;
  final int powerSamples;
  final double? meanPowerW;

  EnergyBenchmarkResult({
    required this.meanMs,
    required this.stdMs,
    required this.throughput,
    required this.runs,
    required this.powerSamples,
    required this.meanPowerW,
  });
}

double _std(List<double> values) {
  if (values.isEmpty) return 0.0;
  final mean = values.reduce((a, b) => a + b) / values.length;
  var acc = 0.0;
  for (final v in values) {
    final d = v - mean;
    acc += d * d;
  }
  return sqrt(acc / values.length);
}

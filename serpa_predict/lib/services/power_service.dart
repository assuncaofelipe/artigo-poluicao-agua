import 'dart:async';

import 'package:flutter/services.dart';

import '../models/power_reading.dart';

class PowerService {
  static const MethodChannel _channel = MethodChannel('power');

  Future<PowerReading?> readPower() async {
    try {
      final res = await _channel.invokeMethod<Map>('readPower');
      if (res == null) return null;
      final current = (res['current_uA'] as num?)?.toDouble() ?? 0.0;
      final voltage = (res['voltage_uV'] as num?)?.toDouble() ?? 0.0;
      if (current == 0.0 || voltage == 0.0) {
        return null;
      }
      return PowerReading(currentUA: current, voltageUV: voltage);
    } catch (_) {
      return null;
    }
  }

  Future<PowerSummary?> sampleAverage({
    Duration duration = const Duration(seconds: 5),
    Duration interval = const Duration(milliseconds: 500),
  }) async {
    final samples = <PowerReading>[];
    final end = DateTime.now().add(duration);
    while (DateTime.now().isBefore(end)) {
      final reading = await readPower();
      if (reading != null) {
        samples.add(reading);
      }
      await Future<void>.delayed(interval);
    }
    if (samples.isEmpty) return null;
    final meanPower =
        samples.map((s) => s.powerW).reduce((a, b) => a + b) / samples.length;
    final meanCurrent =
        samples.map((s) => s.currentA).reduce((a, b) => a + b) / samples.length;
    final meanVoltage =
        samples.map((s) => s.voltageV).reduce((a, b) => a + b) / samples.length;
    return PowerSummary(
      samples: samples.length,
      meanPowerW: meanPower,
      meanCurrentA: meanCurrent,
      meanVoltageV: meanVoltage,
    );
  }
}

class PowerSummary {
  final int samples;
  final double meanPowerW;
  final double meanCurrentA;
  final double meanVoltageV;

  const PowerSummary({
    required this.samples,
    required this.meanPowerW,
    required this.meanCurrentA,
    required this.meanVoltageV,
  });
}

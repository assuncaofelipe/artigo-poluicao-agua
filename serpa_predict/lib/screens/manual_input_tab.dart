import 'package:flutter/material.dart';

import '../services/data_store.dart';
import '../services/model_service.dart';
import '../services/power_service.dart';
import '../models/sample.dart';
import '../utils/pollution.dart';

class ManualInputTab extends StatefulWidget {
  final ModelService model;
  final DataStore store;

  const ManualInputTab({super.key, required this.model, required this.store});

  @override
  State<ManualInputTab> createState() => _ManualInputTabState();
}

class _ManualInputTabState extends State<ManualInputTab> {
  final _formKey = GlobalKey<FormState>();
  final _controllers = <String, TextEditingController>{};
  bool _benchmarking = false;
  BenchmarkResult? _bench;
  bool _energying = false;
  EnergyEstimate? _energy;
  final _power = PowerService();

  @override
  void initState() {
    super.initState();
    for (final col in widget.model.featureCols) {
      _controllers[col] = TextEditingController();
    }
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!widget.model.ready) {
      final msg = widget.model.errorMessage;
      return Center(
        child: Text(
          msg == null ? 'Loading model...' : 'Error loading model:\n$msg',
          textAlign: TextAlign.center,
        ),
      );
    }

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Form(
        key: _formKey,
        child: ListView(
          children: [
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x1A0F172A),
                    blurRadius: 12,
                    offset: Offset(0, 6),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Manual input',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Enter a sample to generate a prediction.',
                    style: TextStyle(color: Color(0xFF64748B)),
                  ),
                  const SizedBox(height: 12),
                  LayoutBuilder(
                    builder: (context, constraints) {
                      final isWide = constraints.maxWidth > 520;
                      final itemWidth = isWide
                          ? (constraints.maxWidth - 16) / 2
                          : constraints.maxWidth;
                      return Wrap(
                        spacing: 16,
                        runSpacing: 12,
                        children: widget.model.featureCols.map((c) {
                          return SizedBox(
                            width: itemWidth,
                            child: TextFormField(
                              controller: _controllers[c],
                              keyboardType: TextInputType.number,
                              decoration: InputDecoration(labelText: c),
                              validator: (v) =>
                                  v == null || v.isEmpty ? 'Required' : null,
                            ),
                          );
                        }).toList(),
                      );
                    },
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: () async {
                        if (!_formKey.currentState!.validate()) return;
                        final input = <String, double>{};
                        for (final c in widget.model.featureCols) {
                          input[c] = double.parse(_controllers[c]!.text);
                        }
                        final pred = widget.model
                            .predictSingle(input, stats: widget.store.stats);
                        widget.store.addSample(Sample.fromMap(input, pred));
                        if (mounted) {
                          showDialog(
                            context: context,
                            builder: (_) => AlertDialog(
                              title: const Text('Prediction'),
                              content: Text(
                                'pH: ${pred.pH.toStringAsFixed(3)}\n'
                                'OD: ${pred.od.toStringAsFixed(3)}\n'
                                'TURB: ${pred.turb.toStringAsFixed(3)}\n'
                                'Index: ${pred.pollution.toStringAsFixed(3)}\n'
                                'Level: ${pollutionLevel(pred.pollution)}',
                              ),
                            ),
                          );
                        }
                      },
                      icon: const Icon(Icons.analytics_outlined),
                      label: const Text('Predict'),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x1A0F172A),
                    blurRadius: 12,
                    offset: Offset(0, 6),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Inference benchmark',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Measures average latency on the device.',
                    style: TextStyle(color: Color(0xFF64748B)),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _benchmarking
                          ? null
                          : () async {
                              setState(() {
                                _benchmarking = true;
                              });
                              try {
                                final res =
                                    widget.model.benchmark(runs: 5000, warmup: 200);
                                if (mounted) {
                                  setState(() {
                                    _bench = res;
                                  });
                                }
                              } finally {
                                if (mounted) {
                                  setState(() {
                                    _benchmarking = false;
                                  });
                                }
                              }
                            },
                      icon: _benchmarking
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.speed),
                      label: Text(_benchmarking ? 'Measuring...' : 'Run benchmark'),
                    ),
                  ),
                  if (_bench != null) ...[
                    const SizedBox(height: 12),
                    Text(
                      'Average latency: ${_bench!.meanMs.toStringAsFixed(3)} ms\n'
                      'Standard deviation: ${_bench!.stdMs.toStringAsFixed(3)} ms\n'
                      'Throughput: ${_bench!.throughput.toStringAsFixed(2)} samples/s\n'
                      'Runs: ${_bench!.runs}',
                      style: const TextStyle(color: Color(0xFF0F172A)),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: 16),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x1A0F172A),
                    blurRadius: 12,
                    offset: Offset(0, 6),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Estimated energy (mJ/inf)',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Measures current via Android and estimates energy per inference.',
                    style: TextStyle(color: Color(0xFF64748B)),
                  ),
                  const SizedBox(height: 12),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton.icon(
                      onPressed: _energying
                          ? null
                          : () async {
                              setState(() {
                                _energying = true;
                              });
                              try {
                                final idle = await _power.sampleAverage(
                                  duration: const Duration(seconds: 5),
                                  interval: const Duration(milliseconds: 500),
                                );
                                final loadBench = await widget.model.benchmarkWithPower(
                                  samplePower: _power.readPower,
                                  runs: 200000,
                                  warmup: 200,
                                  chunk: 1000,
                                );
                                final loadPower = loadBench.meanPowerW;
                                if (idle != null && loadPower != null) {
                                  final deltaP = (loadPower - idle.meanPowerW);
                                  final energyJ = (deltaP > 0 ? deltaP : 0) /
                                      loadBench.throughput;
                                  final energyMJ = energyJ * 1000.0;
                                  if (mounted) {
                                    setState(() {
                                      _energy = EnergyEstimate(
                                        idlePowerW: idle.meanPowerW,
                                        loadPowerW: loadPower,
                                        deltaPowerW: deltaP,
                                        energyPerInfMJ: energyMJ,
                                        throughput: loadBench.throughput,
                                        samples: loadBench.powerSamples,
                                      );
                                    });
                                  }
                                } else if (mounted) {
                                  setState(() {
                                    _energy = null;
                                  });
                                }
                              } finally {
                                if (mounted) {
                                  setState(() {
                                    _energying = false;
                                  });
                                }
                              }
                            },
                      icon: _energying
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.bolt),
                      label: Text(_energying ? 'Measuring...' : 'Measure energy'),
                    ),
                  ),
                  if (_energy != null) ...[
                    const SizedBox(height: 12),
                    Text(
                      'Idle power: ${_energy!.idlePowerW.toStringAsFixed(3)} W\n'
                      'Load power: ${_energy!.loadPowerW.toStringAsFixed(3)} W\n'
                      'ΔP: ${_energy!.deltaPowerW.toStringAsFixed(3)} W\n'
                      'Energy: ${_energy!.energyPerInfMJ.toStringAsFixed(4)} mJ/inf\n'
                      'Throughput: ${_energy!.throughput.toStringAsFixed(2)} samples/s\n'
                      'Power samples: ${_energy!.samples}',
                      style: const TextStyle(color: Color(0xFF0F172A)),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class EnergyEstimate {
  final double idlePowerW;
  final double loadPowerW;
  final double deltaPowerW;
  final double energyPerInfMJ;
  final double throughput;
  final int samples;

  EnergyEstimate({
    required this.idlePowerW,
    required this.loadPowerW,
    required this.deltaPowerW,
    required this.energyPerInfMJ,
    required this.throughput,
    required this.samples,
  });
}

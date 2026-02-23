import 'package:flutter/material.dart';

class PollutionLegend extends StatelessWidget {
  const PollutionLegend({super.key});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Index',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 6),
            _LegendRow(color: Colors.green, label: 'Low (0.00–0.24)'),
            _LegendRow(color: Colors.yellow, label: 'Moderate (0.25–0.49)'),
            _LegendRow(color: Colors.orange, label: 'Poor (0.50–0.74)'),
            _LegendRow(color: Colors.red, label: 'Very poor (0.75–1.00)'),
          ],
        ),
      ),
    );
  }
}

class _LegendRow extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendRow({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 12,
            height: 12,
            color: color,
          ),
          const SizedBox(width: 6),
          Text(label),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';

import '../services/data_store.dart';
import '../services/model_service.dart';
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
          msg == null ? 'Carregando modelo...' : 'Erro ao carregar modelo:\n$msg',
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
                    'Entrada manual',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 4),
                  const Text(
                    'Digite uma amostra para gerar a predição.',
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
                                  v == null || v.isEmpty ? 'Obrigatório' : null,
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
                              title: const Text('Predição'),
                              content: Text(
                                'pH: ${pred.pH.toStringAsFixed(3)}\n'
                                'OD: ${pred.od.toStringAsFixed(3)}\n'
                                'TURB: ${pred.turb.toStringAsFixed(3)}\n'
                                'Índice: ${pred.pollution.toStringAsFixed(3)}\n'
                                'Nível: ${pollutionLevel(pred.pollution)}',
                              ),
                            ),
                          );
                        }
                      },
                      icon: const Icon(Icons.analytics_outlined),
                      label: const Text('Predizer'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

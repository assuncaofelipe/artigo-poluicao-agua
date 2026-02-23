import 'dart:io';

import 'package:csv/csv.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import '../models/sample.dart';
import '../services/data_store.dart';
import '../services/model_service.dart';
import '../utils/pollution.dart';

class CsvTab extends StatefulWidget {
  final ModelService model;
  final DataStore store;

  const CsvTab({super.key, required this.model, required this.store});

  @override
  State<CsvTab> createState() => _CsvTabState();
}

class _CsvTabState extends State<CsvTab>
    with AutomaticKeepAliveClientMixin<CsvTab> {
  String _status = 'No file loaded';
  String? _lastExportPath;
  String? _lastExportContent;

  @override
  Widget build(BuildContext context) {
    super.build(context);
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
                  'Import CSV',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Select a file and generate predictions.',
                  style: TextStyle(color: Color(0xFF64748B)),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () async {
                      final result = await FilePicker.platform.pickFiles(
                        type: FileType.custom,
                        allowedExtensions: ['csv'],
                      );
                      if (result == null || result.files.isEmpty) return;
                      final file = File(result.files.single.path!);
                      final text = await file.readAsString();
                      List<List<dynamic>> rows = const CsvToListConverter(
                        fieldDelimiter: ';',
                      ).convert(text);
                      if (rows.isEmpty) return;
                      if (rows.first.length == 1 &&
                          rows.first.first.toString().contains(',')) {
                        rows = const CsvToListConverter(
                          fieldDelimiter: ',',
                        ).convert(text);
                      }
                      if (rows.isEmpty) return;

                      final header = rows.first
                          .map((e) => e.toString().trim())
                          .toList(growable: false);
                      final samples = <Sample>[];
                      for (var i = 1; i < rows.length; i++) {
                        final row = rows[i];
                        if (row.length != header.length) continue;
                        final map = <String, double>{};
                        for (var j = 0; j < header.length; j++) {
                          final key = header[j].trim();
                          final raw = row[j].toString().replaceAll(',', '.');
                          final v = double.tryParse(raw);
                          if (v == null) continue;
                          map[key] = v;
                        }
                        if (!widget.model.featureCols.every(map.containsKey)) {
                          continue;
                        }
                        final pred = widget.model
                            .predictSingle(map, stats: widget.store.stats);
                        samples.add(Sample.fromMap(map, pred));
                      }
                      widget.store.setSamples(samples);
                      setState(() {
                        _status = 'Loaded: ${samples.length} samples';
                      });
                    },
                    icon: const Icon(Icons.upload_file_outlined),
                    label: const Text('Load CSV'),
                  ),
                ),
                const SizedBox(height: 8),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF1F5F9),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(_status),
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () async {
                      final dir = await getApplicationDocumentsDirectory();
                      final file = File('${dir.path}/predictions.csv');
                      final rows = <List<dynamic>>[
                        [
                          ...widget.model.featureCols,
                          'pH_pred',
                          'OD_pred',
                          'TURB_pred',
                          'pollution_index',
                          'pollution_level'
                        ]
                      ];
                      for (final s in widget.store.samples) {
                        rows.add([
                          ...widget.model.featureCols
                              .map((c) => s.features[c] ?? 0.0),
                          s.pred.pH,
                          s.pred.od,
                          s.pred.turb,
                          s.pred.pollution,
                          pollutionLevel(s.pred.pollution)
                        ]);
                      }
                      final csv = const ListToCsvConverter(fieldDelimiter: ';')
                          .convert(rows);
                      await file.writeAsString(csv);
                      if (mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(content: Text('Saved to ${file.path}')),
                        );
                        setState(() {
                          _lastExportPath = file.path;
                          _lastExportContent = csv;
                        });
                      }
                    },
                    icon: const Icon(Icons.download_outlined),
                    label: const Text('Export predictions'),
                  ),
                ),
                const SizedBox(height: 12),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: _lastExportPath == null
                        ? null
                        : () async {
                            final path = _lastExportPath!;
                            await Share.shareXFiles([XFile(path)],
                                text:
                                    'Predictions generated by Quality Monitoring.');
                          },
                    icon: const Icon(Icons.share_outlined),
                    label: const Text('Share predictions'),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          if (_lastExportContent != null)
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
                    'Exported CSV preview',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Container(
                    height: 200,
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.black12),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: SingleChildScrollView(
                      child: SelectableText(
                        _lastExportContent!,
                        style: const TextStyle(fontSize: 12),
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  @override
  bool get wantKeepAlive => true;
}

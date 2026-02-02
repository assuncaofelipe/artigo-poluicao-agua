import 'dart:convert';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_map_mbtiles/flutter_map_mbtiles.dart';
import 'package:latlong2/latlong.dart';
import 'package:mbtiles/mbtiles.dart';
import 'package:vector_map_tiles/vector_map_tiles.dart';
import 'package:vector_map_tiles_mbtiles/vector_map_tiles_mbtiles.dart';
import 'package:vector_tile_renderer/vector_tile_renderer.dart' as vtr;
import 'package:path_provider/path_provider.dart';

import '../services/data_store.dart';
import '../services/cached_tile_provider.dart';
import '../widgets/pollution_legend.dart';
import '../utils/pollution.dart';

class MapTab extends StatefulWidget {
  final DataStore store;

  const MapTab({super.key, required this.store});

  @override
  State<MapTab> createState() => _MapTabState();
}

class _MapTabState extends State<MapTab> {
  MbTilesTileProvider? _mbtilesProvider;
  MbTiles? _mbtilesDb;
  vtr.Theme? _vectorTheme;
  bool _loadingTheme = false;
  String? _themeError;
  bool _mbtilesIsVector = false;
  String? _mbtilesPath;
  CachedNetworkTileProvider? _onlineTileProvider;
  bool _pickingMbtiles = false;
  bool _loadingTiles = false;

  @override
  void initState() {
    super.initState();
    _initOnlineCache();
    _autoLoadMbtiles();
  }

  @override
  void dispose() {
    _mbtilesProvider?.dispose();
    _mbtilesDb?.dispose();
    _onlineTileProvider?.dispose();
    super.dispose();
  }

  Future<void> _pickMbtiles() async {
    if (_pickingMbtiles) return;
    setState(() {
      _pickingMbtiles = true;
    });
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
    );
    if (!mounted) return;
    setState(() {
      _pickingMbtiles = false;
    });
    if (result == null || result.files.isEmpty) return;
    final path = result.files.single.path;
    if (path == null) return;
    _mbtilesProvider?.dispose();
    _mbtilesDb?.dispose();
    setState(() {
      _loadingTiles = true;
      _mbtilesPath = path;
      _mbtilesDb = MbTiles(mbtilesPath: path, gzip: true);
      final format = _mbtilesDb!.getMetadata().format.toLowerCase();
      _mbtilesIsVector = format == 'pbf' || format == 'mvt';
      _mbtilesProvider =
          _mbtilesIsVector ? null : MbTilesTileProvider.fromPath(path: path);
    });
    if (_mbtilesIsVector) {
      await _ensureVectorTheme();
    }
    _finishLoadingTiles();
  }

  Future<void> _initOnlineCache() async {
    if (_onlineTileProvider != null) return;
    final dir = await getApplicationCacheDirectory();
    setState(() {
      _onlineTileProvider = CachedNetworkTileProvider(cacheDir: dir);
    });
  }

  Future<void> _autoLoadMbtiles() async {
    if (_mbtilesPath != null || _mbtilesDb != null) return;
    try {
      final downloads = await getDownloadsDirectory();
      if (downloads == null) return;
      final file = File(
        '${downloads.path}/osm-2020-02-10-v3.11_south-america_brazil_t.mbtiles',
      );
      if (!await file.exists()) return;
      _mbtilesProvider?.dispose();
      _mbtilesDb?.dispose();
      setState(() {
        _loadingTiles = true;
        _mbtilesPath = file.path;
        _mbtilesDb = MbTiles(mbtilesPath: file.path, gzip: true);
        final format = _mbtilesDb!.getMetadata().format.toLowerCase();
        _mbtilesIsVector = format == 'pbf' || format == 'mvt';
        _mbtilesProvider = _mbtilesIsVector
            ? null
            : MbTilesTileProvider.fromPath(path: file.path);
      });
      if (_mbtilesIsVector) {
        await _ensureVectorTheme();
      }
      _finishLoadingTiles();
    } catch (_) {
      // ignore auto-load errors
    }
  }

  void _finishLoadingTiles() {
    Future.delayed(const Duration(seconds: 2), () {
      if (!mounted) return;
      setState(() {
        _loadingTiles = false;
      });
    });
  }

  void _clearMbtiles() {
    _mbtilesProvider?.dispose();
    _mbtilesDb?.dispose();
    setState(() {
      _mbtilesProvider = null;
      _mbtilesDb = null;
      _mbtilesIsVector = false;
      _mbtilesPath = null;
    });
  }

  Future<void> _ensureVectorTheme() async {
    if (_vectorTheme != null || _loadingTheme) return;
    setState(() {
      _loadingTheme = true;
      _themeError = null;
    });
    try {
      final text = await rootBundle.loadString('assets/map/style.json');
      final jsonMap = jsonDecode(text);
      final theme = vtr.ThemeReader().read(jsonMap);
      setState(() {
        _vectorTheme = theme;
      });
    } catch (e) {
      setState(() {
        _themeError = e.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _loadingTheme = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.store.samples.isEmpty) {
      return const Center(
        child: Text('Carregue um CSV ou insira dados manualmente.'),
      );
    }
    final center =
        widget.store.samples.first.latLng ?? const LatLng(-3.1, -58.5);
    return Stack(
      children: [
        ClipRRect(
          borderRadius: const BorderRadius.only(
            topLeft: Radius.circular(16),
            topRight: Radius.circular(16),
          ),
          child: FlutterMap(
            options: MapOptions(
              initialCenter: center,
              initialZoom: 13,
            ),
            children: [
              if (_mbtilesDb != null && _mbtilesIsVector && _vectorTheme != null)
                VectorTileLayer(
                  theme: _vectorTheme!,
                  tileProviders: TileProviders({
                    'openmaptiles':
                        MbTilesVectorTileProvider(mbtiles: _mbtilesDb!),
                  }),
                  maximumZoom: 18,
                )
              else
                TileLayer(
                  urlTemplate: _mbtilesProvider == null
                      ? 'https://tile.openstreetmap.org/{z}/{x}/{y}.png'
                      : null,
                  userAgentPackageName: 'com.serpa.predict',
                  tileProvider:
                      _mbtilesProvider ?? _onlineTileProvider,
                ),
              CircleLayer(
                circles: widget.store.samples
                    .where((s) => s.latLng != null)
                    .map((s) => CircleMarker(
                          point: s.latLng!,
                          color: pollutionColor(s.pred.pollution),
                          radius: 6,
                          borderStrokeWidth: 1,
                          borderColor: Colors.black54,
                        ))
                    .toList(),
              ),
            ],
          ),
        ),
        Positioned(
          left: 12,
          top: 12,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              boxShadow: const [
                BoxShadow(
                  color: Color(0x1A0F172A),
                  blurRadius: 10,
                  offset: Offset(0, 6),
                ),
              ],
            ),
            child: Text(
              _mbtilesProvider == null && _mbtilesDb == null
                  ? 'Mapa online'
                  : 'Mapa offline (MBTiles)',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
        ),
        Positioned(
          left: 12,
          bottom: 12,
          child: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              boxShadow: const [
                BoxShadow(
                  color: Color(0x1A0F172A),
                  blurRadius: 10,
                  offset: Offset(0, 6),
                ),
              ],
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                ElevatedButton(
                  onPressed: _pickingMbtiles ? null : _pickMbtiles,
                  child: Text(_pickingMbtiles ? 'Abrindo...' : 'Usar MBTiles'),
                ),
                const SizedBox(width: 8),
                if (_mbtilesPath != null)
                  IconButton(
                    onPressed: _clearMbtiles,
                    tooltip: 'Remover MBTiles',
                    icon: const Icon(Icons.close),
                  ),
              ],
            ),
          ),
        ),
        const Positioned(
          right: 12,
          bottom: 12,
          child: PollutionLegend(),
        ),
        if (_mbtilesPath != null)
          Positioned(
            left: 12,
            top: 54,
            child: Container(
              constraints: const BoxConstraints(maxWidth: 220),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x1A0F172A),
                    blurRadius: 10,
                    offset: Offset(0, 6),
                  ),
                ],
              ),
              child: Text(
                'Arquivo: ${_mbtilesPath!.split('/').last}',
                style: const TextStyle(fontSize: 12),
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ),
        if (_loadingTheme)
          Positioned(
            left: 12,
            top: 96,
            child: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x1A0F172A),
                    blurRadius: 10,
                    offset: Offset(0, 6),
                  ),
                ],
              ),
              child: const Text('Carregando estilo vetorial...'),
            ),
          ),
        if (_themeError != null)
          Positioned(
            left: 12,
            top: 96,
            child: Container(
              constraints: const BoxConstraints(maxWidth: 240),
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x1A0F172A),
                    blurRadius: 10,
                    offset: Offset(0, 6),
                  ),
                ],
              ),
              child: Text(
                'Erro no estilo: $_themeError',
                style: const TextStyle(fontSize: 12),
              ),
            ),
          ),
        if (_loadingTiles)
          Positioned(
            left: 12,
            right: 12,
            top: 96,
            child: Card(
              child: Padding(
                padding: const EdgeInsets.all(10),
                child: Row(
                  children: const [
                    SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    SizedBox(width: 10),
                    Text('Carregando tiles...'),
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }
}

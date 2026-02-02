import 'dart:async';
import 'dart:io';
import 'dart:ui' show Codec, ImmutableBuffer;
import 'dart:convert' show utf8;

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/painting.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:http/http.dart';
import 'package:http/retry.dart';

class CachedNetworkTileProvider extends TileProvider {
  final Directory cacheDir;
  final BaseClient _httpClient;
  final bool silenceExceptions;

  CachedNetworkTileProvider({
    required this.cacheDir,
    super.headers,
    BaseClient? httpClient,
    this.silenceExceptions = true,
  }) : _httpClient = httpClient ?? RetryClient(Client());

  @override
  ImageProvider getImage(TileCoordinates coordinates, TileLayer options) {
    final url = getTileUrl(coordinates, options);
    final fallbackUrl = getTileFallbackUrl(coordinates, options);
    return _CachedNetworkImageProvider(
      url: url,
      fallbackUrl: fallbackUrl,
      headers: headers,
      httpClient: _httpClient,
      cacheDir: cacheDir,
      silenceExceptions: silenceExceptions,
    );
  }

  @override
  Future<void> dispose() async {
    _httpClient.close();
    super.dispose();
  }
}

@immutable
class _CachedNetworkImageProvider
    extends ImageProvider<_CachedNetworkImageProvider> {
  final String url;
  final String? fallbackUrl;
  final Map<String, String> headers;
  final BaseClient httpClient;
  final Directory cacheDir;
  final bool silenceExceptions;

  const _CachedNetworkImageProvider({
    required this.url,
    required this.fallbackUrl,
    required this.headers,
    required this.httpClient,
    required this.cacheDir,
    required this.silenceExceptions,
  });

  @override
  ImageStreamCompleter loadImage(
    _CachedNetworkImageProvider key,
    ImageDecoderCallback decode,
  ) =>
      MultiFrameImageStreamCompleter(
        codec: _load(key, decode),
        scale: 1,
        debugLabel: url,
      );

  Future<Codec> _load(
    _CachedNetworkImageProvider key,
    ImageDecoderCallback decode, {
    bool useFallback = false,
  }) async {
    final targetUrl = useFallback ? (fallbackUrl ?? url) : url;
    final file = File(_cachePath(targetUrl));
    try {
      if (await file.exists()) {
        final bytes = await file.readAsBytes();
        final buffer = await ImmutableBuffer.fromUint8List(bytes);
        return decode(buffer);
      }
      final bytes =
          await httpClient.readBytes(Uri.parse(targetUrl), headers: headers);
      await _ensureCacheDir();
      await file.writeAsBytes(bytes, flush: true);
      final buffer = await ImmutableBuffer.fromUint8List(bytes);
      return decode(buffer);
    } catch (err) {
      if (!useFallback && fallbackUrl != null) {
        return _load(key, decode, useFallback: true);
      }
      if (!silenceExceptions) rethrow;
      final buffer =
          await ImmutableBuffer.fromUint8List(TileProvider.transparentImage);
      return decode(buffer);
    }
  }

  String _cachePath(String inputUrl) {
    final hash = sha1.convert(utf8.encode(inputUrl)).toString();
    return '${cacheDir.path}/tiles/$hash.tile';
  }

  Future<void> _ensureCacheDir() async {
    final dir = Directory('${cacheDir.path}/tiles');
    if (!await dir.exists()) {
      await dir.create(recursive: true);
    }
  }

  @override
  SynchronousFuture<_CachedNetworkImageProvider> obtainKey(
    ImageConfiguration configuration,
  ) =>
      SynchronousFuture(this);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is _CachedNetworkImageProvider &&
          fallbackUrl == null &&
          url == other.url);

  @override
  int get hashCode => Object.hashAll([url, if (fallbackUrl != null) fallbackUrl]);
}

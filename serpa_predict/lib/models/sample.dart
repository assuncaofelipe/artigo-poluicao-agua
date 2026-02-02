import 'package:latlong2/latlong.dart';

import '../models/prediction.dart';
import '../utils/geo.dart';

class Sample {
  final Map<String, double> features;
  final Prediction pred;

  Sample({required this.features, required this.pred});

  LatLng? get latLng {
    final x = features['x'];
    final y = features['y'];
    if (x == null || y == null) return null;
    return utmToLatLng(x, y, 21, false);
  }

  static Sample fromMap(Map<String, double> map, Prediction pred) {
    return Sample(features: map, pred: pred);
  }
}


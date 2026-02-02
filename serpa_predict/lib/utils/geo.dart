import 'dart:math';

import 'package:latlong2/latlong.dart';

LatLng utmToLatLng(double easting, double northing, int zone, bool northern) {
  const a = 6378137.0;
  const e = 0.0818191908;
  const e1sq = 0.006739497;
  const k0 = 0.9996;

  var x = easting - 500000.0;
  var y = northing;
  if (!northern) {
    y -= 10000000.0;
  }

  final m = y / k0;
  final mu =
      m / (a * (1 - e * e / 4 - 3 * pow(e, 4) / 64 - 5 * pow(e, 6) / 256));

  final e1 = (1 - sqrt(1 - e * e)) / (1 + sqrt(1 - e * e));
  final j1 = 3 * e1 / 2 - 27 * pow(e1, 3) / 32;
  final j2 = 21 * pow(e1, 2) / 16 - 55 * pow(e1, 4) / 32;
  final j3 = 151 * pow(e1, 3) / 96;
  final j4 = 1097 * pow(e1, 4) / 512;

  final fp = mu +
      j1 * sin(2 * mu) +
      j2 * sin(4 * mu) +
      j3 * sin(6 * mu) +
      j4 * sin(8 * mu);

  final sinFp = sin(fp);
  final cosFp = cos(fp);
  final c1 = e1sq * pow(cosFp, 2);
  final t1 = pow(tan(fp), 2);
  final r1 = a * (1 - e * e) / pow(1 - e * e * pow(sinFp, 2), 1.5);
  final n1 = a / sqrt(1 - e * e * pow(sinFp, 2));
  final d = x / (n1 * k0);

  final q1 = n1 * tan(fp) / r1;
  final q2 = pow(d, 2) / 2;
  final q3 = (5 + 3 * t1 + 10 * c1 - 4 * pow(c1, 2) - 9 * e1sq) *
      pow(d, 4) /
      24;
  final q4 = (61 +
          90 * t1 +
          298 * c1 +
          45 * pow(t1, 2) -
          252 * e1sq -
          3 * pow(c1, 2)) *
      pow(d, 6) /
      720;
  final lat = fp - q1 * (q2 - q3 + q4);

  final q5 = d;
  final q6 = (1 + 2 * t1 + c1) * pow(d, 3) / 6;
  final q7 =
      (5 - 2 * c1 + 28 * t1 - 3 * pow(c1, 2) + 8 * e1sq + 24 * pow(t1, 2)) *
          pow(d, 5) /
          120;
  final lon = (q5 - q6 + q7) / cosFp;

  final lon0 = (zone - 1) * 6 - 180 + 3;
  final latDeg = lat * 180 / pi;
  final lonDeg = lon0 + lon * 180 / pi;
  return LatLng(latDeg, lonDeg);
}


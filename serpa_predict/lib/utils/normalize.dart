double normalizeScalar(double v, double min, double max) {
  if (max - min == 0) return 0;
  return (v - min) / (max - min);
}


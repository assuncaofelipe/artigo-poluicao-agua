class PowerReading {
  final double currentUA;
  final double voltageUV;

  const PowerReading({required this.currentUA, required this.voltageUV});

  double get currentA => currentUA.abs() / 1e6;

  double get voltageV => voltageUV / 1e6;

  double get powerW => currentA * voltageV;
}

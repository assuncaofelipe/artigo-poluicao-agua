#include <Wire.h>
#include <Adafruit_INA219.h>

Adafruit_INA219 ina219;

const unsigned long TELEMETRY_MS = 500;
const float EMA_ALPHA = 0.2;

unsigned long lastSend = 0;

float voltage_V = 0;
float current_mA = 0;
float power_mW = 0;

float powerEMA_mW = 0;
float energy_Wh = 0;

bool emaInitialized = false;

void setup() {
  Serial.begin(9600);
  Wire.begin();

  if (!ina219.begin()) {
    Serial.println("Erro ao encontrar INA219");
    while (1);
  }

  ina219.setCalibration_32V_2A();
}

void loop() {

  unsigned long now = millis();

  if (now - lastSend >= TELEMETRY_MS) {

    float deltaHours = (lastSend == 0) ?
                        (TELEMETRY_MS / 3600000.0) :
                        ((now - lastSend) / 3600000.0);

    lastSend = now;

    voltage_V = ina219.getBusVoltage_V();
    current_mA = ina219.getCurrent_mA();
    power_mW = ina219.getPower_mW();

    if (!emaInitialized) {
      powerEMA_mW = power_mW;
      emaInitialized = true;
    } else {
      powerEMA_mW = EMA_ALPHA * power_mW +
                    (1 - EMA_ALPHA) * powerEMA_mW;
    }

    energy_Wh += (powerEMA_mW / 1000.0) * deltaHours;

    Serial.print("V=");
    Serial.print(voltage_V, 2);
    Serial.print("V | I=");
    Serial.print(current_mA, 2);
    Serial.print("mA | P=");
    Serial.print(powerEMA_mW, 2);
    Serial.print("mW | E=");
    Serial.print(energy_Wh, 5);
    Serial.println("Wh");
  }
}

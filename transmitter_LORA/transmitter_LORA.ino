#include <Wire.h>
#include <MPU6050.h>
#include <SPI.h>
#include <LoRa.h>
#include <math.h>

MPU6050 mpu;

#define SS 5
#define RST 14
#define DIO0 26

void setup() {
  Serial.begin(115200);
  delay(2000);

  Wire.begin(21, 22);
  mpu.initialize();

  Serial.println("MPU6050 Ready ✅");

  SPI.begin(18, 19, 23, 5);

  LoRa.setPins(SS, RST, DIO0);

  if (!LoRa.begin(433E6)) {
    Serial.println("LoRa Failed ❌");
    while (1);
  }

  Serial.println("LoRa Ready ✅");

  Serial.println("ax,ay,az,impact,label");
}

void loop() {
  int16_t ax, ay, az;
  mpu.getAcceleration(&ax, &ay, &az);

  float ax_g = ax / 16384.0;
  float ay_g = ay / 16384.0;
  float az_g = az / 16384.0;

  float impact = sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);

  int label = 0; // 0 = normal, 1 = accident

  // 🚨 Accident Detection
  if (impact > 2.2) {
    label = 1;

    LoRa.beginPacket();
    LoRa.print("ALERT,");
    LoRa.print(impact);
    LoRa.endPacket();

    Serial.println("🚨 ACCIDENT SENT VIA LORA");
    delay(1000); // avoid spam
  }

  // 📊 Dataset Output
  Serial.print(ax_g); Serial.print(",");
  Serial.print(ay_g); Serial.print(",");
  Serial.print(az_g); Serial.print(",");
  Serial.print(impact); Serial.print(",");
  Serial.println(label);

  delay(300);
}
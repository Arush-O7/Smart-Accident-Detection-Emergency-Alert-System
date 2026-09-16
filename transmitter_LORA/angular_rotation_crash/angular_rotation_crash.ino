#include <Wire.h>
#include <MPU6050.h>
#include <SPI.h>
#include <LoRa.h>
#include <math.h>

MPU6050 mpu;

// 🔹 LoRa Pins
#define SS 5
#define RST 14
#define DIO0 26

unsigned long lastSent = 0;
bool alertActive = false; // Tracks if an accident is currently active

void setup() {
  Serial.begin(115200);
  delay(2000);

  // Initialize I2C for MPU6050
  Wire.begin(21, 22);
  mpu.initialize();
  Serial.println("MPU6050 Ready ✅");

  // Initialize LoRa
  SPI.begin(18, 19, 23, 5);
  LoRa.setPins(SS, RST, DIO0);

  if (!LoRa.begin(433E6)) {
    Serial.println("LoRa Failed ❌");
    while (1);
  }
  Serial.println("LoRa Ready ✅");

  // CSV Header for Dataset Collection
  Serial.println("ax,ay,az,gx,gy,gz,impact,rotation,pitch,roll,label");
}

void loop() {
  int16_t ax, ay, az, gx, gy, gz;

  // Read all 6 axes simultaneously
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  // 🔹 Accelerometer (G-force)
  float ax_g = ax / 16384.0;
  float ay_g = ay / 16384.0;
  float az_g = az / 16384.0;
  float impact = sqrt(ax_g * ax_g + ay_g * ay_g + az_g * az_g);

  // 🔹 Gyroscope (Degrees per second)
  float gx_dps = gx / 131.0;
  float gy_dps = gy / 131.0;
  float gz_dps = gz / 131.0;
  float rotation_rate = sqrt(gx_dps * gx_dps + gy_dps * gy_dps + gz_dps * gz_dps);

  // 🔹 Tilt (Orientation / Pitch & Roll)
  float pitch = atan2(-ax_g, sqrt(ay_g * ay_g + az_g * az_g)) * 180.0 / PI;
  float roll  = atan2(ay_g, az_g) * 180.0 / PI;

  int label = 0; // 0 = normal, 1 = accident

  // 🔥 MULTI-CONDITION ACCIDENT DETECTION
  bool highImpactCrash = (impact > 2.2);                           // Straight hard crash
  bool angledCrash     = (impact > 2.2 && rotation_rate > 100);    // Angled crash causing spin
  bool rollover        = (abs(pitch) > 75.0 || abs(roll) > 75.0);  // Vehicle flipped on side/roof
  bool extremeRotation = (rotation_rate > 180);                    // Violent spin without immediate hard impact

  // 🚨 SEND ALERT (With 3-second cooldown to avoid spamming the receiver)
  if ((highImpactCrash || angledCrash || rollover || extremeRotation) && millis() - lastSent > 3000) {
    label = 1;
    alertActive = true; 
    lastSent = millis();

    LoRa.beginPacket();

    if (rollover) {
      LoRa.print("ALERT,ROLLOVER,");
      LoRa.print(max(abs(pitch), abs(roll)));
      Serial.println("🚨 ROLLOVER DETECTED");
    } 
    else if (highImpactCrash) {
      LoRa.print("ALERT,HIGH_IMPACT,");
      LoRa.print(impact);
      Serial.println("🚨 HIGH IMPACT CRASH");
    } 
    else if (angledCrash) {
      LoRa.print("ALERT,ANGLED_CRASH,");
      LoRa.print(impact);
      Serial.println("🚨 ANGLED CRASH");
    } 
    else if (extremeRotation) {
      LoRa.print("ALERT,ROTATION,");
      LoRa.print(rotation_rate);
      Serial.println("🚨 EXTREME ROTATION");
    }

    LoRa.endPacket();
  }

  // 🔁 SEND CLEAR SIGNAL (Only once, 10 seconds after an accident has settled)
  if (impact < 1.2 && alertActive && millis() - lastSent > 10000) {
    LoRa.beginPacket();
    LoRa.print("CLEAR");
    LoRa.endPacket();
    
    alertActive = false; // Reset state so we don't flood the channel
    Serial.println("✅ CLEAR SENT - System Reset");
  }

  // 📊 DATASET LOGGING (Output to Serial Monitor)
  Serial.print(ax_g); Serial.print(",");
  Serial.print(ay_g); Serial.print(",");
  Serial.print(az_g); Serial.print(",");
  Serial.print(gx_dps); Serial.print(",");
  Serial.print(gy_dps); Serial.print(",");
  Serial.print(gz_dps); Serial.print(",");
  Serial.print(impact); Serial.print(",");
  Serial.print(rotation_rate); Serial.print(",");
  Serial.print(pitch); Serial.print(",");
  Serial.print(roll); Serial.print(",");
  Serial.println(label);

  delay(300); // Loop delay
}
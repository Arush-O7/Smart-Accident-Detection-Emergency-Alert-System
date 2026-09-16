#include <SPI.h>
#include <LoRa.h>

#define SS 5
#define RST 14
#define DIO0 26

void setup() {
  Serial.begin(115200);

  SPI.begin(18, 19, 23, 5);
  LoRa.setPins(SS, RST, DIO0);

  if (!LoRa.begin(433E6)) {
    Serial.println("LoRa Failed");
    while (1);
  }

  Serial.println("Receiver Ready 📡");
}

void loop() {
  int packetSize = LoRa.parsePacket();

  if (packetSize) {
    String msg = "";

    while (LoRa.available()) {
      msg += (char)LoRa.read();
    }

    Serial.print("Received: ");
    Serial.println(msg);

    if (msg.startsWith("ALERT")) {
      Serial.println("🚨 ACCIDENT DETECTED 🚨");
    }
  }
}
#include <WiFi.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <SPI.h>
#include <LoRa.h>

// 🔹 LoRa Pins
#define SS 5
#define RST 14
#define DIO0 26

// 🔹 WiFi — fill in before flashing (never commit real credentials)
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// 🔹 SMS API (CircuitDigest Cloud) — fill in before flashing
const char* apiKey = "YOUR_CIRCUITDIGEST_API_KEY";
const char* mobileNumber = "91XXXXXXXXXX";   // country code + number, no "+"

bool alertSent = false;

// ─────────────────────────────────────────
void connectWiFi() {
  WiFi.persistent(false);
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(ssid, password);

  Serial.print("Connecting to WiFi");

  unsigned long startTime = millis();

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");

    if (millis() - startTime > 15000) {
      Serial.println();
      Serial.print("WiFi Status Code: ");
      Serial.println(WiFi.status());
      // 0=IDLE 1=NO_SSID 2=SCAN_COMPLETE
      // 3=CONNECTED 4=CONNECT_FAILED
      // 5=LOST 6=DISCONNECTED

      Serial.println("Timeout! Retrying...");
      WiFi.disconnect();
      delay(1000);
      WiFi.begin(ssid, password);
      startTime = millis();
    }
  }

  Serial.println();
  Serial.println("WiFi Connected ✅");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.print("Signal Strength (RSSI): ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
}

// ─────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(1000);

  // 🌐 WiFi FIRST (before LoRa to avoid RF interference)
  Serial.println("Starting WiFi...");
  connectWiFi();

  // 📡 LoRa Setup
  Serial.println("Starting LoRa...");
  SPI.begin(18, 19, 23, 5);
  LoRa.setPins(SS, RST, DIO0);

  if (!LoRa.begin(433E6)) {
    Serial.println("LoRa Failed ❌");
    while (1);
  }

  Serial.println("LoRa Ready 📡");
  Serial.println("System Ready — Waiting for packets...");
}

// ─────────────────────────────────────────
void loop() {
  // 🔄 Auto-reconnect WiFi if dropped
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi lost! Reconnecting...");
    connectWiFi();
  }

  int packetSize = LoRa.parsePacket();

  if (packetSize) {
    String msg = "";

    while (LoRa.available()) {
      msg += (char)LoRa.read();
    }

    Serial.println("📦 Received: " + msg);
    Serial.print("RSSI: ");
    Serial.println(LoRa.packetRssi());

    // 🚨 Accident detected
    // AI transmitter sends "ALERT,<CRASH|ROLLOVER>,<confidence %>,<peak g or tilt deg>";
    // the older sketches send "ALERT,<TYPE>,<value>".
    if (msg.startsWith("ALERT") && !alertSent) {
      Serial.println("🚨 ACCIDENT DETECTED 🚨");

      String details = describeAlert(msg);
      Serial.println(details);

      sendSMS(details);
      alertSent = true;
    }

    // 🛣️ Non-crash jolt classified by the AI model (speed bump, pothole, harsh manoeuvre): log only
    if (msg.startsWith("EVENT")) {
      Serial.println("🛣️ Road event (no SMS): " + msg.substring(6));
    }

    // 🔁 Reset alert after CLEAR message from sender
    if (msg.startsWith("CLEAR")) {
      alertSent = false;
      Serial.println("✅ Alert Reset — Ready for next event");
    }
  }
}

// ─────────────────────────────────────────
String field(const String &csv, int index) {
  int start = 0;
  for (int i = 0; i < index; i++) {
    start = csv.indexOf(',', start) + 1;
    if (start == 0) return "";
  }
  int end = csv.indexOf(',', start);
  return end < 0 ? csv.substring(start) : csv.substring(start, end);
}

String describeAlert(const String &msg) {
  String type = field(msg, 1);
  if (field(msg, 2).length() == 0) {  // basic transmitter: ALERT,<value>
    return "Vehicle Crash Impact " + type + "g";
  }
  if (field(msg, 3).length() == 0) {  // legacy format: ALERT,<TYPE>,<value>
    return "Vehicle Crash Impact " + field(msg, 2) + "g (" + type + ")";
  }
  String confidence = field(msg, 2);
  String value = field(msg, 3);
  if (type == "ROLLOVER") return "ROLLOVER, tilt " + value + " deg";
  return "AI verified " + type + " (" + confidence + "% confidence), impact " + value + "g";
}

// ─────────────────────────────────────────
void sendSMS(String details) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected ❌ — Attempting reconnect...");
    connectWiFi();
  }

  if (WiFi.status() == WL_CONNECTED) {
    WiFiClientSecure client;
    client.setInsecure(); // Skip SSL cert verification

    HTTPClient http;

    String apiUrl = "https://www.circuitdigest.cloud/api/v1/send_sms?ID=103";

    http.begin(client, apiUrl);
    http.addHeader("Authorization", apiKey);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(10000); // 10 second timeout

    String var1 = "Accident Detection System";
    String var2 = details;

    String payload = "{\"mobiles\":\"" + String(mobileNumber) +
                     "\",\"var1\":\"" + var1 +
                     "\",\"var2\":\"" + var2 + "\"}";

    Serial.println("📤 Sending SMS...");
    Serial.println("Payload: " + payload);

    int httpResponseCode = http.POST(payload);

    Serial.print("HTTP Response Code: ");
    Serial.println(httpResponseCode);

    if (httpResponseCode == 200) {
      Serial.println("✅ SMS Sent Successfully!");
      Serial.println("Response: " + http.getString());
    } else if (httpResponseCode == -1) {
      Serial.println("❌ Connection Failed — Check internet on hotspot");
    } else if (httpResponseCode == 401) {
      Serial.println("❌ Unauthorized — Check API Key");
    } else if (httpResponseCode == 400) {
      Serial.println("❌ Bad Request — Check payload format");
    } else {
      Serial.println("❌ SMS Failed");
      Serial.println("Response: " + http.getString());
    }

    http.end();

  } else {
    Serial.println("❌ Could not reconnect to WiFi — SMS not sent");
  }
}
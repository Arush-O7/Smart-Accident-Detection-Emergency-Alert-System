// Crash detector transmitter: ESP32 + MPU6050 + SX1278
// P(crash) >= 0.80 sends ALERT, 0.50-0.80 starts a cancellable buzzer countdown,
// anything lower is logged as a road event.
#include <Wire.h>
#include <SPI.h>
#include <LoRa.h>
#include <MPU6050.h>
#include "crash_features.h"
#include "crash_model.h"
#include "road_model.h"

#define LORA_SS 5
#define LORA_RST 14
#define LORA_DIO0 26
#define I2C_SDA 21
#define I2C_SCL 22
#define BUZZER_PIN 4
#define CANCEL_BUTTON_PIN 13   // to GND, internal pull-up

#define TRIGGER_G 0.45f
#define EMA_ALPHA 0.02f
#define P_ALERT_NOW 0.80f
#define P_VERIFY 0.50f
#define VERIFY_SECONDS 10
#define ROLLOVER_TILT_DEG 60.0f
#define ROLLOVER_HOLD_MS 3000
#define CLEAR_AFTER_MS 30000
#define STREAM_RAW true            // raw samples on Serial for the dashboard
#define REPORT_ROAD_EVENTS true    // send EVENT packets for non-crash classifications
#define EVENT_REPORT_MIN_MS 5000

// Electronic Cats lib names it BW_42, i2cdevlib uses BW_44
#ifdef MPU6050_DLPF_BW_42
#define DLPF_44HZ MPU6050_DLPF_BW_42
#else
#define DLPF_44HZ MPU6050_DLPF_BW_44
#endif

MPU6050 mpu;

enum State { MONITOR, COLLECTING, VERIFYING, ALERT_SENT, SENSOR_FAULT };
State state = MONITOR;

float ring[CF_WIN][3];        // acceleration, g
float ringGyro[CF_WIN][3];    // angular rate, deg/s
float window_[CF_WIN][3];
float windowGyro[CF_WIN][3];
float feats[CRASH_MODEL_N_FEATURES];          // 30 accelerometer + 7 gyroscope features
float roadProbs[ROAD_MODEL_N_CLASSES];

uint32_t sampleIndex = 0;      // samples since boot
int ringHead = 0;              // next write position
float gEma[3] = {0, 0, 1};
float gUpright[3] = {0, 0, 1}; // calibrated at boot, used by the rollover guard
uint32_t anchor = 0;
float anchorPeak = 0;

unsigned long nextSampleUs = 0;
unsigned long verifyStartMs = 0, alertMs = 0, tiltSinceMs = 0, lastEventReportMs = 0;
int zeroReads = 0;
float pendingP = 0, pendingPeak = 0;

float norm3(const float v[3]) { return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

bool initImu() {
  mpu.initialize();
  if (!mpu.testConnection()) return false;
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_16);   // 2048 LSB/g, a crash easily exceeds 2 g
  mpu.setFullScaleGyroRange(MPU6050_GYRO_FS_2000);   // 16.4 LSB/(deg/s)
  mpu.setDLPFMode(DLPF_44HZ);                        // anti-alias for 100 Hz sampling
  return true;
}

void sendPacket(const char *kind, const char *label, int confidence, float value) {
  LoRa.beginPacket();
  LoRa.printf("%s,%s,%d,%.2f", kind, label, confidence, value);
  LoRa.endPacket();
}

void sendAlert(const char *label, float p, float value) {
  // send 3 times, receiver only sends one SMS
  for (int i = 0; i < 3; i++) {
    sendPacket("ALERT", label, (int)roundf(p * 100), value);
    delay(150 + random(150));
  }
  Serial.printf("# ALERT sent: %s p=%.2f value=%.2f\n", label, p, value);
  digitalWrite(BUZZER_PIN, HIGH);
  alertMs = millis();
  state = ALERT_SENT;
}

void classifyWindow() {
  int start = ringHead;  // ring is full and ringHead is the oldest sample
  for (int i = 0; i < CF_WIN; i++) {
    int j = (start + i) % CF_WIN;
    for (int k = 0; k < 3; k++) { window_[i][k] = ring[j][k]; windowGyro[i][k] = ringGyro[j][k]; }
  }
  unsigned long t0 = micros();
  crash_features(window_, feats);                            // features 0..29 (accelerometer)
  crash_features_gyro(window_, windowGyro, feats + CF_N_FEATURES);   // features 30..36 (gyroscope)
  float pCrash = crash_model_probability(feats);             // model A
  unsigned long dt = micros() - t0;
  float peak = feats[0];

  if (pCrash >= P_VERIFY) {
    Serial.printf("E,CRASH,%.3f,%.2f,%lu\n", pCrash, peak, dt);
    if (pCrash >= P_ALERT_NOW) {
      sendAlert("CRASH", pCrash, peak);
    } else {
      pendingP = pCrash; pendingPeak = peak;
      verifyStartMs = millis();
      state = VERIFYING;
      Serial.printf("# VERIFYING: possible crash p=%.2f, press CANCEL within %d s\n", pCrash, VERIFY_SECONDS);
    }
    return;
  }

  road_model_predict(feats, roadProbs);                      // model B: what kind of jolt was it?
  int best = 0;
  for (int c = 1; c < ROAD_MODEL_N_CLASSES; c++) if (roadProbs[c] > roadProbs[best]) best = c;
  Serial.printf("E,%s,%.3f", ROAD_MODEL_CLASSES[best], pCrash);
  for (int c = 0; c < ROAD_MODEL_N_CLASSES; c++) Serial.printf(",%.3f", roadProbs[c]);
  Serial.printf(",%.2f,%lu\n", peak, dt);
  state = MONITOR;
  if (REPORT_ROAD_EVENTS && best != 0 && millis() - lastEventReportMs > EVENT_REPORT_MIN_MS) {
    sendPacket("EVENT", ROAD_MODEL_CLASSES[best], (int)roundf(roadProbs[best] * 100), peak);
    lastEventReportMs = millis();
  }
}

void onSample(float a[3], float w[3]) {
  for (int k = 0; k < 3; k++) { ring[ringHead][k] = a[k]; ringGyro[ringHead][k] = w[k]; }
  ringHead = (ringHead + 1) % CF_WIN;

  for (int k = 0; k < 3; k++) gEma[k] += EMA_ALPHA * (a[k] - gEma[k]);
  float d[3] = {a[0] - gEma[0], a[1] - gEma[1], a[2] - gEma[2]};
  float dyn = norm3(d);
  uint32_t i = sampleIndex++;

  if (STREAM_RAW) Serial.printf("R,%.3f,%.3f,%.3f\n", a[0], a[1], a[2]);
  if (i < 200) {  // first 2 s: settle and calibrate the upright gravity direction
    for (int k = 0; k < 3; k++) gUpright[k] = gEma[k];
    return;
  }

  // rollover check, separate from the model
  float cosTilt = (gEma[0] * gUpright[0] + gEma[1] * gUpright[1] + gEma[2] * gUpright[2]) /
                  (norm3(gEma) * norm3(gUpright) + 1e-6f);
  float tilt = acosf(constrain(cosTilt, -1.0f, 1.0f)) * 180.0f / PI;
  if (tilt > ROLLOVER_TILT_DEG && state != ALERT_SENT) {
    if (tiltSinceMs == 0) tiltSinceMs = millis();
    if (millis() - tiltSinceMs > ROLLOVER_HOLD_MS) sendAlert("ROLLOVER", 1.0f, tilt);
  } else {
    tiltSinceMs = 0;
  }

  switch (state) {
    case MONITOR:
      if (i >= CF_WIN && dyn > TRIGGER_G) {
        anchor = i; anchorPeak = dyn; state = COLLECTING;
      }
      break;
    case COLLECTING:
      // re-anchor if a bigger jolt comes in
      if (i - anchor > 10 && dyn > max(1.0f, 1.5f * anchorPeak)) { anchor = i; anchorPeak = dyn; }
      anchorPeak = max(anchorPeak, dyn);
      if (i == anchor + CF_N_POST - 1) classifyWindow();
      break;
    default:
      break;
  }
}

void readImu() {
  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  if (ax == 0 && ay == 0 && az == 0) {  // bus error returns zeros
    if (++zeroReads > 50) {
      state = SENSOR_FAULT;
      Serial.println("# SENSOR_FAULT: MPU6050 not responding");
    }
    return;
  }
  zeroReads = 0;
  float a[3] = {ax / 2048.0f, ay / 2048.0f, az / 2048.0f};        // +-16 g -> 2048 LSB/g
  float w[3] = {gx / 16.4f, gy / 16.4f, gz / 16.4f};              // +-2000 deg/s -> 16.4 LSB/(deg/s)
  onSample(a, w);
}

void setup() {
  Serial.begin(115200);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(CANCEL_BUTTON_PIN, INPUT_PULLUP);
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);
  while (!initImu()) {
    Serial.println("# MPU6050 not found, retrying");
    delay(1000);
  }
  Serial.println("# MPU6050 ready (+-16 g, 100 Hz)");

  SPI.begin(18, 19, 23, LORA_SS);
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(433E6)) {
    Serial.println("# LoRa init failed");
    while (true) delay(1000);
  }
  Serial.printf("# LoRa ready. Crash model: %s (%d features), road model: %s\n",
                CRASH_MODEL_NAME, CRASH_MODEL_N_FEATURES, ROAD_MODEL_NAME);
  nextSampleUs = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - nextSampleUs) >= 0) {
    nextSampleUs += 10000;                 // fixed 100 Hz schedule, catches up after slow work
    if (state == SENSOR_FAULT) {
      if (initImu()) { zeroReads = 0; state = MONITOR; Serial.println("# MPU6050 recovered"); }
      else delay(1000);
    } else {
      readImu();
    }
  }

  if (state == VERIFYING) {
    unsigned long elapsed = millis() - verifyStartMs;
    digitalWrite(BUZZER_PIN, (elapsed / 250) % 2);   // beeping countdown
    if (digitalRead(CANCEL_BUTTON_PIN) == LOW) {
      digitalWrite(BUZZER_PIN, LOW);
      Serial.println("# CANCELLED by driver (logged as false positive)");
      state = MONITOR;
    } else if (elapsed > VERIFY_SECONDS * 1000UL) {
      sendAlert("CRASH", pendingP, pendingPeak);
    }
  }

  if (state == ALERT_SENT && millis() - alertMs > CLEAR_AFTER_MS) {
    digitalWrite(BUZZER_PIN, LOW);
    sendPacket("CLEAR", "NONE", 0, 0);
    Serial.println("# CLEAR sent");
    state = MONITOR;
  }
}

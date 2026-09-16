# 🚨 Accident Detection and Emergency Alert System

An IoT-based vehicle accident detection and emergency alert system using LoRa communication, MPU6050 accelerometer/gyroscope sensors, and SMS notifications via WiFi.

## 📋 Overview

This system detects vehicle accidents in real-time by analyzing motion data from an IMU sensor and transmits emergency alerts via LoRa to a receiver station, which then sends SMS notifications to emergency contacts.

**Key Features:**
- 🎯 **Multi-condition accident detection** (high-impact crashes, angled crashes, rollovers, extreme rotation)
- 📡 **Long-range LoRa communication** (433 MHz frequency)
- 📤 **Automated SMS alerts** via WiFi and HTTP API
- 📊 **CSV data logging** for machine learning and analysis
- ⚡ **Low-power communication protocol**
- 🔄 **Auto-reconnection** and failsafe mechanisms

---

## 🤖 AI Crash Classifier (crash vs speed breaker vs pothole)

A jolt is not always an accident. `transmitter_LORA/ml_crash_detector` runs **two models on the ESP32 itself**. Every jolt above 0.45 g opens a 2.5 s window; **model A** decides whether it was a crash, and if it was not, **model B** names what it was.

| | Model A — crash detector | Model B — road-event classifier |
|---|---|---|
| Trained on | **VZCrash**: 31,090 verified real on-road crashes | PVS drives + Driver Behavior dataset |
| Inputs | 30 accelerometer + 7 gyroscope features | 30 accelerometer features |
| Output | P(crash) | NORMAL / SPEED_BUMP / POTHOLE_ROUGH_ROAD / HARSH_MANEUVER |
| Algorithm | LightGBM (400 trees) | XGBoost (250 rounds × 4 classes) |

Measured on VZCrash's held-out test split — 24,174 events, 4,221 real crashes the model never saw:

| Per event, at the alert threshold | Old rule (`impact > 2.2 g`) | Assessment-6 V2 design | **This system** |
|---|---|---|---|
| Average precision | 0.703 | 0.902 | **0.964** |
| Crashes detected | 93.6 % | 84.3 % | **92.3 %** |
| False alarms | 6.98 % | 3.63 % | **2.43 %** |

For reference, the VZCrash paper's own deep network (1.7 M parameters) scores 0.976 AP and its physical baseline 0.896 — this model reaches 0.964 with 37 features inside a microcontroller.

- **Full results, limitations and licences:** [ml/results/REPORT.md](ml/results/REPORT.md)
- **Retrain / reproduce:** [ml/README.md](ml/README.md)
- **Streamlit app (the main interface):** `streamlit run streamlit_app.py` — overview, drive replay, event library, model comparison, and a page to run the models over your own recording. Deploy instructions for Streamlit Community Cloud: [app/README.md](app/README.md)
- **HTML dashboard with live hardware:** open `demo/index.html` in Chrome or Edge to stream from the ESP32 over USB (**ESP32 over USB → Connect ESP32**, Web Serial)

**Decision logic:** P(crash) ≥ 0.80 sends the LoRa `ALERT` immediately; 0.50–0.80 starts a 10 s buzzer countdown the driver can cancel; below that the event is logged and optionally sent as an `EVENT` packet, which the receiver logs without sending an SMS. A separate physics guard raises `ROLLOVER` after 3 s of tilt beyond 60°.

**Flashing:** open `transmitter_LORA/ml_crash_detector/ml_crash_detector.ino` with the `MPU6050` (Electronic Cats) and `LoRa` (Sandeep Mistry) libraries. Optional buzzer on GPIO 4, cancel button from GPIO 13 to GND. It builds to 659 KB (50 % of flash), 45 KB RAM. The updated `reciever_sms.ino` puts the AI verdict and confidence into the SMS.

---

## 💻 Run It on a New Laptop

Needs **Python 3.10+** and **Git**. Nothing below requires downloading any dataset.

```bash
git clone https://github.com/Arush-O7/Vehicle-Accident-Detection-Emergency-Alert-System.git
cd Vehicle-Accident-Detection-Emergency-Alert-System
python -m venv .venv
```

Activate the environment — **Windows:** `.venv\Scripts\activate` · **macOS / Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app opens at http://localhost:8501.

| Part | What you need |
|---|---|
| Streamlit app | the commands above |
| HTML dashboard with live ESP32 | open `demo/index.html` in Chrome or Edge |
| Firmware | Arduino IDE, the ESP32 board package, and the **MPU6050** (Electronic Cats) and **LoRa** (Sandeep Mistry) libraries |
| Paper figures | `pip install -r ml/requirements.txt`, then `python ml/scripts/generate_paper_figures.py` |
| Retraining the models | `pip install -r ml/requirements.txt`, then follow [ml/README.md](ml/README.md) (about 8 GB of data and a free HuggingFace account). **macOS only:** run `brew install libomp` first |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRANSMITTER (Vehicle)                     │
│  ┌────────────────┐         ┌─────────────────┐             │
│  │  MPU6050 IMU   │────────→│  ESP32 + LoRa   │──────────┐  │
│  │  (6-axis)      │         │    Transmitter   │          │  │
│  └────────────────┘         └─────────────────┘          │  │
│                                      ↑                    │  │
│                           Accident Detection Logic       │  │
│                                                           │  │
└───────────────────────────────────────────────────────────┼──┘
                                                             │
                         433 MHz LoRa Link
                                                             │
┌─────────────────────────────────────────────────────────────┼──┐
│                    RECEIVER (Base Station)                   │  │
│  ┌──────────────────────────────────────────────────┐       │  │
│  │         ESP32 + LoRa Receiver                    │←──────┘  │
│  │                                                  │          │
│  │  • Receives LoRa packets                         │          │
│  │  • Connects to WiFi                             │          │
│  │  • Sends SMS via HTTP API                       │          │
│  └──────────────────────────────────────────────────┘          │
│           │                                                     │
│           └──→ [WiFi] → [HTTP API] → [SMS Gateway]             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Vehicle-Accident-Detection-Emergency-Alert-System/
│
├── transmitter_LORA/                      # Vehicle-side transmitter
│   ├── transmitter_LORA.ino               # Basic transmitter (simplified)
│   ├── angular_rotation_crash/
│   │   └── angular_rotation_crash.ino     # Rule-based transmitter
│   └── ml_crash_detector/                 # AI transmitter (recommended)
│       ├── ml_crash_detector.ino          # 100 Hz sampling, trigger, two models, LoRa
│       ├── crash_features.h               # 30 accelerometer + 7 gyroscope features
│       ├── crash_model.h                  # Model A: crash detector (VZCrash)
│       └── road_model.h                   # Model B: speed bump / pothole / harsh driving
│
├── Reciever_LORA/                         # Base station receiver
│   ├── Reciever_LORA.ino                  # Basic receiver (simplified)
│   └── reciever_sms/
│       └── reciever_sms.ino               # Advanced receiver with SMS
│
├── ml/                                    # Datasets, training, model comparison, export
│   └── results/REPORT.md                  # Results report
├── streamlit_app.py                       # Streamlit interface (deployable)
├── app/                                   # App engine + exported models and demo data
├── demo/                                  # HTML dashboard (live ESP32 via Web Serial)
│
└── README.md                              # This file
```

---

## 🔧 Hardware Requirements

### Transmitter (Vehicle)
- **ESP32 microcontroller**
- **MPU6050** (6-axis IMU sensor)
- **LoRa module** (SX1278, 433 MHz)
- **Power supply** (3.3V for ESP32/LoRa, 5V for sensor)
- Wiring connectors

### Receiver (Base Station)
- **ESP32 microcontroller**
- **LoRa module** (SX1278, 433 MHz)
- **WiFi connectivity** (mobile hotspot or router)
- **Power supply** (USB or 5V adapter)
- Wiring connectors

**Pin Configuration (Both modules):**
```
LoRa Pins (SPI):
- CS/SS:  GPIO 5
- RST:    GPIO 14
- DIO0:   GPIO 26
- MOSI:   GPIO 23
- MISO:   GPIO 19
- SCK:    GPIO 18

MPU6050 (Transmitter only):
- SDA:    GPIO 21
- SCL:    GPIO 22
```

---

## 📡 Sensor Specifications

### MPU6050 IMU Sensor
- **6-axis motion tracking:**
  - 3-axis accelerometer (±16G range)
  - 3-axis gyroscope (±2000°/s range)
- **Raw to G-force conversion:** `value / 16384.0`
- **Raw to °/s conversion:** `value / 131.0`

---

## 🚨 Accident Detection Algorithm

The system detects **4 types of accidents:**

### 1. **High-Impact Crash** 
- Condition: `impact > 2.2G`
- Detects frontal/rear collisions with substantial force

### 2. **Angled Crash**
- Condition: `impact > 2.2G AND rotation_rate > 100°/s`
- Detects side-impact collisions with spinning

### 3. **Rollover**
- Condition: `|pitch| > 75° OR |roll| > 75°`
- Detects vehicle tipping on side or roof

### 4. **Extreme Rotation**
- Condition: `rotation_rate > 180°/s`
- Detects violent spinning (skidding, fishtailing)

**Alert Mechanism:**
- Transmitter sends LoRa alert packet when accident detected
- 3-second cooldown prevents alert spam
- Receiver SMS triggered on ALERT signal reception
- System clears alert state on CLEAR signal (after vehicle stabilizes)

---

## 📊 Data Logging Format

The transmitter outputs CSV data on Serial Monitor:

**Headers:** 
```
ax,ay,az,gx,gy,gz,impact,rotation,pitch,roll,label
```

**Example Data:**
```
0.15,-0.02,1.00,5.23,-2.10,1.45,1.02,6.5,8.3,1.2,0
2.85,1.92,2.15,125.4,92.3,-48.2,3.48,145.2,42.1,-28.5,1
```

- **ax, ay, az:** Acceleration in G-force
- **gx, gy, gz:** Angular velocity in °/s
- **impact:** Magnitude of acceleration vector
- **rotation:** Magnitude of rotation rate vector
- **pitch, roll:** Orientation angles in degrees
- **label:** 0=normal, 1=accident

---

## 🚀 Quick Start Guide

### 1. **Setup Hardware**

Assemble the transmitter and receiver modules according to pin configuration above.

### 2. **Install Arduino IDE**

Download from: https://www.arduino.cc/en/software

### 3. **Install Required Libraries**

In Arduino IDE, go to **Sketch → Include Library → Manage Libraries** and install:
- `MPU6050` (by Jeff Rowberg)
- `LoRa` (by Sandeep Mistry)
- `WiFi` (built-in with ESP32)
- `HTTPClient` (built-in with ESP32)

### 4. **Flash Transmitter Code**

**For basic testing:**
1. Open `transmitter_LORA/transmitter_LORA.ino`
2. Select Board: **ESP32 Dev Module**
3. Select Port: (your COM port)
4. Click **Upload**
5. Open Serial Monitor (115200 baud) to view acceleration data

**For production (with advanced detection):**
1. Open `transmitter_LORA/angular_rotation_crash/angular_rotation_crash.ino`
2. Upload same as above
3. Verify MPU6050 initializes with "✅" indicator

### 5. **Flash Receiver Code**

**For basic testing:**
1. Open `Reciever_LORA/Reciever_LORA.ino`
2. Upload to receiver ESP32
3. Serial Monitor will show "Receiver Ready 📡"

**For SMS alerts:**
1. Open `Reciever_LORA/reciever_sms/reciever_sms.ino`
2. **Update WiFi credentials:**
   ```cpp
   const char* ssid = "YOUR_SSID";
   const char* password = "YOUR_PASSWORD";
   ```
3. **Update SMS API credentials:**
   ```cpp
   const char* apiKey = "YOUR_API_KEY";
   const char* mobileNumber = "YOUR_PHONE_NUMBER";
   ```
4. Upload to receiver ESP32

### 6. **Test the System**

1. **Open Serial Monitors** on both devices (set baud to 115200)
2. **Simulate acceleration** on transmitter (shake it gently, then harder)
3. Watch acceleration values on transmitter monitor
4. Transmitter will send ALERT when impact > 2.2G
5. Receiver will receive alert and (if SMS code) send SMS notification

---

## 📤 SMS API Integration

The system uses **CircuitDigest Cloud API** for SMS delivery.

### Configuration:
```cpp
const char* apiUrl = "https://www.circuitdigest.cloud/api/v1/send_sms?ID=103";
const char* apiKey = "YOUR_API_KEY";
const char* mobileNumber = "+91XXXXXXXXXX";  // Include country code
```

### Payload Format:
```json
{
  "mobiles": "+91XXXXXXXXXX",
  "var1": "Accident Detection System",
  "var2": "Vehicle Crash Impact 3.48g"
}
```

### Response Codes:
- `200`: SMS sent successfully
- `401`: API key unauthorized
- `400`: Bad request/invalid payload
- `-1`: Connection failed

---

## 🔌 Serial Monitor Output Examples

### Transmitter (Normal Operation):
```
MPU6050 Ready ✅
LoRa Ready ✅
ax,ay,az,gx,gy,gz,impact,rotation,pitch,roll,label
0.15,-0.02,1.00,5.23,-2.10,1.45,1.02,6.5,8.3,1.2,0
0.18,0.01,0.98,4.15,-1.92,0.82,1.00,5.2,7.1,0.5,0
```

### Transmitter (Accident Detected):
```
🚨 HIGH IMPACT CRASH
ALERT,HIGH_IMPACT,3.48
✅ CLEAR SENT - System Reset
```

### Receiver (Basic):
```
Receiver Ready 📡
Received: ALERT,HIGH_IMPACT,3.48
🚨 ACCIDENT DETECTED 🚨
```

### Receiver (SMS-enabled):
```
WiFi Connected ✅
IP Address: 192.168.43.210
LoRa Ready 📡
Received: ALERT,HIGH_IMPACT,3.48
🚨 ACCIDENT DETECTED 🚨
📤 Sending SMS...
HTTP Response Code: 200
✅ SMS Sent Successfully!
```

---

## ⚙️ Configuration Parameters

### Accident Detection Thresholds (transmitter)
```cpp
// Modify these in angular_rotation_crash.ino for tuning:
bool highImpactCrash = (impact > 2.2);                    // ~22 mph collision
bool angledCrash     = (impact > 2.2 && rotation_rate > 100);
bool rollover        = (abs(pitch) > 75.0 || abs(roll) > 75.0);
bool extremeRotation = (rotation_rate > 180);              // High rotation speed
```

### LoRa Settings
```cpp
LoRa.begin(433E6);  // 433 MHz frequency
// Range: ~5-10 km line-of-sight (depending on antenna)
```

### WiFi Timeout (receiver)
```cpp
if (millis() - startTime > 15000) {  // 15 second timeout
  // Attempt reconnection
}
```
## 👤 Author

**Arush Mishra**

---

## 📚 References

- ESP32 Documentation: https://docs.espressif.com/projects/esp-idf/
- MPU6050 Datasheet: https://invensense.tdk.com/products/motion-tracking/6-axis/
- LoRa Technology: https://lora-alliance.org/
- Arduino LoRa Library: https://github.com/sandeepmistry/arduino-LoRa

---

# AI Crash Classifier: Results Report

**Project:** LoRa-Based Autonomous Vehicular Accident Detection and Emergency Alert System (ESP32 + MPU6050 + SX1278)
**Question:** when the vehicle-mounted sensor feels a jolt, is it a **crash**, or a **speed breaker / pothole / harsh manoeuvre**?

The device runs **two models**:

| | Model A — crash detector | Model B — road-event classifier |
|---|---|---|
| Decides | crash or not | which kind of non-crash jolt |
| Trained on | **VZCrash**: 31,090 verified real on-road crashes (IEEE ITSC 2026) | PVS drives + Driver Behavior dataset |
| Inputs | 30 accelerometer + 7 gyroscope features | 30 accelerometer features |
| Algorithm | LightGBM, 400 trees | XGBoost, 250 rounds × 4 classes |
| On the ESP32 | 168 KB of C arrays | 171 KB of C arrays |

Numbers come from `vzcrash_comparison.csv`, `metrics_vzcrash.json`, `road_model.json` and `firmware_parity.txt`. Reproduce with `ml/README.md`.

---

## 1. Headline results (crash detection)

Trained on the VZCrash train split, threshold chosen on validation, **every number below measured on the untouched test split**: 24,174 events, of which 4,221 are real crashes reviewed by three trained annotators with dashcam footage.

| Metric (per event, at the firmware's P ≥ 0.80 alert threshold) | Old rule `|a| > 2.2 g` | Document's V2 design | **Model A (LightGBM)** |
|---|---|---|---|
| Average precision (ranking quality) | 0.703 | 0.902 | **0.964** |
| Crashes detected | 93.56 % | 84.25 % | **92.25 %** |
| False-alarm rate | 6.98 % | 3.63 % | **2.43 %** |
| Precision | 73.9 % | 83.1 % | **88.9 %** |
| F1 | 0.826 | 0.837 | **0.906** |
| Accuracy | 93.1 % | 94.3 % | **96.6 %** |

For scale, the VZCrash authors' own models on the same dataset: physical baseline **0.896 AP**, CNN-RNN (1.7 M parameters) **0.976 AP**, CNN-Transformer 0.975, Chronos-Bolt 0.966. Model A reaches **0.964 AP with 37 features on a microcontroller**, above the paper's physical baseline and within 1.2 points of its deep network.

At a stricter operating point tuned on validation for ≤ 0.5 % false alarms, Model A detects 72.7 % of crashes with a **0.59 % false-alarm rate**.

## 2. Data

| Source | Role | Size used |
|---|---|---|
| **VZCrash** (Bianconcini et al., IEEE ITSC 2026) | Model A: crash vs non-crash. 16 s of 100 Hz tri-axial accelerometer (g) + gyroscope (deg/s) + 1 Hz GPS speed per event; labels from a 3-way consensus of trained reviewers with dashcam footage | 137,954 train / 27,175 validation / 24,174 test events → 160,097 / 31,252 / 28,026 triggered windows |
| **PVS** (UFSC, Brazil) | Model B: speed bumps, pothole / rough road, normal jolts. 9 drives, 3 vehicles, dashboard + body MPU-9250 at 100 Hz | 3,243 rough-road, 95 speed-bump, 127 normal windows |
| **Driver Behavior** (Ferreira et al., PLOS ONE 2017) | Model B: harsh braking, acceleration, turns, lane changes | 55 harsh, 14 normal manoeuvres |
| NHTSA Crash Test Database | First-generation crash model (superseded by VZCrash); still built by `build_dataset.py` for comparison | 887 crash pulses, 821 tests |

**Only parameters the dataset provides are used**: accelerometer, gyroscope and GPS speed. Vehicle size and the gyroscope-resolution flag are dataset metadata and are deliberately excluded from the model; they are used only to break results down in §5.

**How the data is made to match the hardware.** Every stream is clipped to ±16 g and quantised to 2048 LSB/g (MPU6050 at ±16 g) and the gyroscope to 16.4 LSB/(deg/s) (±2000 deg/s), then cut into windows by the same trigger code the firmware runs.

## 3. Method

1. **Trigger.** Gravity is tracked by an exponential moving average (α = 0.02); a window opens when the dynamic acceleration exceeds **0.45 g**. Events that never reach it are never classified — and count as misses in every number in this report. 100 % of VZCrash crash events do cross it.
2. **Window.** 1.0 s before and 1.5 s after the trigger (250 samples at 100 Hz). A much stronger jolt arriving while the window fills re-anchors it, so a pothole cannot mask a crash 200 ms later.
3. **Features.** Gravity is estimated from the first 0.5 s and the motion is split into **vertical** (bumps, potholes) and **horizontal** (braking, impacts) parts, which makes the model independent of how the box is mounted: peaks, jerk, time above 0.5/1/2/4 g, horizontal ΔV, separated peaks (front and rear axle over a bump), suspension oscillation, post-event vibration and tilt, plus 7 gyroscope features (peak rate, yaw vs tilt rate, rotation through the impact, rotational energy).
4. **Decision.** P(crash) ≥ 0.80 → LoRa `ALERT` immediately. 0.50–0.80 → 10 s buzzer countdown the driver can cancel. Otherwise Model B names the jolt and it is logged (optionally reported as a LoRa `EVENT`, which the receiver logs without sending an SMS).
5. **Rollover** remains an independent physics guard (tilt > 60° held for 3 s).

### 3.1 Why 2.5 s and 37 features, not the document's 200 ms and 8

The Assessment-6 document specifies a 200 ms window, an 8-element feature vector and logistic regression. Measured on identical data, that design is the limiting factor:

| Design | Window | Features | Model | Event AP |
|---|---|---|---|---|
| Document V2 as written | 200 ms | 8 | Logistic regression | 0.902 |
| Document V2 features, stronger model | 200 ms | 8 | XGBoost | 0.930 |
| Proposed, same model family | 2.5 s | 37 | XGBoost | 0.962 |
| **Proposed (selected)** | 2.5 s | 37 | LightGBM | **0.964** |

Replacing only the classifier buys 0.028 AP; replacing the 200 ms window and feature vector buys 0.032 more. The context before and after the impact carries as much information as the impact peak itself.

### 3.2 What the gyroscope and GPS add

| Inputs | Features | Event AP | Recall @ 0.80 | False alarms |
|---|---|---|---|---|
| Accelerometer only | 30 | 0.9559 | 91.7 % | 2.94 % |
| + gyroscope (**shipped**) | 37 | 0.9622 | 92.5 % | 2.63 % |
| + GPS speed | 40 | 0.9677 | 93.4 % | 2.27 % |

(XGBoost, so the three rows differ only by inputs.) The gyroscope is free — the MPU6050 already provides it. Wiring up the NEO-6M GPS and feeding speed change to the model is worth a further ~0.006 AP and about 0.4 points of false-alarm rate; the shipped firmware does not read GPS yet.

## 4. All models compared

Per-event, test split. "Tuned" = threshold chosen on validation for ≤ 0.5 % false alarms.

| Model | Inputs | Event AP | Recall @0.8 | FA @0.8 | Recall @tuned | FA @tuned |
|---|---|---|---|---|---|---|
| V1 rule `|a| > 2.2 g` | accel | 0.7029 | 93.6 % | 6.98 % | — | — |
| Document V2: 200 ms + logistic | doc-200ms | 0.9021 | 84.3 % | 3.63 % | 44.9 % | 0.43 % |
| Document V2 features + XGBoost | doc-200ms | 0.9305 | 88.6 % | 3.39 % | 55.6 % | 0.50 % |
| Logistic Regression | accel+gyro | 0.9368 | 90.8 % | 3.42 % | 51.9 % | 0.47 % |
| Decision Tree | accel+gyro | 0.9013 | 97.6 % | 7.60 % | 46.5 % | 0.49 % |
| Random Forest | accel+gyro | 0.9585 | 89.3 % | 2.34 % | 68.6 % | 0.62 % |
| Extra Trees | accel+gyro | 0.9532 | 88.7 % | 2.58 % | 66.6 % | 0.57 % |
| MLP neural network | accel+gyro | 0.9606 | 94.1 % | 3.11 % | 69.8 % | 0.59 % |
| **LightGBM (selected)** | accel+gyro | **0.9636** | 92.3 % | 2.43 % | 72.7 % | 0.59 % |
| XGBoost | accel+gyro | 0.9622 | 92.5 % | 2.63 % | 71.7 % | 0.59 % |
| XGBoost | accel+gyro+gps | 0.9677 | 93.4 % | 2.27 % | 75.4 % | 0.57 % |

**Selection rule:** highest event AP among models that run on the ESP32 without extra hardware. That is LightGBM; XGBoost is within 0.0014 AP of it and would serve equally well.

## 5. Where Model A still fails

Detection rate at P ≥ 0.80, by how hard the impact actually was:

| Peak dynamic acceleration | Crashes | Detected |
|---|---|---|
| < 1.5 g | 146 | **21.2 %** |
| 1.5 – 3 g | 1,093 | 86.6 % |
| 3 – 6 g | 2,702 | 97.9 % |
| > 6 g | 274 | 99.6 % |

By vehicle size: light 94.0 % (834), medium 94.7 % (2,203), heavy 85.9 % (929), unrecorded 90.8 % (249). Heavy vehicles absorb impacts, so the same collision registers as a smaller jolt.

Gentle impacts remain the open problem: under 1.5 g a real crash and a hard pothole are close to indistinguishable from inertial data alone. The 0.50–0.80 countdown band exists for exactly this case — it asks the driver instead of staying silent.

## 6. Model B: naming the jolt

XGBoost, leave-one-vehicle-out cross-validation on real events. Macro F1 **0.782**, accuracy **95.9 %** (Random Forest 0.750, Logistic Regression 0.585).

| True \ Predicted | Normal | Speed bump | Pothole / rough | Harsh manoeuvre |
|---|---|---|---|---|
| Normal (141) | **125** | 2 | 7 | 7 |
| Speed bump (95) | 1 | **53** | 38 | 3 |
| Pothole / rough road (3,243) | 41 | 31 | **3,167** | 4 |
| Harsh manoeuvre (55) | 5 | 1 | 6 | **43** |

Speed bumps are the weak class (56 % recall, usually confused with rough road): PVS labels road quality per stretch rather than per event, and contains only 95 bump crossings. Recording labelled bumps with this transmitter is the fix.

## 7. Firmware verification

- **Compiles for ESP32** (arduino-cli, esp32 core 3.3.11, board `esp32:esp32:esp32`):
  - transmitter with both models: **658,987 bytes flash (50 %)**, 45,052 bytes RAM (13 %)
  - receiver: 1,028,692 bytes (78 %)
- **C matches Python** (`verify_firmware.py`, 1,987 real windows — VZCrash events and PVS road events):
  - features agree to 1.5 × 10⁻⁴ relative error
  - crash probability agrees to 6.7 × 10⁻⁷
  - **alert decision identical on 100 % of windows**; road class identical on 100 %
- **Browser engine** (`demo/crash_ai.js`) verified the same way, for the live dashboard.
- **Speed:** 31 µs per event on the development laptop for features plus both models. ESP32 timing is printed by the sketch with every event (`E,...,<µs>`) and has not yet been measured on hardware.

## 8. Limitations

1. **Not measured on this hardware yet.** Everything is verified in simulation and on recorded data from other devices (dashcam IMUs, a dashboard MPU-9250). Numbers on your own ESP32 + MPU6050 may differ; recording a few labelled drives and re-running the pipeline is the way to check.
2. **VZCrash negatives are hard on purpose** — near misses, potholes, off-road manoeuvres, harsh braking. A 2.43 % false-alarm rate on that set is not 2.43 % of driving time. On ordinary recorded driving (200 s of PVS drive 7, 119 jolts) the shipped model raises **no** crash alert at all.
3. **Gentle crashes under 1.5 g are mostly missed** (§5).
4. **Model B does not transfer across sensors well.** On unseen iPhone drives it recognised only 1 of 12 confirmed potholes as a pothole, though it raised almost no false crash verdicts (1 in 252). Treat the bump-vs-pothole output as informative, not authoritative.
5. **GPS is not wired in.** The +GPS model is measured but not deployed.
6. **Rollover** is a physics rule, not a trained model: VZCrash has no rollover label.

## 9. Dataset licences (relevant to the patent)

Patenting the **method** (orientation-invariant features, trigger and re-anchoring, two-stage crash/road classification, gated driver verification) is not restricted by these licences. Training a **commercial** product on them is.

| Dataset | Licence |
|---|---|
| VZCrash | CC BY-NC 4.0 — **non-commercial** |
| PVS | CC BY-NC-ND 4.0 — non-commercial, no derivatives |
| Driver Behavior Dataset | No licence stated; ask the authors before commercial use |
| NHTSA Crash Test Database | US Government work (public domain) |

Before selling a device, retrain Model A on your own or commercially licensed crash recordings. The pipeline is unchanged; only the data source swaps.

**Citation for VZCrash:** Bianconcini, T., Monteagudo, H. P., Pjetri, A., Trinci, T., & Taccari, L. "VZCrash: A Large-Scale IMU Dataset of Ego-Vehicle Crashes", 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC).

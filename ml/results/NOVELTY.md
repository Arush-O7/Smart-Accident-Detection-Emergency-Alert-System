# Novelty and Technical Justification

**LoRa-Based Autonomous Vehicular Accident Detection and Emergency Alert System, AI Crash Discrimination Layer**
ESP32 + MPU6050 + SX1278 · Arush Mishra

All figures in this document come from `ml/results/vzcrash_comparison.csv`, `metrics_vzcrash.json`,
`road_model.json`, `pr_curves.json` and `firmware_parity.txt`, reproducible with `ml/README.md`.

---

## 1. Executive summary

The system answers a question that threshold-based crash detectors cannot: **when the vehicle is
jolted, was it a crash, or a speed breaker, a pothole or harsh driving?** Two gradient-boosted tree
models run on the ESP32 itself. Model A decides whether the jolt was a crash from 30 accelerometer
and 7 gyroscope features over a 2.5 s window; Model B names the non-crash jolts. Only crashes raise
a LoRa alert and an SMS.

Measured on the held-out test split of **VZCrash** (24,174 events, of which 4,221 are real crashes
reviewed by three trained annotators against dashcam footage):

| Per event, at the alert threshold P(crash) ≥ 0.80 | Current rule `\|a\| > 2.2 g` | Assessment-6 V2 design | **Proposed system** |
|---|---|---|---|
| Average precision | 0.703 | 0.902 | **0.964** |
| Crashes detected | 93.6 % | 84.3 % | 92.3 % |
| False alarms | 6.98 % | 3.63 % | **2.43 %** |
| Precision | 73.9 % | 83.1 % | **88.9 %** |
| F1 | 0.826 | 0.837 | **0.906** |

For context, the VZCrash authors' own models on the same dataset score **0.896 AP** (physics
baseline) and **0.976 AP** (a 1.7-million-parameter CNN-RNN). The proposed system reaches **0.964 AP
with 37 features inside a microcontroller**, using 339 KB of flash for both models and 24 µs per
decision.

---

## 2. Problem and prior art

### 2.1 What existing systems do

Published low-cost accident detectors, and the first version of this project, declare a crash when
the acceleration magnitude crosses a fixed threshold, typically 2 g to 4 g, sometimes with a
gyroscope threshold for rollover. This is the V1 rule: `|a| > 2.2 g`.

### 2.2 Why that fails, with evidence

1. **Ordinary bad roads exceed the threshold.** In the PVS drives (9 real drives, 3 vehicles), the
   dashboard accelerometer reaches **2.5–3.5 g** on rough and cobblestone stretches. The 2.2 g rule
   fires on **6.3 %** of rough-road jolts, each one an emergency SMS.
2. **On real crash data the rule is barely better than chance at ranking.** On VZCrash its average
   precision is **0.703** against a crash prevalence of 17.5 %, with a **6.98 %** false-alarm rate.
3. **A threshold cannot explain itself.** It outputs one bit. It cannot tell a driver, a fleet
   operator or a road authority that the jolt was a speed breaker rather than a collision.
4. **The sampling rate and range of the original design were inconsistent with the threshold** -
   3.3 Hz sampling cannot resolve a 50–200 ms crash pulse, and a ±2 g range cannot represent 2.2 g.
   Both are corrected here (100 Hz, ±16 g), but correcting them does not solve 1–3.

### 2.3 The nearest prior design within this project

The Assessment-6 document specifies a 200 ms window, an 8-element feature vector and logistic
regression (V2). Implemented exactly as written and trained on the same data, it scores **0.902 AP**
- a real improvement on the threshold, and still 0.062 AP below the proposed design. Section 7.5
shows that the limitation is the 200 ms window, not the classifier.

---

## 3. Claimed novelty

Each claim below is stated as the mechanism, why it is not the obvious choice, and the measurement
that supports it.

### N1, Gravity-referenced, orientation-invariant feature decomposition

Gravity is estimated from the first 0.5 s of every window, and the acceleration is decomposed into a
**vertical** component (along gravity) and a **horizontal** magnitude (across it). Every feature is
computed in that frame.

*Why it is not obvious:* conventional detectors use the resultant magnitude `|a|`, which discards
direction, or raw per-axis values, which depend on how the device is bolted in. The decomposition
separates the two physical regimes that a magnitude cannot: **road anomalies excite the vertical
axis** (suspension travel), while **collisions excite the horizontal plane** (momentum transfer).

*Evidence:* the horizontal features dominate the trained model, `dv_h_kmh` (horizontal velocity
change) and `sustained_h` are the 2nd and 5th most important of 37 features, and `h_energy_frac`
(the horizontal share of event energy) is a further contributor. The device may be mounted in any
orientation without recalibration, which the per-axis approach cannot offer.

### N2, Two-stage on-device inference: detect, then explain

Model A (crash / not crash) runs first. Only if it clears the jolt does Model B name it as
`SPEED_BUMP`, `POTHOLE_ROUGH_ROAD`, `HARSH_MANEUVER` or `NORMAL`. The two models are trained on
different datasets, for different questions, and are exported as separate C headers.

*Why it is not obvious:* the natural design is one multi-class model. Splitting has three concrete
advantages: (a) the safety-critical decision is trained only on **real crash data**, never on
assembled or proxy crashes; (b) the road classes can be retrained from cheap, easily collected
drives without touching the crash detector; (c) a weak road-label vocabulary cannot degrade crash
recall. The cost is one extra inference, which measures at 24 µs total on the host.

*Evidence:* Model A scores 0.964 AP on real crashes; Model B reaches macro F1 0.782 on road events.
A single 5-class model trained on the earlier, assembled crash data reached only 0.815 macro F1
while being unable to claim real-world crash validity.

### N3, Triggered window with re-anchoring

A cheap gravity-referenced trigger (0.45 g over an exponential moving average) opens a 2.5 s window:
1.0 s **before** the trigger from a ring buffer and 1.5 s after. If a substantially stronger jolt
arrives while the window is still filling, `dyn > max(1.0 g, 1.5 × current peak)`, the window
**re-anchors** on the stronger jolt.

*Why it is not obvious:* a plain triggered-window detector is blind for the duration of its own
collection window. A pothole immediately followed by a collision, a realistic sequence on a bad
road, would be classified as a pothole and the crash would never be examined. Re-anchoring closes
that hole without a second buffer or a second classifier pass.

*Evidence:* the mechanism is implemented identically in the Python pipeline, the C firmware and the
browser engine, and verified to produce identical event anchors across all three.

### N4, Confidence-banded, driver-in-the-loop verification

The classifier's probability, not a binary output, drives a three-way decision: **P ≥ 0.80** sends
the alert immediately; **0.50 ≤ P < 0.80** starts a 10 s buzzer countdown the driver can cancel;
below 0.50 the event is logged and named by Model B.

*Why it is not obvious:* it converts the calibrated output of the model into an escalation policy
that spends human attention only on genuinely ambiguous events, and it gives the system a safe
behaviour exactly where the data says it is weakest, gentle impacts (§8.1). A cancelled countdown
is also a labelled false positive, which is the cheapest possible source of on-vehicle training data.

*Evidence:* at P ≥ 0.80 the false-alarm rate is 2.43 %; the 0.50–0.80 band contains the events where
recall would otherwise be lost, and the tuned operating point (§7.2) trades recall for a 0.59 %
false-alarm rate on demand.

### N5, Gyroscope-augmented crash discrimination on a commodity IMU

Seven rotational features are computed in the same gravity frame: peak rate, **yaw rate about
gravity** separated from **tilt rate across it**, rotation integrated through the impact, and
rotational energy after it.

*Why it is not obvious:* rollover detection conventionally uses the gyroscope for sustained tilt,
not for discriminating a collision from a pothole. Separating spin about the gravity axis from
pitch/roll is what makes the signal useful: a collision imparts yaw, a pothole does not.

*Evidence:* `w_yaw_peak` is the **single most important feature of the crash model** (5.8 % of total
gain, highest of 37). Adding the gyroscope raises AP from 0.9559 to 0.9622 on identical windows with
the same algorithm, and cuts false alarms from 2.94 % to 2.63 %. The sensor is already present; the
gain costs nothing in hardware.

### N6, Deterministic tree-ensemble deployment with verified numerical parity

Both ensembles are flattened into pre-order C arrays (feature index, threshold, right-child index)
and compiled into the firmware. Thresholds are converted to float32 with the rounding direction
chosen so the on-device comparison reproduces the training library's decision exactly. A verification
harness compiles the firmware headers on the host and compares them against the Python pipeline on
real windows.

*Why it is not obvious:* the usual embedded-ML route is a neural network plus an interpreter
(TensorFlow Lite Micro), which brings a runtime, a memory arena and quantisation error. Gradient
boosted trees deployed as pure data arrays need no runtime, are bitwise-inspectable, and take a
single-digit microsecond budget, but only if the float32 comparison semantics are handled, which is
the technical contribution here.

*Evidence:* on 1,987 real windows, features agree to **1.5 × 10⁻⁴** relative error, crash probability
to **6.7 × 10⁻⁷**, and the **alert decision is identical on 100 % of windows**; the road class is
identical on 100 %. Total 339 KB of flash for 48,884 nodes, 24 µs per full decision on the host.

### N7, Speed-scaled crash-pulse augmentation for minor collisions (method)

Real crash-test pulses were scaled in amplitude to synthesise lower-speed impacts. Under a
linear-spring crash model, peak deceleration grows with impact speed while pulse duration stays
approximately constant, so scaling by *s* approximates the same structure struck at *s* × the test
speed, down to 8 km/h equivalent.

*Why it is not obvious:* it gives a physically justified way to populate the low-severity region
where public crash-test data is thin, without fabricating waveforms. Used for training and stress
testing only, never for headline metrics.

*Status:* developed for the first-generation model trained on NHTSA crash tests. The shipped crash
model is trained on real on-road crashes and does not require it, but the method remains part of the
disclosure and is directly useful to anyone without access to a dataset such as VZCrash.

### N8, Semantic alert payload over the long-range link

The LoRa frame carries `ALERT,<class>,<confidence %>,<peak g>`, and non-crash events are optionally
reported as `EVENT,<class>,<confidence>,<peak g>`. The fog receiver puts the class and confidence
into the SMS body and logs road events without dispatching an SMS.

*Why it is not obvious:* the alert stops being a bare interrupt and becomes a triaged message. A
responder sees "AI verified CRASH (91 % confidence), impact 3.4 g" rather than a raw number, and the
same channel doubles as a road-quality reporting path at no extra energy cost.

### 3.1 What is *not* claimed

Stating this strengthens the rest: LoRa transmission and the 433 MHz link budget; the MPU6050 and
NEO-6M devices; threshold-based impact detection in general; gradient boosting, logistic regression
and their libraries; the datasets; the cloud SMS API. The novelty is the combination and the
specific mechanisms N1–N8.

---

## 4. System architecture

```
  MPU6050 (100 Hz, ±16 g, ±2000 °/s)
        │  accelerometer + gyroscope
        ▼
  [ Trigger ]  gravity EMA (α = 0.02); fires when |a − g| > 0.45 g
        │      opens a 250-sample window: 100 before + 150 after; re-anchors on a stronger jolt
        ▼
  [ Features ]  30 accelerometer + 7 gyroscope, in the gravity frame
        │
        ├──► [ Model A: LightGBM, 400 trees, 24,400 nodes ] ─► P(crash)
        │            │                       │
        │       P ≥ 0.80              0.50 ≤ P < 0.80
        │            │                       │
        │     LoRa ALERT now        10 s countdown, driver may cancel
        │
        └──► [ Model B: XGBoost, 1,000 trees, 24,484 nodes ] ─► NORMAL | SPEED_BUMP |
                     (only when P < 0.50)                       POTHOLE_ROUGH_ROAD | HARSH_MANEUVER
                                                                  │
                                                        logged, optional EVENT frame

  Independent physics guard: tilt > 60° sustained 3 s  ─►  ROLLOVER alert
```

| Parameter | Value | Reason |
|---|---|---|
| Sampling rate | 100 Hz | resolves a 50–200 ms crash pulse (10–20 samples) |
| Accelerometer range | ±16 g (2048 LSB/g) | real crashes exceed 6 g; ±2 g cannot represent the threshold |
| Gyroscope range | ±2000 °/s (16.4 LSB/°/s) | measured crash rotation reaches 3,036 °/s |
| Anti-alias filter | DLPF 44 Hz | below Nyquist at 100 Hz |
| Trigger | 0.45 g dynamic | catches every VZCrash crash event and labelled speed bump |
| Window | 1.0 s pre + 1.5 s post | §7.5 shows the pre/post context carries as much information as the peak |
| Alert threshold | P ≥ 0.80 | 2.43 % false alarms on hard negatives; 0 on ordinary driving |

---

## 5. Feature set (37 features)

| Group | Features | Physical reasoning |
|---|---|---|
| Impact magnitude | peak dynamic, peak horizontal, peak vertical, h/v ratio, raw peak `\|a\|`, peak jerk | separates direction of the energy, not just its size |
| Impulse shape | samples above 0.5/1/2/4 g, width at half peak, crest factor, peak position | a crash pulse is short and tall; a bump is broad and repeated |
| Velocity change | horizontal ΔV, vertical ΔV over ±300 ms | a collision transfers momentum; a pothole does not |
| Road signature | number of separated peaks, peak span, vertical zero crossings, pre-event roughness | front and rear axles hit a bump twice; suspension then oscillates |
| Post-event state | vibration in the last 0.5 s, mean magnitude, sustained horizontal load, tilt change, gravity-magnitude change | after a crash the vehicle stops, slews or comes to rest tilted |
| Energy | impact energy, post-trigger energy, horizontal energy fraction, early RMS | total and directional energy budget |
| Rotation (gyroscope) | peak rate, mean post rate, rotation through impact, total rotation, **peak yaw**, peak tilt rate, rotational energy | a collision spins the vehicle about gravity; a pothole does not |

**Most important features of the crash model** (gain share): `w_yaw_peak` 5.8 %, `dv_h_kmh` 4.6 %,
`dtheta_impact` 4.2 %, `pre_std` 4.2 %, `sustained_h` 4.1 %, `zc_v` 3.9 %, `pre_h_mean` 3.8 %,
`tilt_change_deg` 3.8 %. No single feature dominates, which is the intended outcome: the model is not
a threshold in disguise.

**Most important features of the road model**: `sustained_h` 14.5 %, `jerk_peak` 11.7 %,
`dv_h_kmh` 9.1 %, `zc_v` 8.8 %, `pre_h_mean` 6.4 %.

---

## 6. Algorithms evaluated and why

| Family | Model | Why it was included |
|---|---|---|
| Rule | V1 threshold `\|a\| > 2.2 g` | the incumbent; the baseline to beat |
| Linear | Logistic regression (8-feature 200 ms, and 37-feature) | the Assessment-6 V2 design; smallest possible footprint |
| Single tree | Decision tree (depth 10) | interpretability ceiling; shows what one tree can do |
| Bagged trees | Random forest, extra trees | variance reduction; strong tabular baseline |
| Boosted trees | **XGBoost**, **LightGBM** | state of the art on tabular features; exportable as plain arrays |
| Neural | MLP (96–48) | tests whether a dense network beats trees on these features |
| Deep (reference) | VZCrash paper's CNN-RNN, CNN-Transformer, Chronos-Bolt | published upper bound on this dataset, 1.3–8.8 M parameters |

**Protocol.** VZCrash's official splits are used unchanged: fit on train (137,954 events → 160,097
triggered windows), choose thresholds on validation (27,175 events), report on test (24,174 events).
Class weights are balanced. Metrics are reported **per event**, which is the deployment question -
did the device raise an alert during this 16 s event?, with crash events that never cross the
trigger counted as misses. Window-level metrics are reported alongside for completeness.

---

## 7. Results

### 7.1 Full comparison (test split, per event)

| Model | Inputs | AP | Recall @0.8 | False alarms @0.8 | Precision @0.8 | F1 @0.8 |
|---|---|---|---|---|---|---|
| V1 rule `\|a\| > 2.2 g` | 30 accel | 0.7029 | 93.56 % | 6.98 % | 73.9 % | 0.826 |
| Document V2: 200 ms + logistic | 8 | 0.9021 | 84.25 % | 3.63 % | 83.1 % | 0.837 |
| Document V2 features + XGBoost | 8 | 0.9305 | 88.63 % | 3.39 % | 84.7 % | 0.866 |
| Logistic regression | 37 | 0.9368 | 90.83 % | 3.42 % | 84.9 % | 0.878 |
| Decision tree | 37 | 0.9013 | 97.63 % | 7.60 % | 73.1 % | 0.836 |
| Random forest | 37 | 0.9585 | 89.29 % | 2.34 % | 89.0 % | 0.891 |
| Extra trees | 37 | 0.9532 | 88.70 % | 2.58 % | 87.9 % | 0.883 |
| MLP neural network | 37 | 0.9606 | 94.10 % | 3.11 % | 86.5 % | 0.901 |
| **LightGBM (selected)** | **37** | **0.9636** | **92.25 %** | **2.43 %** | **88.9 %** | **0.906** |
| XGBoost | 37 | 0.9622 | 92.54 % | 2.63 % | 88.2 % | 0.903 |
| XGBoost + GPS speed | 40 | 0.9677 | 93.37 % | 2.27 % | 89.7 % | 0.915 |

**Selection rule:** highest event AP among models that run on the ESP32 with no extra hardware.
LightGBM wins; XGBoost is within 0.0014 AP and would serve equally well.

### 7.2 Operating points

| Threshold | Recall | False alarms | Use |
|---|---|---|---|
| P ≥ 0.50 | higher recall, 10 s countdown before alerting | - | driver-verified band |
| **P ≥ 0.80** | **92.25 %** | **2.43 %** | shipped: immediate alert |
| P ≥ 0.966 (tuned on validation) | 72.66 % | 0.59 % | conservative fleet setting |

### 7.3 Precision–recall curves

Computed on the test split from the exact model in the firmware (`pr_curves.json`). The proposed
model dominates the document's V2 design across the whole operating range; the V1 rule is a single
point far below both curves.

| Curve | AP | Recall @0.8 | Precision @0.8 |
|---|---|---|---|
| Proposed (LightGBM, 37 features) | 0.9636 | 92.3 % | 88.9 % |
| With GPS speed (40 features) | 0.9677 | 93.4 % | 89.7 % |
| Accelerometer only (30 features) | 0.9559 | 91.7 % | 86.9 % |
| Document V2 (200 ms, 8 features) | 0.9021 | 84.2 % | 83.1 % |
| V1 rule (single point) | 0.7029 | 93.6 % | 73.9 % |

### 7.4 Sensor ablation, what the gyroscope and GPS are worth

| Inputs | Features | AP | Recall @0.8 | False alarms |
|---|---|---|---|---|
| Accelerometer only | 30 | 0.9559 | 91.7 % | 2.94 % |
| **+ gyroscope (shipped)** | **37** | **0.9622** | **92.5 %** | **2.63 %** |
| + GPS speed | 40 | 0.9677 | 93.4 % | 2.27 % |

Same algorithm (XGBoost), same windows, the differences are attributable to the inputs alone. The
gyroscope is free. GPS costs a NEO-6M and is the recommended upgrade path.

### 7.5 Window ablation, why 2.5 s, not 200 ms

| Design | Window | Features | Model | AP |
|---|---|---|---|---|
| Document V2 as written | 200 ms | 8 | Logistic | 0.9021 |
| Document V2 features, stronger model | 200 ms | 8 | XGBoost | 0.9305 |
| Proposed features, same model family | 2.5 s | 37 | XGBoost | 0.9622 |
| **Proposed (selected)** | **2.5 s** | **37** | **LightGBM** | **0.9636** |

Changing only the classifier buys **+0.028 AP**; changing the window and feature vector buys a
further **+0.032 AP**. The pre-impact context and the post-impact settling carry as much information
as the impact peak, which is the reason for claim N3's window geometry.

### 7.6 Crash detection by severity (the honest limit)

| Peak dynamic acceleration | Crashes in test set | Detected at P ≥ 0.80 |
|---|---|---|
| < 1.5 g | 146 | **21.2 %** |
| 1.5 – 3 g | 1,093 | 86.6 % |
| 3 – 6 g | 2,702 | 97.9 % |
| > 6 g | 274 | 99.6 % |

### 7.7 Crash detection by vehicle class

| Vehicle size | Crashes | Detected |
|---|---|---|
| Light duty | 834 | 94.0 % |
| Medium duty | 2,203 | 94.7 % |
| Heavy duty | 929 | 85.9 % |
| Unrecorded | 249 | 90.8 % |

Heavier vehicles absorb impact, so the same collision registers as a smaller jolt, a physically
expected result that a fleet deployment should account for by lowering the threshold per vehicle class.

### 7.8 Model B, naming the jolt

Leave-one-vehicle-out cross-validation on real road events. XGBoost selected: **macro F1 0.782,
accuracy 95.9 %** (random forest 0.750 / 94.0 %, logistic regression 0.585 / 84.5 %).

Confusion matrix (rows = truth, columns = prediction):

| | Normal | Speed bump | Pothole / rough | Harsh manoeuvre |
|---|---|---|---|---|
| **Normal** (141) | **125** | 2 | 7 | 7 |
| **Speed bump** (95) | 1 | **53** | 38 | 3 |
| **Pothole / rough road** (3,243) | 41 | 31 | **3,167** | 4 |
| **Harsh manoeuvre** (55) | 5 | 1 | 6 | **43** |

Per-class F1: pothole/rough road 0.980, normal 0.799, harsh manoeuvre 0.768, speed bump 0.582.

---

## 8. Embedded implementation and verification

| Measure | Value |
|---|---|
| Transmitter firmware, both models | **658,987 bytes (50 % of ESP32 flash)** |
| RAM | 45,052 bytes (13 %) |
| Model A | LightGBM, 400 trees, 24,400 nodes, 168 KB |
| Model B | XGBoost, 1,000 trees (250 rounds × 4 classes), 24,484 nodes, 171 KB |
| Feature + inference time | 24 µs per event on the development host; the sketch prints the on-device figure with every event |
| Receiver firmware | 1,028,692 bytes (78 %) |

**Numerical parity** (`verify_firmware.py`, 1,987 real windows, VZCrash events and PVS road events):

```
feature max relative error:   1.49e-04
crash model max |dP|:         6.67e-07  (99th pct 3.9e-07)
same alert decision (p>=.8):  100.0000%
road model same class:        100.0000%
```

The same computation is implemented three times, Python (training), C (firmware), JavaScript
(dashboard), and all three agree on every decision. This is what makes the reported accuracy a
statement about the deployed device rather than about an offline notebook.

---

## 9. Comparison with published work on the same dataset

| System | Parameters | AP | Runs on |
|---|---|---|---|
| VZCrash physics baseline | - | 0.896 | - |
| Chronos-Bolt | 8.8 M | 0.966 | GPU/server |
| CNN-Transformer | 1.3 M | 0.975 | GPU/server |
| CNN-RNN | 1.7 M | 0.976 | GPU/server |
| **This work** | **37 features, 48.9 k tree nodes** | **0.964** | **ESP32, 339 KB, 24 µs** |

**Caveat, stated plainly:** the published figures are computed over every event in the dataset,
while this system only sees events that cross its 0.45 g trigger, untriggered crashes are counted
as misses here, but the candidate set differs, so the comparison is indicative rather than strictly
like-for-like. The defensible claim is that a microcontroller-sized model lands between the paper's
physics baseline and its deep networks.

---

## 10. Limitations

1. **Gentle crashes.** Below 1.5 g only 21.2 % are detected. The 0.50–0.80 countdown band is the
   mitigation, not a fix.
2. **The negatives are hard by construction.** VZCrash's non-crash events were selected as
   interesting, near misses, potholes, off-road manoeuvres. 2.43 % of *those* is not 2.43 % of
   driving time: over 200 s of ordinary recorded driving (119 classified jolts) the system raised
   no crash alert at all.
3. **Model B does not transfer across sensors.** On unseen iPhone drives it recognised only 1 of 12
   confirmed potholes as a pothole, though it produced almost no false crash verdicts (1 in 252).
4. **No on-vehicle validation yet.** Everything is measured on recorded data from other devices.
   The ESP32 timing, the buzzer and the cancel button have not been exercised in a moving vehicle.
5. **Heavy vehicles are harder** (85.9 % vs 94.7 % for medium duty).
6. **Rollover remains a physics rule**, not a learned model: VZCrash carries no rollover label.

---

## 11. Reproducibility

| Step | Command |
|---|---|
| Crash data | `download_vzcrash.py` (HuggingFace, gated, CC BY-NC 4.0) |
| Road data | `download_data.py` (PVS, Driver Behavior, NHTSA, pothole drives) |
| Windows + features | `build_vzcrash.py`, `build_dataset.py` |
| Model A | `train_vzcrash.py` (12 models, official splits) |
| Model B | `train_road_model.py` (leave-one-vehicle-out) |
| Curves | `pr_curves.py` |
| Export to firmware | `export_model.py` |
| Verify C = Python | `verify_firmware.py` |

Seeds are fixed (42). LightGBM training is not bit-reproducible across runs because of multithreaded
histogram ties, so the invariant enforced is that **the exported header matches the saved model**,
which `verify_firmware.py` checks.

---

## 12. Data sources and licences

| Dataset | Use | Licence |
|---|---|---|
| **VZCrash**, Bianconcini, Monteagudo, Pjetri, Trinci, Taccari, *VZCrash: A Large-Scale IMU Dataset of Ego-Vehicle Crashes*, IEEE ITSC 2026 | Model A training and all crash metrics | CC BY-NC 4.0 (non-commercial) |
| **PVS**, Menegazzo & von Wangenheim, UFSC | Model B: speed bumps, rough roads | CC BY-NC-ND 4.0 |
| **Driver Behavior Dataset**, Ferreira et al., PLOS ONE 2017 | Model B: harsh manoeuvres | no licence stated |
| **NHTSA Vehicle Crash Test Database** | first-generation crash model, N7 method | US Government, public domain |
| **Real-World Pothole Drives**, W. Stading | external transfer test only | CC BY-NC 4.0 |

**Patent implication:** filing on the *method* (N1–N8) is unaffected by these licences. A commercial
product must retrain Model A on owned or commercially licensed crash recordings; the pipeline is
unchanged, only the data source swaps.

---

## 13. Claim structure

**Independent claim (method).** A method of discriminating vehicular collisions from road-surface
disturbances in a vehicle-mounted inertial sensing device, comprising: maintaining a rolling buffer
of tri-axial acceleration and angular rate; estimating a gravity vector; detecting a candidate event
when the gravity-referenced dynamic acceleration exceeds a threshold; forming an analysis window
spanning a pre-trigger interval and a post-trigger interval; **re-anchoring** said window upon a
subsequent larger disturbance; computing a feature vector in which acceleration is decomposed into
components parallel and perpendicular to the estimated gravity vector, and in which angular rate is
decomposed into components about and across said vector; evaluating a first trained ensemble
classifier to produce a collision confidence; and selecting between immediate transmission, a
time-limited operator-cancellable verification state, and classification by a second trained
ensemble classifier into road-disturbance categories, according to bands of said confidence.

**Dependent claims.** The re-anchoring criterion (N3); the horizontal velocity-change feature (N1);
yaw-versus-tilt separation (N5); the two-model split and their independent training corpora (N2);
transmission of class and confidence over a sub-GHz link with receiver-side de-duplication (N8);
the flattened-tree fixed-point/float32 deployment with verified decision parity (N6); the
speed-scaled pulse augmentation used to train for low-severity impacts (N7).

---

*Prepared from the measured results of this project. Every figure is reproducible with the commands
in Section 11; the underlying tables are in `ml/results/`.*

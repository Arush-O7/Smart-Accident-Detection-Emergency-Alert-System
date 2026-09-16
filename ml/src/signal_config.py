"""Constants shared by the Python pipeline, the firmware header generator and the web demo.

Everything here is mirrored 1:1 in firmware (crash_features.h) - change both or neither.
"""
FS = 100                 # Hz, MPU6050 sample rate used by the firmware
N_PRE = 100              # samples kept before the trigger (1.0 s)
N_POST = 150             # samples collected after the trigger (1.5 s)
WIN = N_PRE + N_POST     # 250-sample (2.5 s) event window
N_BASE = 50              # first 0.5 s of the window = gravity / baseline estimate
N_TAIL = 50              # last 0.5 s of the window = post-event state
SEARCH_FROM = N_PRE - 10 # impact search starts 100 ms before the trigger

TRIGGER_G = 0.45         # |a - gravity_ema| above this wakes the classifier (g)
EMA_ALPHA = 0.02         # gravity tracker, time constant ~0.5 s at 100 Hz

ACC_FULL_SCALE_G = 16.0  # MPU6050 configured for +-16 g
ACC_LSB_PER_G = 2048.0
G = 9.80665

CLASSES = ["NORMAL", "SPEED_BUMP", "POTHOLE_ROUGH_ROAD", "HARSH_MANEUVER", "CRASH"]
CRASH = CLASSES.index("CRASH")

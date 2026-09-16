"""Orientation-invariant event features for crash vs road-anomaly classification.

Input: one 250-sample window of 3-axis acceleration in g (gravity included), triggered as in
the firmware. The sensor may be mounted in any orientation: gravity is estimated from the first
0.5 s and the motion is split into a vertical part (bumps, potholes) and a horizontal part
(braking, collisions).

The loops are written the way the C version in firmware/crash_features.h is written, so the two
produce the same numbers. Use float32 throughout for parity with the ESP32.
"""
import math

import numpy as np

from signal_config import N_BASE, N_PRE, N_TAIL, SEARCH_FROM, WIN, G

FEATURE_NAMES = [
    "m_peak",           # peak |dynamic acceleration| (g)
    "h_peak",           # peak horizontal acceleration (g)
    "v_peak",           # peak |vertical| acceleration (g)
    "hv_ratio",         # horizontal / vertical peak ratio
    "amag_peak",        # peak raw |a| incl. gravity (what the old 2.2 g rule used)
    "jerk_peak",        # peak jerk (g/s)
    "n_above_0g5",      # samples with |dyn| > 0.5 g
    "n_above_1g",
    "n_above_2g",
    "n_above_4g",
    "width_half",       # samples above half the peak (pulse width)
    "dv_h_kmh",         # horizontal velocity change around the impact (km/h)
    "dv_v_kmh",         # vertical velocity change around the impact (km/h)
    "pre_std",          # roughness before the event
    "pre_h_mean",       # horizontal load before the event (braking / cornering)
    "post_std",         # vibration in the last 0.5 s (vehicle still moving?)
    "post_m_mean",
    "post_h_mean",      # sustained horizontal load after the event
    "tilt_change_deg",  # orientation change between start and end of window
    "post_gmag_dev",    # change in gravity magnitude start vs end
    "n_peaks",          # separated peaks (front + rear axle hitting a bump)
    "peak_span",        # samples between first and last significant peak
    "energy_impact",    # sum |dyn|^2 around the impact
    "energy_post",      # sum |dyn|^2 after the trigger
    "zc_v",             # vertical zero crossings (suspension oscillation)
    "peak_time",        # impact position relative to the trigger (samples)
    "h_energy_frac",    # share of energy that is horizontal
    "crest",            # peak / rms after the trigger
    "sustained_h",      # max 200 ms mean of horizontal acceleration (g)
    "early_rms",        # rms of |dyn| in the 0.5 s after the trigger
]
# Gyroscope and GPS-speed features, available from VZCrash (and from the MPU6050 / NEO-6M on the
# transmitter). Appended after the 30 accelerometer features, so index 0..29 never change.
GYRO_FEATURE_NAMES = [
    "w_peak",           # peak angular rate (deg/s)
    "w_mean_post",      # mean angular rate in the last 0.5 s
    "dtheta_impact",    # |integrated rotation| around the impact (deg)
    "dtheta_post",      # total rotation travelled after the trigger (deg)
    "w_yaw_peak",       # peak rotation about gravity (spin / yaw)
    "w_tilt_peak",      # peak rotation across gravity (roll / pitch)
    "w_energy_post",    # rotational energy after the trigger
]
SPEED_FEATURE_NAMES = [
    "speed_pre_kmh",    # GPS speed before the event
    "speed_drop_kmh",   # speed lost across the event
    "speed_drop_frac",  # fraction of speed lost
]
FEATURE_NAMES_EXT = FEATURE_NAMES + GYRO_FEATURE_NAMES
FEATURE_NAMES_FULL = FEATURE_NAMES_EXT + SPEED_FEATURE_NAMES
N_FEATURES = len(FEATURE_NAMES)
N_FEATURES_EXT = len(FEATURE_NAMES_EXT)
N_FEATURES_FULL = len(FEATURE_NAMES_FULL)
KMH_PER_G_SAMPLE = G * 3.6 / 100.0   # integrate g at 100 Hz into km/h


def extract(window: np.ndarray) -> np.ndarray:
    a = np.asarray(window, dtype=np.float32)
    assert a.shape == (WIN, 3), a.shape
    f = np.zeros(N_FEATURES, dtype=np.float32)

    g0 = a[:N_BASE].mean(axis=0, dtype=np.float32)
    gmag = float(np.sqrt(np.dot(g0, g0)))
    u = g0 / gmag if gmag > 0.3 else np.array([0, 0, 1], np.float32)

    d = a - g0                                  # dynamic acceleration
    v = d @ u                                   # vertical component
    hvec = d - np.outer(v, u)
    h = np.sqrt((hvec * hvec).sum(axis=1))
    m = np.sqrt((d * d).sum(axis=1))
    amag = np.sqrt((a * a).sum(axis=1))

    s = slice(SEARCH_FROM, WIN)
    ms, hs, vs = m[s], h[s], np.abs(v[s])
    p = SEARCH_FROM + int(np.argmax(ms))
    m_peak = float(m[p])

    f[0] = m_peak
    f[1] = hs.max()
    f[2] = vs.max()
    f[3] = f[1] / (f[2] + 0.05)
    f[4] = amag.max()
    jerk = np.sqrt(((d[1:] - d[:-1]) ** 2).sum(axis=1)) * 100.0
    f[5] = jerk[SEARCH_FROM - 1:].max()
    f[6] = (ms > 0.5).sum()
    f[7] = (ms > 1.0).sum()
    f[8] = (ms > 2.0).sum()
    f[9] = (ms > 4.0).sum()
    f[10] = (ms > 0.5 * m_peak).sum()

    lo, hi = max(p - 10, 0), min(p + 31, WIN)
    f[11] = np.linalg.norm(hvec[lo:hi].sum(axis=0)) * KMH_PER_G_SAMPLE
    f[12] = abs(float(v[lo:hi].sum())) * KMH_PER_G_SAMPLE

    pre = slice(0, SEARCH_FROM)
    f[13] = m[pre].std()
    f[14] = h[pre].mean()
    tail = slice(WIN - N_TAIL, WIN)
    f[15] = m[tail].std()
    f[16] = m[tail].mean()
    f[17] = h[WIN - 90:].mean()

    g1 = a[tail].mean(axis=0)
    g1mag = float(np.sqrt(np.dot(g1, g1)))
    cross = np.cross(g0, g1)
    f[18] = math.degrees(math.atan2(float(np.sqrt(np.dot(cross, cross))), float(np.dot(g0, g1))))
    f[19] = abs(g1mag - gmag)

    thr = max(0.3, 0.3 * m_peak)
    n_peaks, last_peak, first_i, last_i = 0, -1000, -1, -1
    for i in range(SEARCH_FROM, WIN):
        if m[i] > thr:
            if first_i < 0:
                first_i = i
            last_i = i
            left = m[i - 1] if i > 0 else 0.0
            right = m[i + 1] if i + 1 < WIN else 0.0
            if m[i] >= left and m[i] >= right and i - last_peak >= 15:
                n_peaks += 1
                last_peak = i
    f[20] = n_peaks
    f[21] = (last_i - first_i) if first_i >= 0 else 0

    f[22] = (m[lo:hi] ** 2).sum() * 0.01
    energy_post = float((ms ** 2).sum())
    f[23] = energy_post * 0.01

    zc, state = 0, 0
    for i in range(SEARCH_FROM, WIN):
        if v[i] > 0.05:
            if state < 0:
                zc += 1
            state = 1
        elif v[i] < -0.05:
            if state > 0:
                zc += 1
            state = -1
    f[24] = zc
    f[25] = p - N_PRE
    f[26] = float((hs ** 2).sum()) / (energy_post + 1e-6)
    rms = math.sqrt(energy_post / len(ms))
    f[27] = m_peak / (rms + 1e-6)
    csum = np.concatenate([[0.0], np.cumsum(hs, dtype=np.float64)])
    f[28] = ((csum[20:] - csum[:-20]) / 20.0).max()
    f[29] = math.sqrt(float((m[N_PRE:N_PRE + 50] ** 2).mean()))
    return f


def extract_gyro(acc_window: np.ndarray, gyro_window: np.ndarray) -> np.ndarray:
    """The 7 gyroscope features. Same window and gravity convention as extract()."""
    a = np.asarray(acc_window, dtype=np.float32)
    w = np.asarray(gyro_window, dtype=np.float32)
    f = np.zeros(len(GYRO_FEATURE_NAMES), dtype=np.float32)

    g0 = a[:N_BASE].mean(axis=0, dtype=np.float32)
    gmag = float(np.sqrt(np.dot(g0, g0)))
    u = g0 / gmag if gmag > 0.3 else np.array([0, 0, 1], np.float32)

    wmag = np.sqrt((w * w).sum(axis=1))
    yaw = w @ u
    tilt = np.sqrt(np.maximum((w * w).sum(axis=1) - yaw * yaw, 0.0))

    d = a - g0
    p = SEARCH_FROM + int(np.argmax(np.sqrt((d[SEARCH_FROM:] * d[SEARCH_FROM:]).sum(axis=1))))
    lo, hi = max(p - 10, 0), min(p + 31, WIN)
    s = slice(SEARCH_FROM, WIN)

    f[0] = wmag[s].max()
    f[1] = wmag[WIN - N_TAIL:].mean()
    f[2] = np.linalg.norm(w[lo:hi].sum(axis=0)) * 0.01
    f[3] = wmag[s].sum() * 0.01
    f[4] = np.abs(yaw[s]).max()
    f[5] = tilt[s].max()
    f[6] = float((wmag[s] ** 2).sum()) * 1e-4
    return f


def extract_speed(speed_kmh_pre: float, speed_kmh_post: float) -> np.ndarray:
    """The 3 GPS-speed features (needs the NEO-6M; the accel+gyro model does not use these)."""
    drop = float(speed_kmh_pre - speed_kmh_post)
    return np.array([speed_kmh_pre, drop, drop / (speed_kmh_pre + 1.0)], dtype=np.float32)


# --- Assessment-6 document baseline (Section 1.1 "Version 2"): 200 ms window, 8 features.
PDF200_FEATURE_NAMES = [
    "A_peak",     # peak resultant acceleration (g), as the document's serial log prints it
    "w_peak",     # peak resultant angular rate (deg/s)
    "J_peak",     # peak jerk (g/s)
    "tau_ms",     # time above the 0.5 g dynamic threshold (ms)
    "N_v",        # longest run of consecutive violating samples
    "dV_kmh",     # GPS speed lost across the event
    "dTheta_deg", # integrated rotation over the window (deg)
    "A_rms",      # rms dynamic acceleration over the window (g)
]
PDF200_PRE = 5    # the 200 ms window starts 50 ms before the trigger
PDF200_LEN = 20   # 200 ms at 100 Hz


def extract_pdf200(acc_window, gyro_window, speed_pre_kmh, speed_post_kmh):
    """The document's 8-element feature vector over its 200 ms window, for the baseline comparison."""
    a = np.asarray(acc_window, np.float32)[N_PRE - PDF200_PRE: N_PRE - PDF200_PRE + PDF200_LEN]
    w = np.asarray(gyro_window, np.float32)[N_PRE - PDF200_PRE: N_PRE - PDF200_PRE + PDF200_LEN]
    base = np.asarray(acc_window, np.float32)[:N_BASE].mean(axis=0)
    amag = np.sqrt((a * a).sum(axis=1))
    dyn = np.sqrt(((a - base) ** 2).sum(axis=1))
    wmag = np.sqrt((w * w).sum(axis=1))
    jerk = np.sqrt(((a[1:] - a[:-1]) ** 2).sum(axis=1)) * 100.0
    above = dyn > 0.5
    run = best = 0
    for flag in above:
        run = run + 1 if flag else 0
        best = max(best, run)
    return np.array([
        amag.max(), wmag.max(), jerk.max() if len(jerk) else 0.0, float(above.sum()) * 10.0,
        float(best), float(speed_pre_kmh - speed_post_kmh), float(np.linalg.norm(w.sum(axis=0)) * 0.01),
        float(np.sqrt((dyn ** 2).mean())),
    ], dtype=np.float32)


def extract_many(windows: np.ndarray) -> np.ndarray:
    return np.stack([extract(w) for w in windows])

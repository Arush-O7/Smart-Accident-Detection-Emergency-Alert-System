"""Signal conditioning shared by dataset building, evaluation and the demo exporter."""
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from signal_config import ACC_FULL_SCALE_G, ACC_LSB_PER_G, EMA_ALPHA, FS, N_POST, N_PRE, TRIGGER_G, WIN


def resample_uniform(t, x, fs_out=FS):
    """Resample irregularly timed samples (t in s, x shape (N, C)) onto a uniform grid."""
    order = np.argsort(t)
    t, x = t[order], x[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, x = t[keep], x[keep]
    fs_mid = 200
    grid = np.arange(t[0], t[-1], 1.0 / fs_mid)
    mid = np.stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])], axis=1)
    return grid[::fs_mid // fs_out], resample_poly(mid, 1, fs_mid // fs_out, axis=0)


def resample_rate(x, fs_in, fs_out=FS):
    """Anti-aliased resampling between fixed rates (approximates the MPU6050 digital low-pass)."""
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(x, frac.numerator, frac.denominator, axis=0)


def mpu6050(acc_g, rng=None):
    """Make any accelerometer stream look like the MPU6050 on the transmitter (+-16 g, 2048 LSB/g)."""
    a = np.asarray(acc_g, dtype=np.float64)
    if rng is not None:
        a = a + rng.normal(0.0, 0.004, size=a.shape)  # ~400 ug/sqrt(Hz) noise density at 44 Hz
    a = np.clip(a, -ACC_FULL_SCALE_G, ACC_FULL_SCALE_G - 1.0 / ACC_LSB_PER_G)
    return (np.round(a * ACC_LSB_PER_G) / ACC_LSB_PER_G).astype(np.float32)


def gravity_ema(acc_g):
    out = np.empty_like(acc_g)
    e = acc_g[0].astype(np.float64).copy()
    for i in range(len(acc_g)):
        e += EMA_ALPHA * (acc_g[i] - e)
        out[i] = e
    return out


def stream_triggers(acc_g, start=0):
    """Yield window anchor indices exactly as the firmware state machine does.

    A trigger opens a window. While the post-trigger samples are being collected, a much stronger
    jolt (a collision right after a pothole, say) re-anchors the window on the stronger jolt, so a
    small event can never hide a crash.
    """
    dyn = np.linalg.norm(acc_g - gravity_ema(acc_g), axis=1)
    i, n = max(start, N_PRE), len(acc_g)
    while i < n - N_POST:
        if dyn[i] > TRIGGER_G:
            anchor, peak, j = i, dyn[i], i + 1
            while j < anchor + N_POST and j < n - N_POST:
                if j - anchor > 10 and dyn[j] > max(1.0, 1.5 * peak):
                    anchor, peak = j, dyn[j]
                peak = max(peak, dyn[j])
                j += 1
            yield anchor
            i = anchor + N_POST
        else:
            i += 1


def window_at(acc_g, anchor):
    lo = anchor - N_PRE
    if lo < 0 or lo + WIN > len(acc_g):
        return None
    return acc_g[lo:lo + WIN]

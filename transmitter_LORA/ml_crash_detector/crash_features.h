// Orientation-invariant crash / road-anomaly features.
// C port of ml/src/features.py - keep the two in lockstep (ml/scripts/verify_firmware.py checks it).
#pragma once
#include <math.h>
#include <string.h>

#define CF_FS          100
#define CF_N_PRE       100   // samples before the trigger (1.0 s)
#define CF_N_POST      150   // samples after the trigger (1.5 s)
#define CF_WIN         (CF_N_PRE + CF_N_POST)
#define CF_N_BASE      50    // gravity baseline: first 0.5 s
#define CF_N_TAIL      50    // post-event state: last 0.5 s
#define CF_SEARCH_FROM (CF_N_PRE - 10)
#define CF_N_FEATURES        30   // accelerometer features (road model uses these)
#define CF_N_GYRO_FEATURES    7   // gyroscope features (crash model uses accel + gyro)
#define CF_N_FEATURES_EXT    (CF_N_FEATURES + CF_N_GYRO_FEATURES)
#define CF_KMH_PER_G_SAMPLE (9.80665f * 3.6f / 100.0f)

// win: CF_WIN rows of {ax, ay, az} in g, oldest first. out: CF_N_FEATURES values.
static inline void crash_features(const float win[][3], float out[CF_N_FEATURES]) {
  static float d[CF_WIN][3], hv[CF_WIN][3], v[CF_WIN], h[CF_WIN], m[CF_WIN];
  memset(out, 0, sizeof(float) * CF_N_FEATURES);

  float g0[3] = {0, 0, 0};
  for (int i = 0; i < CF_N_BASE; i++)
    for (int k = 0; k < 3; k++) g0[k] += win[i][k];
  for (int k = 0; k < 3; k++) g0[k] /= CF_N_BASE;
  float gmag = sqrtf(g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]);
  float u[3] = {0, 0, 1};
  if (gmag > 0.3f)
    for (int k = 0; k < 3; k++) u[k] = g0[k] / gmag;

  float amag_peak = 0;
  for (int i = 0; i < CF_WIN; i++) {
    float vv = 0, hh = 0, mm = 0, aa = 0;
    for (int k = 0; k < 3; k++) d[i][k] = win[i][k] - g0[k];
    for (int k = 0; k < 3; k++) vv += d[i][k] * u[k];
    for (int k = 0; k < 3; k++) {
      hv[i][k] = d[i][k] - vv * u[k];
      hh += hv[i][k] * hv[i][k];
      mm += d[i][k] * d[i][k];
      aa += win[i][k] * win[i][k];
    }
    v[i] = vv; h[i] = sqrtf(hh); m[i] = sqrtf(mm);
    aa = sqrtf(aa);
    if (aa > amag_peak) amag_peak = aa;
  }

  int p = CF_SEARCH_FROM;
  float h_peak = 0, v_peak = 0, jerk_peak = 0, energy_post = 0, h_energy = 0;
  int n05 = 0, n1 = 0, n2 = 0, n4 = 0;
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) {
    if (m[i] > m[p]) p = i;
    if (h[i] > h_peak) h_peak = h[i];
    if (fabsf(v[i]) > v_peak) v_peak = fabsf(v[i]);
    n05 += m[i] > 0.5f; n1 += m[i] > 1.0f; n2 += m[i] > 2.0f; n4 += m[i] > 4.0f;
    energy_post += m[i] * m[i];
    h_energy += h[i] * h[i];
  }
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) {
    float jx = d[i][0] - d[i - 1][0], jy = d[i][1] - d[i - 1][1], jz = d[i][2] - d[i - 1][2];
    float j = sqrtf(jx * jx + jy * jy + jz * jz) * 100.0f;
    if (j > jerk_peak) jerk_peak = j;
  }
  float m_peak = m[p];
  int width = 0;
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) width += m[i] > 0.5f * m_peak;

  int lo = p - 10 < 0 ? 0 : p - 10;
  int hi = p + 31 > CF_WIN ? CF_WIN : p + 31;
  float sh[3] = {0, 0, 0}, sv = 0, e_imp = 0;
  for (int i = lo; i < hi; i++) {
    for (int k = 0; k < 3; k++) sh[k] += hv[i][k];
    sv += v[i];
    e_imp += m[i] * m[i];
  }

  // variances use deviations from the mean (E[x^2] - mean^2 cancels badly in float32)
  float pre_mean = 0, pre_var = 0, pre_h = 0;
  for (int i = 0; i < CF_SEARCH_FROM; i++) { pre_mean += m[i]; pre_h += h[i]; }
  pre_mean /= CF_SEARCH_FROM;
  for (int i = 0; i < CF_SEARCH_FROM; i++) pre_var += (m[i] - pre_mean) * (m[i] - pre_mean);
  pre_var /= CF_SEARCH_FROM;

  float post_mean = 0, post_var = 0, g1[3] = {0, 0, 0};
  for (int i = CF_WIN - CF_N_TAIL; i < CF_WIN; i++) {
    post_mean += m[i];
    for (int k = 0; k < 3; k++) g1[k] += win[i][k];
  }
  post_mean /= CF_N_TAIL;
  for (int i = CF_WIN - CF_N_TAIL; i < CF_WIN; i++) post_var += (m[i] - post_mean) * (m[i] - post_mean);
  post_var /= CF_N_TAIL;
  for (int k = 0; k < 3; k++) g1[k] /= CF_N_TAIL;
  float post_h = 0;
  for (int i = CF_WIN - 90; i < CF_WIN; i++) post_h += h[i];

  float g1mag = sqrtf(g1[0] * g1[0] + g1[1] * g1[1] + g1[2] * g1[2]);
  // tilt angle via atan2(|g0 x g1|, g0 . g1): acos is too coarse near 0 degrees in float32
  float cx = g0[1] * g1[2] - g0[2] * g1[1], cy = g0[2] * g1[0] - g0[0] * g1[2], cz = g0[0] * g1[1] - g0[1] * g1[0];
  float tilt = atan2f(sqrtf(cx * cx + cy * cy + cz * cz), g0[0] * g1[0] + g0[1] * g1[1] + g0[2] * g1[2]);

  float thr = 0.3f * m_peak > 0.3f ? 0.3f * m_peak : 0.3f;
  int n_peaks = 0, last_peak = -1000, first_i = -1, last_i = -1;
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) {
    if (m[i] > thr) {
      if (first_i < 0) first_i = i;
      last_i = i;
      float right = i + 1 < CF_WIN ? m[i + 1] : 0.0f;
      if (m[i] >= m[i - 1] && m[i] >= right && i - last_peak >= 15) { n_peaks++; last_peak = i; }
    }
  }

  int zc = 0, state = 0;
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) {
    if (v[i] > 0.05f) { if (state < 0) zc++; state = 1; }
    else if (v[i] < -0.05f) { if (state > 0) zc++; state = -1; }
  }

  float run = 0, sustained = 0;
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) {
    run += h[i];
    if (i - CF_SEARCH_FROM >= 20) run -= h[i - 20];
    if (i - CF_SEARCH_FROM >= 19 && run / 20.0f > sustained) sustained = run / 20.0f;
  }
  float early = 0;
  for (int i = CF_N_PRE; i < CF_N_PRE + 50; i++) early += m[i] * m[i];

  float rms = sqrtf(energy_post / (CF_WIN - CF_SEARCH_FROM));
  out[0] = m_peak;
  out[1] = h_peak;
  out[2] = v_peak;
  out[3] = h_peak / (v_peak + 0.05f);
  out[4] = amag_peak;
  out[5] = jerk_peak;
  out[6] = n05; out[7] = n1; out[8] = n2; out[9] = n4;
  out[10] = width;
  out[11] = sqrtf(sh[0] * sh[0] + sh[1] * sh[1] + sh[2] * sh[2]) * CF_KMH_PER_G_SAMPLE;
  out[12] = fabsf(sv) * CF_KMH_PER_G_SAMPLE;
  out[13] = sqrtf(pre_var);
  out[14] = pre_h / CF_SEARCH_FROM;
  out[15] = sqrtf(post_var);
  out[16] = post_mean;
  out[17] = post_h / 90.0f;
  out[18] = tilt * 180.0f / (float)M_PI;
  out[19] = fabsf(g1mag - gmag);
  out[20] = n_peaks;
  out[21] = first_i >= 0 ? last_i - first_i : 0;
  out[22] = e_imp * 0.01f;
  out[23] = energy_post * 0.01f;
  out[24] = zc;
  out[25] = p - CF_N_PRE;
  out[26] = h_energy / (energy_post + 1e-6f);
  out[27] = m_peak / (rms + 1e-6f);
  out[28] = sustained;
  out[29] = sqrtf(early / 50.0f);
}

// Gyroscope features, appended after the 30 accelerometer features.
// win: acceleration in g, gyro: angular rate in deg/s, both CF_WIN rows oldest first.
// C port of extract_gyro() in ml/src/features.py.
static inline void crash_features_gyro(const float win[][3], const float gyro[][3], float out[CF_N_GYRO_FEATURES]) {
  float g0[3] = {0, 0, 0};
  for (int i = 0; i < CF_N_BASE; i++)
    for (int k = 0; k < 3; k++) g0[k] += win[i][k];
  for (int k = 0; k < 3; k++) g0[k] /= CF_N_BASE;
  float gmag = sqrtf(g0[0] * g0[0] + g0[1] * g0[1] + g0[2] * g0[2]);
  float u[3] = {0, 0, 1};
  if (gmag > 0.3f)
    for (int k = 0; k < 3; k++) u[k] = g0[k] / gmag;

  int p = CF_SEARCH_FROM;
  float best = -1;
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) {
    float mm = 0;
    for (int k = 0; k < 3; k++) { float dk = win[i][k] - g0[k]; mm += dk * dk; }
    if (mm > best) { best = mm; p = i; }
  }
  int lo = p - 10 < 0 ? 0 : p - 10;
  int hi = p + 31 > CF_WIN ? CF_WIN : p + 31;

  float w_peak = 0, yaw_peak = 0, tilt_peak = 0, w_sum = 0, w_energy = 0, tail = 0;
  float impact[3] = {0, 0, 0};
  for (int i = CF_SEARCH_FROM; i < CF_WIN; i++) {
    float wmag2 = gyro[i][0] * gyro[i][0] + gyro[i][1] * gyro[i][1] + gyro[i][2] * gyro[i][2];
    float wmag = sqrtf(wmag2);
    float yaw = gyro[i][0] * u[0] + gyro[i][1] * u[1] + gyro[i][2] * u[2];
    float tilt2 = wmag2 - yaw * yaw;
    float tilt = tilt2 > 0 ? sqrtf(tilt2) : 0.0f;
    if (wmag > w_peak) w_peak = wmag;
    if (fabsf(yaw) > yaw_peak) yaw_peak = fabsf(yaw);
    if (tilt > tilt_peak) tilt_peak = tilt;
    w_sum += wmag;
    w_energy += wmag2;
    if (i >= CF_WIN - CF_N_TAIL) tail += wmag;
  }
  for (int i = lo; i < hi; i++)
    for (int k = 0; k < 3; k++) impact[k] += gyro[i][k];

  out[0] = w_peak;
  out[1] = tail / CF_N_TAIL;
  out[2] = sqrtf(impact[0] * impact[0] + impact[1] * impact[1] + impact[2] * impact[2]) * 0.01f;
  out[3] = w_sum * 0.01f;
  out[4] = yaw_peak;
  out[5] = tilt_peak;
  out[6] = w_energy * 1e-4f;
}

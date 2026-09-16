// Crash AI engine for the browser: the same trigger, features and trees as the ESP32 firmware
// (transmitter_LORA/ml_crash_detector): model A detects crashes from 30 accelerometer + 7 gyroscope
// features, model B names the road event. Works as a classic script (window.CrashAI) and in Node.
(function (root) {
  const WIN = 250, N_PRE = 100, N_POST = 150, N_BASE = 50, N_TAIL = 50, SEARCH_FROM = 90;
  const KMH_PER_G_SAMPLE = 9.80665 * 3.6 / 100;

  function features(win) {
    // win: Float32Array(750) of ax,ay,az in g, oldest first
    const f = new Float64Array(30);
    const d = new Float64Array(WIN * 3), hv = new Float64Array(WIN * 3);
    const v = new Float64Array(WIN), h = new Float64Array(WIN), m = new Float64Array(WIN);
    const g0 = [0, 0, 0];
    for (let i = 0; i < N_BASE; i++) for (let k = 0; k < 3; k++) g0[k] += win[i * 3 + k];
    for (let k = 0; k < 3; k++) g0[k] = Math.fround(g0[k] / N_BASE);
    const gmag = Math.hypot(g0[0], g0[1], g0[2]);
    const u = gmag > 0.3 ? g0.map((x) => x / gmag) : [0, 0, 1];
    let amag = 0;
    for (let i = 0; i < WIN; i++) {
      let vv = 0, hh = 0, mm = 0, aa = 0;
      for (let k = 0; k < 3; k++) d[i * 3 + k] = win[i * 3 + k] - g0[k];
      for (let k = 0; k < 3; k++) vv += d[i * 3 + k] * u[k];
      for (let k = 0; k < 3; k++) {
        hv[i * 3 + k] = d[i * 3 + k] - vv * u[k];
        hh += hv[i * 3 + k] ** 2; mm += d[i * 3 + k] ** 2; aa += win[i * 3 + k] ** 2;
      }
      v[i] = vv; h[i] = Math.sqrt(hh); m[i] = Math.sqrt(mm);
      amag = Math.max(amag, Math.sqrt(aa));
    }
    let p = SEARCH_FROM, hPeak = 0, vPeak = 0, jerk = 0, ePost = 0, hE = 0, n05 = 0, n1 = 0, n2 = 0, n4 = 0;
    for (let i = SEARCH_FROM; i < WIN; i++) {
      if (m[i] > m[p]) p = i;
      hPeak = Math.max(hPeak, h[i]); vPeak = Math.max(vPeak, Math.abs(v[i]));
      n05 += m[i] > 0.5; n1 += m[i] > 1; n2 += m[i] > 2; n4 += m[i] > 4;
      ePost += m[i] * m[i]; hE += h[i] * h[i];
      const j = Math.hypot(d[i * 3] - d[i * 3 - 3], d[i * 3 + 1] - d[i * 3 - 2], d[i * 3 + 2] - d[i * 3 - 1]) * 100;
      jerk = Math.max(jerk, j);
    }
    const mPeak = m[p];
    let width = 0;
    for (let i = SEARCH_FROM; i < WIN; i++) width += m[i] > 0.5 * mPeak;
    const lo = Math.max(p - 10, 0), hi = Math.min(p + 31, WIN);
    const sh = [0, 0, 0]; let sv = 0, eImp = 0;
    for (let i = lo; i < hi; i++) { for (let k = 0; k < 3; k++) sh[k] += hv[i * 3 + k]; sv += v[i]; eImp += m[i] * m[i]; }
    const meanStd = (a, b) => {
      let s = 0; for (let i = a; i < b; i++) s += m[i];
      const mu = s / (b - a); let q = 0; for (let i = a; i < b; i++) q += (m[i] - mu) ** 2;
      return [mu, Math.sqrt(q / (b - a))];
    };
    const [, preStd] = meanStd(0, SEARCH_FROM);
    const [postMean, postStd] = meanStd(WIN - N_TAIL, WIN);
    let preH = 0; for (let i = 0; i < SEARCH_FROM; i++) preH += h[i];
    let postH = 0; for (let i = WIN - 90; i < WIN; i++) postH += h[i];
    const g1 = [0, 0, 0];
    for (let i = WIN - N_TAIL; i < WIN; i++) for (let k = 0; k < 3; k++) g1[k] += win[i * 3 + k];
    for (let k = 0; k < 3; k++) g1[k] /= N_TAIL;
    const g1mag = Math.hypot(g1[0], g1[1], g1[2]);
    const cr = [g0[1] * g1[2] - g0[2] * g1[1], g0[2] * g1[0] - g0[0] * g1[2], g0[0] * g1[1] - g0[1] * g1[0]];
    const tilt = Math.atan2(Math.hypot(cr[0], cr[1], cr[2]), g0[0] * g1[0] + g0[1] * g1[1] + g0[2] * g1[2]) * 180 / Math.PI;
    const thr = Math.max(0.3, 0.3 * mPeak);
    let nPeaks = 0, last = -1000, first = -1, lastI = -1;
    for (let i = SEARCH_FROM; i < WIN; i++) {
      if (m[i] > thr) {
        if (first < 0) first = i;
        lastI = i;
        const right = i + 1 < WIN ? m[i + 1] : 0;
        if (m[i] >= m[i - 1] && m[i] >= right && i - last >= 15) { nPeaks++; last = i; }
      }
    }
    let zc = 0, state = 0;
    for (let i = SEARCH_FROM; i < WIN; i++) {
      if (v[i] > 0.05) { if (state < 0) zc++; state = 1; } else if (v[i] < -0.05) { if (state > 0) zc++; state = -1; }
    }
    let run = 0, sustained = 0;
    for (let i = SEARCH_FROM; i < WIN; i++) {
      run += h[i];
      if (i - SEARCH_FROM >= 20) run -= h[i - 20];
      if (i - SEARCH_FROM >= 19) sustained = Math.max(sustained, run / 20);
    }
    let early = 0; for (let i = N_PRE; i < N_PRE + 50; i++) early += m[i] * m[i];
    const rms = Math.sqrt(ePost / (WIN - SEARCH_FROM));
    f.set([mPeak, hPeak, vPeak, hPeak / (vPeak + 0.05), amag, jerk, n05, n1, n2, n4, width,
      Math.hypot(sh[0], sh[1], sh[2]) * KMH_PER_G_SAMPLE, Math.abs(sv) * KMH_PER_G_SAMPLE,
      preStd, preH / SEARCH_FROM, postStd, postMean, postH / 90, tilt, Math.abs(g1mag - gmag),
      nPeaks, first >= 0 ? lastI - first : 0, eImp * 0.01, ePost * 0.01, zc, p - N_PRE,
      hE / (ePost + 1e-6), mPeak / (rms + 1e-6), sustained, Math.sqrt(early / 50)]);
    return f;
  }

  // The 7 gyroscope features (indices 30..36), mirroring extract_gyro() in ml/src/features.py.
  function gyroFeatures(win, gyro) {
    const g0 = [0, 0, 0];
    for (let i = 0; i < N_BASE; i++) for (let k = 0; k < 3; k++) g0[k] += win[i * 3 + k];
    for (let k = 0; k < 3; k++) g0[k] = Math.fround(g0[k] / N_BASE);
    const gmag = Math.hypot(g0[0], g0[1], g0[2]);
    const u = gmag > 0.3 ? g0.map((x) => x / gmag) : [0, 0, 1];
    let p = SEARCH_FROM, best = -1;
    for (let i = SEARCH_FROM; i < WIN; i++) {
      let mm = 0;
      for (let k = 0; k < 3; k++) { const d = win[i * 3 + k] - g0[k]; mm += d * d; }
      if (mm > best) { best = mm; p = i; }
    }
    const lo = Math.max(p - 10, 0), hi = Math.min(p + 31, WIN);
    let wPeak = 0, yawPeak = 0, tiltPeak = 0, wSum = 0, wEnergy = 0, tail = 0;
    const impact = [0, 0, 0];
    for (let i = SEARCH_FROM; i < WIN; i++) {
      const gx = gyro[i * 3], gy = gyro[i * 3 + 1], gz = gyro[i * 3 + 2];
      const w2 = gx * gx + gy * gy + gz * gz, wmag = Math.sqrt(w2);
      const yaw = gx * u[0] + gy * u[1] + gz * u[2];
      const tilt = Math.sqrt(Math.max(w2 - yaw * yaw, 0));
      wPeak = Math.max(wPeak, wmag); yawPeak = Math.max(yawPeak, Math.abs(yaw)); tiltPeak = Math.max(tiltPeak, tilt);
      wSum += wmag; wEnergy += w2;
      if (i >= WIN - N_TAIL) tail += wmag;
    }
    for (let i = lo; i < hi; i++) for (let k = 0; k < 3; k++) impact[k] += gyro[i * 3 + k];
    return [wPeak, tail / N_TAIL, Math.hypot(impact[0], impact[1], impact[2]) * 0.01,
      wSum * 0.01, yawPeak, tiltPeak, wEnergy * 1e-4];
  }

  // Sums the leaf values each tree reaches. `kind` decides how they become a probability:
  // "sigmoid" (LightGBM binary crash model) or "softmax" (XGBoost multi-class road model).
  function margins(model, x) {
    const K = model.n_classes, margin = model.bias.slice();
    const thr = model.thrF32 || (model.thrF32 = Float32Array.from(model.thr));
    const le = model.cmp === "<=";
    for (let t = 0; t < model.roots.length; t++) {
      let i = model.roots[t];
      while (model.feat[i] >= 0) {
        const v = Math.fround(x[model.feat[i]]);
        i = (le ? v <= thr[i] : v < thr[i]) ? i + 1 : model.no[i];
      }
      margin[t % K] += thr[i];
    }
    return margin;
  }

  function predict(model, x) {
    const margin = margins(model, x);
    if (model.kind === "sigmoid") return [1 / (1 + Math.exp(-margin[0]))];
    const mx = Math.max(...margin), e = margin.map((z) => Math.exp(z - mx)), s = e.reduce((a, b) => a + b, 0);
    return e.map((z) => z / s);
  }

  function crashProbability(model, x) {
    return predict(model, x)[0];
  }

  // Streaming detector: feed samples one at a time, get an event when a window completes.
  class Detector {
    constructor(model, { triggerG = model.trigger_g, alpha = model.ema_alpha, roadModel = null } = {}) {
      this.model = model; this.triggerG = triggerG; this.alpha = alpha; this.roadModel = roadModel;
      this.ring = new Float32Array(WIN * 3); this.ringGyro = new Float32Array(WIN * 3);
      this.head = 0; this.i = 0;
      this.ema = null; this.collecting = false; this.anchor = 0; this.peak = 0;
    }
    push(ax, ay, az, gx = 0, gy = 0, gz = 0) {
      const r = this.ring, idx = this.head * 3;
      r[idx] = ax; r[idx + 1] = ay; r[idx + 2] = az;
      this.ringGyro[idx] = gx; this.ringGyro[idx + 1] = gy; this.ringGyro[idx + 2] = gz;
      this.head = (this.head + 1) % WIN;
      if (!this.ema) this.ema = [ax, ay, az];
      const e = this.ema;
      e[0] += this.alpha * (ax - e[0]); e[1] += this.alpha * (ay - e[1]); e[2] += this.alpha * (az - e[2]);
      const dyn = Math.hypot(ax - e[0], ay - e[1], az - e[2]);
      const i = this.i++;
      let event = null;
      if (!this.collecting) {
        if (i >= N_PRE && dyn > this.triggerG) { this.collecting = true; this.anchor = i; this.peak = dyn; }
      } else {
        if (i - this.anchor > 10 && dyn > Math.max(1, 1.5 * this.peak)) { this.anchor = i; this.peak = dyn; }
        this.peak = Math.max(this.peak, dyn);
        if (i === this.anchor + N_POST - 1) {
          const win = new Float32Array(WIN * 3), winGyro = new Float32Array(WIN * 3);
          for (let k = 0; k < WIN; k++) {
            const j = ((this.head + k) % WIN) * 3;
            win[k * 3] = r[j]; win[k * 3 + 1] = r[j + 1]; win[k * 3 + 2] = r[j + 2];
            winGyro[k * 3] = this.ringGyro[j]; winGyro[k * 3 + 1] = this.ringGyro[j + 1]; winGyro[k * 3 + 2] = this.ringGyro[j + 2];
          }
          const x = new Float64Array(37);
          x.set(features(win));
          x.set(gyroFeatures(win, winGyro), 30);
          const pCrash = crashProbability(this.model, x);
          let road = null, label = "CRASH";
          if (this.roadModel) {
            road = predict(this.roadModel, x);
            label = pCrash >= 0.5 ? "CRASH" : this.roadModel.classes[road.indexOf(Math.max(...road))];
          }
          event = { trigger: this.anchor, index: i, window: win, gyroWindow: winGyro, features: x,
            pCrash, roadProbs: road, label };
          this.collecting = false;
        }
      }
      return { dyn, event };
    }
  }

  const api = { WIN, N_PRE, N_POST, features, gyroFeatures, predict, crashProbability, Detector };
  if (typeof module !== "undefined") module.exports = api; else root.CrashAI = api;
})(typeof window !== "undefined" ? window : globalThis);

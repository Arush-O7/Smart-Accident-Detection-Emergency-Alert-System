"""Crash Verdict Console.

    streamlit run streamlit_app.py
"""
import json
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from app.engine import (Detector, classify, load, run_stream, split_components, to_dps, to_g, v1_alarms)

ROOT = Path(__file__).resolve().parent
FS = 100
ALERT_P = 0.80
VERIFY_P = 0.50
NICE = {"NORMAL": "Normal jolt", "SPEED_BUMP": "Speed breaker", "POTHOLE_ROUGH_ROAD": "Pothole / rough road",
        "HARSH_MANEUVER": "Harsh manoeuvre", "CRASH": "Crash", "NEAR_MISS": "Near miss",
        "NORMAL_DRIVING": "Normal driving"}
BLUE, ORANGE, VIOLET, RED, GREEN = "#2a78d6", "#eb6834", "#4a3aa7", "#c8372d", "#2a7f50"

st.set_page_config(page_title="Crash Verdict Console", page_icon="🚗", layout="wide")


@st.cache_resource
def get_models():
    return load()


@st.cache_data
def get_demo():
    d = np.load(ROOT / "app" / "data" / "demo.npz")
    return {k: d[k] for k in d.files}


@st.cache_data
def replay_events():
    models, _ = get_models()
    demo = get_demo()
    acc, gyro = to_g(demo["replay_acc"]), to_dps(demo["replay_gyro"])
    return run_stream(models, acc, gyro), v1_alarms(acc)


@st.cache_data
def library_verdicts():
    models, meta = get_models()
    demo = get_demo()
    out = []
    for i, ev in enumerate(meta["events"]):
        r = classify(models, to_g(demo["event_acc"][i]), to_dps(demo["event_gyro"][i]))
        out.append(r)
    return out


models, meta = get_models()
demo = get_demo()
VZ, ROAD = meta["vzcrash"], meta["road_classes"]
comparison = pd.DataFrame(meta["comparison"])
best = comparison[comparison.model == VZ["selected"]].iloc[0]
v1row = comparison[comparison.model.str.startswith("V1")].iloc[0]


def verdict_chip(p_crash):
    if p_crash >= ALERT_P:
        return "🔴 CRASH - LoRa alert sent, SMS dispatched"
    if p_crash >= VERIFY_P:
        return "🟠 Possible crash - 10 s countdown, driver can cancel"
    return "🟢 No crash - logged only"


def signal_chart(acc_window, height=260, trigger_line=True, t0=None, x_title="seconds from trigger"):
    v, h = split_components(acc_window)
    t = (np.arange(len(v)) - 100) / FS if t0 is None else t0 + np.arange(len(v)) / FS
    df = pd.DataFrame({"time (s)": np.concatenate([t, t]),
                       "g": np.concatenate([v, h]),
                       "component": ["vertical (bumps, potholes)"] * len(t) + ["horizontal (braking, impacts)"] * len(t)})
    chart = alt.Chart(df).mark_line(strokeWidth=1.6).encode(
        x=alt.X("time (s):Q", title=x_title, scale=alt.Scale(nice=False, zero=False)),
        y=alt.Y("g:Q", title="dynamic acceleration (g)"),
        color=alt.Color("component:N", scale=alt.Scale(
            domain=["horizontal (braking, impacts)", "vertical (bumps, potholes)"], range=[BLUE, ORANGE]),
            legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("time (s):Q", format=".2f"), alt.Tooltip("g:Q", format=".2f"), "component:N"])
    layers = [chart]
    if trigger_line:
        layers.append(alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(strokeDash=[4, 4], color="#6f7b84").encode(x="x:Q"))
    return alt.layer(*layers).properties(height=height)


def gyro_chart(gyro_window, height=140):
    mag = np.linalg.norm(gyro_window, axis=1)
    df = pd.DataFrame({"time (s)": (np.arange(len(mag)) - 100) / FS, "deg/s": mag})
    return alt.Chart(df).mark_line(strokeWidth=1.6, color=VIOLET).encode(
        x=alt.X("time (s):Q", title=None), y=alt.Y("deg/s:Q", title="rotation (°/s)"),
        tooltip=[alt.Tooltip("time (s):Q", format=".2f"), alt.Tooltip("deg/s:Q", format=".0f")]
    ).properties(height=height)


def road_bars(probs, height=None):
    df = pd.DataFrame({"class": [NICE[c] for c in ROAD], "probability": probs})
    return alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=3, height=20).encode(
        x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
        y=alt.Y("class:N", sort=[NICE[c] for c in ROAD], title=None,
                axis=alt.Axis(labelLimit=200, labelOverlap=False, labelFontSize=12)),
        tooltip=["class:N", alt.Tooltip("probability:Q", format=".1%")]
    ).properties(height=height or 34 * len(ROAD))


def road_panel(probs, p_crash):
    """Model B only runs when Model A clears the jolt, exactly as on the ESP32."""
    if p_crash is not None and p_crash >= VERIFY_P:
        st.caption("Not run: Model A flagged a possible crash, so the road classifier is skipped - "
                   "the same order the ESP32 uses.")
    else:
        st.altair_chart(road_bars(probs if probs is not None else np.zeros(len(ROAD))), width="stretch")


def crash_meter(p):
    df = pd.DataFrame({"p": [p]})
    bar = alt.Chart(df).mark_bar(color=RED if p >= VERIFY_P else GREEN, height=26).encode(
        x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", title="P(crash) - alert at the black line", titlePadding=8)))
    mark = alt.Chart(pd.DataFrame({"x": [ALERT_P]})).mark_rule(color="#131a1f", strokeWidth=2).encode(x="x:Q")
    return alt.layer(bar, mark).properties(height=80)


VIEWS = ["Overview", "Drive replay", "Event library", "Model comparison",
         "Test your own data", "How it runs on the ESP32"]

with st.sidebar:
    st.title("🚗 Crash Verdict Console")
    st.caption("LoRa accident detection · two AI models on an ESP32 + MPU6050")
    requested = st.query_params.get("view", "")          # ?view=Drive+replay opens that page
    start = VIEWS.index(requested) if requested in VIEWS else 0
    page = st.radio("View", VIEWS, index=start, label_visibility="collapsed")
    if page != requested:
        st.query_params["view"] = page
    st.divider()
    st.metric("Crash detection (AP)", f"{best.event_pr_auc:.3f}", f"{best.event_pr_auc - v1row.event_pr_auc:+.3f} vs old rule")
    st.caption(f"Model A: {meta['models']['crash']['n_trees']} trees · "
               f"Model B: {meta['models']['road']['n_trees']} trees · both on-device")

if page == "Overview":
    st.title("Crash, pothole or speed breaker?")
    st.markdown(
        "Every jolt above **0.45 g** wakes two models on the ESP32. **Model A** decides whether it was a crash, "
        "from 30 accelerometer and 7 gyroscope features, trained on **VZCrash** - 31,090 verified real crashes. "
        "**Model B** then names what the jolt actually was. Everything here runs on data the models never saw.")

    c = st.columns(4)
    c[0].metric("Crash detection (AP)", f"{best.event_pr_auc:.3f}", help="Average precision per event on the held-out test split")
    c[1].metric("Crashes detected", f"{best.event_recall_at_p80:.1%}")
    c[1].caption(f"old rule {v1row.event_recall_at_p80:.1%}, with 3× the false alarms")
    c[2].metric("False alarms", f"{best.event_fpr_at_p80:.2%}", f"{best.event_fpr_at_p80 - v1row.event_fpr_at_p80:+.2%} vs old rule", delta_color="inverse")
    c[3].metric("Firmware size", "659 KB")
    c[3].caption("50% of ESP32 flash, both models")

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("How a jolt is judged")
        st.markdown(f"""
1. **Trigger** - gravity is tracked with an exponential average; a jolt above 0.45 g opens a 2.5 s window
   (1.0 s before, 1.5 s after). A stronger jolt mid-window re-anchors it, so a pothole cannot mask a crash.
2. **Features** - the motion is split into **vertical** (bumps, potholes) and **horizontal** (braking, impacts)
   parts relative to gravity, so the verdict does not depend on how the box is mounted.
3. **Model A** - P(crash) ≥ {ALERT_P:.2f} sends the LoRa alert at once; {VERIFY_P:.2f}–{ALERT_P:.2f} starts a
   10 s buzzer countdown the driver can cancel.
4. **Model B** - anything below that gets a name: speed breaker, pothole / rough road, harsh manoeuvre or normal.
5. **Rollover** - an independent physics guard fires after 3 s of tilt beyond 60°.
        """)
    with right:
        st.subheader("Measured against the field")
        ref = VZ["paper_baselines"]
        bench = pd.DataFrame({
            "model": ["Old rule (|a| > 2.2 g)", "Assessment-6 V2 spec", "VZCrash paper: physics baseline",
                      "This system (37 features, on-device)", "VZCrash paper: CNN-RNN (1.7M params)"],
            "average precision": [v1row.event_pr_auc,
                                  comparison[comparison.model.str.startswith("Document V2:")].iloc[0].event_pr_auc,
                                  ref["physical_baseline"], best.event_pr_auc, ref["cnn_rnn"]]})
        st.altair_chart(alt.Chart(bench).mark_bar(cornerRadiusEnd=3, height=22).encode(
            x=alt.X("average precision:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.condition(alt.datum.model == "This system (37 features, on-device)",
                                alt.value(BLUE), alt.value("#c3ccd2")),
            tooltip=["model:N", alt.Tooltip("average precision:Q", format=".3f")]).properties(height=220),
            width="stretch")
        st.caption("Test split: 24,174 events, 4,221 real crashes, none seen during training.")

elif page == "Drive replay":
    st.title("Drive replay")
    st.caption(meta["replay"]["source"])
    events, v1 = replay_events()
    acc = to_g(demo["replay_acc"])
    total_s = len(acc) / FS
    crash_t = meta["replay"]["crash_time"]

    ctrl = st.columns([1, 1, 2, 3])
    play = ctrl[0].button("▶ Play", type="primary", width="stretch")
    speed = ctrl[1].selectbox("Speed", [5, 10, 20, 40], index=2, label_visibility="collapsed")
    t_now = ctrl[2].slider("Time (s)", 0.0, float(total_s), float(total_s), 0.5, label_visibility="collapsed")

    plot_area = st.empty()
    status_area = st.empty()
    counts_area = st.empty()

    def render(t):
        lo, hi = max(0, int((t - 10) * FS)), int(t * FS)
        seen = [e for e in events if e["t"] <= t]
        v1_seen = [i for i in v1 if i / FS <= t]
        with plot_area.container():
            cols = st.columns([3, 2])
            with cols[0]:
                if hi - lo > 20:
                    st.altair_chart(signal_chart(acc[lo:hi], height=260, trigger_line=False,
                                                 t0=lo / FS, x_title="seconds into the drive"), width="stretch")
                st.caption(f"t = {t:,.1f} s of {total_s:,.0f} s · last 10 seconds shown")
            with cols[1]:
                last = seen[-1] if seen else None
                st.markdown("#### Model A · crash?")
                st.markdown(verdict_chip(last["p_crash"]) if last else "⚪ Monitoring - waiting for a jolt")
                st.altair_chart(crash_meter(last["p_crash"] if last else 0.0), width="stretch")
                st.markdown("#### Model B · what kind of jolt")
                road_panel(last["road"] if last else None, last["p_crash"] if last else None)
        with status_area.container():
            m = st.columns(4)
            alerts = [e for e in seen if e["p_crash"] >= ALERT_P]
            false_v1 = [i for i in v1_seen if abs(i / FS - crash_t) > 2]
            m[0].metric("Events classified", len(seen))
            m[1].metric("AI crash alerts", len(alerts))
            m[2].metric("Old-rule alarms", len(v1_seen), f"{len(false_v1)} false", delta_color="inverse")
            m[3].metric("Ground-truth crash", f"{crash_t:.0f} s")
            m[3].caption("reached" if t >= crash_t else "not reached yet")
        return seen

    if play:
        for t in np.arange(0, total_s + 0.001, speed * 0.1):
            render(min(t, total_s))
            time.sleep(0.02)
        seen = render(total_s)
    else:
        seen = render(t_now)

    st.divider()
    st.subheader("What the AI saw")
    if seen:
        rows = []
        for e in seen:
            label = "CRASH" if e["p_crash"] >= VERIFY_P else ROAD[int(np.argmax(e["road"]))]
            truth = next((s["label"] for s in meta["replay"]["truth"] if s["t"] - 1.0 <= e["t"] <= s["end"] + 1.0), None)
            road_ctx = meta["replay"]["road"][min(int(e["t"]), len(meta["replay"]["road"]) - 1)]
            context = truth or (f"{road_ctx} road" if road_ctx in ("smooth", "rough") else road_ctx)
            rows.append({"time (s)": round(e["t"], 1), "verdict": NICE[label], "P(crash)": round(e["p_crash"], 3),
                         "peak g": round(e["peak_g"], 2), "ΔV (km/h)": round(e["dv_kmh"], 1),
                         "old 2.2 g rule": "ALARM" if e["v1_alarm"] else "quiet",
                         "ground truth": context})
        df = pd.DataFrame(rows).iloc[::-1]
        st.dataframe(df, width="stretch", hide_index=True, height=320,
                     column_config={"P(crash)": st.column_config.ProgressColumn("P(crash)", min_value=0, max_value=1, format="%.3f")})
    st.info("The old rule fires on rough road; the AI classifies those jolts and stays silent, then flags the one real crash.")

elif page == "Event library":
    st.title("Event library")
    st.caption("Real events: VZCrash crashes, near misses and normal driving from the held-out test split, "
               "plus road events from a vehicle the road model never saw.")
    verdicts = library_verdicts()
    labels = []
    for i, ev in enumerate(meta["events"]):
        p = verdicts[i]["p_crash"]
        mark = "🔴" if p >= ALERT_P else ("🟠" if p >= VERIFY_P else "🟢")
        labels.append(f"{mark} {NICE.get(ev['truth'], ev['truth'])} - {ev['desc'][:70]}")

    kinds = ["All"] + sorted({e["truth"] for e in meta["events"]})
    kind = st.selectbox("Filter", kinds, format_func=lambda k: NICE.get(k, k))
    idx = [i for i, e in enumerate(meta["events"]) if kind in ("All", e["truth"])]
    choice = st.selectbox("Event", idx, format_func=lambda i: labels[i])

    ev, r = meta["events"][choice], verdicts[choice]
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(signal_chart(to_g(demo["event_acc"][choice])), width="stretch")
        if ev["source"] == "VZCrash":   # only VZCrash events carry gyroscope data
            st.altair_chart(gyro_chart(to_dps(demo["event_gyro"][choice])), width="stretch")
    with right:
        st.markdown(f"**{ev['source']}** · {ev['why']}")
        st.markdown(f"##### {ev['desc']}")
        st.markdown(verdict_chip(r["p_crash"]))
        st.altair_chart(crash_meter(r["p_crash"]), width="stretch")
        m = st.columns(2)
        m[0].metric("Ground truth", NICE.get(ev["truth"], ev["truth"]))
        m[1].metric("Model A · P(crash)", f"{r['p_crash']:.3f}")
        m[0].metric("Peak jolt", f"{r['peak_g']:.2f} g")
        m[1].metric("Horizontal ΔV", f"{r['dv_kmh']:.1f} km/h")
        m[0].metric("Old 2.2 g rule", "ALARM" if r["v1_alarm"] else "quiet")
        m[1].metric("Peak rotation", f"{r['w_peak']:.0f} °/s" if ev["source"] == "VZCrash" else "n/a")
        st.markdown("###### Model B · what kind of jolt")
        road_panel(r["road"], r["p_crash"])

    crashes = [i for i, e in enumerate(meta["events"]) if e["truth"] == "CRASH"]
    found = sum(verdicts[i]["p_crash"] >= ALERT_P for i in crashes)
    others = [verdicts[i]["p_crash"] for i, e in enumerate(meta["events"]) if e["truth"] != "CRASH"]
    st.success(f"Across this library: {found} of {len(crashes)} real crashes flagged, "
               f"and the highest score on any non-crash event is {max(others):.3f}.")

elif page == "Model comparison":
    st.title("Model comparison")
    st.caption("Trained on the VZCrash train split, threshold tuned on validation, measured on the untouched test "
               "split: 24,174 events, 4,221 real crashes.")

    chart_df = comparison.assign(selected=comparison.model == VZ["selected"])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average precision per event")
        st.altair_chart(alt.Chart(chart_df).mark_bar(cornerRadiusEnd=3, height=18).encode(
            x=alt.X("event_pr_auc:Q", title="average precision", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("model:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
            color=alt.condition(alt.datum.selected, alt.value(BLUE), alt.value("#c3ccd2")),
            tooltip=["model", alt.Tooltip("event_pr_auc:Q", format=".4f")]).properties(height=330),
            width="stretch")
    with c2:
        st.subheader("False alarms per 1,000 events")
        st.altair_chart(alt.Chart(chart_df).mark_bar(cornerRadiusEnd=3, height=18).encode(
            x=alt.X("false_alarms_per_1000_events:Q", title="at P(crash) ≥ 0.80"),
            y=alt.Y("model:N", sort="x", title=None, axis=alt.Axis(labelLimit=240)),
            color=alt.condition(alt.datum.selected, alt.value(BLUE), alt.value("#c3ccd2")),
            tooltip=["model", alt.Tooltip("event_fpr_at_p80:Q", format=".2%")]).properties(height=330),
            width="stretch")

    st.subheader("All metrics")
    show = comparison[["model", "features", "event_pr_auc", "event_recall_at_p80", "event_fpr_at_p80",
                       "event_precision_at_p80", "event_f1_at_p80", "event_recall_tuned", "event_fpr_tuned",
                       "detected_crashes", "missed_crashes"]].rename(columns={
        "features": "inputs", "event_pr_auc": "AP", "event_recall_at_p80": "recall @0.8",
        "event_fpr_at_p80": "false alarms @0.8", "event_precision_at_p80": "precision @0.8",
        "event_f1_at_p80": "F1 @0.8", "event_recall_tuned": "recall @tuned", "event_fpr_tuned": "false alarms @tuned",
        "detected_crashes": "crashes found", "missed_crashes": "crashes missed"})
    st.dataframe(show, width="stretch", hide_index=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.2f%%" if "alarm" in c or "recall" in c or "precision" in c else "%.4f")
                                for c in ["AP", "recall @0.8", "false alarms @0.8", "precision @0.8", "F1 @0.8",
                                          "recall @tuned", "false alarms @tuned"]})

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("What the crash model still misses")
        sev = pd.DataFrame([{"impact": k, "crashes": v["n"], "detected": v["recall"]}
                            for k, v in VZ["crash_recall_by_peak_g"].items()])
        st.dataframe(sev, width="stretch", hide_index=True,
                     column_config={"detected": st.column_config.ProgressColumn("detected", min_value=0, max_value=1, format="%.1f%%")})
        st.caption("Below 1.5 g a real crash and a hard pothole look alike. The 0.50–0.80 countdown band exists "
                   "for exactly that case: it asks the driver instead of staying silent.")
    with c4:
        st.subheader("Model B: naming the jolt")
        cm = pd.DataFrame(meta["road"]["confusion"], index=[NICE[c] for c in ROAD], columns=[NICE[c] for c in ROAD])
        shares = cm.div(cm.sum(axis=1), axis=0)

        def shade(col):
            return [f"background-color: rgba(42, 120, 214, {0.85 * s:.2f}); color: {'white' if s > 0.55 else 'inherit'}"
                    for s in shares[col.name]]

        st.dataframe(cm.style.apply(shade, axis=0), width="stretch")
        st.caption(f"Macro F1 {meta['road']['comparison'][meta['road']['selected']]['macro_f1']:.3f}, "
                   f"accuracy {meta['road']['comparison'][meta['road']['selected']]['accuracy']:.1%}. "
                   "Rows are ground truth, columns the prediction.")

    with st.expander("How to read these results"):
        st.markdown("""
- **The crashes are real.** VZCrash is 16 s of 100 Hz IMU data per event, labelled by three trained reviewers
  with dashcam footage. Nothing in the crash class is simulated.
- **The negatives are hard on purpose** - near misses, potholes, harsh braking, off-road manoeuvres. A 2.4%
  false-alarm rate on that set is not 2.4% of driving time: on ordinary recorded driving the model raises none.
- **The trigger is part of the detector.** Events that never reach 0.45 g are never classified and count as misses.
- **Model B is the weaker half:** PVS labels road quality per stretch, not per event, and holds only 95 bump
  crossings. It separates a bump from a crash reliably, a bump from rough road about half the time.
        """)

elif page == "Test your own data":
    st.title("Test your own data")
    st.markdown("Upload a recording and the same trigger and models will run over it, exactly as they would "
                "on the ESP32. Use the CSV the transmitter prints, or any 100 Hz log.")
    st.caption("Columns: `ax,ay,az` in g (gravity included), optionally `gx,gy,gz` in °/s. "
               "Headers are matched by name; a leading time column is ignored.")

    up = st.file_uploader("CSV file", type=["csv", "txt"])
    demo_cols = st.columns([1, 3])
    use_sample = demo_cols[0].button("Use the replay drive instead")

    data = None
    if up is not None:
        raw = pd.read_csv(up)
        cols = {c.strip().lower(): c for c in raw.columns}
        acc_cols = [cols.get(k) for k in ("ax", "ay", "az")]
        if any(c is None for c in acc_cols):
            numeric = raw.select_dtypes("number")
            acc_cols = list(numeric.columns[:3]) if numeric.shape[1] >= 3 else []
            st.warning(f"No ax/ay/az headers found - using the first three numeric columns: {acc_cols}")
        if acc_cols:
            acc = raw[acc_cols].to_numpy(np.float32)
            if np.abs(acc).max() > 20:          # looks like m/s^2
                acc /= 9.80665
                st.info("Values look like m/s² - divided by 9.81 to get g.")
            gyro_cols = [cols.get(k) for k in ("gx", "gy", "gz")]
            gyro = raw[gyro_cols].to_numpy(np.float32) if all(c is not None for c in gyro_cols) else np.zeros_like(acc)
            data = (acc, gyro, f"{up.name} - {len(acc)} samples ({len(acc) / FS:.1f} s at 100 Hz)")
    elif use_sample:
        data = (to_g(demo["replay_acc"]), to_dps(demo["replay_gyro"]), meta["replay"]["source"])

    if data:
        acc, gyro, label = data
        if len(acc) < 300:
            st.error("Need at least 3 seconds of data (300 samples at 100 Hz).")
        else:
            with st.spinner("Running the detector…"):
                events = run_stream(models, acc, gyro)
                alarms = v1_alarms(acc)
            st.caption(label)
            m = st.columns(4)
            m[0].metric("Jolts classified", len(events))
            m[1].metric("Crash alerts", sum(e["p_crash"] >= ALERT_P for e in events))
            m[2].metric("Countdown (0.5–0.8)", sum(VERIFY_P <= e["p_crash"] < ALERT_P for e in events))
            m[3].metric("Old-rule alarms", len(alarms), delta_color="inverse")
            if events:
                rows = [{"time (s)": round(e["t"], 1),
                         "verdict": NICE["CRASH"] if e["p_crash"] >= VERIFY_P else NICE[ROAD[int(np.argmax(e["road"]))]],
                         "P(crash)": round(e["p_crash"], 3), "peak g": round(e["peak_g"], 2),
                         "ΔV (km/h)": round(e["dv_kmh"], 1), "rotation (°/s)": round(e["w_peak"], 0),
                         "old 2.2 g rule": "ALARM" if e["v1_alarm"] else "quiet"} for e in events]
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                             column_config={"P(crash)": st.column_config.ProgressColumn("P(crash)", min_value=0, max_value=1, format="%.3f")})
                pick = st.selectbox("Inspect an event", range(len(events)),
                                    format_func=lambda i: f"{events[i]['t']:.1f} s · P(crash) {events[i]['p_crash']:.3f}")
                lo = max(0, events[pick]["trigger"] - 100)
                st.altair_chart(signal_chart(acc[lo:lo + 250]), width="stretch")
            else:
                st.info("No jolt in this recording crossed the 0.45 g trigger, so the device would have stayed idle.")
    else:
        st.info("Waiting for a file. On the transmitter, set `#define STREAM_RAW true`, capture the `R,ax,ay,az` "
                "lines from the serial monitor, and save them as CSV with the header `ax,ay,az`.")

else:
    st.title("How it runs on the ESP32")
    c = st.columns(3)
    c[0].metric("Flash used", "659 KB")
    c[0].caption("50% of 1.31 MB")
    c[1].metric("RAM used", "45 KB")
    c[1].caption("13% of 320 KB")
    c[2].metric("Models on-device", f"{meta['models']['crash']['n_nodes'] + meta['models']['road']['n_nodes']:,} nodes")
    c[2].caption("both models")

    st.subheader("The same numbers, three times over")
    st.markdown("The Python pipeline, the C firmware and this app all run the same features and the same trees. "
                "`ml/scripts/verify_firmware.py` compiles the firmware headers on the host and compares them "
                "against Python on 1,987 real windows:")
    st.code(meta["parity"], language="text")

    st.subheader("Flash and test the hardware")
    st.markdown("""
1. Open `transmitter_LORA/ml_crash_detector/ml_crash_detector.ino` in the Arduino IDE (keep the `.h` files beside it).
2. Install the **MPU6050** (Electronic Cats) and **LoRa** (Sandeep Mistry) libraries. Board: **ESP32 Dev Module**.
3. Optional: buzzer on GPIO 4, cancel button from GPIO 13 to GND.
4. Open the Serial Monitor at **115200**. Every classified jolt prints one line:
""")
    st.code("E,POTHOLE_ROUGH_ROAD,0.002,0.001,0.213,0.779,0.007,1.24,2380\n"
            "E,CRASH,0.910,3.42,2405        <- class, P(crash), peak g, microseconds on-device", language="text")
    st.markdown("""
5. To make it fire safely: a **sharp horizontal jerk that stops dead** looks like a crash; a **gentle vertical
   bounce** onto a cushion looks like a bump. Capture those serial lines and drop them into **Test your own data**.
6. The receiver (`Reciever_LORA/reciever_sms`) puts the AI verdict into the SMS: *"AI verified CRASH
   (91% confidence), impact 3.4g"*. Non-crash jolts arrive as `EVENT` packets and are logged without an SMS.
    """)
    st.caption("Live USB streaming needs direct serial access, which a hosted app cannot have - use the HTML "
               "dashboard in `demo/index.html` (Chrome or Edge) for the live ESP32 view.")

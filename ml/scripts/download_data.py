"""Download every real-world dataset used to train and test the crash classifier.

Sources (all publicly downloadable, no login required):
  1. PVS - Passive Vehicular Sensors (UFSC, Brazil). 9 drives, dashboard + body MPU-9250 at 100 Hz,
     per-sample labels for speed bumps and road quality.            -> negatives (bumps, rough roads)
  2. NHTSA Vehicle Crash Test Database (US DOT). Body-mounted accelerometer time histories
     recorded during real crash tests (frontal, side, pole, rear, vehicle-to-vehicle).   -> crash pulses
  3. Driver Behavior Dataset (Ferreira et al., PLOS ONE 2017). Smartphone IMU during aggressive
     braking, turns and lane changes.                                -> harsh-manoeuvre negatives
  4. Real-World Pothole Drives (W. Stading, Austin TX). iPhone IMU (logged at 50 Hz) with human-labelled
     potholes. Used ONLY as an unseen external test set.            -> external false-positive test

Usage:  ml/.venv/bin/python ml/scripts/download_data.py [--nhtsa-tests 700]
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

PVS_SHARE = "https://arquivos.ufsc.br/d/55ee83c2c9d647eaa252/files/?p=/PVS%20{n}/{f}&dl=1"
PVS_COLS = ["timestamp", "acc_x_dashboard", "acc_y_dashboard", "acc_z_dashboard",
            "gyro_x_dashboard", "gyro_y_dashboard", "gyro_z_dashboard",
            "acc_x_above_suspension", "acc_y_above_suspension", "acc_z_above_suspension", "speed"]

NHTSA = "https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results"
DRIVER_BEHAVIOR = "https://raw.githubusercontent.com/jair-jr/driverBehaviorDataset/master/data/{trip}/{f}"
POTHOLE = "https://raw.githubusercontent.com/warstad11/pothole/main/dataset/{f}"

session = requests.Session()
session.headers["User-Agent"] = "accident-detection-research/1.0"


def get(url, **kw):
    for attempt in range(4):
        try:
            r = session.get(url, timeout=180, **kw)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))


def download(url, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get(url, stream=True)
    if r is None:
        return None
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    tmp.rename(dest)
    return dest


# ---------------------------------------------------------------- PVS
def fetch_pvs_drive(n):
    out = RAW / "pvs" / f"pvs{n}.parquet"
    if out.exists() and "acc_z_above_suspension" in pd.read_parquet(out).columns:
        return out
    folder = RAW / "pvs" / f"PVS {n}"
    csv = download(PVS_SHARE.format(n=n, f="dataset_gps_mpu_left.csv"), folder / "dataset_gps_mpu_left.csv")
    lab = download(PVS_SHARE.format(n=n, f="dataset_labels.csv"), folder / "dataset_labels.csv")
    df = pd.read_csv(csv, usecols=PVS_COLS)
    labels = pd.read_csv(lab)
    assert len(df) == len(labels), f"PVS {n}: sensor/label row mismatch"
    df = pd.concat([df, labels], axis=1)
    df.astype({c: "float32" for c in PVS_COLS if c != "timestamp"}).to_parquet(out)
    csv.unlink()  # the parquet keeps only the columns we use (saves ~700 MB)
    return out


# ---------------------------------------------------------------- NHTSA
CRASH_CONFIGS = {"VEHICLE INTO BARRIER", "IMPACTOR INTO VEHICLE", "VEHICLE INTO POLE", "VEHICLE INTO VEHICLE"}
SKIP_TYPES = ("STATIC", "SLED", "PEDESTRIAN", "LOW RISK", "EJECTION", "COLLISION WARNING")
# Rigid body locations away from the crush zone, in order of preference. These see roughly
# what a chassis-mounted MPU6050 would see.
BODY_LOCATIONS = ["VEHICLE CG", "SEAT - LEFT REAR", "SEAT - RIGHT REAR", "FLOORPAN - LEFT REAR",
                  "FLOORPAN - RIGHT REAR", "SILL - LEFT REAR", "SILL - RIGHT REAR", "TRUNK FLOOR - CENTER"]
BAD_CHANNEL = ("FAIL", "WIRE", "CUT", "QUESTION", "NOT RETURN", "NO VALID", "BROKEN", "ANGULAR",
               "QD ", "CF ", "INVALID", "SUSPECT", "RATE")


def nhtsa_catalog():
    cache = RAW / "nhtsa" / "catalog.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    tests, page = [], 0
    while True:
        r = get(f"{NHTSA}/by-search", params=dict(modelYearFrom=1995, count=1000, pageNumber=page)).json()
        tests += r["results"]
        if not r["results"] or len(tests) >= r["meta"]["pagination"]["total"]:
            break
        page += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(tests))
    return tests


def select_tests(tests, n_target, seed=7):
    unique = {t["testNo"]: t for t in tests}  # the paginated catalog repeats some tests
    pool = [t for _, t in sorted(unique.items())
            if t["testConfiguration"] in CRASH_CONFIGS and t["hasInstrumentationInfo"]
            and t["closingSpeed"] and not any(s in (t["testType"] or "") for s in SKIP_TYPES)]
    rng = random.Random(seed)
    # Stratify by configuration and speed band so low-speed crashes are not drowned out
    # by the (very common) 56 km/h NCAP frontal tests.
    strata = {}
    for t in pool:
        band = min(int(t["closingSpeed"] // 15), 5)
        strata.setdefault((t["testConfiguration"], band), []).append(t)
    per = max(1, n_target // len(strata))
    chosen = []
    for key in sorted(strata):
        items = strata[key]
        rng.shuffle(items)
        chosen += items[:per]
    rest = [t for t in pool if t not in chosen]
    rng.shuffle(rest)
    chosen += rest[: max(0, n_target - len(chosen))]
    return chosen


def ascii_url(test_no, curve_no):
    return (f"https://nrd-static.nhtsa.dot.gov/tsv/vehdb/v{test_no // 10000 * 10000:05d}/"
            f"v{test_no // 100 * 100:05d}/v{test_no:05d}/v{test_no:05d}tsv.{curve_no:03d}")


def fetch_nhtsa_test(t):
    tn = t["testNo"]
    folder = RAW / "nhtsa" / f"{tn:05d}"
    meta_path = folder / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    vehicles = get(f"{NHTSA}/get-vehicle-info/{tn}").json()["results"]
    real_vehicles = {v["vehicleNo"] for v in vehicles
                     if (v.get("vehicleMake") or "") not in ("NHTSA", "OTHER")
                     and not any(k in (v.get("vehicleModel") or "") for k in ("IMPACTOR", "BARRIER", "MDB", "SLED"))}
    chans = get(f"{NHTSA}/get-instrumentation-info/{tn}", params=dict(count=2000, pageNumber=0)).json()["results"]
    pulses = []
    for veh in sorted(real_vehicles):
        good = [c for c in chans
                if c["vehicleNo"] == veh and c["sensorType"] == "ACCELEROMETER"
                and c["sensorLocation"] == "NOT APPLICABLE" and "G" in (c["dataMeasurementUnits"] or "")
                and c.get("channelStatus") != "REDUNDANT"
                and not any(b in (c["instrumentationCommentary"] or "").upper() for b in BAD_CHANNEL)]
        for loc in BODY_LOCATIONS:
            axes = {}
            for c in good:
                if (c["sensorAttachment"] or "").strip() == loc:
                    axes.setdefault(c["axisDirofSensor"][0], c)
            if "X" in axes or "Y" in axes:
                pulses.append(dict(vehicleNo=veh, location=loc,
                                   curves={a: c["curveNo"] for a, c in axes.items()}))
                break
    folder.mkdir(parents=True, exist_ok=True)
    kept = []
    for p in pulses:
        ok = True
        for axis, curve in p["curves"].items():
            dest = folder / f"v{p['vehicleNo']}_{axis}.tsv"
            if download(ascii_url(tn, curve), dest) is None:
                # newer tests live under a different static path; ask the API for it
                detail = get(f"{NHTSA}/get-instrumentation-detail-info/{curve}/{tn}")
                url = detail and (detail.json()["results"] or [{}])[0].get("asciiFile")
                if not url or download(url, dest) is None:
                    ok = False
        if ok:
            kept.append(p)
    meta = dict(testNo=tn, testType=t["testType"], configuration=t["testConfiguration"],
                closingSpeed=t["closingSpeed"], impactAngle=t["impactAngle"],
                vehicle=f"{t.get('modelYear')} {t.get('vehicleMake')} {t.get('vehicleModel')}", pulses=kept)
    meta_path.write_text(json.dumps(meta))
    return meta


# ---------------------------------------------------------------- smartphone datasets
def fetch_driver_behavior():
    for trip in (16, 17, 20, 21):
        for f in ("acelerometro_terra.csv", "giroscopio_terra.csv", "groundTruth.csv"):
            download(DRIVER_BEHAVIOR.format(trip=trip, f=f), RAW / "driver_behavior" / str(trip) / f)


def fetch_potholes():
    download(POTHOLE.format(f="labels.csv"), RAW / "potholes" / "labels.csv")
    for d in range(1, 7):
        download(POTHOLE.format(f=f"drives/drive_{d:02d}/sensor.csv"), RAW / "potholes" / f"drive_{d:02d}" / "sensor.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nhtsa-tests", type=int, default=700)
    args = ap.parse_args()

    print("[1/4] Driver Behavior dataset (harsh manoeuvres)")
    fetch_driver_behavior()
    print("[2/4] Real-world pothole drives (external test set)")
    fetch_potholes()

    print("[3/4] PVS drives 1-9 (speed bumps, rough roads)")
    with ThreadPoolExecutor(3) as ex:
        for f in as_completed([ex.submit(fetch_pvs_drive, n) for n in range(1, 10)]):
            print("   ", f.result().name)

    print("[4/4] NHTSA crash tests")
    tests = select_tests(nhtsa_catalog(), args.nhtsa_tests)
    n_pulses, failures = 0, 0
    with ThreadPoolExecutor(8) as ex:
        futs = [ex.submit(fetch_nhtsa_test, t) for t in tests]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                n_pulses += len(f.result()["pulses"])
            except Exception as e:  # a few tests have broken instrumentation records
                failures += 1
                print("    skipped:", e, file=sys.stderr)
            if i % 50 == 0:
                print(f"    {i}/{len(tests)} tests, {n_pulses} usable pulses")
    print(f"    done: {len(tests) - failures} tests, {n_pulses} usable crash pulses")


if __name__ == "__main__":
    main()

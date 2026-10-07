"""Backfills the past week of predicted-vs-actual PM2.5 rows for the OpenAQ source.

Live logging only started recently, so there is no real-time prediction history
for last week. This script reconstructs it as a HINDCAST: for each of the last
N days it pretends to be at a fixed anchor time, builds the 48-hour model input
using ONLY data from before that time (OpenAQ PM2.5 from the nearest live
station + Open-Meteo meteorology), runs all four trained models, and compares
the 24-hour forecast against what OpenAQ actually measured over the next 24
hours. Nothing after the anchor time is used as input, so there is no leakage.

Rows are appended to prediction_log.csv with data_source
"OpenAQ (ground sensor) [backfilled hindcast]" and the actual already filled
in, so they show up in the dashboard's Predicted vs. Actual table. They are
labeled as hindcasts on purpose: report them as a retrospective test, separate
from the real-time rows the live endpoint logs from now on.

Run from the folder that contains main.py, live_data_openaq.py, fill_actuals.py,
config.yaml, scaler.pkl and checkpoints/, with OPENAQ_API_KEY set (.env works):

    python3 backfill_week.py --dry-run     # preview, writes nothing
    python3 backfill_week.py               # last 7 days, all 10 cities
    python3 backfill_week.py --days 5 --anchor-hour-utc 0

Anchor default is 00:00 UTC (08:00 Philippine time). Safe to re-run: an
existing (city, anchor time) row is never duplicated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv

load_dotenv()

from openaq import OpenAQ  # noqa: E402

import main as app_main  # noqa: E402  (loads the trained models + scaler; does not start the server)
from fill_actuals import _retry, fetch_sensor_series  # noqa: E402
from live_data_openaq import (  # noqa: E402
    CITY_COORDS,
    MET_URL,
    MET_VARS,
    _ffill,
    _find_pm25_candidates,
    _require_api_key,
    scale_window,
)

SOURCE_LABEL = "OpenAQ (ground sensor) [backfilled hindcast]"
INPUT_HOURS = 48
HORIZON_HOURS = 24


def fetch_met_df(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(MET_VARS),
        "start_hour": start.strftime("%Y-%m-%dT%H:%M"),
        "end_hour": end.strftime("%Y-%m-%dT%H:%M"),
        "timezone": "UTC",
    }
    r = requests.get(MET_URL, params=params, timeout=30)
    r.raise_for_status()
    h = r.json()["hourly"]
    idx = pd.DatetimeIndex(pd.to_datetime(h["time"], utc=True))
    return pd.DataFrame({v: h[v] for v in MET_VARS}, index=idx).astype("float64")


def as_utc(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    idx = series.index
    series = series.copy()
    series.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return series


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7, help="how many past daily anchors to backfill (default 7)")
    parser.add_argument("--anchor-hour-utc", type=int, default=0, help="UTC hour of each daily anchor (default 0 = 08:00 PHT)")
    parser.add_argument("--min-input-hours", type=int, default=24, help="min real readings in the 48h input window (default 24)")
    parser.add_argument("--min-actual-hours", type=int, default=18, help="min real readings in the 24h actual window (default 18)")
    parser.add_argument("--finetuned", action="store_true",
                        help="use the OpenAQ-fine-tuned checkpoints (main.MODELS_OPENAQ) and label rows accordingly")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    label = f"{app_main.OPENAQ_SOURCE_LABEL} [backfilled hindcast]" if args.finetuned else SOURCE_LABEL
    models = app_main.MODELS_OPENAQ if args.finetuned else app_main.MODELS
    if args.finetuned and any("fine-tuned" not in v for v in app_main.OPENAQ_MODEL_VERSION.values()):
        raise SystemExit("Not all fine-tuned checkpoints were found in checkpoints/ -- copy the four *_openaq_ft.pt files first.")
    now = datetime.now(timezone.utc)
    last_anchor = now.replace(hour=args.anchor_hour_utc, minute=0, second=0, microsecond=0)
    while last_anchor + timedelta(hours=HORIZON_HOURS + 1) > now:  # actual window must be complete
        last_anchor -= timedelta(days=1)
    anchors = [last_anchor - timedelta(days=i) for i in range(args.days - 1, -1, -1)]
    span_start = anchors[0] - timedelta(hours=INPUT_HOURS)
    span_end = anchors[-1] + timedelta(hours=HORIZON_HOURS)
    span_hours = (span_end - span_start) / timedelta(hours=1)
    print(f"Anchors (UTC): {anchors[0]:%Y-%m-%d %H:%M} ... {anchors[-1]:%Y-%m-%d %H:%M}  ({len(anchors)} per city)")

    existing = set()
    if os.path.isfile(app_main.PREDICTION_LOG_PATH):
        with open(app_main.PREDICTION_LOG_PATH, newline="") as f:
            for r in csv.DictReader(f):
                existing.add((r["city"], r["logged_at_utc"], r["data_source"]))

    new_rows: list[dict] = []
    api_key = _require_api_key()
    with OpenAQ(api_key=api_key) as client:
        for city, (lat, lon) in CITY_COORDS.items():
            print(f"\n=== {city} ===")
            try:
                candidates = _find_pm25_candidates(city)
            except RuntimeError as e:
                print(f"  no stations: {e}")
                continue

            chosen = None
            for _, sensor_id, name, dist in candidates:
                series = as_utc(fetch_sensor_series(client, sensor_id, span_start, span_end))
                time.sleep(0.3)
                if len(series) >= 0.6 * span_hours:
                    chosen = (name, dist, series)
                    break
            if chosen is None:
                print("  no nearby station reported >= 60% of the needed hours -- skipped")
                continue
            name, dist, series = chosen
            dist_str = f"{dist:.0f}m" if dist is not None else "distance unknown"
            print(f"  station: {name} ({dist_str}), {len(series)} hourly readings in span")

            try:
                met_df = fetch_met_df(lat, lon, span_start, anchors[-1])
            except requests.RequestException as e:
                print(f"  meteorology fetch failed: {e}")
                continue

            for anchor in anchors:
                key = (city, anchor.isoformat(), label)
                if key in existing:
                    print(f"  {anchor:%m-%d %H:%M}: already logged -- skipped")
                    continue

                in_idx = pd.date_range(anchor - timedelta(hours=INPUT_HOURS), periods=INPUT_HOURS, freq="1h", tz="UTC")
                pm_in = series.reindex(in_idx)
                n_real = int(pm_in.notna().sum())
                if n_real < args.min_input_hours:
                    print(f"  {anchor:%m-%d %H:%M}: only {n_real}h of input data -- skipped")
                    continue
                met_in = met_df.reindex(in_idx)
                if met_in.isna().all().any():
                    print(f"  {anchor:%m-%d %H:%M}: meteorology missing -- skipped")
                    continue

                act_win = series[(series.index >= pd.Timestamp(anchor)) &
                                 (series.index < pd.Timestamp(anchor + timedelta(hours=HORIZON_HOURS)))]
                if len(act_win) < args.min_actual_hours:
                    print(f"  {anchor:%m-%d %H:%M}: only {len(act_win)}h of actual data -- skipped")
                    continue

                x_pm25 = _ffill(pm_in.to_numpy(dtype=np.float32).reshape(-1, 1))
                x_met = _ffill(met_in.to_numpy(dtype=np.float32))
                xp_s, xm_s = scale_window(x_pm25, x_met, app_main.SCALER)
                xp_t = torch.as_tensor(xp_s, dtype=torch.float32).unsqueeze(0)
                xm_t = torch.as_tensor(xm_s, dtype=torch.float32).unsqueeze(0)

                results = {}
                with torch.no_grad():
                    for vname, model in models.items():
                        y = model(xp_t, xm_t)
                        results[vname] = [round(float(v), 2) for v in app_main.inverse_pm25(y.numpy()[0])]

                actual = round(float(act_win.mean()), 2)
                source = (f"OpenAQ '{name}' ({dist_str}), mean of {len(act_win)} hourly readings "
                          f"{anchor:%Y-%m-%d %H:%M}-{anchor + timedelta(hours=HORIZON_HOURS):%Y-%m-%d %H:%M} UTC")
                new_rows.append({
                    "logged_at_utc": anchor.isoformat(),
                    "city": city,
                    "data_source": label,
                    "input_hours_used": n_real,
                    "variant_a_predicted_24h": json.dumps(results["A"]),
                    "variant_b_predicted_24h": json.dumps(results["B"]),
                    "variant_c_predicted_24h": json.dumps(results["C"]),
                    "variant_d_predicted_24h": json.dumps(results["D"]),
                    "actual_pm25": actual,
                    "actual_source": source,
                    "actual_logged_at": datetime.now(timezone.utc).isoformat(),
                })
                c_mean = sum(results["C"]) / len(results["C"])
                print(f"  {anchor:%m-%d %H:%M}: predicted(C) {c_mean:.1f}  actual {actual:.1f}  ({n_real}h input)")

    if not new_rows:
        print("\nNothing to add.")
        return

    print("\nMean absolute error of the 24h-mean forecast vs actual, across new rows:")
    for v, col in zip("ABCD", ["variant_a_predicted_24h", "variant_b_predicted_24h",
                               "variant_c_predicted_24h", "variant_d_predicted_24h"]):
        errs = [abs(r["actual_pm25"] - np.mean(json.loads(r[col]))) for r in new_rows]
        print(f"  Variant {v}: {np.mean(errs):.2f} µg/m³  (n={len(errs)})")

    if args.dry_run:
        print(f"\nDry run: would add {len(new_rows)} rows. Nothing written.")
        return
    path = app_main.PREDICTION_LOG_PATH
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=app_main.LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(new_rows)
    print(f"\nAdded {len(new_rows)} rows to {path}.")


if __name__ == "__main__":
    main()
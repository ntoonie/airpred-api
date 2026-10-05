"""Downloads historical hourly OpenAQ PM2.5 + ERA5 meteorology for fine-tuning.

For each of the 10 NCR cities this:
  1. Takes the nearest registered PM2.5 stations (same search as the live code),
     skipping dead ones (no readings in the last 30 days).
  2. Downloads each remaining station's full hourly history for the date range,
     in 30-day chunks (OpenAQ returns at most 1000 rows per call), caching every
     sensor to data/openaq/raw/sensor_<id>.csv so re-runs resume instead of
     re-downloading.
  3. Picks, per city, the station with the MOST hourly readings in the range
     (ties -> nearest). One fixed station per city, so the series is consistent.
  4. Pulls ERA5 meteorology for the city from Open-Meteo's archive API (the same
     source and 7 variables the models were trained on), UTC timestamps.
  5. Writes data/openaq/<City>.csv with columns
         datetime (UTC, naive), pm25, temperature_2m, relative_humidity_2m,
         precipitation, wind_speed_10m, wind_direction_10m, surface_pressure,
         boundary_layer_height
     plus data/openaq/station_summary.csv (which station, how many hours,
     first/last reading, % of the range covered) -- READ THIS before fine-tuning:
     if a city has only a few hundred hours, it is not enough.

Run from the folder that has live_data_openaq.py and fill_actuals.py, with
OPENAQ_API_KEY in .env:

    python3 fetch_openaq_history.py --check          # station history only, no ERA5, no CSVs
    python3 fetch_openaq_history.py                  # everything, 2025-01-01 -> ~6 days ago
    python3 fetch_openaq_history.py --start 2024-07-01 --cities Manila Pasig

Default start is 2025-01-01 on purpose: the models' MERRA-2 training data ended
2024-12-31, so OpenAQ data from 2025 on never overlaps anything they saw.

Rate limits: ~16 API calls per sensor-year-and-a-half. With --max-stations 3
that is a few hundred calls total; if you hit the hourly cap the script waits
and retries, and anything already downloaded is cached, so just re-run.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import reqduests
from dotenv import load_dotenv

load_dotenv()

from openaq import OpenAQ  # noqa: E402

from fill_actuals import _retry, fetch_sensor_series  # noqa: E402
from live_data_openaq import (  # noqa: E402
    CITY_COORDS,
    MET_VARS,
    _find_pm25_candidates,
    _require_api_key,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
CHUNK_DAYS = 30
ERA5_LAG_DAYS = 6   # ERA5 archive lags real time by about 5 days
FEATURE_COLS = ["pm25"] + list(MET_VARS)


def chunks(start: datetime, end: datetime, days: int = CHUNK_DAYS):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=days), end)
        yield cur, nxt
        cur = nxt


def to_utc(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    idx = series.index
    series = series.copy()
    series.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return series


def load_sensor(client, sensor_id: int, start: datetime, end: datetime, raw_dir: str, label: str) -> pd.Series:
    """Full hourly series for one sensor over [start, end), cached on disk."""
    path = os.path.join(raw_dir, f"sensor_{sensor_id}.csv")
    cached = pd.Series(dtype="float64")
    if os.path.isfile(path):
        df = pd.read_csv(path, parse_dates=["datetime"])
        cached = pd.Series(df["pm25"].values, index=pd.DatetimeIndex(df["datetime"]))
        cached = to_utc(cached)
    have_start = cached.index.min() if len(cached) else None
    have_end = cached.index.max() if len(cached) else None
    if have_start is not None and have_start <= pd.Timestamp(start) + pd.Timedelta(days=2) \
            and have_end >= pd.Timestamp(end) - pd.Timedelta(days=2):
        return cached[(cached.index >= pd.Timestamp(start)) & (cached.index < pd.Timestamp(end))]

    parts = [cached] if len(cached) else []
    todo = list(chunks(start, end))
    for i, (a, b) in enumerate(todo, 1):
        print(f"    {label}: chunk {i}/{len(todo)} ({a:%Y-%m-%d} -> {b:%Y-%m-%d})", end="\r")
        parts.append(to_utc(fetch_sensor_series(client, sensor_id, a, b)))
        time.sleep(0.4)
    print(" " * 90, end="\r")
    s = pd.concat([p for p in parts if len(p)]) if any(len(p) for p in parts) else pd.Series(dtype="float64")
    if len(s):
        s = s[~s.index.duplicated(keep="last")].sort_index()
        out = pd.DataFrame({"datetime": s.index.tz_convert("UTC").tz_localize(None), "pm25": s.values})
        out.to_csv(path, index=False)
    return s[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))] if len(s) else s


def fetch_era5(lat: float, lon: float, start: datetime, end: datetime) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(MET_VARS),
        "timezone": "UTC",
    }
    for attempt in range(5):
        r = requests.get(ARCHIVE_URL, params=params, timeout=120)
        if r.status_code == 429 and attempt < 4:
            time.sleep(20 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    h = r.json()["hourly"]
    idx = pd.DatetimeIndex(pd.to_datetime(h["time"], utc=True))
    return pd.DataFrame({v: h[v] for v in MET_VARS}, index=idx).astype("float64")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2025-01-01", help="UTC start date (default 2025-01-01)")
    ap.add_argument("--end", default=None, help="UTC end date (default: today minus ~6 days, ERA5's lag)")
    ap.add_argument("--cities", nargs="+", default=list(CITY_COORDS))
    ap.add_argument("--max-stations", type=int, default=3, help="live candidate stations to compare per city (default 3)")
    ap.add_argument("--max-km", type=float, default=8.0, help="ignore stations farther than this (default 8 km)")
    ap.add_argument("--out-dir", default="data/openaq")
    ap.add_argument("--check", action="store_true", help="report station history only; write no city CSVs, skip ERA5")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end
           else (datetime.now(timezone.utc) - timedelta(days=ERA5_LAG_DAYS)).replace(hour=0, minute=0, second=0, microsecond=0))
    span_hours = int((end - start) / timedelta(hours=1))
    print(f"Range (UTC): {start:%Y-%m-%d} -> {end:%Y-%m-%d}  ({span_hours} hours)")

    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    summary = []

    with OpenAQ(api_key=_require_api_key()) as client:
        for city in args.cities:
            print(f"\n=== {city} ===")
            try:
                candidates = _find_pm25_candidates(city)
            except RuntimeError as e:
                print(f"  no stations: {e}")
                continue

            live = []
            probe_from, probe_to = end - timedelta(days=30), end
            for _, sensor_id, name, dist in candidates:
                if dist is not None and dist > args.max_km * 1000:
                    continue
                probe = fetch_sensor_series(client, sensor_id, probe_from, probe_to)
                time.sleep(0.3)
                if len(probe) >= 24:
                    live.append((sensor_id, name, dist))
                if len(live) >= args.max_stations:
                    break
            if not live:
                print(f"  no live station within {args.max_km:.0f} km -- skipped")
                continue

            best = None
            for sensor_id, name, dist in live:
                s = load_sensor(client, sensor_id, start, end, raw_dir, f"{name[:28]}")
                n = len(s)
                first = s.index.min() if n else None
                last = s.index.max() if n else None
                dist_str = f"{dist:.0f}m" if dist is not None else "?"
                print(f"  {name} ({dist_str}): {n} hourly readings "
                      f"({100 * n / span_hours:.0f}% of range), first {first:%Y-%m-%d}" if n else
                      f"  {name} ({dist_str}): 0 readings in range")
                if n and (best is None or n > best[0] or (n == best[0] and (dist or 1e9) < (best[3] or 1e9))):
                    best = (n, name, sensor_id, dist, s, first, last)

            if best is None:
                print("  no readings in range for any candidate -- skipped")
                continue
            n, name, sensor_id, dist, s, first, last = best
            print(f"  -> using {name}: {n} hours, {first:%Y-%m-%d} to {last:%Y-%m-%d}")
            summary.append({
                "city": city, "station": name, "sensor_id": sensor_id,
                "distance_m": None if dist is None else round(dist),
                "hours": n, "pct_of_range": round(100 * n / span_hours, 1),
                "first": first.isoformat(), "last": last.isoformat(),
            })
            if args.check:
                continue

            lat, lon = CITY_COORDS[city]
            met = fetch_era5(lat, lon, start, end)
            idx = pd.date_range(start, end - timedelta(hours=1), freq="1h", tz="UTC")
            df = pd.DataFrame(index=idx)
            df["pm25"] = s.reindex(idx)
            for v in MET_VARS:
                df[v] = met[v].reindex(idx)
            df.index = df.index.tz_localize(None)   # naive UTC, matches the training pipeline's frames
            df.index.name = "datetime"
            df = df[FEATURE_COLS]
            nan_pct = (100 * df.isna().mean()).round(1).to_dict()
            print(f"  NaN % per column: {nan_pct}")
            df.reset_index().to_csv(os.path.join(args.out_dir, f"{city}.csv"), index=False)

    if summary:
        out = pd.DataFrame(summary)
        out.to_csv(os.path.join(args.out_dir, "station_summary.csv"), index=False)
        print("\nSummary:")
        print(out[["city", "station", "distance_m", "hours", "pct_of_range", "first", "last"]].to_string(index=False))
        print(f"\nWrote {os.path.join(args.out_dir, 'station_summary.csv')}")


if __name__ == "__main__":
    main()

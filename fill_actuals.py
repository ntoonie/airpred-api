"""Auto-fills the 'actual_pm25' column in prediction_log.csv from OpenAQ.

For each logged prediction whose 24-hour forecast window has fully elapsed
and whose actual value is still blank, this fetches the real hourly PM2.5
readings from the nearest live OpenAQ station for that city over exactly the
24 hours the model predicted, and writes the reading for the forecast's 24th hour (the last hour of that window)
into the row, with the station name, distance and exact hour recorded in
actual_source. This matches the dashboard, which shows the 24th-hour forecast.

The window is the 24 whole hours starting at the next full hour after the
prediction was logged -- the same hours the model forecast. A row is only
filled if the station has at least --min-hours readings in that window;
otherwise it is skipped and left blank rather than filled with a thin mean.

Run from the same folder as main.py, live_data_openaq.py and
prediction_log.csv, with OPENAQ_API_KEY set (the .env file works):

    python3 fill_actuals.py              # fill everything that's ready
    python3 fill_actuals.py --dry-run    # show what it would fill, write nothing
    python3 fill_actuals.py --min-hours 20

Safe to re-run: rows that already have an actual are never touched. Avoid
running it at the exact moment the API server is appending a new row.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from openaq import OpenAQ  # noqa: E402

from live_data_openaq import _find_pm25_candidates, _require_api_key  # noqa: E402

LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.csv")
WINDOW_HOURS = 24
AVAILABILITY_BUFFER_H = 1  # wait this long after the window ends for sensor upload lag


def _retry(fn, *args, max_retries=5, **kwargs):
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # OpenAQ client raises on 429
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"    (rate limited, waiting {delay:.0f}s...)")
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def window_for(logged_at: datetime) -> tuple[datetime, datetime]:
    """The 24 whole hours the forecast covers: [next full hour, +24h)."""
    start = logged_at.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return start, start + timedelta(hours=WINDOW_HOURS)


def hour24_reading(window: pd.Series, end, tolerance_h: int):
    """The sensor reading for the forecast's 24th hour, i.e. the hour starting at
    end - 1h (the last hour of the 24-hour window). Uses the exact hour when
    present, else the nearest reading within +/- tolerance_h hours. Returns
    (value, timestamp) or None."""
    target = pd.Timestamp(end) - pd.Timedelta(hours=1)
    if len(window) == 0:
        return None
    gaps = abs(window.index - target)          # TimedeltaIndex
    pos = int(gaps.argmin())
    if gaps[pos] > pd.Timedelta(hours=tolerance_h):
        return None
    return float(window.iloc[pos]), window.index[pos]


def fetch_sensor_series(client, sensor_id: int, start: datetime, end: datetime) -> pd.Series:
    resp = _retry(
        client.measurements.list,
        sensors_id=sensor_id,
        data="hours",
        datetime_from=start,
        datetime_to=end,
        limit=1000,
    )
    records = getattr(resp, "results", None) or []
    if not records:
        # Empty series must still have a (tz-aware) DatetimeIndex: newer pandas gives an empty
        # Series a RangeIndex, and comparing that with a Timestamp raises TypeError.
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], tz="UTC"))
    times = [pd.Timestamp(r.period.datetime_from.utc) for r in records]
    values = [r.value for r in records]
    s = pd.Series(values, index=pd.DatetimeIndex(times)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tolerance-hours", type=int, default=1,
                        help="if the exact 24th-hour reading is missing, accept the nearest one within this many hours (default 1; 0 = exact only)")
    parser.add_argument("--min-hours", type=int, default=18,
                        help="minimum hourly readings required in the window (default 18 of 24)")
    parser.add_argument("--dry-run", action="store_true", help="report only, do not write the CSV")
    args = parser.parse_args()

    if not os.path.isfile(LOG_PATH):
        raise SystemExit(f"No log file at {LOG_PATH} yet -- make a live prediction first.")
    with open(LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("prediction_log.csv has no rows yet.")
    fieldnames = list(rows[0].keys())

    now = datetime.now(timezone.utc)
    pending: dict[str, list[tuple[dict, datetime, datetime]]] = {}
    not_ready = 0
    for row in rows:
        if row.get("actual_pm25"):
            continue
        logged_at = datetime.fromisoformat(row["logged_at_utc"])
        start, end = window_for(logged_at)
        if now < end + timedelta(hours=AVAILABILITY_BUFFER_H):
            not_ready += 1
            continue
        pending.setdefault(row["city"], []).append((row, start, end))

    n_pending = sum(len(v) for v in pending.values())
    print(f"{n_pending} row(s) ready to fill, {not_ready} still waiting for their 24h window to finish.")
    if not n_pending:
        return

    filled = skipped = 0
    api_key = _require_api_key()
    with OpenAQ(api_key=api_key) as client:
        for city, items in pending.items():
            print(f"\n=== {city}: {len(items)} row(s) ===")
            try:
                candidates = _find_pm25_candidates(city)
            except RuntimeError as e:
                print(f"  no stations: {e}")
                skipped += len(items)
                continue

            span_start = min(s for _, s, _ in items)
            span_end = max(e for _, _, e in items)
            series_cache: dict[int, pd.Series] = {}

            def series_for(sensor_id: int) -> pd.Series:
                if sensor_id not in series_cache:
                    series_cache[sensor_id] = fetch_sensor_series(client, sensor_id, span_start, span_end)
                    time.sleep(0.3)
                return series_cache[sensor_id]

            for row, start, end in items:
                chosen = None
                for _, sensor_id, name, dist in candidates:
                    s = series_for(sensor_id)
                    window = s[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]
                    if len(window) >= args.min_hours and hour24_reading(window, end, args.tolerance_hours) is not None:
                        chosen = (name, dist, window)
                        break
                label = f"{row['logged_at_utc'][:16]}"
                if chosen is None:
                    print(f"  {label}: no station had >= {args.min_hours}h in the window -- skipped")
                    skipped += 1
                    continue
                name, dist, window = chosen
                value, ts = hour24_reading(window, end, args.tolerance_hours)
                value = round(value, 2)
                dist_str = f"{dist:.0f}m" if dist is not None else "distance unknown"
                source = (f"OpenAQ '{name}' ({dist_str}), hour-24 reading at {ts:%Y-%m-%d %H:%M} UTC "
                          f"(forecast window {start:%Y-%m-%d %H:%M}-{end:%Y-%m-%d %H:%M} UTC)")
                print(f"  {label}: actual {value} µg/m³ (hour 24 @ {ts:%m-%d %H:%M})  <- {name}")
                if not args.dry_run:
                    row["actual_pm25"] = value
                    row["actual_source"] = source
                    row["actual_logged_at"] = datetime.now(timezone.utc).isoformat()
                filled += 1

    if args.dry_run:
        print(f"\nDry run: would fill {filled}, skip {skipped}. Nothing written.")
        return
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFilled {filled}, skipped {skipped}. Wrote {LOG_PATH}.")


if __name__ == "__main__":
    main()
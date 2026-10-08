"""Builds a live AIRPRED input window using REAL ground-sensor PM2.5 readings
from OpenAQ (openaq.org), instead of GEOS-CF's modeled estimate.

WHY USE OPENAQ:
    It gives real, measured PM2.5 in µg/m³ at point locations, so it can serve
    as ground truth for checking predictions. Open-Meteo still supplies
    meteorology, same as the GEOS-CF pipeline -- only the PM2.5 source changes.

    CAVEAT (corrected): AIRPRED's TRAINING PM2.5 was MERRA-2-derived (coarse
    ~50 km grid; one cell covers all 10 NCR cities), NOT ground-station data.
    Point sensors in urban Manila read much higher than that grid-cell mean,
    so feeding OpenAQ values into the model is a train/inference distribution
    shift, and the forecasts come out systematically low vs. OpenAQ actuals.
    Disclose this; do not present OpenAQ-input forecasts as in-distribution.

STILL BE HONEST ABOUT:
  - These are mostly low-cost optical sensors (Clarity, sensor.community,
    etc.), not EPA/DENR reference-grade monitors -- disclose the instrument
    type if asked, same as any sensor's limitations in a methods section.
  - Coverage was verified with diagnose_openaq_coverage.py: all 10 NCR
    cities have a live (hourly-reporting) PM2.5 station within ~4.3km,
    out of a dense local network (373/500 candidate stations tested were
    live as of that run -- see openaq_coverage_report.csv). Station
    availability can still change over time (sensors go offline), which is
    why fetch_live_window_openaq() always re-checks live status at request
    time rather than trusting a fixed list -- it raises a clear RuntimeError
    naming the city if nothing is found, and never silently substitutes
    another city's station.
  - Low-cost sensors drop out more than a modeled grid (GEOS-CF) does.
    Gaps are forward-filled the same way live_data_geoscf_s3.py handles
    missing meteorology, and every fill is disclosed in `warnings`.

SETUP:
    pip install openaq pandas numpy requests
    Get a free API key: sign up at https://explore.openaq.org, then:
        export OPENAQ_API_KEY="your-key-here"

USAGE (from main.py):
    from live_data_openaq import fetch_live_window_openaq, scale_window as scale_window_openaq
    x_pm25, x_met, n_hours, warnings = fetch_live_window_openaq("Manila")
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from openaq import OpenAQ

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")
PM25_PARAMETER_ID = 2          # OpenAQ's fixed id for the pm25 parameter
SEARCH_RADIUS_M = 25_000       # OpenAQ's maximum allowed radius
MIN_HOURS_PREFERRED = 24       # a station with at least this many hourly rows is accepted immediately
MAX_EXTRA_STATIONS = 6         # once some data is found, try at most this many more candidates for a fuller window
TARGET_HOURS = 48              # how far back we TRY to pull; fewer is fine (length-agnostic model)

MET_URL = "https://api.open-meteo.com/v1/forecast"

# Must match scaler.pkl's feature_names_in_[1:] exactly -- same list used by
# live_data.py / live_data_geoscf.py / live_data_geoscf_s3.py.
MET_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "boundary_layer_height",
]

# Same 10 cities/coordinates as the other live_data_*.py files -- keep these
# in sync if a coordinate ever changes.
CITY_COORDS = {
    "Manila": (14.5995, 120.9842),
    "Quezon_City": (14.6760, 121.0437),
    "Caloocan": (14.6760, 120.9663),
    "Valenzuela": (14.7011, 120.9830),
    "Pasig": (14.5764, 121.0851),
    "Makati": (14.5547, 121.0244),
    "Mandaluyong": (14.5794, 121.0359),
    "Navotas": (14.6667, 120.9417),
    "Pasay": (14.5378, 120.9972),
    "San_Juan": (14.6000, 121.0333),
}

# Resolved city -> (location_id, sensor_id, station_name, distance_m) of the
# station that ACTUALLY had live hourly data last time, so repeat requests
# don't re-run the candidate search+trial loop every call. Cleared for a
# city if that cached station stops returning data (see fetch_live_window_openaq).
_STATION_CACHE: dict[str, tuple[int, int, str, float | None]] = {}

CANDIDATE_SEARCH_LIMIT = 50  # how many nearby pm25 locations to consider, not just the #1 match.
# IMPORTANT: this was originally 10, which was too small. OpenAQ's API does
# NOT reliably return the N nearest results first when a city has a dense
# cluster of candidates (confirmed via diagnose_openaq_coverage.py -- a
# limit=10 search for Manila missed several live stations under 5km away and
# fell back to one 8km away, while a limit=50 search found them). 50 was
# confirmed to surface a live station within ~4.3km for all 10 cities.


def _require_api_key() -> str:
    if not OPENAQ_API_KEY:
        raise RuntimeError(
            "OPENAQ_API_KEY is not set. Sign up for a free key at "
            "https://explore.openaq.org, then `export OPENAQ_API_KEY=...` "
            "before starting the API server."
        )
    return OPENAQ_API_KEY


# OpenAQ allows only ~60 requests/minute. The map asks for all 10 cities at once, so:
#  * the nearby-station list (which almost never changes) is cached for 6 hours,
#  * every OpenAQ call retries briefly on HTTP 429, and
#  * a rate limit that persists becomes a RuntimeError (HTTP 503), not a 500 crash.
_CANDIDATE_TTL_S = 6 * 3600
_CANDIDATE_CACHE: dict[str, tuple[float, list]] = {}
_OPENAQ_LOCK = threading.Lock()   # one OpenAQ lookup at a time -> no bursts


def _openaq_call(fn):
    for wait in (3, 8, None):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ != "HTTPRateLimitError":
                raise
            if wait is None:
                raise RuntimeError(
                    "OpenAQ rate limit reached (too many requests). Wait a minute and retry; "
                    "results are cached for 10 minutes once fetched."
                ) from e
            time.sleep(wait)


def _find_pm25_candidates(city: str) -> list[tuple[int, int, str, float | None]]:
    """Returns up to CANDIDATE_SEARCH_LIMIT (location_id, sensor_id, name,
    distance_m) tuples with a pm25 sensor within SEARCH_RADIUS_M of the
    city's coordinates, sorted nearest-first. Sorted explicitly in Python
    rather than trusting the API's default order (not documented as
    distance-sorted, and in practice didn't behave like it was). Does NOT
    check whether each station actually has recent hourly data -- some are
    filter-based reference networks (e.g. SPARTAN) that report integrated
    multi-day samples, not hourly readings, and will look fine here but
    return zero rows from the hourly-measurements endpoint. That check
    happens per-candidate in fetch_live_window_openaq."""
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city '{city}'. Known: {list(CITY_COORDS)}")

    lat, lon = CITY_COORDS[city]
    api_key = _require_api_key()

    hit = _CANDIDATE_CACHE.get(city)
    if hit and time.time() - hit[0] < _CANDIDATE_TTL_S:
        return list(hit[1])

    with OpenAQ(api_key=api_key) as client:
        resp = _openaq_call(lambda: client.locations.list(
            coordinates=(lat, lon),
            radius=SEARCH_RADIUS_M,
            parameters_id=PM25_PARAMETER_ID,
            limit=CANDIDATE_SEARCH_LIMIT,
        ))

    results = getattr(resp, "results", None) or []
    if not results:
        raise RuntimeError(
            f"No OpenAQ station reporting pm25 found within "
            f"{SEARCH_RADIUS_M / 1000:.0f}km of {city} "
            f"({lat}, {lon}). This city has no confirmed live ground-sensor "
            "coverage -- fall back to live_data_geoscf_s3.py for this city."
        )

    candidates = []
    for station in results:
        sensor_id = None
        for sensor in station.sensors:
            if sensor.parameter.name == "pm25":
                sensor_id = sensor.id
                break
        if sensor_id is None:
            continue  # matched the location search but has no pm25 sensor on it -- skip
        candidates.append((station.id, sensor_id, station.name, getattr(station, "distance", None)))

    candidates.sort(key=lambda c: (c[3] if c[3] is not None else float("inf")))

    if not candidates:
        raise RuntimeError(
            f"{len(results)} OpenAQ location(s) found near {city}, but none "
            "actually carry a pm25 sensor -- inspect manually."
        )
    _CANDIDATE_CACHE[city] = (time.time(), list(candidates))
    return candidates


def find_pm25_station(city: str) -> tuple[int, int, str, float | None]:
    """Back-compat wrapper: nearest pm25 candidate only, no hourly-data
    check. Prefer fetch_live_window_openaq(), which tries candidates in
    order and skips ones with no live hourly data."""
    return _find_pm25_candidates(city)[0]


def _fetch_met_aligned(lat: float, lon: float, start_utc, end_utc) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(MET_VARS),
        "start_hour": pd.Timestamp(start_utc).strftime("%Y-%m-%dT%H:%M"),
        "end_hour": pd.Timestamp(end_utc).strftime("%Y-%m-%dT%H:%M"),
        "timezone": "UTC",
    }
    r = requests.get(MET_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["hourly"]


def _ffill(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for col in range(out.shape[1]):
        series = out[:, col]
        mask = np.isnan(series)
        if mask.all():
            series[:] = 0.0
            continue
        idx = np.where(~mask, np.arange(len(series)), 0)
        np.maximum.accumulate(idx, out=idx)
        series[mask] = series[idx[mask]]
        if np.isnan(series[0]):
            first_valid = series[~np.isnan(series)][0]
            series[: np.argmax(~np.isnan(series))] = first_valid
    return out


def fetch_live_window_openaq(city: str) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Returns (x_pm25, x_met, n_hours, warnings) using real OpenAQ ground
    sensor readings for PM2.5 and Open-Meteo for meteorology. Like
    fetch_live_window_geoscf, n_hours is whatever's actually available
    (length-agnostic model, no padding).

    x_pm25: shape (n_hours, 1)
    x_met:  shape (n_hours, 7), columns in MET_VARS order
    """
    warnings: list[str] = []
    lat, lon = CITY_COORDS[city]
    api_key = _require_api_key()
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=TARGET_HOURS)

    # Try the cached station first (it worked last time); if it's gone dry,
    # fall through to a fresh candidate search instead of trusting the cache.
    candidates: list[tuple[int, int, str, float | None]] = []
    if city in _STATION_CACHE:
        candidates.append(_STATION_CACHE[city])
    seen_sensors = {c[1] for c in candidates}
    for c in _find_pm25_candidates(city):
        if c[1] not in seen_sensors:
            candidates.append(c)
            seen_sensors.add(c[1])

    records = []
    station_name = distance_m = sensor_id = None
    tried: list[str] = []
    extra_tries = 0
    with OpenAQ(api_key=api_key) as client:
        for _, cand_sensor_id, cand_name, cand_dist in candidates:
            resp = _openaq_call(lambda: client.measurements.list(
                sensors_id=cand_sensor_id,
                data="hours",
                datetime_from=start,
                datetime_to=now,
                limit=TARGET_HOURS + 2,
            ))
            cand_records = getattr(resp, "results", None) or []
            tried.append(f"{cand_name} ({len(cand_records)} hourly rows)")
            # Keep the candidate with the most hourly rows seen so far.
            if len(cand_records) > len(records):
                records, sensor_id, station_name, distance_m = cand_records, cand_sensor_id, cand_name, cand_dist
            # Stop as soon as a station gives a reasonably full window.
            if len(records) >= MIN_HOURS_PREFERRED:
                break
            # A station returned something but is thin: look a few candidates
            # further for a fuller one, then settle for the best seen.
            if records:
                extra_tries += 1
                if extra_tries > MAX_EXTRA_STATIONS:
                    break

    if not records:
        _STATION_CACHE.pop(city, None)
        raise RuntimeError(
            f"{len(candidates)} OpenAQ station(s) found near {city}, but none "
            f"returned hourly data in the last {TARGET_HOURS}h: {'; '.join(tried)}. "
            "Some nearby stations may be reference networks (e.g. SPARTAN) that "
            "report multi-day integrated samples, not hourly readings -- not "
            "usable as a live input regardless of distance. Fall back to "
            "live_data_geoscf_s3.py for this city."
        )

    _STATION_CACHE[city] = (None, sensor_id, station_name, distance_m)

    times = [pd.Timestamp(r.period.datetime_from.utc) for r in records]
    values = [r.value for r in records]
    series = pd.Series(values, index=pd.DatetimeIndex(times)).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    n_hours = len(series)
    dist_note = f"{distance_m:.0f}m away" if distance_m is not None else "distance unknown"
    warnings.append(
        f"PM2.5 input is sourced from OpenAQ station '{station_name}' "
        f"({dist_note} from {city}'s reference coordinates) -- a real "
        "ground sensor reading, not a model estimate. Likely a low-cost "
        "optical sensor (e.g. Clarity), not an EPA/DENR reference-grade "
        "monitor; disclose this if asked."
    )
    if len(tried) > 1:
        warnings.append(
            f"{len(tried)} OpenAQ station(s) were tried (nearest first); the one "
            f"with the most hourly data was used: {'; '.join(tried)}."
        )
    warnings.append(
        f"Using {n_hours} real hour(s) of OpenAQ PM2.5 context out of "
        f"{TARGET_HOURS} requested -- NOT the model's validated 48-hour "
        "window. Low-cost sensors have more reporting gaps than GEOS-CF's "
        "modeled grid; disclose the actual hour count if asked."
    )
    if n_hours < 6:
        warnings.append(
            f"Only {n_hours} hour(s) available -- this station may be "
            "intermittently offline right now. Consider falling back to "
            "live_data_geoscf_s3.py for this request."
        )

    x_pm25 = series.values.reshape(-1, 1).astype(np.float32)

    met = _fetch_met_aligned(lat, lon, series.index.min(), series.index.max())
    met_rows = np.array([met[v][:n_hours] for v in MET_VARS], dtype=np.float32).T

    if met_rows.shape[0] != n_hours:
        n = min(met_rows.shape[0], n_hours)
        warnings.append(
            f"Meteorology returned {met_rows.shape[0]} rows for a "
            f"{n_hours}-hour PM2.5 window -- trimming both to {n} rows."
        )
        met_rows = met_rows[:n]
        x_pm25 = x_pm25[:n]
        n_hours = n

    nan_pm25 = int(np.isnan(x_pm25).sum())
    if nan_pm25:
        warnings.append(f"{nan_pm25} NaN PM2.5 reading(s) from the sensor -- forward-filling.")
        x_pm25 = _ffill(x_pm25)

    nan_met = int(np.isnan(met_rows).sum())
    if nan_met:
        warnings.append(f"{nan_met} NaN met values -- forward-filling.")
        met_rows = _ffill(met_rows)

    return x_pm25, met_rows, n_hours, warnings


def scale_window(x_pm25: np.ndarray, x_met: np.ndarray, scaler):
    """Identical logic to the other live_data_*.py scale_window functions,
    duplicated so this file has no import dependency on the others."""
    names = list(scaler.feature_names_in_)
    pm25_idx = names.index("pm25")
    met_idx = [names.index(v) for v in MET_VARS]

    pm25_mean, pm25_scale = scaler.mean_[pm25_idx], scaler.scale_[pm25_idx]
    met_mean = scaler.mean_[met_idx]
    met_scale = scaler.scale_[met_idx]

    x_pm25_scaled = (x_pm25 - pm25_mean) / pm25_scale
    x_met_scaled = (x_met - met_mean) / met_scale
    return x_pm25_scaled.astype(np.float32), x_met_scaled.astype(np.float32)


if __name__ == "__main__":
    # NOTE: this calls the REAL fetch_live_window_openaq() for every city,
    # not just find_pm25_station(). find_pm25_station() only tells you the
    # nearest station that HAS a pm25 sensor -- it never checks whether that
    # station actually reports hourly data. In testing, the "nearest" match
    # was frequently a dead station (0 hourly rows) and the real data came
    # from a station several candidates further down the list. Only a full
    # fetch_live_window_openaq() run proves a city has usable live coverage.
    print("=== Checking REAL OpenAQ live coverage for all 10 cities ===")
    print("(this calls the full candidate-trial fetch for each city, not just")
    print(" the nearest sensor match -- slower, but tells the truth)\n")
    summary = {}
    for city in CITY_COORDS:
        try:
            x_pm25, x_met, n_hours, warnings = fetch_live_window_openaq(city)
            _, sensor_id, station_name, distance_m = _STATION_CACHE[city]
            dist_str = f"{distance_m:.0f}m" if distance_m is not None else "?"
            print(f"  {city:15s} -> OK: '{station_name}' ({dist_str} away), {n_hours}h of real hourly data")
            summary[city] = (True, station_name, n_hours)
        except RuntimeError as e:
            print(f"  {city:15s} -> NO LIVE COVERAGE: {e}")
            summary[city] = (False, None, 0)

    ok_cities = [c for c, (ok, _, _) in summary.items() if ok]
    print(f"\n=== Summary: {len(ok_cities)}/10 cities have real live hourly OpenAQ coverage ===")
    for city in ok_cities:
        _, station, n_hours = summary[city]
        print(f"  {city}: '{station}', {n_hours}h")

    if "Manila" in summary and summary["Manila"][0]:
        print("\n=== Full warnings for Manila (as an example) ===")
        _, _, _, warnings = fetch_live_window_openaq("Manila")
        for w in warnings:
            print(f"  - {w}")
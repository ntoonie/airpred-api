"""Extracts the latest 48 hourly pm25_rh35 values for the 10 AIRPRED NCR
cities directly from NASA's public GEOS-CF v2 "latest forecast" Zarr store,
without loading the full ~41GB dataset into RAM.

STORE:
    s3://smce-geos-cf-public/geos-cf-v2-fcst-latest.zarr/

WHY THIS ONE, NOT EARTH ENGINE OR OPENDAP:
  - NASA's own direct OPeNDAP "latest" endpoint for this product was checked
    and is frozen at Jan 2026 (dead).
  - Earth Engine's copy works but lags ~3 days behind real time and costs
    one network round-trip per city per 48-image time series.
  - This Zarr store is described by NASA as the latest forecast cycle's
    output -- if "now" falls inside its time range (the script checks this
    for you), the hours up to "now" are the model's own very-short-range
    output from its most recent initialization: the closest thing to a
    live PM2.5 reading available without waiting weeks for reanalysis.
    Hours AFTER "now" in this same array are genuine forward forecast, not
    historical input -- do not feed those into AIRPRED's 48-hour input
    window. The script separates these for you.

STILL TRUE REGARDLESS OF SOURCE: GEOS-CF is a model estimate (NASA GMAO
composition forecast), not a ground observation, and has not been
validated against your test set. It shares a modeling lineage with the
MERRA-2 data your model trained on (unlike CAMS/Open-Meteo), which is the
honest reason to prefer it, not a claim of equal or better accuracy.

VARIABLE-LENGTH LIVE WINDOW (fetch_live_window_geoscf):
  The 48-row requirement above turned out to be a self-imposed constraint,
  not a model requirement. AIRPRED's architecture was confirmed (by direct
  review of tcn.py, variants.py, attention.py) to be length-agnostic:
  CausalConv1d pads/slices per-input-length, ForecastHead does global
  average pooling over time, and the cross-modal attention module is
  standard nn.MultiheadAttention -- none of it hardcodes 48. So instead of
  padding/stitching to force exactly 48 hours, fetch_live_window_geoscf()
  below just returns however many real hours the current GEOS-CF cycle
  actually has right now (no fabricated or backfilled hours), and the
  model is fed that length directly. This is explicitly OUTSIDE the
  48-hour window the model was trained/validated on -- always surface the
  actual hour count used, and do not present this as equivalent to your
  test-set results.

SETUP:
    pip install xarray zarr s3fs fsspec pandas numpy requests

USAGE:
    python3 live_data_geoscf_s3.py
    # prints a 48-row x 10-city PM2.5 table (fixed-48 version, may fail
    # early in a cycle) AND a single-city variable-length live window
    # (always succeeds if the cycle has at least 1 past hour), saves the
    # fixed-48 table to geoscf_live_pm25.csv if available

USAGE (from main.py, variable-length version):
    from live_data_geoscf_s3 import fetch_live_window_geoscf, scale_window
    x_pm25, x_met, n_hours, warnings = fetch_live_window_geoscf("Manila")
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import xarray as xr

ZARR_STORE = "s3://smce-geos-cf-public/geos-cf-v2-fcst-latest.zarr/"
PM25_VAR = "pm25_rh35"

MET_URL = "https://api.open-meteo.com/v1/forecast"

# Column order MUST match scaler.pkl's feature_names_in_[1:] exactly --
# same list as live_data.py / live_data_geoscf.py.
MET_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "boundary_layer_height",
]

# Same 10 cities / coordinates as live_data.py and live_data_geoscf.py --
# keep these three files in sync if you ever change a coordinate.
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


def open_latest(zarr_store: str = ZARR_STORE) -> xr.Dataset:
    """Opens the store lazily -- no data is downloaded until a variable is
    both selected (indexed down to a small slice) and computed."""
    return xr.open_zarr(
        zarr_store,
        storage_options={"anon": True},
        consolidated=True,
    )


def _find_pm25_var(ds: xr.Dataset) -> str:
    if PM25_VAR in ds.data_vars:
        return PM25_VAR
    candidates = [v for v in ds.data_vars if "pm25" in v.lower()]
    if not candidates:
        raise KeyError(
            f"No variable matching 'pm25' found. Available vars: {list(ds.data_vars)}"
        )
    # prefer an exact/close match to pm25_rh35 (total PM2.5, not a species
    # sub-component like pm25bc_rh35, pm25du_rh35, etc.)
    for c in candidates:
        if c.lower() == PM25_VAR:
            return c
    print(
        f"WARNING: exact var '{PM25_VAR}' not found, using closest match "
        f"'{candidates[0]}' -- available pm25-like vars: {candidates}"
    )
    return candidates[0]


def extract_city_table(ds: xr.Dataset | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (df, info).

    df: DataFrame indexed by UTC datetime, one column per city, pm25 in ug/m3,
        covering the full time range the store currently has (up to 120 rows).
    info: dict with 'now_index' (row index closest to current time, or None
          if "now" falls outside the store's range) and 'var_name' used.
    """
    if ds is None:
        ds = open_latest()

    var_name = _find_pm25_var(ds)

    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    time_name = "time" if "time" in ds.coords else "forecast_time"

    cities = list(CITY_COORDS.keys())
    lats = xr.DataArray([CITY_COORDS[c][0] for c in cities], dims="city")
    lons = xr.DataArray([CITY_COORDS[c][1] for c in cities], dims="city")

    da = ds[var_name]
    if "lev" in da.dims:
        da = da.isel(lev=0)  # single pressure level per earlier inspection

    # Vectorized nearest-neighbor selection for all 10 cities at once.
    # This is still lazy -- nothing is downloaded yet.
    sel = da.sel({lat_name: lats, lon_name: lons}, method="nearest")
    sel = sel.assign_coords(city=("city", cities))

    # NOW trigger the actual S3 read -- only the (time, city) subset,
    # not the full global grid, gets pulled down.
    print(f"Fetching {var_name} for {len(cities)} cities x "
          f"{sel.sizes[time_name]} time steps from S3 (lazy -> computing now)...")
    values = sel.compute()

    times = pd.to_datetime(values[time_name].values)
    df = pd.DataFrame(values.values, index=times, columns=cities)
    df.index.name = "datetime_utc"

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if df.index.min() <= now <= df.index.max():
        now_index = int((df.index <= now).sum()) - 1
    else:
        now_index = None

    info = {"var_name": var_name, "now_index": now_index, "now_utc": now}
    return df, info


def latest_48h_window(df: pd.DataFrame, now_index: int | None) -> pd.DataFrame:
    """Returns the 48 rows ending at (or nearest before) "now", i.e. the
    historical-input-shaped slice AIRPRED expects. Raises if fewer than 48
    rows are available at or before "now"."""
    if now_index is None:
        raise RuntimeError(
            "Current time does not fall inside this Zarr store's time range "
            "-- the store may not have refreshed, or your clock/timezone is "
            "off. Falling back to the most recent 48 rows available instead "
            "would silently use forecast-only or stale data -- check "
            "manually before proceeding."
        )
    start = now_index - 47
    if start < 0:
        raise RuntimeError(
            f"Only {now_index + 1} rows available at/before 'now', need 48. "
            "The latest forecast cycle may have just initialized -- try "
            "again in a few hours, or fall back to live_data.py / "
            "live_data_geoscf.py for tonight."
        )
    return df.iloc[start : now_index + 1]


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


def fetch_live_window_geoscf(city: str) -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    """Returns (x_pm25, x_met, n_hours, warnings) using WHATEVER real hours
    the current GEOS-CF S3 cycle has available right now -- no padding, no
    backfill, no fixed 48. n_hours is almost certainly NOT 48; AIRPRED's
    architecture is confirmed length-agnostic (see module docstring), so
    this is fed to the model as-is.

    x_pm25: shape (n_hours, 1)
    x_met:  shape (n_hours, 7), columns in MET_VARS order
    """
    if city not in CITY_COORDS:
        raise ValueError(f"Unknown city '{city}'. Known: {list(CITY_COORDS)}")

    warnings: list[str] = []
    lat, lon = CITY_COORDS[city]

    ds = open_latest()
    var_name = _find_pm25_var(ds)
    lat_name = "lat" if "lat" in ds.coords else "latitude"
    lon_name = "lon" if "lon" in ds.coords else "longitude"
    time_name = "time" if "time" in ds.coords else "forecast_time"

    da = ds[var_name]
    if "lev" in da.dims:
        da = da.isel(lev=0)
    sel = da.sel({lat_name: lat, lon_name: lon}, method="nearest")

    print(f"[S3] fetching {var_name} for {city}...")
    values = sel.compute()
    times = pd.to_datetime(values[time_name].values)
    series = pd.Series(np.asarray(values.values, dtype=np.float32), index=times)

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    if not (series.index.min() <= now <= series.index.max()):
        raise RuntimeError(
            f"'now' ({now}) falls outside this GEOS-CF cycle's range "
            f"({series.index.min()} to {series.index.max()}) -- the store "
            "may need a refresh, or check your clock/timezone."
        )
    now_index = int((series.index <= now).sum()) - 1
    past = series.iloc[: now_index + 1]
    n_hours = len(past)
    cycle_start = series.index.min()

    warnings.append(
        f"Using {n_hours} real hour(s) of GEOS-CF PM2.5 context (cycle "
        f"initialized {cycle_start} UTC; 'now' is {now} UTC) -- NOT the "
        "model's validated 48-hour window. Disclose the actual hour count "
        "if asked; treat this as an architecture-robustness demo, not "
        "validated accuracy."
    )
    if n_hours < 6:
        warnings.append(
            f"Only {n_hours} hour(s) available -- this cycle likely just "
            "initialized. Consider waiting, or fall back to live_data.py "
            "(Open-Meteo, always a full 48h) for a more substantial window."
        )

    x_pm25 = past.values.reshape(-1, 1).astype(np.float32)

    met = _fetch_met_aligned(lat, lon, past.index.min(), past.index.max())
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

    nan_met = int(np.isnan(met_rows).sum())
    if nan_met:
        warnings.append(f"{nan_met} NaN met values -- forward-filling.")
        met_rows = _ffill(met_rows)

    warnings.append(
        "PM2.5 input is sourced from NASA GEOS-CF v2 (GMAO composition "
        "forecast, same modeling family as your MERRA-2 training data) via "
        "the public S3 Zarr 'latest' forecast cycle, not validated against "
        "your test set."
    )

    return x_pm25, met_rows, n_hours, warnings


def scale_window(x_pm25: np.ndarray, x_met: np.ndarray, scaler):
    """Identical logic to live_data.scale_window / live_data_geoscf.scale_window,
    duplicated so this file has no import dependency on the others. Works for
    any window length since scaling is elementwise/per-column."""
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
    print("=== Variable-length live window (Manila) ===")
    try:
        x_pm25, x_met, n_hours, warnings = fetch_live_window_geoscf("Manila")
        print(f"n_hours used: {n_hours}")
        print(f"x_pm25 shape: {x_pm25.shape}  x_met shape: {x_met.shape}")
        print(f"Last 3 PM2.5 readings: {x_pm25[-3:].ravel()}")
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")
    except RuntimeError as e:
        print(f"Variable-length fetch failed: {e}")

    print("\n=== Fixed 48-hour table, all 10 cities (may fail early in a cycle) ===")
    df_full, info = extract_city_table()
    print(f"\nFull store range: {df_full.index.min()} to {df_full.index.max()} UTC "
          f"({len(df_full)} rows)")
    print(f"Current time (UTC): {info['now_utc']}")

    if info["now_index"] is None:
        print(
            "\n'Now' falls OUTSIDE this store's time range -- every row here "
            "is either already-past forecast output with no fresher cycle "
            "loaded, or entirely future forecast. Inspect df_full manually "
            "before using it as AIRPRED input."
        )
    else:
        hours_in = info["now_index"]
        hours_after = len(df_full) - info["now_index"] - 1
        print(
            f"'Now' sits at row {hours_in} of {len(df_full)} -- "
            f"{hours_in} hours of this run are in the past (nowcast/short-range), "
            f"{hours_after} hours are forecast into the future."
        )
        try:
            window = latest_48h_window(df_full, info["now_index"])
            print(f"\n48-hour input window: {window.index.min()} to "
                  f"{window.index.max()} UTC")
            print(window.round(2))
            window.to_csv("geoscf_live_pm25.csv")
            print("\nSaved to geoscf_live_pm25.csv")
        except RuntimeError as e:
            print(f"\n{e}")
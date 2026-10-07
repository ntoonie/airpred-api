"""Meteorology swap test: is the live feed hurting Variant C?

Variant D ignores meteorology, Variant C uses it, so if the live meteorology is
wrong, C is penalised and D is not. This script feeds the SAME PM2.5 input and
the SAME actual to the models under three meteorology conditions and compares:

  A  ERA5, timestamp-aligned          what the models were TRAINED on (reference)
  B  Open-Meteo forecast API, timestamp-aligned
                                      what backfill_week.py feeds (live source, correct alignment)
  S  B, but PM2.5 compressed          only hours that have a PM2.5 reading are kept (like the live
                                      code does), meteorology still matched by TIMESTAMP
  C  live-style assembly              what fetch_live_window_openaq() actually does: the same
                                      compressed PM2.5, but meteorology taken POSITIONALLY
                                      ([:n_hours]) -- misaligned whenever the sensor has gaps

Reading the result (per variant, MAE in ug/m3, same anchors in every column):
  * D must be IDENTICAL under A and B (it ignores meteorology). If not, there is a bug.
  * B - A:  the cost of the live METEOROLOGY SOURCE (forecast API vs ERA5).
  * S - B:  the cost of dropping the gap hours from the PM2.5 input (affects D too).
  * C - S:  the cost of the live ALIGNMENT BUG (position vs timestamp); D must show exactly 0.
  * If C's MAE barely moves across A/B/C, the live feed is not the reason D >= C.
  * If C gets clearly worse in B or C while D does not, that is a feed problem worth fixing.
  Look at the paired differences and p-values, not just the means.

Needs the per-city history CSVs from fetch_openaq_history.py (data/openaq/<City>.csv: OpenAQ
PM2.5 + ERA5 meteorology, UTC). Run from the API folder (next to main.py):

    python3 swap_test_met.py --days 45                 # original models
    python3 swap_test_met.py --days 45 --finetuned     # fine-tuned models
    python3 swap_test_met.py --days 30 --cities Manila Pasig

The forecast API only serves roughly the last 3 months, so the window must be recent:
the script uses the last --days days that the CSVs cover. If the forecast request is
rejected, lower --days or re-run fetch_openaq_history.py to refresh the CSVs.
Nothing is written to prediction_log.csv; results go to swap_test_met_results.csv.
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import torch
from scipy import stats

import main as app_main  # noqa: E402  (loads models + scaler; does not start the server)
from live_data_openaq import (  # noqa: E402
    CITY_COORDS,
    MET_URL,
    MET_VARS,
    _ffill,
    scale_window,
)

INPUT_HOURS = 48
HORIZON_HOURS = 24
CACHE_DIR = "swap_cache"
CONDITIONS = {"A": "ERA5, aligned", "B": "forecast API, aligned", "S": "B + PM2.5 compressed", "C": "live-style (positional)"}


def load_city_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC")
    df = df.set_index("datetime").sort_index()
    bad = (df["pm25"] < 0) | (df["pm25"] > 500)          # same cleaning rule as training
    df.loc[bad, "pm25"] = np.nan
    return df


def fetch_forecast_met(lat: float, lon: float, start, end) -> pd.DataFrame:
    """Hourly meteorology from the SAME endpoint the live code uses (MET_URL), UTC."""
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(MET_VARS),
        "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"fc_{lat:.4f}_{lon:.4f}_{params['start_date']}_{params['end_date']}.csv")
    if os.path.isfile(cache):                         # reruns (e.g. --finetuned) reuse the download
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.index = pd.DatetimeIndex(df.index).tz_localize("UTC") if df.index.tz is None else df.index
        return df
    last_err = None
    for attempt in range(5):
        try:
            r = requests.get(MET_URL, params=params, timeout=120)
        except requests.RequestException as e:       # timeouts happen; wait and retry
            last_err = e
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(20 * (attempt + 1))
            continue
        if r.status_code >= 400:
            raise SystemExit(
                f"Forecast API rejected the request ({r.status_code}): {r.text[:200]}\n"
                "The window is probably too old for this endpoint -- use a smaller --days "
                "and refresh the CSVs with fetch_openaq_history.py."
            )
        h = r.json()["hourly"]
        idx = pd.DatetimeIndex(pd.to_datetime(h["time"], utc=True))
        df = pd.DataFrame({v: h[v] for v in MET_VARS}, index=idx).astype("float64")
        df.to_csv(cache)
        return df
    raise SystemExit(f"Forecast API did not answer after 5 tries ({last_err}). Re-run; finished cities are cached.")


def predict_mean(models: dict, x_pm25: np.ndarray, x_met: np.ndarray) -> dict:
    xp, xm = scale_window(x_pm25, x_met, app_main.SCALER)
    xp_t = torch.as_tensor(xp, dtype=torch.float32).unsqueeze(0)
    xm_t = torch.as_tensor(xm, dtype=torch.float32).unsqueeze(0)
    out = {}
    with torch.no_grad():
        for v, m in models.items():
            out[v] = float(np.mean(app_main.inverse_pm25(m(xp_t, xm_t).numpy()[0])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/openaq")
    ap.add_argument("--days", type=int, default=45, help="number of daily anchors (default 45)")
    ap.add_argument("--anchor-hour-utc", type=int, default=0)
    ap.add_argument("--cities", nargs="+", default=list(CITY_COORDS))
    ap.add_argument("--min-input-hours", type=int, default=24)
    ap.add_argument("--min-actual-hours", type=int, default=18)
    ap.add_argument("--finetuned", action="store_true", help="use main.MODELS_OPENAQ (fine-tuned) instead of the original models")
    ap.add_argument("--out", default="swap_test_met_results.csv")
    args = ap.parse_args()

    models = app_main.MODELS_OPENAQ if args.finetuned else app_main.MODELS
    print(f"Models: {'OpenAQ fine-tuned' if args.finetuned else 'original'}   variants: {list(models)}")

    rows, met_diffs, shifts = [], [], []
    for city in args.cities:
        path = os.path.join(args.data_dir, f"{city}.csv")
        if not os.path.isfile(path):
            print(f"{city}: {path} not found -- skipped")
            continue
        df = load_city_csv(path)
        pm, era = df["pm25"], df[MET_VARS]
        last = df.index.max() - timedelta(hours=HORIZON_HOURS)
        last_anchor = last.replace(hour=args.anchor_hour_utc, minute=0, second=0, microsecond=0)
        if last_anchor > last:
            last_anchor -= timedelta(days=1)
        anchors = [last_anchor - timedelta(days=i) for i in range(args.days - 1, -1, -1)]
        lat, lon = CITY_COORDS[city]
        fc = fetch_forecast_met(lat, lon, anchors[0] - timedelta(hours=INPUT_HOURS), anchors[-1])
        used = 0
        for anchor in anchors:
            in_idx = pd.date_range(anchor - timedelta(hours=INPUT_HOURS), periods=INPUT_HOURS, freq="1h", tz="UTC")
            pm_in = pm.reindex(in_idx)
            n_real = int(pm_in.notna().sum())
            act = pm[(pm.index >= anchor) & (pm.index < anchor + timedelta(hours=HORIZON_HOURS))].dropna()
            if n_real < args.min_input_hours or len(act) < args.min_actual_hours:
                continue
            era_in, fc_in = era.reindex(in_idx), fc.reindex(in_idx)
            if era_in.isna().all().any() or fc_in.isna().all().any():
                continue
            actual = float(act.mean())

            x_pm = _ffill(pm_in.to_numpy(dtype=np.float32).reshape(-1, 1))
            pred = {
                "A": predict_mean(models, x_pm, _ffill(era_in.to_numpy(dtype=np.float32))),
                "B": predict_mean(models, x_pm, _ffill(fc_in.to_numpy(dtype=np.float32))),
            }
            # Conditions S and C: only hours WITH a reading are kept, as in the live code.
            real = pm_in.dropna()
            x_real = real.to_numpy(dtype=np.float32).reshape(-1, 1)
            pred["S"] = predict_mean(models, _ffill(x_real), _ffill(fc.reindex(real.index).to_numpy(dtype=np.float32)))
            # C: the meteorology block is contiguous min..max and is cut by position -- the live bug.
            met_span = fc.loc[real.index.min():real.index.max()]
            met_rows = met_span.to_numpy(dtype=np.float32)[: len(real)]
            n = min(len(real), len(met_rows))
            pred["C"] = predict_mean(
                models,
                _ffill(real.to_numpy(dtype=np.float32).reshape(-1, 1)[:n]),
                _ffill(met_rows[:n]),
            )
            shifts.append(len(met_span) - len(real))      # hours of meteorology that fall off the end / shift

            met_diffs.append((era_in - fc_in).abs().mean().to_dict() | {"city": city})
            for cond, per_var in pred.items():
                for v, p in per_var.items():
                    rows.append({"city": city, "anchor": anchor.isoformat(), "variant": v, "condition": cond,
                                 "pred": p, "actual": actual, "abs_err": abs(p - actual)})
            used += 1
        print(f"{city}: {used} usable anchors")

    if not rows:
        raise SystemExit("No usable anchors. Check the CSVs and --days.")
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    n_anchor = res.groupby(["city", "anchor"]).ngroups
    print(f"\n{n_anchor} (city, anchor) cases; identical cases in every column below.\n")

    # 1. MAE per variant x condition
    print("MAE (ug/m3)  -- lower is better")
    print(f"{'':8s}" + "".join(f"{'cond ' + c:>10s}" for c in CONDITIONS)
          + f"{'B-A':>9s}{'S-B':>9s}{'C-S':>9s}")
    for v in models:
        m = {c: res[(res.variant == v) & (res.condition == c)]["abs_err"].mean() for c in CONDITIONS}
        print(f"{'Var ' + v:8s}" + "".join(f"{m[c]:10.2f}" for c in CONDITIONS)
              + f"{m['B'] - m['A']:+9.2f}{m['S'] - m['B']:+9.2f}{m['C'] - m['S']:+9.2f}")
    print("  " + "; ".join(f"{c} = {d}" for c, d in CONDITIONS.items()))

    # 2. sanity: D must not care about meteorology
    if "D" in models:
        d = res[res.variant == "D"].pivot_table(index=["city", "anchor"], columns="condition", values="pred")
        g1, g2 = float((d["A"] - d["B"]).abs().max()), float((d["S"] - d["C"]).abs().max())
        ok = g1 < 1e-4 and g2 < 1e-4
        print(f"\nSanity check: Variant D differs by at most {g1:.6f} (A vs B) and {g2:.6f} (S vs C) -- "
              f"{'OK, D ignores meteorology' if ok else 'PROBLEM: D should not change'}")

    # 3. paired tests for the variants that use meteorology
    print("\nPaired differences in absolute error (negative = better), per anchor:")
    for v in models:
        if v == "D":
            continue
        e = res[res.variant == v].pivot_table(index=["city", "anchor"], columns="condition", values="abs_err")
        for a, b, label in (("B", "A", "forecast-API met vs ERA5"), ("S", "B", "PM2.5 gap hours dropped"),
                            ("C", "S", "positional vs timestamp met alignment")):
            diff = e[a] - e[b]
            if float(diff.abs().max()) < 1e-9:
                print(f"  Var {v}: cond {a} - cond {b} ({label}): identical inputs/outputs")
                continue
            t = stats.ttest_rel(e[a], e[b])
            print(f"  Var {v}: cond {a} - cond {b} ({label}): {diff.mean():+.2f}  "
                  f"worse in {int((diff > 0).sum())}/{len(diff)}  p={t.pvalue:.3f}")

    # 4. how different are the two meteorology sources?
    md = pd.DataFrame(met_diffs).drop(columns="city").mean()
    scale = dict(zip(app_main.SCALER.feature_names_in_, app_main.SCALER.scale_))
    print("\nERA5 vs forecast-API meteorology, mean |difference| per variable (and as a fraction of the training std):")
    for v in MET_VARS:
        print(f"  {v:24s}{md[v]:10.3f}   {md[v] / scale[v]:6.2f} std")
    if float(md.max()) < 1e-6:
        print("\nWARNING: ERA5 and forecast-API meteorology are IDENTICAL here, so conditions A and B are the "
              "same input and 'B - A' says nothing about the source. Either the CSV meteorology was fetched from "
              "the forecast API, or the two endpoints agree. Check ARCHIVE_URL in fetch_openaq_history.py and "
              "compare a few raw values before concluding anything about the source.")
    print(f"\nLive-style assembly: on average {np.mean(shifts):.1f} h of meteorology per window are shifted/dropped "
          f"because the sensor had gaps (0 = no gaps).")
    print(f"\nPer-anchor results written to {args.out}")


if __name__ == "__main__":
    main()
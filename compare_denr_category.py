"""Compare AIRPRED's OpenAQ-based forecast with the EMB-DENR PM2.5 AQI category.

Question this answers: "the forecast that was made for 10/8/2026 6:00 PM, did it land in the
same AQI category that EMB-DENR reported at 10/8/2026 6:00 PM?"

How the forecast is chosen
  A forecast made at time T is a 24-hour forecast, so its HOUR-24 value is the prediction
  for T + 24 h. To predict 10/8 6:00 PM (PHT) we need a forecast made at 10/7 6:00 PM (PHT).
  For each city this script takes the LIVE OpenAQ row in prediction_log.csv that was logged
  closest to that time (within --tolerance-hours) and uses its hour-24 value from Variant C.
  Backfilled-hindcast rows are ignored: they are anchored at 8:00 AM PHT, so their hour 24
  falls at about 7:00 AM, not 6:00 PM.

Category rule (DAO 2020-14, PM2.5 ug/m3): Good <= 25, Fair <= 35, Unhealthy for Sensitive
  Groups <= 45, Very Unhealthy <= 55, Acutely Unhealthy <= 90, Emergency above. These are the
  same bands as EMB's AQI scale (Good 0-50, Fair 51-100, ...), so the categories are comparable.

Usage (run next to prediction_log.csv, any Python 3):
    python3 compare_denr_category.py
    python3 compare_denr_category.py --tolerance-hours 6
    python3 compare_denr_category.py --made-at "2026-10-07 18:00"     # PHT, override
    python3 compare_denr_category.py --use-mean                       # mean of the 24 h instead of hour 24

Nothing is written; it only reads the log.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone

PHT = timezone(timedelta(hours=8))

# EMB-DENR "PM 2.5" table, "Generated Data as of 10/8/2026 6:00:00 PM" (read from the screenshot).
# city key (as in prediction_log.csv) -> (station, AQI, remark). Pasig has no station in that table.
EMB_PM25 = {
    "Manila": ("Manila (Mehan Garden)", 60, "Fair"),
    "Quezon_City": ("Quezon City (Ateneo)", 58, "Fair"),
    "Caloocan": ("Caloocan (Univ. of East)", 82, "Fair"),
    "Valenzuela": ("Valenzuela (Punturin SHS)", 77, "Fair"),
    "Makati": ("Makati (Univ. of Makati)", 42, "Good"),
    "Mandaluyong": ("Mandaluyong (NCMH)", 54, "Fair"),
    "Navotas": ("Navotas (City Hall)", 89, "Fair"),
    "Pasay": ("Pasay (Phil. Airlines Cmpd.)", 47, "Good"),
    "San_Juan": ("San Juan City (Pinaglabanan)", 58, "Fair"),
}
EMB_TIME_PHT = datetime(2026, 10, 8, 18, 0, tzinfo=PHT)

DENR = [(25.0, "Good"), (35.0, "Fair"), (45.0, "Unhealthy for Sensitive Groups"),
        (55.0, "Very Unhealthy"), (90.0, "Acutely Unhealthy"), (float("inf"), "Emergency")]
BANDS = [(0, 25, 0, 50), (25.1, 35, 51, 100), (35.1, 45, 101, 150), (45.1, 55, 151, 200),
         (55.1, 90, 201, 300), (91, 500, 301, 500)]


def category(c: float) -> str:
    return next(name for top, name in DENR if c <= top)


def approx_aqi(c: float) -> int:
    for lo, hi, ilo, ihi in BANDS:
        if c <= hi:
            return round((ihi - ilo) / (hi - lo) * (max(c, lo) - lo) + ilo)
    return 500


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default="prediction_log.csv")
    ap.add_argument("--made-at", default=None,
                    help='when the forecast was made, PHT, "YYYY-MM-DD HH:MM" (default: EMB time minus 24 h)')
    ap.add_argument("--tolerance-hours", type=float, default=3.0,
                    help="max gap between the wanted time and the logged row (default 3)")
    ap.add_argument("--use-mean", action="store_true", help="use the 24 h mean instead of hour 24")
    args = ap.parse_args()

    if args.made_at:
        made_at = datetime.strptime(args.made_at, "%Y-%m-%d %H:%M").replace(tzinfo=PHT)
    else:
        made_at = EMB_TIME_PHT - timedelta(hours=24)
    print(f"EMB-DENR reading time : {EMB_TIME_PHT:%Y-%m-%d %H:%M} PHT")
    print(f"Forecast wanted from  : {made_at:%Y-%m-%d %H:%M} PHT (+/- {args.tolerance_hours:g} h), "
          f"{'24 h mean' if args.use_mean else 'hour-24 value'}, Variant C, live OpenAQ rows only\n")

    rows = []
    with open(args.log, newline="") as f:
        for r in csv.DictReader(f):
            src = (r.get("data_source") or "").lower()
            if "openaq" not in src or "backfilled hindcast" in src:
                continue
            try:
                t = datetime.fromisoformat(r["logged_at_utc"])
                preds = json.loads(r["variant_c_predicted_24h"])
            except (ValueError, KeyError, TypeError):
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            rows.append((r["city"], t, preds, r.get("input_hours_used")))

    header = f"{'City':<13}{'Logged (PHT)':<18}{'gap h':>6}{'C ug/m3':>9}{'C AQI~':>8}  {'C category':<30}{'EMB AQI':>8}  {'EMB':<6}  Match"
    print(header)
    print("-" * len(header))
    n = hit = within1 = 0
    order = [c[0] for c in DENR]
    for city, (station, emb_aqi, emb_cat) in EMB_PM25.items():
        cand = [(abs((t - made_at).total_seconds()) / 3600, t, p, h) for c, t, p, h in rows if c == city]
        if not cand:
            print(f"{city:<13}no live OpenAQ rows in the log")
            continue
        gap, t, preds, hrs = min(cand, key=lambda x: x[0])
        if gap > args.tolerance_hours:
            print(f"{city:<13}nearest live row is {gap:.1f} h away ({t.astimezone(PHT):%m-%d %H:%M}) -- skipped")
            continue
        val = sum(preds) / len(preds) if args.use_mean else preds[-1]
        cat = category(val)
        ok = cat == emb_cat
        n += 1
        hit += ok
        names = [x[1] for x in DENR]
        within1 += abs(names.index(cat) - names.index(emb_cat)) <= 1
        print(f"{city:<13}{t.astimezone(PHT):%m-%d %H:%M}{'':<5}{gap:>6.1f}{val:>9.1f}{approx_aqi(val):>8}  "
              f"{cat:<30}{emb_aqi:>8}  {emb_cat:<6}  {'yes' if ok else 'NO'}")

    print()
    if n:
        print(f"Same category as EMB-DENR: {hit}/{n} ({100 * hit / n:.0f}%)   within one category: {within1}/{n}")
    else:
        print("No city could be compared. Increase --tolerance-hours, or check that live OpenAQ rows "
              "were logged around that time (rows are written when the dashboard or map loads).")
    print("\nNotes: Pasig has no EMB PM2.5 station in that table, so it is left out. EMB's averaging time "
          "for the 'real time' value is not stated on the page. With n this small and EMB values all in "
          "Good/Fair, treat this as an illustration, not a result.")


if __name__ == "__main__":
    main()

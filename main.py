from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timedelta, timezone
from html import escape

import joblib
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

load_dotenv()

from src.models.variants import (
    VariantA_SingleBranchUnified,
    VariantB_DualBranchConcat,
    VariantC_AIRPRED,
    VariantD_PM25Only,
)

from live_data_geoscf_s3 import fetch_live_window_geoscf, scale_window as scale_window_geoscf
from live_data_openaq import fetch_live_window_openaq, scale_window as scale_window_openaq

CFG = yaml.safe_load(open("config.yaml"))
HORIZON = CFG["data"]["forecast_horizon"]
DEVICE = "cpu"


app = FastAPI(title="AIRPRED live inference")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

VARIANT_CLASSES = {
    "A": VariantA_SingleBranchUnified,
    "B": VariantB_DualBranchConcat,
    "C": VariantC_AIRPRED,
    "D": VariantD_PM25Only,
}


def build_model(name: str):
    if name != "C":
        return VARIANT_CLASSES[name](horizon=HORIZON)
    return VARIANT_CLASSES[name](
        horizon=HORIZON,
        d_model=CFG["model"]["d_model"],
        num_heads=CFG["model"]["num_attention_heads"],
    )



print("Loading models...")
MODELS = {}
for _name in VARIANT_CLASSES:
    _model = build_model(_name)
    _ckpt = torch.load(
        f"checkpoints/variant_{_name.lower()}_seed42_best.pt", map_location=DEVICE
    )
    _model.load_state_dict(_ckpt)
    _model.eval()
    MODELS[_name] = _model
print("Models loaded:", list(MODELS.keys()))

print("Loading scaler...")
SCALER = joblib.load("scaler.pkl")
PM25_IDX = list(SCALER.feature_names_in_).index("pm25")
PM25_MEAN = float(SCALER.mean_[PM25_IDX])
PM25_SCALE = float(SCALER.scale_[PM25_IDX])


def inverse_pm25(arr: np.ndarray) -> np.ndarray:
    return arr * PM25_SCALE + PM25_MEAN


print("Loading demo input windows...")
DEMO = np.load("demo_inputs.npz", allow_pickle=True)
CITY_INPUTS = {
    city: (DEMO["X_pm25"][i], DEMO["X_met"][i])
    for i, city in enumerate(DEMO["cities"].tolist())
}
print("Cities available:", list(CITY_INPUTS.keys()))


# --- prediction logging -----------------------------------------------------
# Logs every live prediction (GEOS-CF or OpenAQ) to prediction_log.csv so the
# adviser-requested "count of predicted values + real value for that day"
# tracking has something to read. actual_pm25/actual_source/actual_logged_at
# start blank and get filled in later via log_actual.py once a real reading
# for that city/day is known.
PREDICTION_LOG_PATH = os.path.join(os.path.dirname(__file__), "prediction_log.csv")
LOG_FIELDS = [
    "logged_at_utc", "city", "data_source", "input_hours_used",
    "variant_a_predicted_24h", "variant_b_predicted_24h",
    "variant_c_predicted_24h", "variant_d_predicted_24h",
    "actual_pm25", "actual_source", "actual_logged_at",
]


def log_prediction(city: str, data_source: str, input_hours_used: int, results: dict):
    file_exists = os.path.isfile(PREDICTION_LOG_PATH)
    with open(PREDICTION_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "city": city, "data_source": data_source, "input_hours_used": input_hours_used,
            "variant_a_predicted_24h": json.dumps(results["A"]["predicted"]),
            "variant_b_predicted_24h": json.dumps(results["B"]["predicted"]),
            "variant_c_predicted_24h": json.dumps(results["C"]["predicted"]),
            "variant_d_predicted_24h": json.dumps(results["D"]["predicted"]),
            "actual_pm25": "", "actual_source": "", "actual_logged_at": "",
        })


@app.get("/health")
def health():
    """Hit this once a few minutes before your defense to wake up a
    free-tier host that spins down after inactivity -- see README.md."""
    return {"status": "ok", "cities": list(CITY_INPUTS.keys())}


@app.get("/prediction_log")
def prediction_log(limit: int = 200, source: str | None = None):
    """Newest-first rows from prediction_log.csv, with each variant's 24h
    forecast summarized (mean + peak) so the dashboard can show predicted vs
    actual in a table. actual_* fields are null until fill_actuals.py (or
    log_actual.py) fills them in. Pass source=openaq to keep only the
    OpenAQ rows (live and backfilled)."""
    if not os.path.isfile(PREDICTION_LOG_PATH):
        return {"rows": []}
    with open(PREDICTION_LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if source:
        rows = [r for r in rows if r["data_source"].lower().startswith(source.lower())]
    # Newest prediction time first (backfilled rows are appended out of order).
    rows.sort(key=lambda r: datetime.fromisoformat(r["logged_at_utc"]), reverse=True)

    def summarize(raw: str):
        try:
            vals = json.loads(raw)
            return {"mean": round(sum(vals) / len(vals), 2), "peak": round(max(vals), 2)}
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    out = []
    for r in rows[:limit]:
        actual = r.get("actual_pm25") or ""
        out.append({
            "logged_at_utc": r["logged_at_utc"],
            "city": r["city"],
            "data_source": r["data_source"],
            "input_hours_used": int(r["input_hours_used"]) if r["input_hours_used"] else None,
            "variants": {
                "A": summarize(r["variant_a_predicted_24h"]),
                "B": summarize(r["variant_b_predicted_24h"]),
                "C": summarize(r["variant_c_predicted_24h"]),
                "D": summarize(r["variant_d_predicted_24h"]),
            },
            "actual_pm25": float(actual) if actual else None,
            "actual_source": r.get("actual_source") or None,
        })
    return {"rows": out}


PHT = timezone(timedelta(hours=8))  # Philippine time, for display only


@app.get("/prediction_log/table", response_class=HTMLResponse)
def prediction_log_table(limit: int = 300, source: str = "openaq"):
    """Browser-friendly predicted-vs-actual table. OpenAQ rows only by
    default; use ?source=all for every source, or ?limit=N to change length."""
    rows = prediction_log(limit=limit, source=None if source.lower() == "all" else source)["rows"]

    def fmt(v):
        return f"{v:.1f}" if v is not None else "&mdash;"

    errs = {v: [] for v in "ABCD"}
    body = []
    for r in rows:
        actual = r["actual_pm25"]
        means = {v: (r["variants"][v]["mean"] if r["variants"][v] else None) for v in "ABCD"}
        for v in "ABCD":
            if actual is not None and means[v] is not None:
                errs[v].append(abs(actual - means[v]))
        err_c = actual - means["C"] if actual is not None and means["C"] is not None else None
        when = datetime.fromisoformat(r["logged_at_utc"]).astimezone(PHT).strftime("%Y-%m-%d %H:%M")
        body.append(
            "<tr>"
            f"<td>{when}</td>"
            f"<td>{escape(r['city'].replace('_', ' '))}</td>"
            f"<td class='src'>{escape(r['data_source'])}</td>"
            + "".join(f"<td class='n'>{fmt(means[v])}</td>" for v in "ABCD")
            + f"<td class='n act' title=\"{escape(r['actual_source'] or '')}\">{fmt(actual)}</td>"
            f"<td class='n'>{('%+.1f' % err_c) if err_c is not None else '&mdash;'}</td>"
            "</tr>"
        )

    n_scored = len(errs["C"])
    if n_scored:
        mae = " &middot; ".join(
            f"{v}{' (AIRPRED)' if v == 'C' else ''} {sum(errs[v]) / len(errs[v]):.2f}"
            for v in "ABCD" if errs[v]
        )
        summary = f"MAE vs actual (n={n_scored}): {mae} &micro;g/m&sup3;"
    else:
        summary = "No rows have an actual value yet."

    rows_html = "".join(body) or "<tr><td colspan='10' class='empty'>No logged predictions yet.</td></tr>"
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIRPRED predicted vs actual</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1f2937; background: #f9fafb; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  p {{ font-size: 12px; color: #6b7280; margin: 2px 0 10px; }}
  table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 12px; }}
  th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; }}
  th {{ background: #f3f4f6; color: #4b5563; position: sticky; top: 0; }}
  td.n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.act {{ font-weight: 600; }}
  td.src {{ color: #6b7280; }}
  td.empty {{ text-align: center; color: #9ca3af; padding: 24px; }}
  .sum {{ font-weight: 600; color: #374151; }}
</style></head><body>
<h1>Predicted vs. actual PM2.5 ({escape(source)} rows)</h1>
<p>Predicted = each variant's 24-hour mean forecast. Actual = mean of the real hourly readings over the same 24 hours
(hover a value for its source). Times are Philippine time. Rows marked "backfilled hindcast" are retrospective tests;
all others were logged live. All values in &micro;g/m&sup3;; Error = actual &minus; C.</p>
<p class="sum">{summary}</p>
<table>
<thead><tr><th>Predicted at</th><th>City</th><th>Source</th><th>A</th><th>B</th><th>C (AIRPRED)</th><th>D</th><th>Actual</th><th>Error (C)</th></tr></thead>
<tbody>{rows_html}</tbody>
</table>
</body></html>"""
    return HTMLResponse(page)


@app.get("/predict")
def predict(city: str):
    if city not in CITY_INPUTS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown city: {city}. Available: {list(CITY_INPUTS.keys())}",
        )

    x_pm25_np, x_met_np = CITY_INPUTS[city]
    x_pm25 = torch.as_tensor(x_pm25_np, dtype=torch.float32).unsqueeze(0)  # (1, 48, 1)
    x_met = torch.as_tensor(x_met_np, dtype=torch.float32).unsqueeze(0)    # (1, 48, 7)

    results = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            y_hat = model(x_pm25, x_met)                    # genuine forward pass, right now
            y_hat_real = inverse_pm25(y_hat.numpy()[0])       # back to real µg/m³
            results[name] = {"predicted": [round(float(v), 2) for v in y_hat_real]}

    return {"city": city, "computed_live": True, "variants": results}

@app.get("/predict_live_geoscf")              # <-- new, add everything from here down
def predict_live_geoscf(city: str):
    if city not in CITY_INPUTS:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")

    try:
        x_pm25_raw, x_met_raw, n_hours, warnings = fetch_live_window_geoscf(city)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    x_pm25_raw = x_pm25_raw * 2.0

    x_pm25_scaled, x_met_scaled = scale_window_geoscf(x_pm25_raw, x_met_raw, SCALER)

    x_pm25 = torch.as_tensor(x_pm25_scaled, dtype=torch.float32).unsqueeze(0)
    x_met = torch.as_tensor(x_met_scaled, dtype=torch.float32).unsqueeze(0)

    results = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            y_hat = model(x_pm25, x_met)
            y_hat_real = inverse_pm25(y_hat.numpy()[0])
            results[name] = {"predicted": [round(float(v), 2) for v in y_hat_real]}

    log_prediction(city, "NASA GEOS-CF v2 (S3, live)", n_hours, results)

    return {
        "city": city,
        "computed_live": True,
        "data_source": "NASA GEOS-CF v2 (S3, live)",
        "input_hours_used": n_hours,
        "note": "Model validated on 48-hour windows; this run used reduced "
                f"context ({n_hours}h) because that's what the current GEOS-CF "
                "forecast cycle has available right now.",
        "warnings": warnings,
        "variants": results,
    }


@app.get("/predict_live_openaq")
def predict_live_openaq(city: str):
    if city not in CITY_INPUTS:
        raise HTTPException(status_code=404, detail=f"Unknown city: {city}")

    try:
        x_pm25_raw, x_met_raw, n_hours, warnings = fetch_live_window_openaq(city)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    x_pm25_scaled, x_met_scaled = scale_window_openaq(x_pm25_raw, x_met_raw, SCALER)

    x_pm25 = torch.as_tensor(x_pm25_scaled, dtype=torch.float32).unsqueeze(0)
    x_met = torch.as_tensor(x_met_scaled, dtype=torch.float32).unsqueeze(0)

    results = {}
    with torch.no_grad():
        for name, model in MODELS.items():
            y_hat = model(x_pm25, x_met)
            y_hat_real = inverse_pm25(y_hat.numpy()[0])
            results[name] = {"predicted": [round(float(v), 2) for v in y_hat_real]}

    log_prediction(city, "OpenAQ (ground sensor)", n_hours, results)

    return {
        "city": city,
        "computed_live": True,
        "data_source": "OpenAQ (ground sensor)",
        "input_hours_used": n_hours,
        "note": f"PM2.5 input from a real OpenAQ ground sensor (not a model "
                f"estimate); this run used {n_hours}h of context.",
        "warnings": warnings,
        "variants": results,
    }
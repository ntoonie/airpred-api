"""Live inference API for the AIRPRED demo.

Loads all four trained checkpoints ONCE at startup, then computes a genuine
forward pass through each of them on every /predict request -- this is real
inference, not a lookup into precomputed results.

RUN LOCALLY:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
    # then: curl "http://localhost:8000/predict?city=Manila"

FILES THIS FOLDER NEEDS BEFORE IT WILL START (see README.md for exact steps):
    checkpoints/variant_a_seed42_best.pt
    checkpoints/variant_b_seed42_best.pt
    checkpoints/variant_c_seed42_best.pt
    checkpoints/variant_d_seed42_best.pt
    scaler.pkl
    demo_inputs.npz          <- from export_demo_inputs.py
    config.yaml              <- copy of airpred-ml's configs/config.yaml
    src/models/{tcn.py, attention.py, variants.py, __init__.py}
    src/__init__.py
"""
from __future__ import annotations

import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models.variants import (
    VariantA_SingleBranchUnified,
    VariantB_DualBranchConcat,
    VariantC_AIRPRED,
    VariantD_PM25Only,
)

CFG = yaml.safe_load(open("config.yaml"))
HORIZON = CFG["data"]["forecast_horizon"]
DEVICE = "cpu"  # CPU is fine here -- this model is small; one forward pass
                # over a single 48-hour window is milliseconds, no GPU needed
                # for a demo serving one visitor at a time.

app = FastAPI(title="AIRPRED live inference")

# CORS: lets the deployed Next.js frontend call this API from the browser.
# "*" is fine to get the demo working; once you have your actual Vercel URL,
# narrow this to allow_origins=["https://your-app.vercel.app"] instead.
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


# ---- Load everything ONCE at process startup, not per-request ----
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


@app.get("/health")
def health():
    """Hit this once a few minutes before your defense to wake up a
    free-tier host that spins down after inactivity -- see README.md."""
    return {"status": "ok", "cities": list(CITY_INPUTS.keys())}


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

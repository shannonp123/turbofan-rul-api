import os, json
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from rul_model import predict_rul, SENSORS, L

BASE = os.path.dirname(os.path.abspath(__file__))
EX = json.load(open(os.path.join(BASE, "examples.json")))
INDEX = open(os.path.join(BASE, "index.html"), encoding="utf-8").read()

app = FastAPI(title="Turbofan RUL Prediction API", version="1.0",
              description="Predicts Remaining Useful Life (cycles) of a jet engine from its recent sensor readings (LSTM).")

class PredictRequest(BaseModel):
    engine_id: Union[int, str] = Field(..., description="Engine identifier")
    readings: List[Dict[str, float]] = Field(..., min_length=1,
        description="Recent cycles, oldest first. Each item needs the sensors: " + ", ".join(SENSORS))

class PredictResponse(BaseModel):
    engine_id: Union[int, str]
    predicted_RUL: float
    zone: str
    recommendation: str
    cycles_received: int
    warning: Optional[str] = None
    model: str = "LSTM (30-cycle window)"

@app.get("/health")
def health():
    return {"status": "ok", "model": "LSTM", "window": L, "required_sensors": SENSORS}

@app.get("/example/{unit}")
def example(unit: int):
    e = EX["engines"].get(str(unit))
    if e is None:
        raise HTTPException(status_code=404, detail="Engine must be between 1 and 100")
    return {"engine_id": unit, "true_RUL": e["true_RUL"],
            "readings": [dict(zip(EX["sensors"], row)) for row in e["values"]]}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    rows = []
    for i, r in enumerate(req.readings):
        missing = [s for s in SENSORS if s not in r]
        if missing:
            raise HTTPException(status_code=422, detail=f"Reading {i} is missing sensors: {missing}")
        rows.append([r[s] for s in SENSORS])
    p, n, warning = predict_rul(rows)
    p = round(p, 1)
    if p <= 30:
        zone, rec = "Urgent", "Schedule servicing immediately."
    elif p <= 60:
        zone, rec = "Soon", "Plan servicing in the coming flights."
    else:
        zone, rec = "Safe", "No servicing needed now; keep monitoring."
    return PredictResponse(engine_id=req.engine_id, predicted_RUL=p, zone=zone,
                           recommendation=rec, cycles_received=n, warning=warning)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return INDEX

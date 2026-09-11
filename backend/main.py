"""
main.py

FastAPI Web Service for Food Delivery Time Prediction.
Endpoints:
  - GET  /health           -> Service health & model status
  - POST /predict          -> Main prediction with 6 user fields & auto-derived features
  - GET  /api/autocomplete -> Nominatim search proxy for frontend address autocomplete
"""

import json, time, logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import httpx
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    MODEL_PATH,
    REQUEST_LOG_PATH,
    NOMINATIM_URL,
    NOMINATIM_USER_AGENT,
    HTTP_TIMEOUT
)
from predictor import get_model, predict_delivery

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("delivery_predictor.api")

app = FastAPI(
    title="Food Delivery Time Prediction API",
    description="Production-grade FastAPI service utilizing Stacking Ensemble ML for real-time delivery estimation.",
    version="1.0.0"
)

# Enable CORS for local and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response schemas
class PredictRequest(BaseModel):
    from_address: str = Field(..., description="Pickup restaurant address or landmark")
    to_address: str = Field(..., description="Customer delivery drop address or landmark")
    vehicle_type: str = Field("motorcycle", description="One of: motorcycle, scooter, electric_scooter, bicycle")
    vehicle_condition: int = Field(1, ge=0, le=3, description="Vehicle condition integer (0, 1, 2, or 3)")
    type_of_order: str = Field("Meal", description="One of: Snack, Meal, Drinks, Buffet")
    multiple_deliveries: int = Field(1, ge=0, le=3, description="Number of concurrent deliveries (0, 1, 2, 3)")
    pickup_lat: Optional[float] = None
    pickup_lon: Optional[float] = None
    drop_lat: Optional[float] = None
    drop_lon: Optional[float] = None


class PredictResponse(BaseModel):
    predicted_minutes: float
    weather: str
    traffic_level: str
    distance_km: float
    osrm_road_distance_km: Optional[float] = None
    osrm_duration_min: Optional[float] = None
    route_geometry: Optional[Dict[str, Any]] = None
    fallbacks_used: List[str]
    breakdown: Dict[str, Any]


def log_transaction(req_payload: Dict[str, Any], res_payload: Dict[str, Any], duration_ms: float):
    """Log prediction request/response to local JSONL file for debugging and monitoring."""
    try:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round(duration_ms, 2),
            "request": req_payload,
            "response": res_payload
        }
        with open(REQUEST_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.error(f"Failed to log transaction: {e}")


@app.on_event("startup")
async def startup_event():
    """Warm up the model artifact at server startup."""
    try:
        model = get_model()
        logger.info(f"Startup complete. Loaded model: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Model startup error: {e}")


@app.get("/health")
async def health():
    """Healthcheck endpoint verifying model readiness."""
    try:
        model = get_model()
        return {
            "status": "ok",
            "model_loaded": model is not None,
            "model_path": MODEL_PATH,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "model_loaded": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Accepts the 6 user-provided fields, derives all external data via live APIs,
    and returns predicted delivery minutes + full explanatory breakdown.
    """
    t0 = time.time()
    req_dict = req.dict()
    try:
        result = await predict_delivery(req_dict)
        dur_ms = (time.time() - t0) * 1000.0
        log_transaction(req_dict, result, dur_ms)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction pipeline error: {str(e)}")


@app.get("/api/autocomplete")
async def autocomplete(q: str = Query(..., min_length=2)):
    """
    Address autocomplete proxy using OpenStreetMap Nominatim.
    Biased towards India for optimal relevance.
    """
    try:
        url = f"{NOMINATIM_URL}/search"
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        params = {
            "q": q,
            "format": "json",
            "countrycodes": "in",
            "limit": 5,
            "addressdetails": 1
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                results = []
                for item in resp.json():
                    results.append({
                        "display_name": item.get("display_name", ""),
                        "lat": float(item.get("lat", 0.0)),
                        "lon": float(item.get("lon", 0.0))
                    })
                return results
    except Exception as e:
        logger.warning(f"Autocomplete proxy failed: {e}")

    return []

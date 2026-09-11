"""
config.py

Configuration settings for FastAPI delivery prediction backend.
Reads configuration from environment variables or defaults.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file from backend/ directory if present (pure Python fallback)
ENV_PATH = Path(__file__).resolve().parent / ".env"
if ENV_PATH.is_file():
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
    except Exception:
        pass

# Path to delivery_time_predictor.pkl (loaded by reference, read-only)
_nb_model = BASE_DIR / "notebooks" / "delivery_time_predictor.pkl"
_root_model = BASE_DIR / "delivery_time_predictor.pkl"
DEFAULT_MODEL_PATH = str(_nb_model if _nb_model.is_file() else _root_model)
MODEL_PATH = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)

# OpenWeatherMap API Key (Free tier)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# External APIs
NOMINATIM_URL = os.getenv("NOMINATIM_URL", "https://nominatim.openstreetmap.org")
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "FoodDeliveryPredictionDemoApp/1.0 (contact: demo-student@outlook.com)")

OSRM_ROUTE_URL = os.getenv("OSRM_ROUTE_URL", "https://router.project-osrm.org/route/v1/driving")

# Logging
LOG_DIR = BASE_DIR / "backend" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_LOG_PATH = str(LOG_DIR / "requests.jsonl")

# Network timeout in seconds for external APIs
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "4.0"))

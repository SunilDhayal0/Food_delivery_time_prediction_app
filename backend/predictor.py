"""
predictor.py

Feature assembly and live prediction service.
Derives all required features using live APIs (Nominatim, OSRM, OpenWeather)
with graceful fallbacks and detailed diagnostic breakdown.
"""

import sys, os, time, math, logging, json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import httpx
import joblib
import numpy as np
import pandas as pd

# Add D:\testing to sys.path so clean_data_utils is available for unpickling
PARENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PARENT_DIR))
import clean_data_utils

from config import (
    MODEL_PATH,
    OPENWEATHER_API_KEY,
    NOMINATIM_URL,
    NOMINATIM_USER_AGENT,
    OSRM_ROUTE_URL,
    HTTP_TIMEOUT
)
from mappings import (
    map_openweather_condition,
    map_wmo_weather_code,
    map_reverse_geocode_to_city,
    extract_city_code,
    compute_traffic_density,
    fallback_traffic_density,
    is_festival_date,
    MEDIAN_DELIVERY_PERSON_AGE,
    MEDIAN_DELIVERY_PERSON_RATING
)

logger = logging.getLogger("delivery_predictor.service")

# Global singleton predictor
_PREDICTOR_MODEL = None

def get_model():
    """Load the trained DeliveryTimePredictor from the notebooks folder by reference."""
    global _PREDICTOR_MODEL
    if _PREDICTOR_MODEL is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        logger.info(f"Loading delivery_time_predictor from: {MODEL_PATH}")
        _PREDICTOR_MODEL = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    return _PREDICTOR_MODEL


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine distance in kilometers matching training data formula."""
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371.0 * c


async def geocode_address(client: httpx.AsyncClient, address: str) -> Tuple[Optional[float], Optional[float], Dict[str, Any], bool]:
    """Geocode an address string using Nominatim. Returns: (lat, lon, address_details, is_fallback)"""
    if not address or not address.strip():
        return 12.9716, 77.5946, {}, True  # Default to Bengaluru center

    try:
        url = f"{NOMINATIM_URL}/search"
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        params = {"q": address, "format": "json", "addressdetails": 1, "limit": 1}
        resp = await client.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200 and resp.json():
            data = resp.json()[0]
            lat = float(data["lat"])
            lon = float(data["lon"])
            details = data.get("address", {})
            return lat, lon, details, False
    except Exception as e:
        logger.warning(f"Geocoding failed for '{address}': {e}. Using fallback coordinates.")

    # Fallback coordinates (central urban coordinates in India)
    return 12.9716, 77.5946, {}, True


async def reverse_geocode_coords(client: httpx.AsyncClient, lat: float, lon: float) -> Tuple[Dict[str, Any], bool]:
    """Reverse-geocode latitude/longitude into address details using Nominatim."""
    try:
        url = f"{NOMINATIM_URL}/reverse"
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        params = {"lat": lat, "lon": lon, "format": "json"}
        resp = await client.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200 and resp.json():
            return resp.json().get("address", {}), False
    except Exception as e:
        logger.warning(f"Reverse geocode failed for ({lat}, {lon}): {e}")

    return {}, True


async def get_osrm_route(
    client: httpx.AsyncClient,
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> Tuple[float, float, Optional[Dict[str, Any]], bool]:
    """
    Query OSRM for driving distance (meters), duration (seconds), and route geometry.
    Returns: (duration_sec, distance_meters, route_geometry_geojson, is_fallback)
    """
    try:
        url = f"{OSRM_ROUTE_URL}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        resp = await client.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                route = data["routes"][0]
                geom = route.get("geometry")
                return float(route["duration"]), float(route["distance"]), geom, False
    except Exception as e:
        logger.warning(f"OSRM routing failed: {e}. Falling back to Haversine speed estimation.")

    # Fallback: estimate from Haversine distance assuming 30 km/h average speed
    dist_km = haversine_distance_km(lat1, lon1, lat2, lon2)
    dist_m = dist_km * 1000.0
    duration_s = (dist_km / 30.0) * 3600.0
    return duration_s, dist_m, None, True


async def get_live_weather(
    client: httpx.AsyncClient,
    lat: float,
    lon: float
) -> Tuple[str, Dict[str, Any], bool]:
    """
    Query OpenWeatherMap API for current weather at coordinates.
    Returns: (weather_condition_string, raw_weather_info, is_fallback)
    """
    # 1. Primary Option: If OPENWEATHER_API_KEY is supplied, query OpenWeatherMap
    if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY.strip():
        try:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHER_API_KEY.strip(),
                "units": "metric"
            }
            resp = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                weather_item = data.get("weather", [{}])[0]
                wid = weather_item.get("id", 800)
                main = weather_item.get("main", "Clear")
                desc = weather_item.get("description", "clear sky")
                wind_speed_kmh = float(data.get("wind", {}).get("speed", 0.0)) * 3.6
                temp_c = data.get("main", {}).get("temp", 25.0)

                condition_str, is_clean = map_openweather_condition(wid, main, desc, wind_speed_kmh)
                info = {
                    "provider": "OpenWeatherMap Live API",
                    "id": wid,
                    "main": main,
                    "description": desc,
                    "temperature_c": round(temp_c, 1),
                    "wind_speed_kmh": round(wind_speed_kmh, 1),
                    "clean_match": is_clean
                }
                return condition_str, info, False
        except Exception as e:
            logger.warning(f"OpenWeather API call failed: {e}. Trying Open-Meteo.")

    # 2. Keyless Real-Time Live Weather API: Open-Meteo (Global, free, no API key required)
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "current": "weather_code,temperature_2m,wind_speed_10m"
        }
        resp = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            current = data.get("current", {})
            wmo_code = int(current.get("weather_code", 0))
            temp_c = float(current.get("temperature_2m", 28.0))
            wind_kmh = float(current.get("wind_speed_10m", 10.0))

            condition_str, label = map_wmo_weather_code(wmo_code, wind_kmh)
            info = {
                "provider": "Open-Meteo Live Satellite/Station API (Keyless)",
                "wmo_code": wmo_code,
                "condition": label,
                "temperature_c": round(temp_c, 1),
                "wind_speed_kmh": round(wind_kmh, 1),
                "clean_match": True
            }
            return condition_str, info, False
    except Exception as e:
        logger.warning(f"Open-Meteo live weather query failed: {e}. Using seasonal fallback.")

    # 3. Last-resort fallback: Heuristic based on current hour and season in India
    now = datetime.now()
    month = now.month
    if month in (7, 8):  # Monsoon
        fallback_cond = "conditions Stormy"
    elif month in (12, 1):  # Winter fog
        fallback_cond = "conditions Fog"
    elif 6 <= now.hour <= 17:
        fallback_cond = "conditions Sunny"
    else:
        fallback_cond = "conditions Cloudy"

    info = {
        "provider": "Fallback (Climate/Hour Heuristic)",
        "inferred_from": f"Month {month}, Hour {now.hour}",
        "temperature_c": 28.0,
        "wind_speed_kmh": 12.0
    }
    return fallback_cond, info, True


async def predict_delivery(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction pipeline:
    1. Geocode pickup & drop
    2. Query OSRM for traffic ratio
    3. Query OpenWeather for weather
    4. Reverse geocode pickup for City type
    5. Derive temporal and distance features
    6. Run Stacking Ensemble model
    7. Return formatted prediction + diagnostic breakdown
    """
    model = get_model()
    now = datetime.now()

    # User input extraction
    from_addr = payload.get("from_address", "")
    to_addr = payload.get("to_address", "")
    pickup_lat_in = payload.get("pickup_lat")
    pickup_lon_in = payload.get("pickup_lon")
    drop_lat_in = payload.get("drop_lat")
    drop_lon_in = payload.get("drop_lon")

    vehicle_type = str(payload.get("vehicle_type", "motorcycle")).strip().lower()
    vehicle_condition = int(payload.get("vehicle_condition", 1))
    type_of_order = str(payload.get("type_of_order", "Meal")).strip().capitalize() + " "
    multiple_deliveries = int(payload.get("multiple_deliveries", 1))

    fallbacks_used = []

    async with httpx.AsyncClient() as client:
        # 1. Coordinates derivation
        if pickup_lat_in is not None and pickup_lon_in is not None:
            pickup_lat, pickup_lon = float(pickup_lat_in), float(pickup_lon_in)
            pickup_addr_details, _ = await reverse_geocode_coords(client, pickup_lat, pickup_lon)
            pickup_geo_fb = False
        else:
            pickup_lat, pickup_lon, pickup_addr_details, pickup_geo_fb = await geocode_address(client, from_addr)
            if pickup_geo_fb:
                fallbacks_used.append("pickup_geocoding")

        if drop_lat_in is not None and drop_lon_in is not None:
            drop_lat, drop_lon = float(drop_lat_in), float(drop_lon_in)
            drop_geo_fb = False
        else:
            drop_lat, drop_lon, _, drop_geo_fb = await geocode_address(client, to_addr)
            if drop_geo_fb:
                fallbacks_used.append("drop_geocoding")

        # 2. OSRM Route & Traffic Density
        osrm_duration_s, osrm_distance_m, osrm_geom, osrm_fb = await get_osrm_route(client, pickup_lat, pickup_lon, drop_lat, drop_lon)
        if osrm_fb:
            traffic_density = fallback_traffic_density(now)
            traffic_ratio = 1.5
            fallbacks_used.append("traffic_osrm_fallback_to_time_of_day")
        else:
            traffic_density, traffic_ratio = compute_traffic_density(osrm_duration_s, osrm_distance_m, vehicle_type)

        # 3. Live Weather
        weather_condition, weather_info, weather_fb = await get_live_weather(client, pickup_lat, pickup_lon)
        if weather_fb:
            fallbacks_used.append("weather_openweather_fallback")

        # 4. City Type and City Code
        city_type = map_reverse_geocode_to_city(pickup_addr_details)
        city_code = extract_city_code(pickup_addr_details)

    # 5. Computed Distance & Timestamps
    haversine_km = haversine_distance_km(pickup_lat, pickup_lon, drop_lat, drop_lon)
    festival_str = is_festival_date(now)

    # Rider ID prefix from city_code
    rider_id = f"{city_code}RES19DEL01"

    order_date_str = now.strftime("%d-%m-%Y")
    order_time_str = now.strftime("%H:%M:%S")
    # Average kitchen prep delay ~12 minutes
    picked_time_str = (now + timedelta(minutes=12)).strftime("%H:%M:%S")

    # Construct the single raw row DataFrame expected by DeliveryTimePredictor
    raw_order_df = pd.DataFrame([{
        "ID": "0xLIVE",
        "Delivery_person_ID": rider_id,
        "Delivery_person_Age": MEDIAN_DELIVERY_PERSON_AGE,          # Imputed median from training data
        "Delivery_person_Ratings": MEDIAN_DELIVERY_PERSON_RATING,    # Imputed median from training data
        "Restaurant_latitude": pickup_lat,
        "Restaurant_longitude": pickup_lon,
        "Delivery_location_latitude": drop_lat,
        "Delivery_location_longitude": drop_lon,
        "Order_Date": order_date_str,
        "Time_Orderd": order_time_str,
        "Time_Order_picked": picked_time_str,
        "Weatherconditions": weather_condition,
        "Road_traffic_density": traffic_density,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": type_of_order,
        "Type_of_vehicle": f"{vehicle_type} ",
        "multiple_deliveries": multiple_deliveries,
        "Festival": festival_str,
        "City": city_type
    }])

    # 6. Execute model inference
    predicted_arr = model.predict(raw_order_df)
    predicted_minutes = float(predicted_arr[0])

    # 7. Assemble comprehensive breakdown
    return {
        "predicted_minutes": round(max(5.0, predicted_minutes), 1),
        "weather": weather_condition.replace("conditions ", "").strip(),
        "traffic_level": traffic_density.strip(),
        "distance_km": round(haversine_km, 2),
        "osrm_road_distance_km": round(osrm_distance_m / 1000.0, 2),
        "osrm_duration_min": round(osrm_duration_s / 60.0, 1),
        "route_geometry": osrm_geom,
        "fallbacks_used": fallbacks_used,
        "breakdown": {
            "pickup_coordinates": {"lat": round(pickup_lat, 5), "lon": round(pickup_lon, 5)},
            "drop_coordinates": {"lat": round(drop_lat, 5), "lon": round(drop_lon, 5)},
            "weather_condition": weather_condition,
            "weather_details": weather_info,
            "traffic_density": traffic_density.strip(),
            "traffic_ratio": round(traffic_ratio, 2),
            "city_classification": city_type.strip(),
            "city_code": city_code,
            "vehicle_type": vehicle_type,
            "vehicle_condition": vehicle_condition,
            "type_of_order": type_of_order.strip(),
            "multiple_deliveries": multiple_deliveries,
            "festival": festival_str.strip(),
            "order_date": order_date_str,
            "order_time": order_time_str,
            "estimated_pickup_time": picked_time_str,
            "imputed_rider_age": MEDIAN_DELIVERY_PERSON_AGE,
            "imputed_rider_rating": MEDIAN_DELIVERY_PERSON_RATING,
        }
    }

"""
mappings.py

Explicit mapping tables and classification logic for auto-derived features.
Target category strings match ground truth in training data exactly, character-for-character,
including quirks, prefixes, and trailing spaces.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Tuple

logger = logging.getLogger('delivery_predictor.mappings')

# ===========================================================================
# 1. Weather Conditions Mapping
# ===========================================================================
# Exact training strings:
# 'conditions Cloudy', 'conditions Fog', 'conditions Sandstorms',
# 'conditions Stormy', 'conditions Sunny', 'conditions Windy'

VALID_WEATHER_CATEGORIES = {
    'conditions Cloudy',
    'conditions Fog',
    'conditions Sandstorms',
    'conditions Stormy',
    'conditions Sunny',
    'conditions Windy',
}

def map_openweather_condition(
    weather_id: int,
    main_text: str = '',
    description: str = '',
    wind_speed_kmh: float = 0.0
) -> Tuple[str, bool]:
    """
    Map OpenWeather condition code / text to exact training Weatherconditions.
    Returns: (mapped_condition, is_clean_match)
    """
    main_lower = (main_text or '').strip().lower()
    desc_lower = (description or '').strip().lower()

    # Extreme wind check: if strong winds (> 28 km/h), prioritize Windy
    if wind_speed_kmh >= 28.0:
        return 'conditions Windy', True

    # 1. Check exact OpenWeather ID ranges
    # 2xx: Thunderstorms -> Stormy
    if 200 <= weather_id <= 232:
        return 'conditions Stormy', True

    # 3xx: Drizzle -> Stormy
    if 300 <= weather_id <= 321:
        return 'conditions Stormy', True

    # 5xx: Rain -> Stormy
    if 500 <= weather_id <= 531:
        return 'conditions Stormy', True

    # 6xx: Snow / Sleet -> Cloudy
    if 600 <= weather_id <= 622:
        return 'conditions Cloudy', True

    # 7xx: Atmosphere
    if weather_id in (701, 741, 721):  # Mist, Fog, Haze
        return 'conditions Fog', True
    if weather_id in (711, 731, 751, 761, 762):  # Smoke, Dust/Sand whirls, Sand, Dust, Ash
        return 'conditions Sandstorms', True
    if weather_id in (771, 781):  # Squalls, Tornado
        return 'conditions Windy', True

    # 800: Clear -> Sunny
    if weather_id == 800:
        return 'conditions Sunny', True

    # 801-804: Clouds -> Cloudy
    if 801 <= weather_id <= 804:
        return 'conditions Cloudy', True

    # 2. Text-based fallback if code was unknown / zero
    if any(w in desc_lower or w in main_lower for w in ['storm', 'thunder', 'rain', 'drizzle', 'shower', 'downpour']):
        return 'conditions Stormy', True
    if any(w in desc_lower or w in main_lower for w in ['fog', 'mist', 'haze']):
        return 'conditions Fog', True
    if any(w in desc_lower or w in main_lower for w in ['sand', 'dust', 'smoke', 'ash']):
        return 'conditions Sandstorms', True
    if any(w in desc_lower or w in main_lower for w in ['wind', 'squall', 'tornado', 'gale', 'breeze']):
        return 'conditions Windy', True
    if any(w in desc_lower or w in main_lower for w in ['cloud', 'overcast', 'partly']):
        return 'conditions Cloudy', True
    if any(w in desc_lower or w in main_lower for w in ['clear', 'sun']):
        return 'conditions Sunny', True

    # Unmapped code -> log explicitly and fall back to Sunny
    logger.warning(
        f'Unmapped weather condition: id={weather_id}, main="{main_text}", desc="{description}". '
        'Defaulting to "conditions Sunny".'
    )
    return 'conditions Sunny', False


def map_wmo_weather_code(wmo_code: int, wind_speed_kmh: float = 0.0) -> Tuple[str, str]:
    """
    Map WMO weather code (Open-Meteo standard) to exact training Weatherconditions.
    Returns: (mapped_condition, condition_label)
    WMO codes:
      0: Clear sky -> Sunny
      1, 2, 3: Mainly clear, partly cloudy, and overcast -> Cloudy
      45, 48: Fog and depositing rime fog -> Fog
      51, 53, 55: Drizzle -> Stormy
      61, 63, 65: Rain -> Stormy
      71, 73, 75: Snow fall -> Cloudy
      80, 81, 82: Rain showers -> Stormy
      85, 86: Snow showers -> Cloudy
      95, 96, 99: Thunderstorm -> Stormy
    """
    if wind_speed_kmh >= 28.0:
        return 'conditions Windy', 'Windy'

    if wmo_code == 0:
        return 'conditions Sunny', 'Clear / Sunny'
    if wmo_code in (1, 2, 3):
        return 'conditions Cloudy', 'Partly Cloudy' if wmo_code in (1, 2) else 'Overcast'
    if wmo_code in (45, 48):
        return 'conditions Fog', 'Foggy / Hazy'
    if wmo_code in (51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99):
        return 'conditions Stormy', 'Rain / Storm'
    if wmo_code in (71, 73, 75, 77, 85, 86):
        return 'conditions Cloudy', 'Snow / Overcast'

    return 'conditions Sunny', 'Clear'


# ===========================================================================
# 2. City Type (Classification from Reverse Geocoding)
# ===========================================================================
# Exact training strings: 'Metropolitian ', 'Urban ', 'Semi-Urban '
# Note the source typo: 'Metropolitian ' (with an 'i' before an)

METROPOLITAN_CITIES = {
    'mumbai', 'bombay', 'delhi', 'new delhi', 'bengaluru', 'bangalore',
    'hyderabad', 'chennai', 'madras', 'kolkata', 'calcutta', 'pune',
    'ahmedabad', 'jaipur', 'surat', 'lucknow', 'kanpur', 'nagpur',
    'indore', 'thane', 'bhopal', 'visakhapatnam', 'patna', 'vadodara',
    'ghaziabad', 'ludhiana', 'agra', 'nashik', 'faridabad', 'meerut',
    'rajkot', 'varanasi', 'srinagar', 'aurangabad', 'dhanbad', 'amritsar',
    'navi mumbai', 'allahabad', 'prayagraj', 'howrah', 'ranchi', 'gwalior',
    'jabalpur', 'coimbatore', 'vijayawada', 'jodhpur', 'madurai', 'raipur',
    'kota', 'chandigarh', 'guwahati', 'solapur', 'mysore', 'mysuru',
    'gurgaon', 'gurugram', 'noida', 'kochi', 'cochin', 'dehradun'
}

# Training city prefixes mapping
CITY_NAME_PREFIX_MAP = {
    'bangalore': 'BANG', 'bengaluru': 'BANG',
    'mumbai': 'MUM', 'bombay': 'MUM', 'navi mumbai': 'MUM', 'thane': 'MUM',
    'delhi': 'DEH', 'new delhi': 'DEH', 'noida': 'DEH', 'gurgaon': 'DEH', 'gurugram': 'DEH',
    'hyderabad': 'HYD', 'secunderabad': 'HYD',
    'chennai': 'CHEN', 'madras': 'CHEN',
    'kolkata': 'KOL', 'calcutta': 'KOL', 'howrah': 'KOL',
    'pune': 'PUNE',
    'jaipur': 'JAP',
    'surat': 'SUR',
    'indore': 'INDO',
    'bhopal': 'BHP',
    'vadodara': 'VAD',
    'coimbatore': 'COIMB',
    'kochi': 'KOC', 'cochin': 'KOC',
    'ludhiana': 'LUDH',
    'agra': 'AGR',
    'allahabad': 'ALH', 'prayagraj': 'ALH',
    'aurangabad': 'AURG',
    'dehradun': 'DEH',
    'goa': 'GOA', 'panaji': 'GOA',
    'kanpur': 'KNP',
    'mysore': 'MYS', 'mysuru': 'MYS',
    'ranchi': 'RANCHI'
}

def map_reverse_geocode_to_city(address: Dict[str, Any]) -> str:
    """
    Classify reverse-geocoded place into 'Metropolitian ', 'Urban ', or 'Semi-Urban '.
    """
    if not address:
        return 'Metropolitian '

    city = (address.get('city') or address.get('state_district') or '').strip().lower()
    town = (address.get('town') or address.get('municipality') or address.get('suburb') or '').strip().lower()
    village = (address.get('village') or address.get('hamlet') or address.get('isolated_dwelling') or '').strip().lower()

    if city and (city in METROPOLITAN_CITIES or any(m in city for m in METROPOLITAN_CITIES)):
        return 'Metropolitian '
    if town and (town in METROPOLITAN_CITIES or any(m in town for m in METROPOLITAN_CITIES)):
        return 'Metropolitian '

    # If city is present but not in top metro set, it is Urban
    if city or town:
        return 'Urban '

    # If it is clearly rural / village / hamlet
    if village or address.get('county') or address.get('road'):
        return 'Semi-Urban '

    return 'Metropolitian '


def extract_city_code(address: Dict[str, Any]) -> str:
    """
    Extract a valid training rider city prefix (e.g. 'BANG', 'MUM', 'JAP').
    """
    if not address:
        return 'BANG'

    for field in ['city', 'town', 'municipality', 'state_district', 'state']:
        val = (address.get(field) or '').strip().lower()
        if not val:
            continue
        for name, code in CITY_NAME_PREFIX_MAP.items():
            if name in val:
                return code

    return 'BANG'


# ===========================================================================
# 3. Traffic Density Computation & Empirical Fallback
# ===========================================================================
# Exact training strings: 'Low ', 'Medium ', 'High ', 'Jam '

FREE_FLOW_SPEED_KMH = {
    'bicycle': 15.0,
    'electric_scooter': 25.0,
    'scooter': 35.0,
    'motorcycle': 40.0,
}

def compute_traffic_density(
    osrm_duration_sec: float,
    osrm_distance_meters: float,
    vehicle_type: str = 'motorcycle'
) -> Tuple[str, float]:
    """
    Determine traffic density via ratio of actual OSRM duration to free-flow duration.
    Returns: (density_string, traffic_ratio)
    """
    free_speed_kmh = FREE_FLOW_SPEED_KMH.get(vehicle_type.strip().lower(), 35.0)
    free_speed_mps = free_speed_kmh / 3.6  # m/s

    if osrm_distance_meters <= 0:
        return 'Low ', 1.0

    free_flow_duration_sec = osrm_distance_meters / free_speed_mps
    ratio = max(0.5, osrm_duration_sec / max(free_flow_duration_sec, 1.0))

    # Tuned thresholds based on urban congestion indices:
    if ratio < 1.25:
        return 'Low ', ratio
    elif ratio < 1.65:
        return 'Medium ', ratio
    elif ratio < 2.25:
        return 'High ', ratio
    else:
        return 'Jam ', ratio


def fallback_traffic_density(dt: datetime = None) -> str:
    """
    Empirical fallback table built from food_delivery_data.csv Road_traffic_density vs Time_Orderd:
      Hours 00-10: Low
      Hours 11-14: High  (Lunch rush peak)
      Hours 15-18: Medium (Afternoon lull)
      Hours 19-21: Jam   (Dinner rush peak)
      Hours 22-23: Low   (Late night)
    """
    now = dt or datetime.now()
    hour = now.hour
    if hour <= 10 or hour >= 22:
        return 'Low '
    elif 11 <= hour <= 14:
        return 'High '
    elif 15 <= hour <= 18:
        return 'Medium '
    else:  # 19 <= hour <= 21
        return 'Jam '


# ===========================================================================
# 4. Festival Detection
# ===========================================================================
# Exact training strings: 'Yes ', 'No '

INDIAN_FESTIVALS_MONTH_DAY = {
    (1, 14),   # Makar Sankranti / Pongal
    (1, 26),   # Republic Day
    (2, 15),   # Maha Shivaratri
    (3, 3),    # Holi (Chhoti Holi)
    (3, 4),    # Holi
    (3, 20),   # Eid-ul-Fitr
    (3, 21),   # Eid-ul-Fitr
    (3, 27),   # Ram Navami
    (4, 14),   # Ambedkar Jayanti / Baisakhi / Tamil New Year
    (5, 27),   # Bakrid / Eid al-Adha
    (8, 15),   # Independence Day
    (8, 28),   # Raksha Bandhan
    (9, 4),    # Janmashtami
    (9, 14),   # Ganesh Chaturthi
    (10, 2),   # Gandhi Jayanti
    (10, 19),  # Navratri / Maha Saptami
    (10, 20),  # Dussehra / Vijayadashami
    (10, 29),  # Karwa Chauth
    (11, 8),   # Dhanteras
    (11, 10),  # Diwali
    (11, 12),  # Bhai Dooj
    (11, 24),  # Guru Nanak Jayanti
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve
    (1, 1),    # New Year's Day
}

def is_festival_date(dt: datetime = None) -> str:
    now = dt or datetime.now()
    if (now.month, now.day) in INDIAN_FESTIVALS_MONTH_DAY:
        return 'Yes '
    return 'No '


# ===========================================================================
# 5. Non-live Feature Imputations (Only Age and Ratings)
# ===========================================================================
# Derived from food_delivery_data.csv median on clean rows:
MEDIAN_DELIVERY_PERSON_AGE = 30.0     # Delivery rider age cannot be observed by customer pre-dispatch
MEDIAN_DELIVERY_PERSON_RATING = 4.7  # Delivery rider rating cannot be observed by customer pre-dispatch

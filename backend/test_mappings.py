"""
test_mappings.py

Unit tests for:
1. OpenWeather condition code/text -> Weatherconditions mapping
2. Reverse geocoding address dict -> City type mapping
3. Traffic density calculation and fallback
4. Festival date detection
"""

import sys, os
from datetime import datetime

# Add current backend dir to path
sys.path.insert(0, os.path.dirname(__file__))

from mappings import (
    map_openweather_condition,
    map_reverse_geocode_to_city,
    extract_city_code,
    compute_traffic_density,
    fallback_traffic_density,
    is_festival_date,
    VALID_WEATHER_CATEGORIES
)

def test_weather_mappings():
    print("Testing OpenWeather mappings...")
    test_cases = [
        # (id, main, desc, wind, expected_cat, expected_clean)
        (800, 'Clear', 'clear sky', 5.0, 'conditions Sunny', True),
        (801, 'Clouds', 'few clouds', 10.0, 'conditions Cloudy', True),
        (804, 'Clouds', 'overcast clouds', 12.0, 'conditions Cloudy', True),
        (211, 'Thunderstorm', 'thunderstorm with rain', 20.0, 'conditions Stormy', True),
        (501, 'Rain', 'moderate rain', 15.0, 'conditions Stormy', True),
        (301, 'Drizzle', 'drizzle', 8.0, 'conditions Stormy', True),
        (741, 'Fog', 'fog', 4.0, 'conditions Fog', True),
        (701, 'Mist', 'mist', 3.0, 'conditions Fog', True),
        (711, 'Smoke', 'smoke', 6.0, 'conditions Sandstorms', True),
        (761, 'Dust', 'dust', 10.0, 'conditions Sandstorms', True),
        (771, 'Squall', 'squalls', 22.0, 'conditions Windy', True),
        (800, 'Clear', 'clear with high wind', 35.0, 'conditions Windy', True),
        (9999, 'BizarreWeather', 'weird phenomenon', 5.0, 'conditions Sunny', False),  # unmapped fallback
    ]

    for wid, main, desc, wind, expected_cat, expected_clean in test_cases:
        res_cat, res_clean = map_openweather_condition(wid, main, desc, wind)
        assert res_cat in VALID_WEATHER_CATEGORIES, f"Invalid category: {res_cat}"
        assert res_cat == expected_cat, f"Mismatch for {wid}/{main}: got '{res_cat}', expected '{expected_cat}'"
        assert res_clean == expected_clean, f"Clean mismatch for {wid}: got {res_clean}, expected {expected_clean}"
        print(f"  [PASS] {wid:4d} ({main:12s}, wind={wind:4.1f}) -> '{res_cat}' (clean={res_clean})")

def test_city_mappings():
    print("\nTesting Reverse Geocoding City mappings...")
    test_cases = [
        ({'city': 'Bengaluru', 'state': 'Karnataka'}, 'Metropolitian ', 'BANG'),
        ({'city': 'Mumbai', 'state': 'Maharashtra'}, 'Metropolitian ', 'MUM'),
        ({'city': 'New Delhi', 'state': 'Delhi'}, 'Metropolitian ', 'DEH'),
        ({'city': 'Jaipur', 'state': 'Rajasthan'}, 'Metropolitian ', 'JAP'),
        ({'city': 'Kochi', 'state': 'Kerala'}, 'Metropolitian ', 'KOC'),
        ({'town': 'Dharmapuri', 'state': 'Tamil Nadu'}, 'Urban ', 'BANG'),  # smaller urban center
        ({'village': 'Rampur', 'county': 'Bareilly', 'state': 'Uttar Pradesh'}, 'Semi-Urban ', 'BANG'), # rural
        ({}, 'Metropolitian ', 'BANG'), # empty fallback
    ]

    valid_cities = {'Metropolitian ', 'Urban ', 'Semi-Urban '}
    for addr, expected_city, expected_code in test_cases:
        res_city = map_reverse_geocode_to_city(addr)
        res_code = extract_city_code(addr)
        assert res_city in valid_cities, f"Invalid city string '{res_city}'"
        assert res_city == expected_city, f"Mismatch for {addr}: got '{res_city}', expected '{expected_city}'"
        print(f"  [PASS] {str(addr):60s} -> city='{res_city}', code='{res_code}'")

def test_traffic_mappings():
    print("\nTesting Traffic Density & Fallbacks...")
    # 5 km distance, motorcycle free-flow is ~40 km/h (450s)
    # 300s -> ratio ~ 0.66 -> Low
    # 600s -> ratio ~ 1.33 -> Medium
    # 900s -> ratio ~ 2.0  -> High
    # 1500s -> ratio ~ 3.3 -> Jam
    density, ratio = compute_traffic_density(300, 5000, 'motorcycle')
    assert density == 'Low ', f"Expected Low , got {density}"
    density, ratio = compute_traffic_density(600, 5000, 'motorcycle')
    assert density == 'Medium ', f"Expected Medium , got {density}"
    density, ratio = compute_traffic_density(900, 5000, 'motorcycle')
    assert density == 'High ', f"Expected High , got {density}"
    density, ratio = compute_traffic_density(1500, 5000, 'motorcycle')
    assert density == 'Jam ', f"Expected Jam , got {density}"

    # Test time-of-day fallbacks
    assert fallback_traffic_density(datetime(2026, 9, 10, 8, 30)) == 'Low '
    assert fallback_traffic_density(datetime(2026, 9, 10, 13, 0)) == 'High '
    assert fallback_traffic_density(datetime(2026, 9, 10, 16, 30)) == 'Medium '
    assert fallback_traffic_density(datetime(2026, 9, 10, 20, 0)) == 'Jam '
    assert fallback_traffic_density(datetime(2026, 9, 10, 23, 15)) == 'Low '
    print("  [PASS] Traffic density & time-of-day fallbacks verified")

def test_festival_mappings():
    print("\nTesting Festival mappings...")
    assert is_festival_date(datetime(2026, 11, 10)) == 'Yes '  # Diwali
    assert is_festival_date(datetime(2026, 8, 15)) == 'Yes '   # Independence Day
    assert is_festival_date(datetime(2026, 12, 25)) == 'Yes '  # Christmas
    assert is_festival_date(datetime(2026, 6, 15)) == 'No '    # Regular day
    print("  [PASS] Festival dates verified")

if __name__ == '__main__':
    test_weather_mappings()
    test_city_mappings()
    test_traffic_mappings()
    test_festival_mappings()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")

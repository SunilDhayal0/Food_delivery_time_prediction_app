"""
clean_data_utils.py

Reconstructed data-cleaning module for the food-delivery time prediction project.

This file was rebuilt by reverse-engineering:
  1. 01_data_cleaning.ipynb   -> the exploratory notebook where the original
                                 cleaning logic (column renaming, missing-value
                                 handling, outlier removal, feature extraction,
                                 haversine distance) was developed and tested.
  2. 04_experiment_3_model_comparison.ipynb -> the notebook that actually
                                 IMPORTED and CALLED clean_data_utils, which
                                 shows the exact public function signature
                                 (`perform_data_cleaning(raw_df, saved_data_path=...)`)
                                 and the exact final column set/names expected
                                 downstream (e.g. `pickup_time_minutes`, not the
                                 `pickup_time_minutess` typo that exists in the
                                 exploratory notebook, and a `distance_type`
                                 column that never appears in the exploratory
                                 notebook at all).

Verification performed while rebuilding this file:
  - Running this module's `perform_data_cleaning` on food_delivery_data.csv
    reproduces the EXACT dataset shape (45502, 26 before the experiment's own
    column drops) and the EXACT per-column missing-value counts printed in
    04_experiment_3_model_comparison.ipynb cell 4 output, e.g.:
        age 1854, ratings 1908, weather 525, traffic 510,
        multiple_deliveries 993, festival 228, city_type 1198,
        pickup_time_minutes 1640, order_time_of_day 1640, distance 3630.

Post-verification fix (applied after user review, changes results vs. the
very first reconstructed version of this file):
  - `pickup_time_minutes` could go negative (e.g. -1435) for orders placed
    right before midnight and picked up just after (e.g. 23:50 -> 00:05).
    This happened because `order_time`/`order_picked_time` only carry a time
    of day, no date, so both get parsed onto the same default date and the
    subtraction wraps around. Fixed by adding 1440 minutes (24 hours) back
    whenever the raw difference comes out negative. NaN values are left
    untouched (NaN < 0 is False, so they pass through unchanged).

Known assumption (flagged explicitly, confirm/correct if results don't match):
  - `distance_type` does not exist anywhere in 01_data_cleaning.ipynb, so its
    binning rule could not be recovered from source. Based on the observed
    distance distribution (min ~1.47 km, quartiles ~4.66 / 9.19 / 13.68 km,
    max ~20.97 km) and confirmation from the user to use a best-effort guess,
    this file buckets `distance` using round-number km cutoffs:
        distance <= 5   -> 'short'
        distance <= 10  -> 'medium'
        distance <= 15  -> 'long'
        distance > 15   -> 'very_long'
    (NaN distance -> NaN distance_type, matching the 3630/3630 missing-count
    match observed between `distance` and `distance_type` in the experiment
    notebook's printed missing-value summary.)
    If your real results differ, this is the one place to adjust.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column renaming
# ---------------------------------------------------------------------------
def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Lower-case all columns and rename them to clean, consistent names."""
    df_lower = data.rename(str.lower, axis=1)
    renamed = df_lower.rename({
        "delivery_person_id": "rider_id",
        "delivery_person_age": "age",
        "delivery_person_ratings": "ratings",
        "delivery_location_latitude": "delivery_latitude",
        "delivery_location_longitude": "delivery_longitude",
        "time_orderd": "order_time",
        "time_order_picked": "order_picked_time",
        "weatherconditions": "weather",
        "road_traffic_density": "traffic",
        "city": "city_type",
        "time_taken(min)": "time_taken",
    }, axis=1)
    if 'time_taken' in renamed.columns:
        renamed['time_taken'] = (
            renamed['time_taken'].astype(str).str.replace('(min) ', '', regex=False).astype(int)
        )
    return renamed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def time_of_day(ser: pd.Series) -> pd.Series:
    """
    Bucket an hour-of-day series into morning/afternoon/evening/night
    (0-6 falls through to 'after_midnight'). Missing hours stay missing
    (NaN in, NaN out) instead of being swallowed into 'after_midnight'.
    """
    result = np.select(
        condlist=[
            ser.between(6, 12, inclusive='left'),
            ser.between(12, 17, inclusive='left'),
            ser.between(17, 20, inclusive='left'),
            ser.between(20, 24, inclusive='left'),
        ],
        choicelist=['morning', 'afternoon', 'evening', 'night'],
        default='after_midnight',
    )
    return pd.Series(result, index=ser.index).where(ser.notna(), np.nan)


def fix_pickup_time_rollover(pickup_minutes: pd.Series) -> pd.Series:
    """
    Correct midnight-rollover errors in pickup duration.

    order_time / order_picked_time carry only a time-of-day (no date), so an
    order placed just before midnight and picked up just after midnight
    (e.g. 23:50 -> 00:05) produces a large negative difference (-1435)
    instead of the real +15 minutes. Adding back 1440 minutes (24 hours)
    corrects this. NaN stays NaN, since `NaN < 0` evaluates to False.
    """
    return np.where(pickup_minutes < 0, pickup_minutes + 1440, pickup_minutes)


def assign_distance_type(distance: pd.Series) -> pd.Series:
    """
    Bucket haversine distance (km) into short/medium/long/very_long.

    NOTE: see module docstring - this rule was not recoverable from
    01_data_cleaning.ipynb and is a best-effort reconstruction.
    """
    return pd.cut(
        distance,
        bins=[-np.inf, 5, 10, 15, np.inf],
        labels=['short', 'medium', 'long', 'very_long'],
    ).astype(object).where(distance.notna(), np.nan)


def clean_lat_long(data: pd.DataFrame, threshold: float = 1) -> pd.DataFrame:
    """Replace bogus (near-zero) lat/long values with NaN."""
    location_columns = [
        'restaurant_latitude', 'restaurant_longitude',
        'delivery_latitude', 'delivery_longitude',
    ]
    return data.assign(**{
        col: np.where(data[col] < threshold, np.nan, data[col].values)
        for col in location_columns
    })


def calculate_harversine_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `distance` column (km) computed via the haversine formula."""
    location_columns = [
        'restaurant_latitude', 'restaurant_longitude',
        'delivery_latitude', 'delivery_longitude',
    ]
    lat1 = df[location_columns[0]]
    lon1 = df[location_columns[1]]
    lat2 = df[location_columns[2]]
    lon2 = df[location_columns[3]]

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return df.assign(distance=distance)


# ---------------------------------------------------------------------------
# Main cleaning routine
# ---------------------------------------------------------------------------
def data_cleaning(data: pd.DataFrame, drop_anomalies: bool = True) -> pd.DataFrame:
    """
    Core row/column cleaning + feature extraction step.

    Removes underage-rider rows and the "rating == 6" anomaly rows
    (recomputed fresh from the incoming data rather than relying on
    hard-coded indices, since this must work on any raw dataframe).
    If drop_anomalies is False (e.g. inference/live prediction), rows are not dropped.
    """
    cleaned = data.copy()
    if drop_anomalies:
        minors_index = cleaned[cleaned['age'].astype(float) < 18].index.tolist() if 'age' in cleaned.columns else []
        six_star_index = cleaned.loc[cleaned['ratings'].astype(float) == 6].index.tolist() if 'ratings' in cleaned.columns else []
        if minors_index:
            cleaned = cleaned.drop(index=minors_index)
        if six_star_index:
            cleaned = cleaned.drop(index=six_star_index)

    drop_id_cols = [c for c in ['id'] if c in cleaned.columns]
    if drop_id_cols:
        cleaned = cleaned.drop(columns=drop_id_cols)

    return (
        cleaned
            .assign(
                # city column extracted from the rider id
                city_name=lambda df_: (df_['rider_id'].str.split("RES").str.get(0)),
                # convert age to float
                age=lambda df_: df_['age'].astype(float),
                # convert ratings to float
                ratings=lambda df_: df_['ratings'].astype(float),
                # absolute value for location based columns
                restaurant_latitude=lambda df_: pd.to_numeric(df_['restaurant_latitude'], errors='coerce').abs(),
                restaurant_longitude=lambda df_: pd.to_numeric(df_['restaurant_longitude'], errors='coerce').abs(),
                delivery_latitude=lambda df_: pd.to_numeric(df_['delivery_latitude'], errors='coerce').abs(),
                delivery_longitude=lambda df_: pd.to_numeric(df_['delivery_longitude'], errors='coerce').abs(),

                # order_date -> day / month / day_of_week / is_weekend
                _order_date_dt=lambda df_: pd.to_datetime(
                    df_['order_date'], dayfirst=True, errors='coerce'
                ),
                order_day=lambda df_: df_['_order_date_dt'].dt.day,
                order_month=lambda df_: df_['_order_date_dt'].dt.month,
                order_day_of_week=lambda df_: df_['_order_date_dt'].dt.day_name().str.lower(),
                is_weekend=lambda df_: (
                    df_['_order_date_dt'].dt.day_name().isin(['Saturday', 'Sunday']).astype(int)
                ),

                # order_time / order_picked_time -> pickup duration + hour + time-of-day
                _order_time_dt=lambda df_: pd.to_datetime(
                    df_['order_time'].replace("NaN ", np.nan), format='mixed', errors='coerce'
                ),
                _order_picked_time_dt=lambda df_: pd.to_datetime(
                    df_['order_picked_time'].replace("NaN ", np.nan), format='mixed', errors='coerce'
                ),
                pickup_time_minutes=lambda df_: fix_pickup_time_rollover(
                    (df_['_order_picked_time_dt'] - df_['_order_time_dt']).dt.total_seconds() / 60
                ),
                order_time_hour=lambda df_: df_['_order_time_dt'].dt.hour,
                order_time_of_day=lambda df_: time_of_day(df_['order_time_hour']),

                # categorical clean-up
                weather=lambda df_: (
                    df_['weather'].str.replace("conditions ", "").str.lower().replace("nan", np.nan)
                ),
                traffic=lambda df_: df_['traffic'].str.rstrip().str.lower(),
                type_of_order=lambda df_: df_['type_of_order'].str.rstrip().str.lower(),
                type_of_vehicle=lambda df_: df_['type_of_vehicle'].str.rstrip().str.lower(),
                city_type=lambda df_: df_['city_type'].str.rstrip().str.lower(),
                multiple_deliveries=lambda df_: pd.to_numeric(df_['multiple_deliveries'], errors='coerce'),
                festival=lambda df_: df_['festival'].str.rstrip().str.lower(),
            )
            .drop(columns=[
                '_order_date_dt', '_order_time_dt', '_order_picked_time_dt',
                'order_date', 'order_time', 'order_picked_time',
            ])
    )


def perform_data_cleaning(raw_df: pd.DataFrame, saved_data_path: str = None) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline: raw Kaggle-style food-delivery data in,
    fully cleaned + feature-engineered dataframe out.

    Steps:
      1. Normalize the literal "NaN " string values to real NaN.
      2. Rename columns to clean, lower_snake_case names.
      3. Run row-level cleaning + feature extraction (`data_cleaning`).
      4. Fix bogus (near-zero) lat/long values (`clean_lat_long`).
      5. Compute haversine `distance` between restaurant and delivery point.
      6. Bucket `distance` into `distance_type`.

    If `saved_data_path` is given, the cleaned dataframe is also written to
    that path as a CSV (matching how this function is used in the modelling
    notebooks, e.g. `clean_data_utils.perform_data_cleaning(raw_df,
    saved_data_path='/content/cleaned_data.csv')`).
    """
    df = raw_df.replace("NaN ", np.nan)
    df = change_column_names(df)

    cleaned_data = (
        df.pipe(data_cleaning)
          .pipe(clean_lat_long)
          .pipe(calculate_harversine_distance)
          .assign(distance_type=lambda df_: assign_distance_type(df_['distance']))
    )

    if saved_data_path is not None:
        cleaned_data.to_csv(saved_data_path, index=False)

    return cleaned_data


# ---------------------------------------------------------------------------
# DeliveryTimePredictor Class (For live prediction & deployment)
# ---------------------------------------------------------------------------
class DeliveryTimePredictor:
    """
    Self-contained predictor for the delivery-time website / live prediction API.
    Wraps: raw-data cleaning -> stacking ensemble -> inverse power-transform.
    Usage:
        import joblib
        predictor = joblib.load('delivery_time_predictor.pkl')
        minutes = predictor.predict(raw_dataframe)
    """
    DEFAULT_IMPUTE = {
        'age': 30.0,
        'ratings': 4.7,
        'weather': 'fog',
        'traffic': 'medium',
        'vehicle_condition': 1.0,
        'type_of_order': 'snack',
        'type_of_vehicle': 'motorcycle',
        'multiple_deliveries': 1.0,
        'festival': 'no',
        'city_type': 'metropolitian',
        'city_name': 'JAP',
        'order_month': 3.0,
        'order_day_of_week': 'wednesday',
        'is_weekend': 0.0,
        'pickup_time_minutes': 10.0,
        'order_time_of_day': 'night',
        'distance': 9.22,
        'distance_type': 'medium'
    }

    def __init__(self, stacking_model, power_transformer, drop_cols=None):
        self.stacking_model = stacking_model
        self.power_transformer = power_transformer
        self.drop_cols = drop_cols or [
            'rider_id', 'restaurant_latitude', 'restaurant_longitude',
            'delivery_latitude', 'delivery_longitude', 'order_day', 'order_time_hour'
        ]

    def _clean(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = (
            raw_df.copy()
            .replace("NaN ", np.nan)
            .pipe(change_column_names)
            .pipe(lambda d: data_cleaning(d, drop_anomalies=False))
            .pipe(clean_lat_long)
            .pipe(calculate_harversine_distance)
            .pipe(lambda d: d.assign(distance_type=assign_distance_type(d['distance'])))
        )
        existing = [c for c in self.drop_cols if c in df.columns]
        cleaned = df.drop(columns=existing)
        for col, default_val in self.DEFAULT_IMPUTE.items():
            if col in cleaned.columns:
                cleaned[col] = cleaned[col].fillna(default_val)
        return cleaned

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        X = self._clean(raw_df)
        if 'time_taken' in X.columns:
            X = X.drop(columns='time_taken')
        y_pred_pt = self.stacking_model.predict(X)
        y_pred = self.power_transformer.inverse_transform(np.asarray(y_pred_pt).reshape(-1, 1)).ravel()
        return y_pred


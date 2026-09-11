"""
app.py

Gradio Web Application for DeliveryTime-X.
Compatible with Hugging Face Spaces (Gradio SDK).
Users enter only 6 fields; all other features are derived live via external APIs and spatial algorithms.
"""

import sys, os, asyncio
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd

# Add current dir and backend dir to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "backend"))

import clean_data_utils
from backend.predictor import predict_delivery


def run_prediction_sync(from_address, to_address, vehicle_type, vehicle_condition, type_of_order, multiple_deliveries):
    if not from_address or not to_address:
        return "⚠️ Please enter both pickup and drop addresses.", "", "", "", "", ""

    payload = {
        "from_address": str(from_address).strip(),
        "to_address": str(to_address).strip(),
        "vehicle_type": str(vehicle_type).strip().lower(),
        "vehicle_condition": int(vehicle_condition),
        "type_of_order": str(type_of_order).strip(),
        "multiple_deliveries": int(multiple_deliveries)
    }

    try:
        # Run async predict_delivery in synchronous Gradio handler
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        res = loop.run_until_complete(predict_delivery(payload))
        loop.close()

        predicted_mins = f"{res.get('predicted_minutes', 0)} mins"
        traffic = f"{res.get('traffic_level', 'Medium').upper()} TRAFFIC"
        weather_info = res.get("breakdown", {}).get("weather_details", {})
        weather_str = f"{res.get('weather', 'Clear')} ({weather_info.get('temperature_c', '--')}°C, Wind: {weather_info.get('wind_speed_kmh', '--')} km/h)"
        distance_str = f"Road: {res.get('osrm_road_distance_km', '--')} km (Direct: {res.get('distance_km', '--')} km)"
        city_str = res.get("breakdown", {}).get("city_classification", "Metropolitian")
        festival_str = res.get("breakdown", {}).get("festival", "No")

        return predicted_mins, traffic, weather_str, distance_str, city_str, festival_str
    except Exception as e:
        return f"Error: {str(e)}", "", "", "", "", ""


# Build modern Gradio UI
with gr.Blocks(title="DeliveryTime-X | Real-Time Food Delivery ETA") as demo:
    gr.Markdown(
        """
        # ⚡ DeliveryTime-X — Real-Time ML-Powered Food Delivery Estimation
        ### Powered by a Stacking Ensemble Regressor (`LightGBM` + `RandomForest` via `LinearRegression`).
        Fill in the **6 fields below** — all other features (coordinates, live weather, OSRM road traffic, and timestamps) are auto-derived live from external APIs.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Order & Route Inputs (6 Fields)")
            from_input = gr.Textbox(label="1. Pickup Address ('From' Restaurant)", placeholder="e.g. Koramangala 4th Block, Bengaluru", value="Koramangala 4th Block, Bengaluru")
            to_input = gr.Textbox(label="2. Drop Address ('To' Customer)", placeholder="e.g. Indiranagar 100ft Road, Bengaluru", value="Indiranagar 100ft Road, Bengaluru")
            
            with gr.Row():
                vehicle_type = gr.Dropdown(choices=["motorcycle", "scooter", "electric_scooter", "bicycle"], value="motorcycle", label="3. Vehicle Type")
                vehicle_cond = gr.Dropdown(choices=[0, 1, 2, 3], value=1, label="4. Vehicle Condition (0=Poor, 3=New)")

            with gr.Row():
                order_type = gr.Dropdown(choices=["Meal", "Snack", "Drinks", "Buffet"], value="Meal", label="5. Type of Order")
                multi_del = gr.Dropdown(choices=[0, 1, 2, 3], value=1, label="6. Concurrent Deliveries")

            predict_btn = gr.Button("⚡ Predict Delivery Time", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Real-Time ML Prediction & Live Factors")
            eta_output = gr.Textbox(label="Estimated Delivery Time", interactive=False)
            traffic_output = gr.Textbox(label="Live Traffic Density (OSRM Route-Delay Ratio)", interactive=False)
            weather_output = gr.Textbox(label="Live Weather (Open-Meteo Satellite/Station API)", interactive=False)
            distance_output = gr.Textbox(label="Route Distance", interactive=False)
            city_output = gr.Textbox(label="City Zone Classification", interactive=False)
            festival_output = gr.Textbox(label="Indian Festival Rush Impact", interactive=False)

    predict_btn.click(
        fn=run_prediction_sync,
        inputs=[from_input, to_input, vehicle_type, vehicle_cond, order_type, multi_del],
        outputs=[eta_output, traffic_output, weather_output, distance_output, city_output, festival_output]
    )

    gr.Markdown("---")
    gr.Markdown("Dataset: 45,500+ historical records across Indian cities. Model: Stacking Ensemble (Test RMSE: 3.765 mins, R²: 0.8379).")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

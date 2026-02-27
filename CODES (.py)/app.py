"""
Flood Prediction Web Application
Flask backend serving prediction API and dashboard
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import json
import os

app = Flask(__name__)

# ─── Load Model & Artifacts ───────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(BASE, 'models')

model = joblib.load(os.path.join(MODELS_PATH, 'best_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))

with open(os.path.join(MODELS_PATH, 'feature_names.json')) as f:
    feature_names = json.load(f)

with open(os.path.join(MODELS_PATH, 'metrics.json')) as f:
    metrics = json.load(f)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', metrics=metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        rainfall_mm = float(data.get('rainfall_mm', 0))
        rainfall_3day_avg = float(data.get('rainfall_3day_avg', 0))
        rainfall_7day_avg = float(data.get('rainfall_7day_avg', 0))
        river_level_m = float(data.get('river_level_m', 0))
        temperature_c = float(data.get('temperature_c', 25))
        humidity_pct = float(data.get('humidity_pct', 60))
        wind_speed_kmh = float(data.get('wind_speed_kmh', 10))
        soil_moisture_pct = float(data.get('soil_moisture_pct', 40))
        elevation_m = float(data.get('elevation_m', 50))
        distance_to_river_km = float(data.get('distance_to_river_km', 5))
        drainage_quality = int(data.get('drainage_quality', 1))

        # Engineered features
        rainfall_river_interaction = rainfall_mm * river_level_m
        low_elevation_near_river = int(elevation_m < 20 and distance_to_river_km < 2)
        high_risk_conditions = int(rainfall_mm > 30 and soil_moisture_pct > 60)

        features = [
            rainfall_mm, rainfall_3day_avg, rainfall_7day_avg,
            river_level_m, temperature_c, humidity_pct, wind_speed_kmh,
            soil_moisture_pct, elevation_m, distance_to_river_km,
            drainage_quality, rainfall_river_interaction,
            low_elevation_near_river, high_risk_conditions
        ]

        features_scaled = scaler.transform([features])
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])

        # Risk level
        if probability < 0.25:
            risk_level = 'Low'
            risk_color = '#4CAF50'
        elif probability < 0.55:
            risk_level = 'Moderate'
            risk_color = '#FF9800'
        elif probability < 0.75:
            risk_level = 'High'
            risk_color = '#FF5722'
        else:
            risk_level = 'Critical'
            risk_color = '#F44336'

        # Actionable recommendations
        recommendations = []
        if probability > 0.5:
            recommendations.append("⚠️ Activate early warning systems and alert residents")
        if rainfall_mm > 40:
            recommendations.append("🌧️ Extreme rainfall detected — monitor drainage systems")
        if river_level_m > 7:
            recommendations.append("🌊 River levels critical — consider evacuation of low-lying areas")
        if soil_moisture_pct > 70:
            recommendations.append("💧 Soil saturation high — runoff risk elevated")
        if elevation_m < 10:
            recommendations.append("📍 Low elevation area — prioritize flood barrier deployment")
        if not recommendations:
            recommendations.append("✅ Conditions normal — continue routine monitoring")

        return jsonify({
            'prediction': prediction,
            'probability': round(probability * 100, 1),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendations': recommendations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

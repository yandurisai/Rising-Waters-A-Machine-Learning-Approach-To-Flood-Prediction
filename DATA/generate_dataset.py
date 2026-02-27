"""
Generate synthetic flood prediction dataset for training/testing
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

def generate_flood_dataset(n_samples=5000):
    dates = [datetime(2015, 1, 1) + timedelta(days=i) for i in range(n_samples)]
    
    rainfall = np.random.exponential(scale=15, size=n_samples)
    rainfall_3day = np.convolve(rainfall, np.ones(3)/3, mode='same')
    rainfall_7day = np.convolve(rainfall, np.ones(7)/7, mode='same')
    
    river_level = np.random.normal(loc=3.5, scale=1.2, size=n_samples)
    river_level = np.clip(river_level + 0.3 * rainfall, 0, 15)
    
    temperature = 25 + 10 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 2, n_samples)
    humidity = np.clip(60 + 0.5 * rainfall + np.random.normal(0, 5, n_samples), 0, 100)
    
    wind_speed = np.random.exponential(scale=10, size=n_samples)
    soil_moisture = np.clip(30 + 0.4 * rainfall_7day + np.random.normal(0, 5, n_samples), 0, 100)
    
    elevation = np.random.choice([5, 10, 20, 50, 100, 200], size=n_samples, p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
    distance_to_river = np.random.exponential(scale=2, size=n_samples)
    drainage_quality = np.random.choice([0, 1, 2], size=n_samples, p=[0.2, 0.5, 0.3])  # poor, medium, good
    
    # Flood label based on realistic conditions
    flood_score = (
        0.3 * (rainfall > 40).astype(int) +
        0.25 * (rainfall_3day > 25).astype(int) +
        0.2 * (river_level > 7).astype(int) +
        0.1 * (soil_moisture > 70).astype(int) +
        0.1 * (elevation < 10).astype(int) +
        0.05 * (distance_to_river < 1).astype(int)
    ) - 0.1 * (drainage_quality == 2)
    
    noise = np.random.normal(0, 0.05, n_samples)
    flood = (flood_score + noise > 0.35).astype(int)
    
    df = pd.DataFrame({
        'date': dates,
        'rainfall_mm': np.round(rainfall, 2),
        'rainfall_3day_avg': np.round(rainfall_3day, 2),
        'rainfall_7day_avg': np.round(rainfall_7day, 2),
        'river_level_m': np.round(river_level, 2),
        'temperature_c': np.round(temperature, 2),
        'humidity_pct': np.round(humidity, 2),
        'wind_speed_kmh': np.round(wind_speed, 2),
        'soil_moisture_pct': np.round(soil_moisture, 2),
        'elevation_m': elevation,
        'distance_to_river_km': np.round(distance_to_river, 2),
        'drainage_quality': drainage_quality,
        'flood': flood
    })
    
    return df

if __name__ == "__main__":
    df = generate_flood_dataset(5000)
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), 'flood_data.csv'), index=False)
    print(f"Dataset generated: {len(df)} samples")
    print(f"Flood rate: {df['flood'].mean():.2%}")
    print(df.head())

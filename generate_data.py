"""
generate_data.py
Generates synthetic cold-chain sensor data for vaccine logistics.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

NUM_SAMPLES = 5000
OUTPUT_PATH = os.path.join("data", "cold_chain_data.csv")


def generate_dataset(n: int = NUM_SAMPLES) -> pd.DataFrame:
    """Generate a synthetic dataset simulating cold-chain sensor readings."""

    # Shipment metadata
    shipment_ids = [f"SHP-{str(i+1).zfill(5)}" for i in range(n)]
    vaccine_types = np.random.choice(
        ["COVID-19-mRNA", "Polio", "Hepatitis-B", "Influenza", "Rotavirus"],
        size=n,
        p=[0.30, 0.20, 0.20, 0.15, 0.15],
    )
    transport_modes = np.random.choice(
        ["Air", "Road", "Rail", "Sea"], size=n, p=[0.25, 0.50, 0.15, 0.10]
    )
    routes = np.random.choice(
        ["Delhi-Mumbai", "Chennai-Kolkata", "Jaipur-Hyderabad", "Bengaluru-Pune", "Lucknow-Ahmedabad"],
        size=n,
    )

    # Sensor readings
    temperature_avg = np.random.normal(loc=4.0, scale=2.5, size=n)          # °C; ideal 2–8 °C
    temperature_max = temperature_avg + np.abs(np.random.normal(0, 1.5, n))
    temperature_min = temperature_avg - np.abs(np.random.normal(0, 1.0, n))
    humidity = np.random.normal(loc=55, scale=10, size=n).clip(20, 95)       # %
    vibration_level = np.random.exponential(scale=0.5, size=n)              # g-force
    door_open_count = np.random.poisson(lam=3, size=n)
    transit_duration_hrs = np.random.uniform(6, 72, size=n)
    power_outage_mins = np.random.exponential(scale=5, size=n)

    # Derived features
    temp_excursion = (temperature_max > 8) | (temperature_min < 2)
    temp_excursion_duration_mins = np.where(
        temp_excursion,
        np.random.uniform(10, 120, size=n),
        0.0,
    )

    # Failure label (binary)
    failure_score = (
        (temperature_max > 10).astype(float) * 0.40
        + (temperature_min < 0).astype(float) * 0.25
        + (humidity > 75).astype(float) * 0.10
        + (vibration_level > 1.0).astype(float) * 0.10
        + (door_open_count > 5).astype(float) * 0.10
        + (power_outage_mins > 10).astype(float) * 0.05
    )
    noise = np.random.uniform(0, 0.15, size=n)
    failure = ((failure_score + noise) > 0.35).astype(int)

    df = pd.DataFrame(
        {
            "shipment_id": shipment_ids,
            "vaccine_type": vaccine_types,
            "transport_mode": transport_modes,
            "route": routes,
            "temperature_avg_c": temperature_avg.round(2),
            "temperature_max_c": temperature_max.round(2),
            "temperature_min_c": temperature_min.round(2),
            "humidity_pct": humidity.round(2),
            "vibration_g": vibration_level.round(4),
            "door_open_count": door_open_count,
            "transit_duration_hrs": transit_duration_hrs.round(2),
            "power_outage_mins": power_outage_mins.round(2),
            "temp_excursion": temp_excursion.astype(int),
            "temp_excursion_duration_mins": temp_excursion_duration_mins.round(2),
            "cold_chain_failure": failure,
        }
    )
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    total = len(df)
    failures = df["cold_chain_failure"].sum()
    print(f"Dataset generated: {total} records → {OUTPUT_PATH}")
    print(f"Failure rate: {failures}/{total} ({failures/total*100:.1f}%)")

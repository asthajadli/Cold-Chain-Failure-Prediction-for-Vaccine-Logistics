# ❄️ Cold-Chain Failure Prediction for Vaccine Logistics

A machine learning system that predicts cold-chain failures in vaccine supply chains using simulated IoT sensor data. Built with Python, scikit-learn, and Streamlit.

---

## 📌 Problem Statement

Vaccines are highly temperature-sensitive. A single cold-chain break — a power outage, refrigeration malfunction, or improper handling — can render entire shipments ineffective and put lives at risk. This project uses machine learning to **predict failure before it happens**, enabling proactive intervention.

---

## 🗂️ Project Structure

```
Cold-Chain-Failure-Prediction-for-Vaccine-Logistics/
│
├── data/
│   └── cold_chain_data.csv        # Synthetic sensor dataset (5,000 records)
│
├── models/
│   ├── cold_chain_model.pkl       # Trained Random Forest classifier
│   ├── le_vaccine.pkl             # Label encoder for vaccine type
│   ├── le_transport.pkl           # Label encoder for transport mode
│   └── evaluation_report.json    # Accuracy, ROC-AUC, feature importances
│
├── generate_data.py               # Synthetic data generator
├── train_model.py                 # Model training & evaluation script
├── app.py                         # Streamlit web app (dashboard + predictor)
└── requirements.txt               # Python dependencies
```

---

## 🧠 ML Pipeline

| Step | Details |
|------|---------|
| **Data** | 5,000 synthetic shipment records with 14 features |
| **Features** | Temperature (avg/min/max), humidity, vibration, door opens, transit time, power outages, excursion duration, vaccine type, transport mode |
| **Model** | Random Forest Classifier (200 trees, balanced class weight) |
| **Evaluation** | Train/test split 80/20, accuracy & ROC-AUC reported |

---

## 📊 Features

- **🔮 Failure Predictor** — Input shipment parameters and get an instant risk score with a gauge chart
- **📊 Data Explorer** — Interactive plots for temperature distributions, vaccine types, and transport modes
- **📈 Model Performance** — View accuracy, ROC-AUC, confusion matrix, and feature importances

---

## 🌡️ Key Sensor Features

| Feature | Description |
|---------|-------------|
| `temperature_avg_c` | Average shipment temperature (°C) |
| `temperature_max_c` | Maximum recorded temperature |
| `temperature_min_c` | Minimum recorded temperature |
| `humidity_pct` | Relative humidity percentage |
| `vibration_g` | Vibration level in g-force units |
| `door_open_count` | Number of times storage door was opened |
| `transit_duration_hrs` | Total shipment transit time |
| `power_outage_mins` | Total minutes of power outage |
| `temp_excursion` | Binary flag: 1 if temperature went out of range |
| `temp_excursion_duration_mins` | Duration of temperature excursion |
---

## 📄 License

MIT License — feel free to fork, use, and build on this project.

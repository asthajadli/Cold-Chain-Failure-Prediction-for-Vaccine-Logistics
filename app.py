"""
app.py
Streamlit web application for Cold-Chain Failure Prediction.
Run with: streamlit run app.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cold-Chain Failure Predictor",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH  = os.path.join("models", "cold_chain_model.pkl")
REPORT_PATH = os.path.join("models", "evaluation_report.json")
DATA_PATH   = os.path.join("data",   "cold_chain_data.csv")

VACCINE_TYPES    = ["COVID-19-mRNA", "Hepatitis-B", "Influenza", "Polio", "Rotavirus"]
TRANSPORT_MODES  = ["Air", "Rail", "Road", "Sea"]

# ─── Load Assets ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_report():
    with open(REPORT_PATH) as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ─── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/fluency/96/cold-storage.png", width=80
)
st.sidebar.title("❄️ Cold-Chain Predictor")
st.sidebar.markdown("Predict whether a vaccine shipment will suffer a cold-chain failure.")

page = st.sidebar.radio(
    "Navigate",
    ["🔮 Predict Failure", "📊 Data Explorer", "📈 Model Performance"],
)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Predict
# ════════════════════════════════════════════════════════════════════════════
if page == "🔮 Predict Failure":
    st.title("🔮 Cold-Chain Failure Prediction")
    st.markdown("Fill in the shipment details below to get an instant risk assessment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌡️ Temperature")
        temp_avg = st.slider("Avg Temperature (°C)", -10.0, 30.0, 4.0, 0.1)
        temp_max = st.slider("Max Temperature (°C)", temp_avg, 35.0, max(temp_avg + 1, 6.0), 0.1)
        temp_min = st.slider("Min Temperature (°C)", -15.0, temp_avg, min(temp_avg - 1, 2.0), 0.1)
        temp_excursion = int(temp_max > 8 or temp_min < 2)
        excursion_dur  = st.slider("Excursion Duration (mins)", 0, 180, 0) if temp_excursion else 0

    with col2:
        st.subheader("📦 Shipment Info")
        vaccine_type   = st.selectbox("Vaccine Type", VACCINE_TYPES)
        transport_mode = st.selectbox("Transport Mode", TRANSPORT_MODES)
        transit_hrs    = st.slider("Transit Duration (hrs)", 1.0, 72.0, 24.0, 0.5)
        door_opens     = st.slider("Door Open Count", 0, 20, 3)

    with col3:
        st.subheader("⚡ Environment")
        humidity       = st.slider("Humidity (%)", 10.0, 95.0, 55.0, 0.5)
        vibration      = st.slider("Vibration (g-force)", 0.0, 3.0, 0.3, 0.01)
        power_outage   = st.slider("Power Outage (mins)", 0.0, 60.0, 0.0, 0.5)

    # Encode categoricals (must match training label order)
    vaccine_enc   = sorted(VACCINE_TYPES).index(vaccine_type)
    transport_enc = sorted(TRANSPORT_MODES).index(transport_mode)

    features = np.array([[
        temp_avg, temp_max, temp_min,
        humidity, vibration, door_opens,
        transit_hrs, power_outage,
        temp_excursion, excursion_dur,
        vaccine_enc, transport_enc,
    ]])

    if st.button("🚀 Predict Risk", use_container_width=True):
        try:
            model = load_model()
            prob  = model.predict_proba(features)[0][1]
            label = model.predict(features)[0]

            st.markdown("---")
            r1, r2, r3 = st.columns(3)
            r1.metric("Failure Probability", f"{prob*100:.1f}%")
            r2.metric("Prediction", "⚠️ FAILURE" if label else "✅ SAFE")
            r3.metric("Risk Level",
                "🔴 High" if prob > 0.7 else "🟡 Medium" if prob > 0.4 else "🟢 Low")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Failure Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": "crimson" if prob > 0.5 else "steelblue"},
                    "steps": [
                        {"range": [0,  40], "color": "#d4edda"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "value": 50},
                },
            ))
            gauge.update_layout(height=300)
            st.plotly_chart(gauge, use_container_width=True)

        except FileNotFoundError:
            st.error("Model not found. Please run `python train_model.py` first.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Explorer
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.title("📊 Dataset Explorer")
    try:
        df = load_data()
        st.markdown(f"**{len(df):,} shipment records** | **{df['cold_chain_failure'].mean()*100:.1f}% failure rate**")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(df, x="temperature_avg_c", color="cold_chain_failure",
                               barmode="overlay", nbins=50, title="Avg Temperature Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.pie(df, names="vaccine_type", title="Shipments by Vaccine Type")
            st.plotly_chart(fig, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.box(df, x="transport_mode", y="temperature_max_c",
                         color="cold_chain_failure", title="Max Temp by Transport Mode")
            st.plotly_chart(fig, use_container_width=True)
        with c4:
            fig = px.scatter(df.sample(500), x="humidity_pct", y="vibration_g",
                             color=df.sample(500)["cold_chain_failure"].map({0:"Safe",1:"Failure"}),
                             title="Humidity vs Vibration (sample 500)")
            st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Show raw data"):
            st.dataframe(df.head(200), use_container_width=True)
    except FileNotFoundError:
        st.error("Dataset not found. Run `python generate_data.py` first.")

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Performance
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Evaluation")
    try:
        rpt = load_report()
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy",  f"{rpt['accuracy']*100:.2f}%")
        m2.metric("ROC-AUC",   f"{rpt['roc_auc']:.4f}")
        m3.metric("Test Size",  rpt["test_size"])

        st.subheader("Feature Importance")
        feat_df = pd.DataFrame(
            rpt["feature_importance"].items(), columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)
        fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Confusion Matrix")
        cm = np.array(rpt["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, labels={"x":"Predicted","y":"Actual"},
                        x=["Safe","Failure"], y=["Safe","Failure"],
                        color_continuous_scale="Blues", title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error("Evaluation report not found. Run `python train_model.py` first.")

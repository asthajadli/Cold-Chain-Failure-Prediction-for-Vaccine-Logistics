"""
train_model.py
Trains a Random Forest classifier to predict cold-chain failures
in vaccine logistics shipments.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

DATA_PATH   = os.path.join("data", "cold_chain_data.csv")
MODEL_PATH  = os.path.join("models", "cold_chain_model.pkl")
REPORT_PATH = os.path.join("models", "evaluation_report.json")

FEATURES = [
    "temperature_avg_c",
    "temperature_max_c",
    "temperature_min_c",
    "humidity_pct",
    "vibration_g",
    "door_open_count",
    "transit_duration_hrs",
    "power_outage_mins",
    "temp_excursion",
    "temp_excursion_duration_mins",
    "vaccine_type_enc",
    "transport_mode_enc",
]
TARGET = "cold_chain_failure"


def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    le_vaccine = LabelEncoder()
    le_transport = LabelEncoder()
    df["vaccine_type_enc"] = le_vaccine.fit_transform(df["vaccine_type"])
    df["transport_mode_enc"] = le_transport.fit_transform(df["transport_mode"])

    # Persist encoders alongside model
    os.makedirs("models", exist_ok=True)
    joblib.dump(le_vaccine,   os.path.join("models", "le_vaccine.pkl"))
    joblib.dump(le_transport, os.path.join("models", "le_transport.pkl"))
    return df


def train(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)
    cm      = confusion_matrix(y_test, y_pred).tolist()

    # Feature importance
    importance = dict(
        zip(FEATURES, model.feature_importances_.round(4).tolist())
    )
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    evaluation = {
        "accuracy": round(acc, 4),
        "roc_auc":  round(roc_auc, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importance": importance,
        "train_size": len(X_train),
        "test_size":  len(X_test),
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(REPORT_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)

    print("=" * 55)
    print("  Cold-Chain Failure Prediction — Training Complete")
    print("=" * 55)
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  Model saved  → {MODEL_PATH}")
    print(f"  Report saved → {REPORT_PATH}")
    print("=" * 55)
    print("\nTop-5 Feature Importances:")
    for feat, imp in list(importance.items())[:5]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<35} {bar} {imp:.4f}")
    return model


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at '{DATA_PATH}'. Run generate_data.py first.")
        raise SystemExit(1)

    df = load_and_preprocess(DATA_PATH)
    train(df)

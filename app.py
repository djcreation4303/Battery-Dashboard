import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained models
sei_model = joblib.load("sei_model_comressed.pkl")
ir_model = joblib.load("ir_model_comressed.pkl")
soh_model = joblib.load("soh_model_comressed.pkl")

st.set_page_config(page_title="🔋 Battery Safety Predictor", layout="centered")
st.title("🔋 Lithium-ion Battery Safety & Health Predictor")
st.markdown("Enter battery details below. The app will predict SEI, IR, SOH and compute CSI with a safety classification.")

# Battery ID mapping (for user-friendly selection)
battery_id_map = {
    0: "B0005", 1: "B0006", 2: "B0007", 3: "B0018", 4: "B0025", 5: "B0026",
    6: "B0027", 7: "B0028", 8: "B0029", 9: "B0030", 10: "B0031", 11: "B0032",
    12: "B0033", 13: "B0034", 14: "B0036", 15: "B0038", 16: "B0039", 17: "B0040",
    18: "B0042", 19: "B0043", 20: "B0044", 21: "B0046", 22: "B0047", 23: "B0048"
}

# User Inputs
st.header("📥 Input Battery Parameters")

battery_id_input = st.selectbox(
    "Select Battery ID (Label Encoded)", 
    options=list(battery_id_map.keys()),
    format_func=lambda x: f"{battery_id_map[x]} → {x}"
)

cycle_count = st.number_input("Cycle Count", min_value=0, value=200)
depth_of_discharge = st.slider("Depth of Discharge (%)", 0, 100, 80)
storage_time_months = st.number_input("Storage Time (months)", min_value=0, value=6)
battery_age_months = st.number_input("Battery Age (months)", min_value=0, value=12)
ambient_temperature = st.slider("Ambient Temperature (°C)", 15, 45, 25)
voltage = st.slider("Current Voltage (V)", 3.2, 4.2, 3.7)

# Create input DataFrame
input_df = pd.DataFrame([[
    battery_id_input,
    cycle_count,
    depth_of_discharge,
    storage_time_months,
    battery_age_months,
    ambient_temperature,
    voltage
]], columns=[
    "battery_id", "cycle_count", "depth_of_discharge",
    "storage_time_months", "battery_age_months", 
    "ambient_temperature", "voltage"
])

# Feature sets for each model
features_sei = input_df[[
    "battery_id", "cycle_count", "depth_of_discharge",
    "storage_time_months", "battery_age_months", "voltage"
]]

features_ir = input_df[[
    "battery_id", "cycle_count", "depth_of_discharge",
    "ambient_temperature", "battery_age_months", "voltage"
]]

# Run Prediction
if st.button("🔍 Predict Health & Safety"):
    # Predict SEI and IR
    sei_pred = sei_model.predict(features_sei)[0]
    ir_pred = ir_model.predict(features_ir)[0]

    # SOH prediction input
    soh_input = pd.DataFrame([[
        sei_pred, ir_pred, battery_age_months, cycle_count, voltage, depth_of_discharge, battery_id_input
    ]], columns=[
        "SEI", "IR", "battery_age_months", "cycle_count", "voltage", "depth_of_discharge", "battery_id"
    ])
    soh_pred = soh_model.predict(soh_input)[0]

    # --- CSI Calculation ---
    # SEI Scoring
    if sei_pred <= 0.4:
        sei_score = 1
    elif sei_pred <= 0.7:
        sei_score = 1 - (sei_pred - 0.4) / 0.3
    else:
        sei_score = 0.2

    # Normalize IR (min=2.25, max=25.56)
    ir_norm = (25.56 - ir_pred) / (25.56 - 2.25)
    ir_norm = max(0, min(ir_norm, 1))  # clamp between 0 and 1

    # Normalize SOH (min=70, max=122)
    soh_norm = (soh_pred - 70) / (122 - 70)
    soh_norm = max(0, min(soh_norm, 1))

    # CSI formula
    csi = 0.35 * sei_score + 0.30 * ir_norm + 0.35 * soh_norm

    # Safety Category
    if csi >= 0.8:
        category = "✅ Safe"
    elif csi >= 0.6:
        category = "⚠️ Moderate"
    elif csi >= 0.4:
        category = "⚠️ Warning"
    else:
        category = "❌ Critical"

    # Output results
    st.subheader("📊 Prediction Results")
    st.markdown(f"🔬 **Predicted SEI Thickness:** `{sei_pred:.4f} nm`")
    st.markdown(f"🔌 **Predicted Internal Resistance (IR):** `{ir_pred:.2f} mΩ`")
    st.markdown(f"💚 **Predicted State of Health (SOH):** `{soh_pred:.2f} %`")
    st.markdown(f"🛡️ **Calculated CSI:** `{csi:.3f}` → **{category}**")

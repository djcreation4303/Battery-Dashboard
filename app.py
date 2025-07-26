import streamlit as st
import pandas as pd
import numpy as np
import joblib  # ← Use joblib instead of pickle

# Load trained models using joblib
sei_model = joblib.load("sei_model.pkl")
ir_model = joblib.load("ir_model.pkl")
soh_model = joblib.load("soh_model.pkl")

st.set_page_config(page_title="Battery Safety Predictor", layout="centered")
st.title("🔋 Lithium-ion Battery Safety & Health Prediction")

st.markdown("Enter the battery details below. The app will predict SEI, IR, SOH and calculate CSI with safety categorization.")

# Maps for encoding
chemistry_map = {'LFP': 0, 'NMC': 1}
charging_map = {'normal': 1, 'fast': 0, 'overnight': 2}

# User Inputs
st.header("📥 User Inputs")

cycle_count = st.number_input("Cycle Count", min_value=0, value=200)
charge_rate = st.number_input("Charge Rate (C)", min_value=0.0, value=1.0)
discharge_rate = st.number_input("Discharge Rate (C)", min_value=0.0, value=1.0)
depth_of_discharge = st.slider("Depth of Discharge (%)", 0, 100, 80)
storage_time_months = st.number_input("Storage Time (months)", min_value=0, value=6)
battery_age_months = st.number_input("Battery Age (months)", min_value=0, value=12)
ambient_temperature = st.slider("Ambient Temperature (°C)", 15, 45, 25)
current_voltage = st.slider("Current Voltage (V)", 3.2, 4.2, 3.7)

charging_input = st.selectbox("Charging Behavior", list(charging_map.keys()))
chemistry_input = st.selectbox("Chemistry Type", list(chemistry_map.keys()))

# Encode categorical inputs
charging_behavior_encoded = charging_map[charging_input]
chemistry_type_encoded = chemistry_map[chemistry_input]

# Create input DataFrame
input_features = pd.DataFrame([[
    cycle_count,	
    depth_of_discharge,
    battery_age_months,
    chemistry_type_encoded,	
    charging_behavior_encoded,
    ambient_temperature,
    storage_time_months,
    current_voltage,
   
]], columns=[
    "cycle_count",	
    "depth_of_discharge",
    "battery_age_months",
    "chemistry_type_encoded",	
    "charging_behavior_encoded",
    "ambient_temperature",
    "storage_time_months",
    "current_voltage"
])

input_features_sei = input_features[[
    "cycle_count",
    "depth_of_discharge",
    "storage_time_months",
    "battery_age_months",
    "current_voltage",
    "chemistry_type_encoded",
    "charging_behavior_encoded"
]]

input_features_ir = input_features[[
    "cycle_count",
    "depth_of_discharge",
    "ambient_temperature",
    "battery_age_months",
    "chemistry_type_encoded",
    "charging_behavior_encoded",
    "current_voltage"
]]




# Prediction logic
if st.button("🔍 Predict Battery Health & Safety"):
    sei_pred = sei_model.predict(input_features_sei)[0]
    ir_pred = ir_model.predict(input_features_ir)[0]

    soh_input = pd.DataFrame([[sei_pred, ir_pred, battery_age_months, cycle_count, current_voltage, depth_of_discharge, chemistry_type_encoded]], 
                             columns=["SEI", "IR", "battery_age_months", "cycle_count", "current_voltage", "depth_of_discharge", "chemistry_type_encoded"])
    soh_pred = soh_model.predict(soh_input)[0]

    # SEI scoring
    if sei_pred <= 0.4:
        sei_score = 1
    elif sei_pred <= 0.7:
        sei_score = 1 - (sei_pred - 0.4) / (0.7 - 0.4)
    else:
        sei_score = 0.2

    # Normalization
    ir_norm = (122 - ir_pred) / (122 - 63)
    soh_norm = (soh_pred - 50) / (100 - 50)

    # CSI calculation
    csi = 0.35 * sei_score + 0.30 * ir_norm + 0.35 * soh_norm

    # Category
    if csi >= 0.8:
        category = "Safe ✅"
    elif csi >= 0.6:
        category = "Moderate ⚠️"
    elif csi >= 0.4:
        category = "Warning ⚠️"
    else:
        category = "Critical ❌"

    # Display predictions
    st.markdown(f"**Predicted SEI:** `{sei_pred:.3f}`")
    st.markdown(f"**Predicted IR:** `{ir_pred:.2f} mΩ`")
    st.markdown(f"**Predicted SOH:** `{soh_pred:.2f} %`")
    st.markdown(f"**Calculated CSI:** `{csi:.3f}` → **{category}**")

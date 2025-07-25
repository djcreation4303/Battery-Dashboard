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
charging_map = {'normal': 0, 'fast': 1, 'overnight': 2}

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
    charge_rate,
    discharge_rate,
    depth_of_discharge,
    storage_time_months,
    battery_age_months,
    chemistry_type_encoded,	
    charging_behavior_encoded,
    current_voltage,
    ambient_temperature
]], columns=[
    "cycle_count",	
    "charge_rate",
    "discharge_rate",
    "depth_of_discharge",
    "storage_time_months",
    "battery_age_months",
    "chemistry_type_encoded",	
    "charging_behavior_encoded",
    "current_voltage",
    "ambient_temperature"
])

# Prediction logic
if st.button("🔍 Predict Battery Health & Safety"):
    input_features = input_features[sei_model.feature_names_in_]


    sei_pred = sei_model.predict(input_features)[0]
    ir_pred = ir_model.predict(input_features)[0]

    soh_pred = soh_model.predict(
        pd.DataFrame([[sei_pred, ir_pred]], columns=["SEI", "IR"])
    )[0]

    st.write("SEI:", sei_pred
    st.write("IR:", ir_pred)
    st.write("SOH:", soh_pred)

    # Calculate CSI
    csi = ((1 - sei_pred) * 0.4 + (110 - ir_pred) / 110 * 0.3 + soh_pred / 100 * 0.3)

    # Categorize
    if csi >= 0.8:
        category = "Safe ✅"
    elif csi >= 0.6:
        category = "Moderate ⚠️"
    elif csi >= 0.4:
        category = "Warning ⚠️"
    else:
        category = "Critical ❌"

    # Display
    st.header("📊 Results")
    st.markdown(f"**Predicted SEI:** `{sei_pred:.3f}`")
    st.markdown(f"**Predicted IR:** `{ir_pred:.2f} mΩ`")
    st.markdown(f"**Predicted SOH:** `{soh_pred:.2f} %`")
    st.markdown(f"**Calculated CSI:** `{csi:.3f}`")
    st.markdown(f"### 🛡️ Safety Category: **{category}**")

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
sei_model = joblib.load("sei_model_comressed.pkl")
ir_model = joblib.load("ir_model_comressed.pkl")
soh_model = joblib.load("soh_model_comressed.pkl")

# Battery ID mapping
battery_id_map = {
    'B0005': 0, 'B0006': 1, 'B0007': 2, 'B0018': 3,
    'B0025': 4, 'B0026': 5, 'B0027': 6, 'B0028': 7,
    'B0029': 8, 'B0030': 9, 'B0031': 10, 'B0032': 11,
    'B0033': 12, 'B0034': 13, 'B0036': 14, 'B0038': 15,
    'B0039': 16, 'B0040': 17, 'B0042': 18, 'B0043': 19,
    'B0044': 20, 'B0046': 21, 'B0047': 22, 'B0048': 23
}

# Streamlit setup
st.set_page_config(page_title="Battery Safety Predictor", layout="centered")
st.title("🔋 Lithium-ion Battery Safety & Health Prediction")
st.markdown("Enter the battery details below to predict SEI, IR, SOH and assess safety (CSI).")

# Battery ID dropdown
battery_selection = st.selectbox("Select Battery ID", options=[f"{k} (→ {v})" for k, v in battery_id_map.items()])
battery_id_encoded = battery_id_map[battery_selection.split()[0]]

# User Inputs
st.header("📥 Battery Input Parameters")

cycle_number = st.number_input("Cycle Number", min_value=0, value=150)
voltage_measured = st.number_input("Voltage Measured (V)", min_value=3.0, max_value=4.5, value=3.7)
current_measured = st.number_input("Current Measured (A)", min_value=0.0, max_value=5.0, value=1.0)
temperature_measured = st.number_input("Temperature Measured (°C)", min_value=15.0, max_value=60.0, value=25.0)
soc = st.slider("State of Charge (SoC) %", 0, 100, 80)

# Prediction Trigger
if st.button("🔍 Predict Health and Safety"):

    # Prepare input for SEI model
    sei_input = pd.DataFrame([[
        cycle_number, voltage_measured, current_measured, temperature_measured, soc, battery_id_encoded
    ]], columns=["cycle_number", "Voltage_measured", "Current_measured", "Temperature_measured", "SoC", "battery_id_encoded"])

    sei_pred = sei_model.predict(sei_input)[0]

    # Prepare input for IR model (using SEI output)
    ir_input = sei_input.copy()
    ir_input["SEI_pred"] = sei_pred
    ir_pred = ir_model.predict(ir_input)[0]

    # Prepare input for SOH model (using SEI and IR)
    soh_input = pd.DataFrame([[
        sei_pred, ir_pred, cycle_number, voltage_measured, current_measured, temperature_measured, soc
    ]], columns=["SEI", "IR", "cycle_number", "Voltage_measured", "Current_measured", "Temperature_measured", "SoC"])

    soh_pred = soh_model.predict(soh_input)[0]

    # CSI Logic
    # Normalize values
    sei_score = 1 - (sei_pred - 0.0657) / (0.425 - 0.0657)
    sei_score = max(0, min(sei_score, 1))

    ir_score = 1 - (ir_pred - 2.25) / (25.56 - 2.25)
    ir_score = max(0, min(ir_score, 1))

    soh_score = (soh_pred - 70) / (122 - 70)
    soh_score = max(0, min(soh_score, 1))

    # Final CSI
    csi = round(0.35 * sei_score + 0.30 * ir_score + 0.35 * soh_score, 3)

    # Category
    if csi >= 0.8:
        category = "Safe ✅"
    elif csi >= 0.6:
        category = "Moderate ⚠️"
    elif csi >= 0.4:
        category = "Warning ⚠️"
    else:
        category = "Critical ❌"

    # Display results
    st.markdown(f"**Predicted SEI (nm):** `{sei_pred:.4f}`")
    st.markdown(f"**Predicted IR (mΩ):** `{ir_pred:.2f}`")
    st.markdown(f"**Predicted SOH (%):** `{soh_pred:.2f}`")
    st.markdown(f"**Calculated CSI:** `{csi}` → **{category}**")


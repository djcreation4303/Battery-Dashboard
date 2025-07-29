import streamlit as st
import pandas as pd
import joblib
import numpy as np

# === Load Compressed Models ===
sei_model = joblib.load("soh_model_comressed.pkl")
ir_model = joblib.load("ir_model_comressed.pkl")
soh_model = joblib.load("soh_model_comressed.pkl")

# === Battery ID Mapping ===
battery_id_map = {
    'B0005': 0, 'B0006': 1, 'B0007': 2, 'B0018': 3,
    'B0025': 4, 'B0026': 5, 'B0027': 6, 'B0028': 7,
    'B0029': 8, 'B0030': 9, 'B0031': 10, 'B0032': 11,
    'B0033': 12, 'B0034': 13, 'B0036': 14, 'B0038': 15,
    'B0039': 16, 'B0040': 17, 'B0042': 18, 'B0043': 19,
    'B0044': 20, 'B0046': 21, 'B0047': 22, 'B0048': 23
}

# === Streamlit Page Settings ===
st.set_page_config(page_title="🔋 Battery Safety & SOH Dashboard", layout="centered")
st.title("🔋 Lithium-ion Battery Safety & Health Predictor")
st.markdown("Enter the input parameters to predict **SEI**, **IR**, **SOH**, and get the safety score (**CSI**).")

# === Battery ID Selection ===
battery_display = [f"{k} (→ {v})" for k, v in battery_id_map.items()]
selected_battery = st.selectbox("Select Battery ID", options=battery_display)
battery_id_encoded = battery_id_map[selected_battery.split()[0]]

# === User Inputs ===
st.header("📥 Input Parameters")

cycle_number = st.number_input("Cycle Number", min_value=0, max_value=5000, value=150)
voltage_measured = st.number_input("Voltage Measured (V)", min_value=2.5, max_value=5.0, value=3.7)
current_measured = st.number_input("Current Measured (A)", min_value=0.0, max_value=10.0, value=1.0)
temperature_measured = st.number_input("Temperature (°C)", min_value=0.0, max_value=100.0, value=25.0)
soc = st.slider("State of Charge (SoC) %", min_value=0, max_value=100, value=80)


# === Prediction Trigger ===
if st.button("🔍 Predict"):

    # === 1. SEI Prediction ===
    sei_input = pd.DataFrame([[cycle_number, voltage_measured, current_measured,
                               temperature_measured, soc, battery_id_encoded]],
                             columns=["cycle_number", "Voltage_measured", "Current_measured",
                                      "Temperature_measured", "SoC", "battery_id_encoded"])

    st.write("Model expects these features:", sei_model.feature_names_in_)
    st.write("You provided these features:", list(sei_input.columns))

    sei_pred = sei_model.predict(sei_input)[0]

    # === 2. IR Prediction ===
    ir_input = sei_input.copy()
    ir_input["SEI_pred"] = sei_pred
    ir_pred = ir_model.predict(ir_input)[0]

    # === 3. SOH Prediction ===
        # === 3. SOH Prediction ===
    soh_input = pd.DataFrame([[cycle_number, sei_pred, ir_pred, battery_id_encoded]],
                             columns=["cycle_number", "SEI_pred", "IR_pred", "battery_id_encoded"])
    soh_pred = soh_model.predict(soh_input)[0]


    # === 4. CSI Calculation ===
    sei_score = 1 - (sei_pred - 0.0657) / (0.425 - 0.0657)
    sei_score = max(0, min(sei_score, 1))

    ir_score = 1 - (ir_pred - 2.25) / (25.56 - 2.25)
    ir_score = max(0, min(ir_score, 1))

    soh_score = (soh_pred - 70) / (122 - 70)
    soh_score = max(0, min(soh_score, 1))

    csi = round(0.35 * sei_score + 0.30 * ir_score + 0.35 * soh_score, 3)

    # === 5. Safety Category ===
    if csi >= 0.8:
        category = "🟢 Safe"
    elif csi >= 0.6:
        category = "🟡 Moderate"
    elif csi >= 0.4:
        category = "🟠 Warning"
    else:
        category = "🔴 Critical"

    # === 6. Results ===
    st.subheader("📊 Prediction Results")
    st.markdown(f"**Predicted SEI Thickness (nm):** `{sei_pred:.4f}`")
    st.markdown(f"**Predicted Internal Resistance (mΩ):** `{ir_pred:.2f}`")
    st.markdown(f"**Predicted SOH (%):** `{soh_pred:.2f}`")
    st.markdown(f"**CSI Score:** `{csi}` → **{category}**")

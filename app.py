import streamlit as st
import numpy as np
import joblib

# ------------ Load Models ------------
sei_model = joblib.load("sei_model.pkl")
ir_model = joblib.load("ir_model.pkl")
soh_model = joblib.load("soh_model.pkl")

# ------------ Page Settings ------------
st.set_page_config(page_title="Battery Safety Checker", layout="centered")
st.title("🔋 Battery Safety Check using CSI")
st.write("Enter battery usage details to predict SEI, IR, SOH, and compute CSI with safety category.")

# ------------ User Input ------------
st.header("📥 Battery Input Parameters")


current_voltage = st.number_input("Current Voltage (V)", value=3.7, step=0.01)

ambient_temp = st.number_input("Ambient Temperature (°C)", value=25.0, step=0.5)
cycle_count = st.number_input("Charge Cycles Completed", value=300, step=1)

charging_behavior_encoded = st.selectbox("Charging Behavior", ["Normal", "Fast", "Overnight"])
chemistry_type_encoded = st.selectbox("Chemistry Type", ["LFP", "NMC"])

# ------------ Feature Encoding ------------
charging_map = {"Normal": 0, "Fast": 1, "Overnight": 2}
chemistry_map = {"LFP": 0, "NMC": 1}

input_features = np.array([[
    current_voltage,
    ambient_temp,
    cycle_count,
    charging_map[charging_behavior],
    chemistry_map[chemistry_type]
]])

# ------------ Predictions ------------
if st.button("🔍 Predict Safety"):

    # Predict SEI and IR
    sei = sei_model.predict(input_features)[0]
    ir = ir_model.predict(input_features)[0]

    # Now predict SOH using SEI, IR + selected features
    soh_input = np.hstack((input_features, [sei, ir]))
    soh = soh_model.predict([soh_input])[0]

    # Calculate CSI
    csi = soh / (sei * ir) if sei * ir != 0 else 0

    # Categorize CSI
    if csi > 0.9:
        category = "✅ Safe"
    elif csi > 0.7:
        category = "⚠️ Moderate"
    elif csi > 0.5:
        category = "🟠 Warning"
    else:
        category = "🔴 Critical"

    # ------------ Output ------------
    st.subheader("🧠 Model Predictions")
    st.metric("Predicted SEI", f"{sei:.3f}")
    st.metric("Predicted IR (mΩ)", f"{ir:.2f}")
    st.metric("Predicted SOH (%)", f"{soh:.2f}")
    st.metric("CSI Score", f"{csi:.4f}")
    st.metric("Safety Category", category)

    with st.expander("📊 Input Summary"):
        st.write({
            "Min Voltage": min_voltage,
            "Max Voltage": max_voltage,
            "Avg Voltage": avg_voltage,
            "Battery Temp": temperature,
            "Ambient Temp": ambient_temp,
            "Charge Cycles": charge_cycles,
            "Charging": charging_behavior,
            "Chemistry": chemistry_type,
            "SEI": round(sei, 3),
            "IR": round(ir, 2),
            "SOH": round(soh, 2),
            "CSI": round(csi, 4),
            "Category": category
        })




              

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(layout="wide")
st.title("🔋 Battery Dashboard")

# 1. Load data
@st.cache_data
def load_data():
    return pd.read_csv("Battery_Ageing_With_CSI_Final.csv")


df = load_data()

# 2. Load trained model
@st.cache_resource
def load_model():
    model_path = "soh_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("❌ Model file 'soh_model.pkl' not found in directory!")
        return None

model = load_model()

# 3. Metrics
st.subheader("📊 Battery Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("🔁 Cycles", df["Cycle"].max())
col2.metric("🔋 Avg SOH (%)", f"{df['SOH (%)'].mean():.2f}")
col3.metric("⚡ Avg Voltage", f"{df['Voltage'].mean():.2f} V")

# 4. Plots
st.subheader("📈 Battery Trends")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
sns.lineplot(data=df, x="Cycle", y="SOH (%)", ax=ax1)
ax1.set_title("SOH (%) vs Cycle")
ax1.set_ylabel("SOH (%)")
ax1.set_xlabel("Cycle")

sns.lineplot(data=df, x="Cycle", y="CSI", ax=ax2)
ax2.set_title("CSI vs Cycle")
ax2.set_ylabel("CSI")
ax2.set_xlabel("Cycle")

st.pyplot(fig)

# 5. SOH Prediction Based on User Input
st.subheader("🧠 Predict SOH Based on Input")

if model:
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            cycle = st.number_input("Cycle Number", min_value=0, step=1)
            resistance = st.number_input("Resistance_norm", min_value=0.0, step=0.01)
            capacity = st.number_input("Capacity_norm", min_value=0.0, step=0.01)
        with col2:
            temp = st.number_input("Temp_norm", min_value=0.0, step=0.01)
            voltage = st.number_input("Voltage_norm", min_value=0.0, step=0.01)

        submitted = st.form_submit_button("Predict SOH")

        if submitted:
            input_df = pd.DataFrame([{
                "Resistance_norm": resistance,
                "Capacity_norm": capacity,
                "Temp_norm": temp,
                "Voltage_norm": voltage
            }])

            prediction = model.predict(input_df)[0]
            st.success(f"🔮 Predicted SOH at Cycle {cycle}: **{prediction:.2f}%**")


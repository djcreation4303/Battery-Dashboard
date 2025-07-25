import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load models
sei_model = pickle.load(open("sei_model.pkl", "rb"))
ir_model = pickle.load(open("ir_model.pkl", "rb"))
soh_model = pickle.load(open("soh_model.pkl", "rb"))

# User inputs
cycle_count = st.number_input("Cycle Count", min_value=0)
avg_temp = st.number_input("Ambient Temperature (°C)")
charge_rate = st.number_input("Charge Rate (C)")
discharge_rate = st.number_input("Discharge Rate (C)")
depth_of_discharge = st.number_input("Depth of Discharge (%)")
storage_time = st.number_input("Storage Time (months)")
battery_age = st.number_input("Battery Age (months)")
current_voltage = st.number_input("Current Voltage (V)")

# Dropdowns
charging_map = {'normal': 0, 'fast': 1, 'overnight': 2}
charging_behavior = st.selectbox("Charging Behavior", list(charging_map.keys()))
charging_behavior_encoded = charging_map[charging_behavior]

chemistry_map = {'NMC': 0, 'LFP': 1}
chemistry_type = st.selectbox("Chemistry Type", list(chemistry_map.keys()))
chemistry_type_encoded = chemistry_map[chemistry_type]

# Final feature vector (must match training)
input_data = pd.DataFrame([[
    cycle_count, avg_temp, charge_rate, discharge_rate,
    depth_of_discharge, storage_time, battery_age,
    chemistry_type_encoded, charging_behavior_encoded,
    current_voltage
]], columns=[
    'cycle_count', 'ambient_temp', 'charge_rate', 'discharge_rate',
    'depth_of_discharge', 'storage_time_months', 'battery_age_months',
    'chemistry_type_encoded', 'charging_behavior_encoded', 'current_voltage'
])

# Predict
sei = sei_model.predict(input_data)[0]
ir = ir_model.predict(input_data)[0]

# Then use sei, ir to get SOH and CSI


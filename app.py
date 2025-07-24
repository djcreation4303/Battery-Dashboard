import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Title
st.title("🔋 Battery Health Dashboard")
st.markdown("Visualize CSI Score, SOH Prediction, and Degradation Trends")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("Battery_Ageing_With_CSI_Final.csv")

df = load_data()

# Show Data Preview
st.subheader("📄 Raw Data Preview")
st.dataframe(df.head())

# Plot: SOH vs Cycle
st.subheader("📈 SOH (%) vs Cycle_Number")
fig1, ax1 = plt.subplots()
sns.lineplot(data=df, x="Cycle_Number", y="SOH (%)", ax=ax1)
st.pyplot(fig1)

# Plot: CSI Score vs Cycle
if "CSI" in df.columns:
    st.subheader("📊 CSI vs Cycle_Number")
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=df, x="Cycle_Number", y="CSI", ax=ax2)
    st.pyplot(fig2)

# Category Count Plot
if "Category" in df.columns:
    st.subheader("🧠 CSI Category Count")
    fig3, ax3 = plt.subplots()
    sns.countplot(data=df, x="Category", ax=ax3)
    st.pyplot(fig3)

# Upload Trained Model to Predict SOH
st.subheader("📤 Upload Trained SOH Model")
model_file = st.file_uploader("Upload .pkl model", type=["pkl"])

if model_file is not None:
    model = joblib.load(model_file)
    
    # Feature Inputs (use normalized ones you used in training)
    input_cols = ['Cycle_Number',  'Resistance_norm', 'Capacity_norm', 'Temp_norm', 'SEI_norm']
    
    if all(col in df.columns for col in input_cols):
        X = df[input_cols]
        st.success("✅ Predicting SOH based on uploaded model")
        predictions = model.predict(X)
        df['Predicted_SOH'] = predictions
        st.write(df[['Cycle_Number', 'Predicted_SOH']].head())

        # Plot: Predicted SOH vs Cycle
        st.subheader(" Predicted SOH vs Cycle")
        fig4, ax4 = plt.subplots()
        sns.lineplot(x=df["Cycle_Number"], y=df["Predicted_SOH"], ax=ax4)
        st.pyplot(fig4)
    else:
        st.warning("⚠️ Missing normalized input columns in your data.")

# Footer
st.markdown("---")
st.caption("Made with  by Team Voltcrafters")

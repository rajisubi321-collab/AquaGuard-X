import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AquaGuard-X",
    page_icon="💧",
    layout="wide"
)

st.title("💧 AquaGuard-X")
st.subheader("AI-Based Water Quality Deterioration Early Warning System")

# Load files
data = pd.read_csv("aquaguard_risk_data.csv")
model = joblib.load("aquaguard_future_prediction.pkl")

st.sidebar.header("Sensor Readings")

ph = st.sidebar.number_input(
    "pH",
    min_value=0.0,
    max_value=14.0,
    value=7.0
)

tds = st.sidebar.number_input(
    "TDS (ppm)",
    min_value=0.0,
    value=350.0
)

turbidity = st.sidebar.number_input(
    "Turbidity (NTU)",
    min_value=0.0,
    value=4.0
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    value=27.0
)

# Display sensor values
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("pH", f"{ph:.2f}")

with col2:
    st.metric("TDS", f"{tds:.0f} ppm")

with col3:
    st.metric("Turbidity", f"{turbidity:.1f} NTU")

with col4:
    st.metric("Temperature", f"{temperature:.1f} °C")

# Prediction
input_data = pd.DataFrame({
    "ph": [ph],
    "tds": [tds],
    "turbidity": [turbidity],
    "temperature": [temperature],
    "ph_change": [0],
    "tds_change": [0],
    "turbidity_change": [0],
    "temperature_change": [0]
})

probability = model.predict_proba(input_data)[0][1]

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Future Deterioration Probability")
    st.metric(
        "Probability",
        f"{probability * 100:.2f}%"
    )

with col2:
    st.subheader("System Status")

    if probability >= 0.7:
        st.error("🔴 HIGH RISK")
    elif probability >= 0.4:
        st.warning("🟡 WARNING")
    else:
        st.success("🟢 SAFE")

# Graph
st.markdown("---")

st.subheader("📈 Water Quality History")

if "time" in data.columns and "turbidity" in data.columns:

    fig, ax = plt.subplots()

    ax.plot(
        data["time"],
        data["turbidity"]
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("Turbidity (NTU)")
    ax.set_title("Turbidity Trend")

    st.pyplot(fig)

else:
    st.info("Required graph columns are not available in the CSV file.")

st.markdown("---")

st.subheader("🔍 AI Explanation")

if probability >= 0.7:

    st.write(
        "The model indicates a high possibility of future "
        "water-quality deterioration based on the current readings."
    )

elif probability >= 0.4:

    st.write(
        "The model indicates a moderate deterioration signal. "
        "Continuous monitoring is recommended."
    )

else:

    st.write(
        "The current sensor readings indicate a relatively stable condition."
    )

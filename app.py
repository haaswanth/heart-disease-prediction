import streamlit as st
import pandas as pd
import pickle
import joblib

# -----------------------------
# Load saved files
# -----------------------------

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)


# -----------------------------
# Page settings
# -----------------------------

st.set_page_config(
    page_title="Haaswanth Heart Disease Prediction",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details below to predict the result.")


# -----------------------------
# User inputs
# -----------------------------

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=40
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=0,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=700,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl?",
    ["No", "Yes"]
)

max_hr = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)

sex = st.selectbox(
    "Sex",
    ["M", "F"]
)

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ASY", "ATA", "NAP", "TA"]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["LVH", "Normal", "ST"]
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["N", "Y"]
)

st_slope = st.selectbox(
    "ST Slope",
    ["Down", "Flat", "Up"]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    # Start with all columns as 0
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=columns
    )

    # Numerical values
    input_data["Age"] = age
    input_data["RestingBP"] = resting_bp
    input_data["Cholesterol"] = cholesterol
    input_data["FastingBS"] = 1 if fasting_bs == "Yes" else 0
    input_data["MaxHR"] = max_hr
    input_data["Oldpeak"] = oldpeak

    # One-hot encoded values
    input_data[f"Sex_{sex}"] = 1

    input_data[f"ChestPainType_{chest_pain}"] = 1

    input_data[f"RestingECG_{resting_ecg}"] = 1

    input_data[f"ExerciseAngina_{exercise_angina}"] = 1

    input_data[f"ST_Slope_{st_slope}"] = 1

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
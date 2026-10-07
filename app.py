import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤️",
    layout="centered"
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

MODEL_PATH = Path(__file__).parent / "model" / "heart_disease_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Application Title
# --------------------------------------------------

st.title("AI-Based Heart Disease Risk Prediction")

st.write(
    "Enter the patient's health information below to estimate "
    "the predicted heart disease risk using a machine learning model."
)

st.info(
    "This application is an educational machine-learning project "
    "and is not intended for medical diagnosis."
)


# --------------------------------------------------
# Patient Information
# --------------------------------------------------

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex_label = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=600,
        value=200
    )

    fbs_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        ["No", "Yes"]
    )

with col2:
    cp_label = st.selectbox(
        "Chest Pain Type",
        [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ]
    )

    restecg_label = st.selectbox(
        "Resting ECG Result",
        [
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        ]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang_label = st.selectbox(
        "Exercise-Induced Angina?",
        ["No", "Yes"]
    )


# --------------------------------------------------
# Additional Clinical Information
# --------------------------------------------------

st.subheader("Additional Clinical Information")

col3, col4 = st.columns(2)

with col3:

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope_label = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        [
            "Upsloping",
            "Flat",
            "Downsloping"
        ]
    )

with col4:

    ca = st.selectbox(
        "Number of Major Vessels",
        [0, 1, 2, 3]
    )

    thal_label = st.selectbox(
        "Thalassemia Result",
        [
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        ]
    )


# --------------------------------------------------
# Convert User-Friendly Values to Model Values
# --------------------------------------------------

sex = 1 if sex_label == "Male" else 0

cp_mapping = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-anginal Pain": 3,
    "Asymptomatic": 4
}

fbs = 1 if fbs_label == "Yes" else 0

restecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

exang = 1 if exang_label == "Yes" else 0

slope_mapping = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}

thal_mapping = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}

cp = cp_mapping[cp_label]
restecg = restecg_mapping[restecg_label]
slope = slope_mapping[slope_label]
thal = thal_mapping[thal_label]


# --------------------------------------------------
# Create Input DataFrame
# --------------------------------------------------

input_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs],
    "restecg": [restecg],
    "thalach": [thalach],
    "exang": [exang],
    "oldpeak": [oldpeak],
    "slope": [slope],
    "ca": [ca],
    "thal": [thal]
})


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.divider()

if st.button(
    "Predict Heart Disease Risk",
    type="primary",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    probability_percentage = probability * 100

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error(
            "Higher Risk of Heart Disease"
        )

    else:

        st.success(
            "Lower Risk of Heart Disease"
        )

    st.metric(
        "Estimated Model Probability",
        f"{probability_percentage:.2f}%"
    )

    st.caption(
        "This probability is generated by the machine-learning model "
        "and should not be interpreted as a medical diagnosis."
    )
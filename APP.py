import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# === Streamlit Page Config ===
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Risk Prediction")
st.markdown("This app uses a Multiple Linear Regression model to predict the likelihood of heart disease.")

# === Sidebar Model Selector ===
st.sidebar.header("Select Model Iteration")

# Get all pkl files in the models folder
model_dir = "models"
available_models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

if not available_models:
    st.error(f"No .pkl model files found in '{model_dir}'. Please add your model file and restart the app.")
    st.stop()  # stops the app safely
else:
    selected_model = st.sidebar.selectbox("Choose a model iteration:", available_models)
    MODEL_PATH = os.path.join(model_dir, selected_model)


# === Load Model ===
@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data

model_bundle = load_model(MODEL_PATH)
model = model_bundle["model"]
preprocessor = model_bundle["preprocessor"]
scaler = model_bundle["scaler"]

# === Sidebar Inputs ===
st.sidebar.header("Input Parameters")

def user_input_features():
    age = st.sidebar.number_input("Age", 20, 100, 50)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    chest_pain_type = st.sidebar.number_input("Chest Pain Type (0–3)", 0, 3, 1)
    resting_blood_pressure = st.sidebar.number_input("Resting BP", 80, 200, 120)
    cholesterol = st.sidebar.number_input("Cholesterol", 100, 600, 200)
    fasting_blood_sugar = st.sidebar.selectbox("Fasting Blood Sugar >120", [1, 0])
    resting_ecg = st.sidebar.number_input("Resting ECG (0–2)", 0, 2, 1)
    max_heart_rate = st.sidebar.number_input("Max Heart Rate", 60, 220, 150)
    exercise_induced_angina = st.sidebar.selectbox("Exercise Angina (1=Yes, 0=No)", [1, 0])
    st_depression = st.sidebar.number_input("ST Depression", 0.0, 6.0, 1.0)
    st_slope = st.sidebar.number_input("ST Slope (0–2)", 0, 2, 1)
    num_major_vessels = st.sidebar.number_input("Major Vessels (0–3)", 0, 3, 0)
    thalassemia = st.sidebar.number_input("Thalassemia (0–3)", 0, 3, 1)

    data = {
        'age': age,
        'sex': sex,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': fasting_blood_sugar,
        'resting_ecg': resting_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': exercise_induced_angina,
        'st_depression': st_depression,
        'st_slope': st_slope,
        'num_major_vessels': num_major_vessels,
        'thalassemia': thalassemia
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# === Display Input Data ===
st.subheader("User Input:")
st.write(input_df)

# === Make Prediction ===
X_proc = preprocessor.transform(input_df)
X_scaled = scaler.transform(X_proc)
prediction = model.predict(X_scaled)

st.subheader("Model Prediction:")
st.metric(label="Predicted Heart Disease Score", value=f"{prediction[0]:.3f}")

# === Optional Graph Display ===
graph_path = os.path.join("assets", "performance_graphs.png")
if os.path.exists(graph_path):
    st.subheader("Training Performance Graph")
    st.image(graph_path, caption="MSE and R² Trends", use_column_width=True)

st.markdown("---")
st.caption(f"Active model: {selected_model}")

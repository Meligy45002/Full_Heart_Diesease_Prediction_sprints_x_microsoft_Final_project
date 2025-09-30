import streamlit as st
import joblib
import numpy as np
import pandas as pd

model = joblib.load("final_model.pkl")

st.title("❤️ Heart Disease Prediction App")

# Example features (adjust depending on your selected features)
age = st.number_input("Age", 20, 100, 50)
chol = st.number_input("Cholesterol", 100, 600, 200)
thalach = st.number_input("Max Heart Rate Achieved", 50, 220, 150)
cp = st.selectbox("Chest Pain Type (0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic)", [0,1,2,3])

if st.button("Predict"):
    # Adjust order of features to match model training
    features = np.array([[age, chol, thalach, cp]])
    prediction = model.predict(features)[0]
    st.subheader("Result:")
    st.write("✅ No Heart Disease" if prediction == 0 else "⚠️ Heart Disease Detected")



import numpy as np
import pandas as pd
import joblib
import streamlit as st

scaler = joblib.load('scaler.pkl')  # Load the scaler
le_gender = joblib.load('label_encoder_gender.pkl')  # Load the label encoder for gender
le_diabetic = joblib.load('label_encoder_diabetic.pkl')  # Load the label encoder for diabetic
le_smoker = joblib.load('label_encoder_smoker.pkl')  # Load the label encoder for smoker
model = joblib.load('best_model.pkl')  # Load the best model

st.set_page_config(page_title="Insurance Claim Prediction", layout="centered")
st.title("Health Insurance Payment Prediction App")
st.write("Enter the details below to predict the health insurance payment.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

    with col2:
        blood_pressure = st.number_input("Blood Pressure", min_value=60, max_value=200, value=120)
        gender = st.selectbox("Gender", options=le_gender.classes_)
        diabetic = st.selectbox("Diabetic", options=le_diabetic.classes_) 
        smoker = st.selectbox("Smoker", options=le_smoker.classes_)

    submitted = st.form_submit_button("Predict Payment")

if submitted:
    input_data = pd.DataFrame({
        'age': [age],
        'bmi': [bmi],
        'children': [children],
        'bloodpressure': [blood_pressure],
        'gender': [gender],
        'diabetic': [diabetic],
        'smoker': [smoker]
    })

    input_data['gender'] = le_gender.transform(input_data['gender'])
    input_data['diabetic'] = le_diabetic.transform(input_data['diabetic'])
    input_data['smoker'] = le_smoker.transform(input_data['smoker'])

    num_col = ['age', 'bmi', 'bloodpressure', 'children']

    print("Scaler expects:", scaler.feature_names_in_)
    print("num_col:", num_col)
    print("Input columns:", input_data[num_col].columns.tolist())

    input_data[num_col] = scaler.transform(input_data[num_col])

    input_data = input_data[['age', 'gender', 'bmi', 'bloodpressure', 'children', 'diabetic', 'smoker']]

    prediction = model.predict(input_data)

    st.success(f"Estimated Health Insurance Payment: ${prediction[0]:,.2f}")
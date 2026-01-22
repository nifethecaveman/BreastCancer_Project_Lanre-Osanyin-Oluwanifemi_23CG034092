import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('model/breast_cancer_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.title("Breast Cancer Prediction System")
st.write("Enter the tumor features below to predict if it is Benign or Malignant.")

# Input fields for the 5 selected features
radius = st.number_input("Radius Mean", value=14.0)
texture = st.number_input("Texture Mean", value=19.0)
perimeter = st.number_input("Perimeter Mean", value=92.0)
area = st.number_input("Area Mean", value=650.0)
smoothness = st.number_input("Smoothness Mean", value=0.1)

if st.button("Predict"):
    # Prepare input for prediction
    input_data = np.array([[radius, texture, perimeter, area, smoothness]])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    result = "Malignant" if prediction[0] == 1 else "Benign"
    
    st.header(f"The tumor is predicted to be: {result}")
    st.warning("Disclaimer: This is for educational purposes only.")
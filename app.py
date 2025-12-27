import numpy as np
import pandas as pd
import streamlit as st
import joblib
le_gender=joblib.load("label_encoder_gender.pkl")
le_diabetic=joblib.load("label_encoder_diabetic.pkl")
le_region=joblib.load("label_encoder_region.pkl")
le_smoker=joblib.load("label_encoder_smoker.pkl")
model=joblib.load("model.pkl")

st.set_page_config(page_title="Insurance claim predictor",layout="centered")
st.title("Health Insurance Payment Prediction App")
st.write("Enter the details below to estimate your insurance payment amount.")


with st.form("input_form"):
    col1,col2=st.columns(2)
    with col1:
        age=st.number_input("Age",min_value=0,max_value=100,value=30,step=1)
        bmi=st.number_input("BMI",min_value=10.0,max_value=60.0,value=25.5,step=0.1)
        children=st.number_input("Number of children",min_value=0,max_value=8,value=0,step=1)
        

    with col2:
        bloodpressure=st.number_input("Blood Pressure",min_value=60,max_value=200,value=120,step=1)
        gender=st.selectbox("Gender",options=le_gender.classes_)
        diabetic=st.selectbox("Diabetic",options=le_diabetic.classes_)
        smoker=st.selectbox("Smoker",options=le_smoker.classes_)
        region=st.selectbox("Region",options=le_region.classes_)

    submitted=st.form_submit_button("Predict Payment")

if submitted:
    input_data=pd.DataFrame({
        "age":[age],
        "gender":[gender],
        "bmi":[bmi],
        "bloodpressure":[bloodpressure],
        "diabetic":[diabetic],
        "children":[children],
        "smoker":[smoker],
        "region":[region]
    })

    input_data["gender"]=le_gender.transform(input_data["gender"])
    input_data["diabetic"]=le_diabetic.transform(input_data["diabetic"])
    input_data["smoker"]=le_smoker.transform(input_data["smoker"]) 
    input_data["region"] = le_region.transform(input_data["region"])


    prediction =model.predict(input_data)[0]
    st.success(f"**Estimated Insurance Payment Amount:** ${prediction:,.2f}")    
    



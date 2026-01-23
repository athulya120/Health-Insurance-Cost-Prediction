import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- LOAD MODELS & ENCODERS ----------------
le_gender = joblib.load("label_encoder_gender.pkl")
le_diabetic = joblib.load("label_encoder_diabetic.pkl")
le_region = joblib.load("label_encoder_region.pkl")
le_smoker = joblib.load("label_encoder_smoker.pkl")
model = joblib.load("model.pkl")

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("insurance (1).csv")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Insurance Claim Predictor",
    layout="centered"
)

st.title("🏥 Health Insurance Payment Prediction App")
st.write("Enter the details below to estimate your insurance payment amount.")

# ======================================================
# 📊 DATA VISUALIZATION SECTION
# ======================================================
st.subheader("📊 Data Insights")

col1, col2 = st.columns(2)

with col1:
    st.write("BMI Distribution")
    fig, ax = plt.subplots()
    ax.hist(df["bmi"], bins=20,color="y")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.write("Smoker vs Insurance Charges")
    fig, ax = plt.subplots()
    sns.boxplot(x="smoker", y="claim", data=df, ax=ax,color="y")
    st.pyplot(fig)


st.success("EDA section loaded successfully")

# ======================================================
# 📝 USER INPUT FORM
# ======================================================
st.subheader("📝 Enter Patient Details")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.5)
        children = st.number_input("Number of Children", 0, 8, 0)

    with col2:
        bloodpressure = st.number_input("Blood Pressure", 60, 200, 120)
        gender = st.selectbox("Gender", le_gender.classes_)
        diabetic = st.selectbox("Diabetic", le_diabetic.classes_)
        smoker = st.selectbox("Smoker", le_smoker.classes_)
        region = st.selectbox("Region", le_region.classes_)

    submitted = st.form_submit_button("🔮 Predict Payment")

# ======================================================
# 🤖 PREDICTION & VISUALIZATION
# ======================================================
if submitted:
    input_data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "bmi": [bmi],
        "bloodpressure": [bloodpressure],
        "diabetic": [diabetic],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Encoding
    input_data["gender"] = le_gender.transform(input_data["gender"])
    input_data["diabetic"] = le_diabetic.transform(input_data["diabetic"])
    input_data["smoker"] = le_smoker.transform(input_data["smoker"])
    input_data["region"] = le_region.transform(input_data["region"])

    # Prediction
    prediction = model.predict(input_data)[0]

    st.success(f"💰 **Estimated Insurance Payment Amount:** ${prediction:,.2f}")

    # ---------------- INPUT SUMMARY ----------------
    st.subheader("🔍 Input Summary")

    input_vis = pd.DataFrame({
        "Feature": ["Age", "BMI", "Children", "Blood Pressure"],
        "Value": [age, bmi, children, bloodpressure]
    })

    st.bar_chart(input_vis.set_index("Feature"))

    # ---------------- PREDICTION COMPARISON ----------------
    st.subheader("📈 Prediction Comparison")

    avg_charge = df["claim"].mean()

    comparison = pd.DataFrame({
        "Type": ["Average Insurance Cost", "Your Prediction"],
        "Amount": [avg_charge, prediction]
    })

    st.bar_chart(comparison.set_index("Type"),color="#2ca02c")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("🧠 Model Feature Importance (Random Forest)")

    feature_names = [
        "age", "gender", "bmi", "bloodpressure",
        "diabetic", "children", "smoker", "region"
    ]

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(feat_imp["Feature"], feat_imp["Importance"],color="y")
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")

    st.pyplot(fig)

    st.info(
        "Higher importance means the feature has more influence on the prediction. "
        "This improves transparency and trust in the model."
    )

# Run using:
# python -m streamlit run app.py

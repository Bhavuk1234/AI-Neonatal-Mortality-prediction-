import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="Bulk Neonatal Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Bulk Neonatal Mortality Prediction")

st.write("""
Upload an Excel or CSV file containing newborn details.

The AI model will predict mortality risk for every newborn.
""")

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

model = joblib.load("xgboost_model.pkl")

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Excel or CSV File",
    type=["xlsx", "csv"]
)

# -------------------------------------------------------
# RUN ONLY AFTER FILE IS UPLOADED
# -------------------------------------------------------

if uploaded_file is not None:

    # Read file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("✅ File Uploaded Successfully")

    st.subheader("Uploaded Data")

    st.dataframe(df)

    # -------------------------------------------------------
    # FEATURE COLUMNS
    # -------------------------------------------------------

    feature_columns = [

        "BirthWeight",
        "GestationalAge",
        "APGAR5",
        "Resuscitation",
        "CryAtBirth",
        "Breastfeeding1Hr",
        "MaternalAge",
        "MaternalHb",
        "MaternalHTN",
        "ANCVisits",
        "MultiplePregnancy",
        "ModeOfDelivery",
        "CongenitalAnomaly",
        "Sex",
        "HighRiskPregnancy",
        "SNCUAdmission"

    ]

    # Select features

    X = df[feature_columns]

    # Predict probability

    probabilities = model.predict_proba(X)

    df["RiskProbability"] = probabilities[:, 1] * 100

    # -------------------------------------------------------
    # Risk Category
    # -------------------------------------------------------

    def risk_category(prob):

        if prob < 5:
            return "🟢 Low"

        elif prob < 15:
            return "🟡 Moderate"

        elif prob < 40:
            return "🟠 High"

        else:
            return "🔴 Critical"

    df["RiskCategory"] = df["RiskProbability"].apply(risk_category)

    # Sort

    df = df.sort_values(
        by="RiskProbability",
        ascending=False
    )

    st.subheader("Prediction Results")

    st.dataframe(df)
    df.to_csv("prediction_results.csv", index=False)
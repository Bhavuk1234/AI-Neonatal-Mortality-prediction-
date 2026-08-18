import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI-Based Neonatal Mortality Prediction",
    page_icon="👶",
    layout="wide"
)


model = joblib.load("xgboost_model.pkl")
explainer = shap.TreeExplainer(model)



st.title("👶 AI-Based Neonatal Mortality Prediction")

st.markdown("""
This AI model predicts the **probability of neonatal death**
using maternal and neonatal clinical parameters.
""")

st.markdown("---")



left, right = st.columns(2)

with left:

    birth_weight = st.number_input(
        "Birth Weight (kg)",
        min_value=0.5,
        max_value=5.5,
        value=2.8,
        step=0.1
    )

    gestational_age = st.number_input(
        "Gestational Age (weeks)",
        min_value=24,
        max_value=42,
        value=38
    )

    apgar5 = st.slider(
        "APGAR Score (5 min)",
        0,
        10,
        8
    )

    resuscitation = st.selectbox(
        "Resuscitation Required",
        ["No", "Yes"]
    )

    cry_at_birth = st.selectbox(
        "Cry at Birth",
        ["Yes", "No"]
    )

    breastfeeding = st.selectbox(
        "Breastfeeding within 1 Hour",
        ["Yes", "No"]
    )

    maternal_age = st.number_input(
        "Maternal Age (years)",
        min_value=15,
        max_value=50,
        value=25
    )

    maternal_hb = st.number_input(
        "Maternal Hemoglobin (g/dL)",
        min_value=5.0,
        max_value=18.0,
        value=11.0,
        step=0.1
    )
    # =====================================================
# RIGHT COLUMN
# =====================================================

with right:

    maternal_htn = st.selectbox(
        "Maternal Hypertension",
        ["No", "Yes"]
    )

    anc_visits = st.number_input(
        "ANC Visits",
        min_value=0,
        max_value=10,
        value=4
    )

    multiple_pregnancy = st.selectbox(
        "Multiple Pregnancy",
        ["No", "Yes"]
    )

    mode_of_delivery = st.selectbox(
        "Mode of Delivery",
        ["C-section", "Vaginal"]
    )

    congenital_anomaly = st.selectbox(
        "Congenital Anomaly",
        ["No", "Yes"]
    )

    sex = st.selectbox(
        "Sex of Baby",
        ["Female", "Male"]
    )

    high_risk_pregnancy = st.selectbox(
        "High Risk Pregnancy",
        ["No", "Yes"]
    )

    sncu_admission = st.selectbox(
        "SNCU Admission",
        ["No", "Yes"]
    )

# =====================================================
# ENCODING
# =====================================================

resuscitation = 1 if resuscitation == "Yes" else 0
cry_at_birth = 1 if cry_at_birth == "Yes" else 0
breastfeeding = 1 if breastfeeding == "Yes" else 0
maternal_htn = 1 if maternal_htn == "Yes" else 0
multiple_pregnancy = 1 if multiple_pregnancy == "Yes" else 0
congenital_anomaly = 1 if congenital_anomaly == "Yes" else 0
high_risk_pregnancy = 1 if high_risk_pregnancy == "Yes" else 0
sncu_admission = 1 if sncu_admission == "Yes" else 0

# IMPORTANT:
# Must match the encoding used during model training.

mode_of_delivery = 0 if mode_of_delivery == "C-section" else 1
sex = 0 if sex == "Female" else 1

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("🔍 Predict Neonatal Risk", use_container_width=True):

    input_data = np.array([[
        birth_weight,
        gestational_age,
        apgar5,
        resuscitation,
        cry_at_birth,
        breastfeeding,
        maternal_age,
        maternal_hb,
        maternal_htn,
        anc_visits,
        multiple_pregnancy,
        mode_of_delivery,
        congenital_anomaly,
        sex,
        high_risk_pregnancy,
        sncu_admission
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.markdown("---")

    st.header("Prediction Result")

    st.metric(
        "Probability of Neonatal Death",
        f"{probability:.2f}%"
    )

    st.progress(min(int(probability),100))

    if probability < 5:

        st.success("🟢 LOW RISK")

    elif probability < 15:

        st.info("🟡 MODERATE RISK")

    elif probability < 40:

        st.warning("🟠 HIGH RISK")

    else:

        st.error("🔴 CRITICAL RISK")

    # ============================================
    # SHAP
    # ============================================

    shap_values = explainer.shap_values(input_data)

    feature_names = [

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

    # Compatible with different SHAP versions
    if isinstance(shap_values, list):
        shap_array = shap_values[0][0]
    else:
        shap_array = shap_values[0]

    importance = pd.DataFrame({

        "Feature": feature_names,
        "SHAP": shap_array

    })

    importance["ABS"] = importance["SHAP"].abs()

    importance = importance.sort_values(

        by="ABS",

        ascending=False

    )

    st.markdown("---")

    st.subheader("Why did the model predict this risk?")

    for _, row in importance.head(5).iterrows():

        if row["SHAP"] > 0:

            st.write(
                f"🔴 **{row['Feature']}** increased the predicted risk "
                f"(SHAP = {row['SHAP']:.3f})"
            )

        else:

            st.write(
                f"🟢 **{row['Feature']}** reduced the predicted risk "
                f"(SHAP = {row['SHAP']:.3f})"
            )

    st.subheader("Feature Contribution")

    st.bar_chart(
        importance.set_index("Feature")["SHAP"]
    )
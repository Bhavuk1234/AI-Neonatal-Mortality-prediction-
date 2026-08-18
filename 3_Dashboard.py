import streamlit as st
import pandas as pd
st.title("📊 Neonatal Dashboard")
df = pd.read_csv("prediction_results.csv")
total = len(df)

low = len(df[df["RiskCategory"]=="🟢 Low"])

moderate = len(df[df["RiskCategory"]=="🟡 Moderate"])

high = len(df[df["RiskCategory"]=="🟠 High"])

critical = len(df[df["RiskCategory"]=="🔴 Critical"])

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Babies", total)

col2.metric("🟢 Low Risk", low)

col3.metric("🟡 Moderate", moderate)

col4.metric("🟠 High Risk", high)

col5.metric("🔴 Critical", critical)

st.subheader("Risk Category Distribution")
risk_counts = df["RiskCategory"].value_counts()

st.bar_chart(risk_counts)
st.subheader("Birth Weight Distribution")

st.bar_chart(
    df["BirthWeight"].value_counts(bins=10)
)
st.subheader("Gestational Age")
st.bar_chart(
    df["GestationalAge"].value_counts(bins=10)
)
st.subheader("Top High Risk Babies")

top = df.sort_values(
    "RiskProbability",
    ascending=False
)

st.dataframe(
    top[
        [
            "BabyID",
            "RiskProbability",
            "RiskCategory"
        ]
    ].head(20)
)

csv = df.to_csv(index=False)

st.download_button(
    "Download Report",
    csv,
    "prediction_report.csv",
    "text/csv"
)
import streamlit as st

st.set_page_config(
    page_title="AI Neonatal Mortality System",
    page_icon="👶",
    layout="wide"
)

st.title("👶 AI Neonatal Mortality Prediction")

st.write("""
Welcome to the AI Neonatal Mortality Prediction System.

Use the left sidebar to navigate between pages.

Available Modules:

• 👶 Single Prediction

• 📊 Bulk Prediction

• 📈 Dashboard
""")
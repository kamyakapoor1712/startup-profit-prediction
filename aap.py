import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Startup Profit Predictor",
    layout="centered",
    page_icon="💼"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("mlr_predictor (1).joblib")

model = load_model()

# State encoding
state_mapping = {"California": 0, "Florida": 1, "New York": 2}

# Title and description
st.title("💼 Startup Profit Predictor")
st.markdown("""
Welcome to the **Startup Profit Predictor App**.  
This tool uses a trained machine learning model to estimate startup profits based on your inputs.
""")

# Sidebar for input
st.sidebar.header("Input Features")

rd = st.sidebar.number_input("R&D Spend", min_value=0.0, format="%.2f", help="Amount spent on R&D in USD")
admin = st.sidebar.number_input("Administration", min_value=0.0, format="%.2f", help="Administration expenses in USD")
marketing = st.sidebar.number_input("Marketing Spend", min_value=0.0, format="%.2f", help="Marketing budget in USD")
state = st.sidebar.selectbox("State", options=list(state_mapping.keys()), help="State where the startup operates")

# Button
if st.sidebar.button("Predict Profit"):
    try:
        state_encoded = state_mapping[state]
        input_data = np.array([[rd, admin, marketing, state_encoded]])
        prediction = model.predict(input_data)
        st.success(f"💰 **Estimated Profit:** ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")

# Optional: Expandable section to view sample data
with st.expander("📊 View Sample Startup Data (Optional)"):
    try:
        df = pd.read_csv("50_Startups.csv")
        st.dataframe(df.head())
    except:
        st.info("Upload '50_Startups.csv' to display sample data.")

# Footer
st.markdown("---")
st.caption("Made by pythrive| Model: startup profit prediction")

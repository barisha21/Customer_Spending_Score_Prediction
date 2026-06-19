
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Customer Spending Analytics",
    page_icon="📊",
    layout="wide"
)

# ---------------------------
# LOAD FILES
# ---------------------------
model = pickle.load(open("artifacts/models/best_model.pkl", "rb"))
scaler = pickle.load(open("artifacts/models/scaler.pkl", "rb"))
gender_encoder = pickle.load(open("artifacts/models/gender_encoder.pkl", "rb"))
profession_encoder = pickle.load(open("artifacts/models/profession_encoder.pkl", "rb"))

# CHANGE THIS TO YOUR REAL DATASET NAME
# Example:
df = pd.read_csv("C:\\Users\\Admin\\Documents\\Full_Stack_Internship\\Customer_Spending_Score_Prediction\\data\\Customers.csv")

# ---------------------------
# TITLE
# ---------------------------
st.title("💳 Customer Spending Score Prediction")
st.write("Professional Analytics Dashboard")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["select","Male", "Female"])
age = st.sidebar.slider("Age",18, 80, 0)
annual_income = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=1000,
    value=1000
)

profession = st.sidebar.selectbox("Profession", ["select","Artist",
        "Doctor",
        "Engineer",
        "Executive",
        "Healthcare",
        "Lawyer",
        "Marketing",
        "Entertainment"])

work_experience = st.sidebar.slider(
    "Work Experience",
    0,
    40,
    0
)

family_size = st.sidebar.slider(
    "Family Size",
    1,
    10,
    1
)

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
income_per_family = annual_income / family_size

experience_ratio = (
    work_experience / age if age != 0 else 0
)

# ---------------------------
# ENCODING
# ---------------------------
try:
    gender_encoded = gender_encoder.transform(
        [gender]
    )[0]
except:
    gender_encoded = 0

try:
    profession_encoded = profession_encoder.transform(
        [profession]
    )[0]
except:
    profession_encoded = 0

# ---------------------------
# INPUT DATAFRAME
# ---------------------------
input_df = pd.DataFrame({
    "Gender": [gender_encoded],
    "Age": [age],
    "Annual Income ($)": [annual_income],
    "Profession": [profession_encoded],
    "Work Experience": [work_experience],
    "Family Size": [family_size],
    "Income_Per_Family": [income_per_family],
    "Experience_Ratio": [experience_ratio]
})

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🚀 Predict Spending Score"):

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(
        input_scaled
    )[0]

    # KPI
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Age", age)
    c2.metric("Income", f"${annual_income:,.0f}")
    c3.metric("Family Size", family_size)
    c4.metric("Score", round(prediction, 2))

    # Gauge Chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={
                "text":
                "Predicted Spending Score"
            },
            gauge={
                "axis": {
                    "range": [0, 100]
                },
                "steps": [
                    {
                        "range": [0, 30],
                        "color": "red"
                    },
                    {
                        "range": [30, 70],
                        "color": "orange"
                    },
                    {
                        "range": [70, 100],
                        "color": "green"
                    }
                ]
            }
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # Category
    if prediction < 30:
        st.error(
            "Low Spending Customer"
        )
    elif prediction < 70:
        st.warning(
            "Medium Spending Customer"
        )
    else:
        st.success(
            "High Value Customer"
        )

    # Download
    result = pd.DataFrame({
        "Age": [age],
        "Income": [annual_income],
        "Prediction": [prediction]
    })
    import plotly.express as px

    fig = px.scatter(
        df,
        x="Annual Income ($)",
        y="Spending Score (1-100)",
        color="Profession",
        title="Income vs Spending Score"
    )

    st.plotly_chart(fig, width="stretch")
    st.download_button(
        "📥 Download Prediction",
        result.to_csv(index=False),
        file_name="prediction.csv"
    )


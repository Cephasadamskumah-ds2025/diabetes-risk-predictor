import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap
from pathlib import Path

# Project root path
BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD MODEL AND EXPLAINER
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_DIR / "xgb_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(MODEL_DIR / "explainer.pkl", "rb") as f:
        explainer = pickle.load(f)

    with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    return model, explainer, feature_names
model, explainer, feature_names = load_model()

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("🏥 Diabetes Risk Prediction")
st.markdown(
    "**Early Detection System** — Enter patient details to "
    "assess diabetes risk using an XGBoost model trained on "
    "229,474 CDC survey respondents (ROC-AUC: 0.82)"
)
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR — PATIENT INPUT
# ─────────────────────────────────────────
st.sidebar.header("🧾 Patient Details")
st.sidebar.markdown("Adjust the sliders and selections below:")

# Clinical indicators
st.sidebar.subheader("Clinical Indicators")
HighBP = st.sidebar.selectbox(
    "High Blood Pressure", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
HighChol = st.sidebar.selectbox(
    "High Cholesterol", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
CholCheck = st.sidebar.selectbox(
    "Cholesterol Checked in Last 5 Years", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
BMI = st.sidebar.slider("BMI", 10, 80, 27)
Stroke = st.sidebar.selectbox(
    "Ever had a Stroke", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
HeartDiseaseorAttack = st.sidebar.selectbox(
    "Heart Disease or Attack", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Lifestyle
st.sidebar.subheader("Lifestyle Factors")
PhysActivity = st.sidebar.selectbox(
    "Physically Active (last 30 days)", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
Fruits = st.sidebar.selectbox(
    "Eats Fruit Daily", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
Veggies = st.sidebar.selectbox(
    "Eats Vegetables Daily", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
HvyAlcoholConsump = st.sidebar.selectbox(
    "Heavy Alcohol Consumption", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
Smoker = st.sidebar.selectbox(
    "Smoker (100+ cigarettes lifetime)", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Healthcare access
st.sidebar.subheader("Healthcare Access")
AnyHealthcare = st.sidebar.selectbox(
    "Has Health Insurance", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)
NoDocbcCost = st.sidebar.selectbox(
    "Could Not See Doctor Due to Cost", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# General health
st.sidebar.subheader("General Health")
GenHlth = st.sidebar.slider(
    "General Health (1=Excellent, 5=Poor)", 1, 5, 3
)
MentHlth = st.sidebar.slider(
    "Poor Mental Health Days (last 30 days)", 0, 30, 0
)
PhysHlth = st.sidebar.slider(
    "Poor Physical Health Days (last 30 days)", 0, 30, 0
)
DiffWalk = st.sidebar.selectbox(
    "Difficulty Walking", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Demographics
st.sidebar.subheader("Demographics")
Sex = st.sidebar.selectbox(
    "Sex", [0, 1],
    format_func=lambda x: "Male" if x == 1 else "Female"
)
Age = st.sidebar.slider(
    "Age Group (1=18-24, 13=80+)", 1, 13, 7
)
Education = st.sidebar.slider(
    "Education Level (1=None, 6=College)", 1, 6, 4
)
Income = st.sidebar.slider(
    "Income Level (1=<$10K, 8=>$75K)", 1, 8, 5
)

# ─────────────────────────────────────────
# BUILD INPUT DATAFRAME
# ─────────────────────────────────────────
input_data = pd.DataFrame([{
    'HighBP': HighBP,
    'HighChol': HighChol,
    'CholCheck': CholCheck,
    'BMI': BMI,
    'Smoker': Smoker,
    'Stroke': Stroke,
    'HeartDiseaseorAttack': HeartDiseaseorAttack,
    'PhysActivity': PhysActivity,
    'Fruits': Fruits,
    'Veggies': Veggies,
    'HvyAlcoholConsump': HvyAlcoholConsump,
    'AnyHealthcare': AnyHealthcare,
    'NoDocbcCost': NoDocbcCost,
    'GenHlth': GenHlth,
    'MentHlth': MentHlth,
    'PhysHlth': PhysHlth,
    'DiffWalk': DiffWalk,
    'Sex': Sex,
    'Age': Age,
    'Education': Education,
    'Income': Income
}])

# Ensure feature order matches training data
input_data = input_data[feature_names]

# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
prob = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]

# ─────────────────────────────────────────
# MAIN PANEL — RESULTS
# ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Diabetes Risk Score", f"{prob:.1%}")

with col2:
    if prob < 0.30:
        risk_level = "🟢 LOW RISK"
    elif prob < 0.60:
        risk_level = "🟡 MODERATE RISK"
    else:
        risk_level = "🔴 HIGH RISK"
    st.metric("Risk Level", risk_level)

with col3:
    st.metric("Prediction Confidence", f"{max(prob, 1-prob):.1%}")

st.markdown("---")

# Risk gauge bar
st.subheader("Risk Probability")
st.progress(float(prob))
if prob < 0.30:
    st.success(f"✅ Low diabetes risk — {prob:.1%} probability")
elif prob < 0.60:
    st.warning(f"⚠️ Moderate diabetes risk — {prob:.1%} probability")
else:
    st.error(f"🚨 High diabetes risk — {prob:.1%} probability")

st.markdown("---")

# ─────────────────────────────────────────
# SHAP EXPLANATION
# ─────────────────────────────────────────
st.subheader("🔍 Why This Prediction? — SHAP Explanation")
st.markdown(
    "The chart below shows which factors are **increasing** "
    "(red) or **decreasing** (blue) this patient's diabetes risk."
)

shap_values = explainer.shap_values(input_data)
fig, ax = plt.subplots(figsize=(10, 6))

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_data.iloc[0].values,
        feature_names=feature_names
    ),
    show=False
)

fig = plt.gcf()
fig.set_size_inches(10, 6)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# ─────────────────────────────────────────
# PATIENT SUMMARY TABLE
# ─────────────────────────────────────────
st.subheader("📋 Patient Input Summary")

summary = pd.DataFrame({
    'Feature': feature_names,
    'Value': input_data.iloc[0].values
})

col_a, col_b = st.columns(2)
with col_a:
    st.dataframe(
        summary.iloc[:11],
        use_container_width=True,
        hide_index=True
    )
with col_b:
    st.dataframe(
        summary.iloc[11:],
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown(
    "**⚠️ Disclaimer:** This tool is for educational and "
    "portfolio demonstration purposes only. It is not a "
    "substitute for professional medical advice, diagnosis, "
    "or treatment."
)
st.markdown(
    "**Model:** XGBoost · **Dataset:** CDC BRFSS 2015 · "
    "**ROC-AUC:** 0.82 · **Recall:** 0.78 · "
    "**Built by:** Cephas Adams Kumah"
)
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OptiCover | Health Insurance Predictor",
    page_icon="🛡️",
    layout="wide" 
)

# ── Custom CSS (High Contrast, No Neon) ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    /* Stops the stretch by capping the width */
    .block-container {
        max-width: 950px !important;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    .stApp { background-color: #0d1117; } 

    /* Headers - Replaced neon glow with a sleek gradient matching the button */
    .main-title { 
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; 
        text-align: center; 
        font-size: 3.8rem; 
        margin-bottom: 0rem; 
        padding-bottom: 0; 
    }
    .sub-title { color: #8b949e; font-weight: 600; text-align: center; font-size: 1.2rem; margin-top: 0.5rem; text-transform: uppercase; letter-spacing: 1px;}
    .desc-text { color: #f0f6fc; text-align: center; margin-bottom: 2.5rem; font-size: 1.05rem; }

    /* Form Container styling */
    div[data-testid="stForm"] {
        background: #161b22;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        border: 1px solid #30363d;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: transform 0.2s, opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; transform: scale(1.02); color: white;}

    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin-top: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid #3a506b;
    }
    .result-card h2 { color: #00e5ff; font-size: 1.1rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1.5px;}
    .result-card h1 { color: #ffffff; font-size: 4rem; margin: 0; font-weight: 700; }

    /* Section Labels */
    .section-label {
        font-weight: 700;
        font-size: 1.2rem;
        color: #58a6ff;
        border-left: 4px solid #58a6ff;
        padding-left: 0.8rem;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
        background: rgba(88, 166, 255, 0.1);
        padding-top: 0.4rem;
        padding-bottom: 0.4rem;
        border-radius: 0 4px 4px 0;
    }

    /* HIGH CONTRAST LABELS */
    .stSelectbox label, .stNumberInput label, div[data-testid="stForm"] label { 
        font-weight: 600 !important; 
        color: #ffffff !important; 
        font-size: 1rem !important;
        letter-spacing: 0.3px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_young = load(r"C:\Users\Rachel Priya R\Downloads\project_1_model_retraining_resources\artifacts\model_young.joblib")
    model_rest = load(r"C:\Users\Rachel Priya R\Downloads\project_1_model_retraining_resources\artifacts\model_rest.joblib")
    scaler_young_dict = load(r"C:\Users\Rachel Priya R\Downloads\project_1_model_retraining_resources\artifacts\scaler_young.joblib")
    scaler_rest_dict = load(r"C:\Users\Rachel Priya R\Downloads\project_1_model_retraining_resources\artifacts\scaler_rest.joblib")
    
    return model_young, model_rest, scaler_young_dict, scaler_rest_dict

model_young, model_rest, scaler_young_dict, scaler_rest_dict = load_artifacts()

# ── Risk-score helpers ─────────────────────────────────────────────────────────
RISK_SCORES = {
    "diabetes": 6,
    "heart disease": 8,
    "high blood pressure": 6,
    "thyroid": 5,
    "no disease": 0,
    "none": 0,
}

_RISK_MIN, _RISK_MAX = 0, 14

def compute_normalized_risk(medical_history: str) -> float:
    parts = medical_history.lower().split(" & ")
    d1 = parts[0].strip() if len(parts) > 0 else "none"
    d2 = parts[1].strip() if len(parts) > 1 else "none"
    raw = RISK_SCORES.get(d1, 0) + RISK_SCORES.get(d2, 0)
    return (raw - _RISK_MIN) / (_RISK_MAX - _RISK_MIN)

# ── Preprocessing ──────────────────────────────────────────────────────────────
INSURANCE_MAP = {"Gold": 3, "Silver": 2, "Bronze": 1}

FEATURE_COLS = [
    "age", "number_of_dependants", "income_lakhs",
    "insurance_plan", "genetical_risk", "normalized_risk_score",
    "gender_Male",
    "region_Northwest", "region_Southeast", "region_Southwest",
    "marital_status_Unmarried",
    "bmi_category_Obesity", "bmi_category_Overweight", "bmi_category_Underweight",
    "smoking_status_Occasional", "smoking_status_Regular",
    "employment_status_Salaried", "employment_status_Self-Employed",
]

COLS_TO_SCALE = ["age", "number_of_dependants", "income_lakhs", "insurance_plan", "genetical_risk"]

def preprocess(age, number_of_dependants, income_lakhs, insurance_plan,
               genetical_risk, gender, region, marital_status,
               bmi_category, smoking_status, employment_status, medical_history, scaler):

    row = {
        "age":                             age,
        "number_of_dependants":            number_of_dependants,
        "income_lakhs":                    income_lakhs,
        "insurance_plan":                  INSURANCE_MAP[insurance_plan],
        "genetical_risk":                  genetical_risk,
        "normalized_risk_score":           compute_normalized_risk(medical_history),
        "gender_Male":                     1 if gender == "Male" else 0,
        "region_Northwest":                1 if region == "Northwest" else 0,
        "region_Southeast":                1 if region == "Southeast" else 0,
        "region_Southwest":                1 if region == "Southwest" else 0,
        "marital_status_Unmarried":        1 if marital_status == "Unmarried" else 0,
        "bmi_category_Obesity":            1 if bmi_category == "Obesity" else 0,
        "bmi_category_Overweight":         1 if bmi_category == "Overweight" else 0,
        "bmi_category_Underweight":        1 if bmi_category == "Underweight" else 0,
        "smoking_status_Occasional":       1 if smoking_status == "Occasional" else 0,
        "smoking_status_Regular":          1 if smoking_status == "Regular" else 0,
        "employment_status_Salaried":      1 if employment_status == "Salaried" else 0,
        "employment_status_Self-Employed": 1 if employment_status == "Self-Employed" else 0,
    }

    df = pd.DataFrame([row])[FEATURE_COLS]         
    df[COLS_TO_SCALE] = scaler.transform(df[COLS_TO_SCALE])
    return df


# ── UI Layout ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🛡️ OptiCover</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Health Insurance Premium Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="desc-text">Enter the policyholder\'s details to estimate their <b>annual premium amount</b>.</p>', unsafe_allow_html=True)

with st.form("premium_form"):

    # ── Personal details ──
    st.markdown('<p class="section-label"> Personal Details</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col3:
        marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])

    col4, col5 = st.columns(2)
    with col4:
        region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
    with col5:
        number_of_dependants = st.number_input("Number of Dependants", min_value=0, max_value=20, value=0, step=1)

    # ── Financial details ──
    st.markdown('<p class="section-label"> Financial Details</p>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)
    with col6:
        income_lakhs = st.number_input("Annual Income (₹ Lakhs)", min_value=0.0, max_value=200.0, value=10.0, step=0.5, format="%.1f")
    with col7:
        employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Freelancer"])
    with col8:
        insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])

    # ── Health details ──
    st.markdown('<p class="section-label"> Health Details</p>', unsafe_allow_html=True)
    col9, col10 = st.columns(2)
    with col9:
        bmi_category = st.selectbox("BMI Category", ["Normal", "Obesity", "Overweight", "Underweight"])
    with col10:
        smoking_status = st.selectbox("Smoking Status", ["No Smoking", "Occasional", "Regular"])

    col11, col12 = st.columns(2)
    with col11:
        genetical_risk = st.number_input("Genetical Risk Score (0 – 5)", min_value=0, max_value=5, value=0, step=1)
    with col12:
        medical_history = st.selectbox("Medical History", [
            "No Disease",
            "Diabetes",
            "High blood pressure",
            "Thyroid",
            "Heart disease",
            "Diabetes & High blood pressure",
            "Diabetes & Thyroid",
            "Diabetes & Heart disease",
            "High blood pressure & Heart disease",
        ])

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Centers the button
    _, btn_col, _ = st.columns([1, 1.5, 1])
    with btn_col:
        submitted = st.form_submit_button("Predict Annual Premium", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    if age <= 25:
        model      = model_young
        scaler     = scaler_young_dict['scaler']
    else:
        model      = model_rest
        scaler     = scaler_rest_dict['scaler']

    input_df = preprocess(
        age, number_of_dependants, income_lakhs, insurance_plan,
        genetical_risk, gender, region, marital_status,
        bmi_category, smoking_status, employment_status, medical_history, scaler
    )

    prediction = model.predict(input_df)[0]

    st.markdown(f"""
    <div class="result-card">
        <h2>Estimated Annual Premium</h2>
        <h1>₹ {prediction:,.2f}</h1>
    </div>
    """, unsafe_allow_html=True)
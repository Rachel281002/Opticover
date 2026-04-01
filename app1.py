import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OptiCover | Health Insurance Predictor",
    page_icon="🛡️",
    layout="wide", # Uses the full width of the monitor
    initial_sidebar_state="collapsed"
)

# ── Custom CSS (Ultra-Compact, No Scrolling) ───────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    /* Remove top padding to pull everything up */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important; /* Lets it stretch side-to-side */
    }

    .stApp { background-color: #0d1117; } 

    /* Headers */
    .main-title { 
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; 
        font-size: 2.5rem; 
        margin-bottom: 0; 
        padding-bottom: 0; 
    }
    .sub-title { color: #8b949e; font-weight: 600; font-size: 1rem; margin-top: 0.2rem; text-transform: uppercase; letter-spacing: 1px;}

    /* Form Container styling - More compact */
    div[data-testid="stForm"] {
        background: #161b22;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #0072ff 0%, #00c6ff 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        margin-top: 1rem;
        transition: transform 0.2s, opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; transform: scale(1.02); color: white;}

    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid #3a506b;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .result-card.empty {
        background: #161b22;
        border: 1px dashed #30363d;
        box-shadow: none;
    }
    .result-card h2 { color: #00e5ff; font-size: 1.2rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1.5px;}
    .result-card h1 { color: #ffffff; font-size: 3.5rem; margin: 0; font-weight: 700; }
    .result-card p { color: #8b949e; font-size: 1rem; }

    /* Section Labels */
    .section-label {
        font-weight: 700;
        font-size: 1rem;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.3rem;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }

    /* HIGH CONTRAST LABELS */
    .stSelectbox label, .stNumberInput label, div[data-testid="stForm"] label { 
        font-weight: 600 !important; 
        color: #ffffff !important; 
        font-size: 0.9rem !important;
    }
    
    /* Tweak input box heights to save space */
    .stSelectbox > div > div, .stNumberInput > div > div {
        min-height: 2.5rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_young = load("model_young.joblib")
    model_rest = load("model_rest.joblib")
    scaler_young_dict = load("scaler_young.joblib")
    scaler_rest_dict = load("scaler_rest.joblib")
    return model_young, model_rest, scaler_young_dict, scaler_rest_dict

model_young, model_rest, scaler_young_dict, scaler_rest_dict = load_artifacts()

# ── Risk-score helpers ─────────────────────────────────────────────────────────
RISK_SCORES = {
    "diabetes": 6, "heart disease": 8, "high blood pressure": 6,
    "thyroid": 5, "no disease": 0, "none": 0,
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
    "gender_Male", "region_Northwest", "region_Southeast", "region_Southwest",
    "marital_status_Unmarried", "bmi_category_Obesity", "bmi_category_Overweight", 
    "bmi_category_Underweight", "smoking_status_Occasional", "smoking_status_Regular",
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

# ── UI Layout (Side-by-Side) ───────────────────────────────────────────────────

# Header
st.markdown('<h1 class="main-title">🛡️ OptiCover</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Health Insurance Premium Predictor</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Split screen: 70% for the Form, 30% for the Results
left_col, right_col = st.columns([2.2, 1])

with left_col:
    with st.form("premium_form"):
        # Pack inputs into 3 tight columns to prevent vertical scrolling
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown('<p class="section-label">Personal</p>', unsafe_allow_html=True)
            age = st.number_input("Age", min_value=1, max_value=100, value=30, step=1)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])
            region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

        with c2:
            st.markdown('<p class="section-label">Financial</p>', unsafe_allow_html=True)
            income_lakhs = st.number_input("Income (₹ Lakhs)", min_value=0.0, max_value=200.0, value=10.0, step=0.5, format="%.1f")
            employment_status = st.selectbox("Employment", ["Salaried", "Self-Employed", "Freelancer"])
            number_of_dependants = st.number_input("Dependants", min_value=0, max_value=20, value=0, step=1)
            insurance_plan = st.selectbox("Plan Tier", ["Bronze", "Silver", "Gold"])

        with c3:
            st.markdown('<p class="section-label">Health</p>', unsafe_allow_html=True)
            bmi_category = st.selectbox("BMI", ["Normal", "Obesity", "Overweight", "Underweight"])
            smoking_status = st.selectbox("Smoking", ["No Smoking", "Occasional", "Regular"])
            genetical_risk = st.number_input("Genetic Risk (0-5)", min_value=0, max_value=5, value=0, step=1)
            medical_history = st.selectbox("Medical History", [
                "No Disease", "Diabetes", "High blood pressure", "Thyroid", "Heart disease",
                "Diabetes & High blood pressure", "Diabetes & Thyroid", "Diabetes & Heart disease",
                "High blood pressure & Heart disease"
            ])

        submitted = st.form_submit_button("Predict Annual Premium")

# ── Prediction & Result display in Right Column ────────────────────────────────
with right_col:
    if submitted:
        if age <= 25:
            model = model_young
            scaler = scaler_young_dict['scaler']
        else:
            model = model_rest
            scaler = scaler_rest_dict['scaler']

        input_df = preprocess(
            age, number_of_dependants, income_lakhs, insurance_plan,
            genetical_risk, gender, region, marital_status,
            bmi_category, smoking_status, employment_status, medical_history, scaler
        )

        prediction = model.predict(input_df)[0]

        st.markdown(f"""
        <div class="result-card">
            <h2>Estimated Premium</h2>
            <h1>₹ {prediction:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Placeholder before user clicks predict
        st.markdown("""
        <div class="result-card empty">
            <h2 style="color: #8b949e;">Ready to Calculate</h2>
            <p>Fill out the details on the left and click <b>Predict</b> to see the estimated annual premium here.</p>
        </div>
        """, unsafe_allow_html=True)
import streamlit as st
import numpy as np
import pickle
import pandas as pd

# ─────────────────────────────────────────────
#  LOAD MODEL + ENCODERS + SCALER
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("loan_model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["target_encoders"], bundle["scaler"]

model, target_encoders, scaler = load_artifacts()
GLOBAL_MEAN = target_encoders["global_mean"]

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Alpha Dreamers | Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:   #0b1e3d;
    --navy2:  #112952;
    --gold:   #c9a84c;
    --gold2:  #f0c96b;
    --red:    #e05252;
    --green:  #3dbf82;
    --card:   #142240;
    --border: rgba(201,168,76,0.25);
    --text:   #e8eaf0;
    --muted:  #8a9bbf;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--text) !important;
}

[data-testid="stSidebar"] {
    background: var(--navy2) !important;
    border-right: 1px solid var(--border);
}

.app-header {
    background: linear-gradient(135deg, var(--navy2) 0%, #1a3460 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.app-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    color: var(--gold2);
    margin: 0;
}
.app-header p { color: var(--muted); margin: 0.2rem 0 0; font-size: 0.88rem; }

.sec-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: var(--gold);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

.result-card {
    border-radius: 12px;
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: 0.5rem;
}
.result-card.default {
    background: linear-gradient(135deg, rgba(224,82,82,0.12), rgba(224,82,82,0.04));
    border: 2px solid rgba(224,82,82,0.45);
    box-shadow: 0 0 40px rgba(224,82,82,0.15);
}
.result-card.safe {
    background: linear-gradient(135deg, rgba(61,191,130,0.12), rgba(61,191,130,0.04));
    border: 2px solid rgba(61,191,130,0.45);
    box-shadow: 0 0 40px rgba(61,191,130,0.15);
}
.flag-number {
    font-family: 'Playfair Display', serif;
    font-size: 7rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-card.default .flag-number { color: var(--red); text-shadow: 0 0 30px rgba(224,82,82,0.4); }
.result-card.safe    .flag-number { color: var(--green); text-shadow: 0 0 30px rgba(61,191,130,0.4); }
.flag-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}
.result-card.default .flag-label { color: var(--red); }
.result-card.safe    .flag-label { color: var(--green); }
.flag-sub { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.9rem; }
.flag-rec {
    font-size: 0.82rem;
    color: var(--gold2);
    background: rgba(201,168,76,0.1);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    display: inline-block;
}

.metric-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .m-label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .m-value { font-family: 'Playfair Display', serif; font-size: 1.5rem; font-weight: 700; }
.metric-card.gold  .m-value { color: var(--gold2); }
.metric-card.green .m-value { color: var(--green); }
.metric-card.red   .m-value { color: var(--red); }

label { color: var(--muted) !important; font-size: 0.82rem !important; }

.stButton > button {
    background: linear-gradient(135deg, #c9a84c, #f0c96b) !important;
    color: #0b1e3d !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DROPDOWN OPTIONS
#  Pulled from your saved training data
# ─────────────────────────────────────────────
PROFESSIONS = sorted(target_encoders["Profession"].keys())
CITIES      = sorted(target_encoders["CITY"].keys())
STATES      = sorted(target_encoders["STATE"].keys())


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 1.5rem;'>
        <div style='font-size:2.2rem;'>🏦</div>
        <div style='font-family:Playfair Display,serif; color:#f0c96b;
                    font-size:1.1rem; font-weight:600;'>Alpha Dreamers</div>
        <div style='color:#8a9bbf; font-size:0.75rem;'>Banking Consortium</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#8a9bbf; line-height:2;'>
    <b style='color:#f0c96b;'>Risk Flag Guide</b><br>
    🟢 <b style='color:#3dbf82;'>Flag 0</b> — Low default risk<br>
    🔴 <b style='color:#e05252;'>Flag 1</b> — High default risk<br><br>
    <b style='color:#f0c96b;'>Model</b><br>
    loan_model.pkl
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <h1>Loan Default Prediction System</h1>
        <p>Alpha Dreamers Banking Consortium · Enter customer details to predict the default risk flag</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FORM + RESULT LAYOUT
# ─────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:

    # ── Financial Info ──────────────────────────
    st.markdown('<div class="sec-label">💰 Financial Information</div>', unsafe_allow_html=True)
    f1, f2 = st.columns(2)
    with f1:
        Income = st.number_input("Annual Income (Pula)", min_value=0, value=55000, step=1000)
    with f2:
        Age = st.number_input("Age", min_value=18, max_value=95, value=35)

    st.markdown("---")

    # ── Personal Details ────────────────────────
    st.markdown('<div class="sec-label">👤 Personal Details</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        Experience        = st.number_input("Work Experience (yrs)", min_value=0, max_value=50, value=8)
    with p2:
        CURRENT_JOB_YRS   = st.number_input("Current Job (yrs)",     min_value=0, max_value=50, value=3)
    with p3:
        CURRENT_HOUSE_YRS = st.number_input("Current House (yrs)",   min_value=0, max_value=50, value=2)

    p4, p5, p6 = st.columns(3)
    with p4:
        Married_Single  = st.selectbox("Marital Status",  ["single", "married"])
    with p5:
        House_Ownership = st.selectbox("House Ownership", ["rented", "owned", "norent_noown"])
    with p6:
        Car_Ownership   = st.selectbox("Car Ownership",   ["no", "yes"])

    st.markdown("---")

    # ── Profession & Location ───────────────────
    st.markdown('<div class="sec-label">💼 Profession & Location</div>', unsafe_allow_html=True)
    l1, l2, l3 = st.columns(3)
    with l1:
        Profession = st.selectbox("Profession", PROFESSIONS)
    with l2:
        CITY  = st.selectbox("City",  CITIES)
    with l3:
        STATE = st.selectbox("State", STATES)

    st.markdown("---")
    predict_btn = st.button("Predict Default Risk Flag")


# ─────────────────────────────────────────────
#  RESULT PANEL
# ─────────────────────────────────────────────
with col_result:
    st.markdown('<div class="sec-label">📊 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:

        # ── Step 1: Target encoding ────────────────
        prof_enc  = target_encoders["Profession"].get(Profession, GLOBAL_MEAN)
        city_enc  = target_encoders["CITY"].get(CITY,             GLOBAL_MEAN)
        state_enc = target_encoders["STATE"].get(STATE,            GLOBAL_MEAN)

        # ── Step 2: One-hot encoding ───────────────
        # drop_first=True dropped: "married", "norent_noown", "no"
        married_single_single  = 1 if Married_Single  == "single" else 0
        house_ownership_owned  = 1 if House_Ownership == "owned"  else 0
        house_ownership_rented = 1 if House_Ownership == "rented" else 0
        car_ownership_yes      = 1 if Car_Ownership   == "yes"    else 0

        # ── Step 3: Build feature row ──────────────
        input_df = pd.DataFrame([{
            "Income":                   Income,
            "Age":                      Age,
            "Experience":               Experience,
            "Profession":               prof_enc,
            "CITY":                     city_enc,
            "STATE":                    state_enc,
            "CURRENT_JOB_YRS":          CURRENT_JOB_YRS,
            "CURRENT_HOUSE_YRS":        CURRENT_HOUSE_YRS,
            "Married/Single_single":    married_single_single,
            "House_Ownership_owned":    house_ownership_owned,
            "House_Ownership_rented":   house_ownership_rented,
            "Car_Ownership_yes":        car_ownership_yes,
        }])

        # ── Step 4: Scale and predict ──────────────
        input_scaled = scaler.transform(input_df)
        risk_flag    = int(model.predict(input_scaled)[0])

        # ── Result display ─────────────────────────
        if risk_flag == 1:
            st.markdown("""
            <div class="result-card default">
                <div class="flag-number">1</div>
                <div class="flag-label">⚠️ High Default Risk</div>
                <div class="flag-sub">This customer is <b>likely to default</b> on their loan.</div>
                <div class="flag-rec">Decline application or escalate for manual review</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card safe">
                <div class="flag-number">0</div>
                <div class="flag-label">✅ Low Default Risk</div>
                <div class="flag-sub">This customer is <b>unlikely to default</b> on their loan.</div>
                <div class="flag-rec">Proceed with the standard loan approval process</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Snapshot metrics ───────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)

        inc_color = "green" if Income     >= 50000 else "red"
        age_color = "green" if Age        <= 45    else "gold"
        exp_color = "green" if Experience >= 5     else "gold"

        m1.markdown(f"""<div class="metric-card {inc_color}">
            <div class="m-label">Income</div>
            <div class="m-value">P{Income/1000:.0f}k</div></div>""",
            unsafe_allow_html=True)

        m2.markdown(f"""<div class="metric-card {age_color}">
            <div class="m-label">Age</div>
            <div class="m-value">{Age}</div></div>""",
            unsafe_allow_html=True)

        m3.markdown(f"""<div class="metric-card {exp_color}">
            <div class="m-label">Experience</div>
            <div class="m-value">{Experience} yrs</div></div>""",
            unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background:#142240; border:1px dashed rgba(201,168,76,0.25);
                    border-radius:12px; padding:4rem 2rem; text-align:center; color:#8a9bbf;'>
            <div style='font-size:2.5rem; margin-bottom:0.8rem;'>🚩</div>
            <div style='font-size:0.92rem; line-height:1.7;'>
                Fill in the customer details<br>and click
                <b style='color:#f0c96b;'>Predict Default Risk Flag</b><br>
                to get <b style='color:#e8eaf0;'>Flag 0</b> or <b style='color:#e8eaf0;'>Flag 1</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="ChurnGuard", page_icon="🔥", layout="wide")

# ─── CUSTOM CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.main { background: #0f0f0f; }
.block-container { padding: 1.5rem 2rem !important; }

/* Hide streamlit default stuff */
#MainMenu, footer, header { visibility: hidden; }

/* Metric cards */
.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: #444;
}
.metric-val { font-size: 2rem; font-weight: 600; margin: 0.3rem 0; }
.metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-sub { font-size: 0.7rem; color: #555; margin-top: 4px; }

/* Customer cards */
.customer-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin-bottom: 10px;
    transition: transform 0.2s, border-color 0.2s;
    cursor: pointer;
}
.customer-card:hover { transform: translateX(4px); }
.customer-card.high { border-left: 3px solid #e74c3c; }
.customer-card.med  { border-left: 3px solid #f39c12; }
.customer-card.low  { border-left: 3px solid #2ecc71; }

/* Pills */
.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 600;
}
.pill-red    { background: #2d1010; color: #e74c3c; border: 1px solid #5a1a1a; }
.pill-amber  { background: #2d1e08; color: #f39c12; border: 1px solid #5a3a10; }
.pill-green  { background: #0d2d10; color: #2ecc71; border: 1px solid #1a5a20; }

/* Action chips */
.chip {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.7rem;
    background: #252525;
    border: 1px solid #333;
    color: #aaa;
    margin: 3px 2px;
}

/* Section title */
.section-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.5rem 0 0.75rem;
}

/* Top bar */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1f1f1f;
}
.logo { font-size: 1.2rem; font-weight: 600; color: white; }
.live-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #2ecc71; display: inline-block;
    animation: pulse 2s infinite; margin-right: 6px;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

.stTabs [data-baseweb="tab-list"] {
    background: #1a1a1a;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #2a2a2a;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #888;
    font-size: 0.8rem;
}
.stTabs [aria-selected="true"] {
    background: #252525 !important;
    color: white !important;
}

div[data-testid="stButton"] button {
    background: #e74c3c !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.85 !important;
}
</style>
""", unsafe_allow_html=True)

# ─── LOGIN ───
def check_password():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        st.markdown("""
        <div style='text-align:center;padding:3rem 0'>
            <div style='font-size:2rem;font-weight:700;color:white;margin-bottom:0.5rem'>🔥 ChurnGuard</div>
            <div style='color:#888;font-size:0.9rem;margin-bottom:2rem'>AI-powered churn prediction platform</div>
        </div>
        """, unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            if st.button("Login →", use_container_width=True):
                if username == "admin" and password == "churn123":
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Wrong credentials!")
        st.stop()

check_password()

# ─── TRAIN MODEL ───
@st.cache_resource
def train_model():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, X.columns.tolist(), encoders

model, feature_cols, encoders = train_model()

# ─── TOP BAR ───
st.markdown("""
<div class='topbar'>
    <div class='logo'>🔥 ChurnGuard</div>
    <div style='display:flex;align-items:center;gap:12px'>
        <span><span class='live-dot'></span><span style='color:#888;font-size:0.8rem'>Live</span></span>
        <span style='color:#555;font-size:0.8rem'>7,032 customers loaded</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ───
tab1, tab2, tab3 = st.tabs(["🔍 Single Customer", "📁 Bulk Upload", "📊 Analytics"])

# ══════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER
# ══════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Customer Details</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with col3:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.slider("Monthly Charges (₹)", 0, 5000, 999)
        total_charges = monthly_charges * tenure
        st.markdown(f"<div style='color:#888;font-size:0.85rem;margin-top:8px'>Total Charges: <b style='color:white'>₹{total_charges:,}</b></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Churn Risk", use_container_width=True):
        input_data = {
            'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
            'PhoneService': phone_service, 'MultipleLines': multiple_lines,
            'InternetService': internet_service, 'OnlineSecurity': online_security,
            'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
            'TechSupport': tech_support, 'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies, 'Contract': contract,
            'PaperlessBilling': paperless, 'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
        }

        df_input = pd.DataFrame([input_data])
        for col in df_input.select_dtypes(include='object').columns:
            if col in encoders:
                df_input[col] = encoders[col].transform(df_input[col])

        prob = model.predict_proba(df_input)[0][1]
        pct = round(prob * 100, 1)

        # Color
        if prob >= 0.7:
            color = "#e74c3c"
            risk_label = "HIGH RISK"
            pill_class = "pill-red"
        elif prob >= 0.4:
            color = "#f39c12"
            risk_label = "MEDIUM RISK"
            pill_class = "pill-amber"
        else:
            color = "#2ecc71"
            risk_label = "LOW RISK"
            pill_class = "pill-green"

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            number={'suffix': '%', 'font': {'size': 40, 'color': color}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#444'},
                'bar': {'color': color, 'thickness': 0.25},
                'bgcolor': '#1a1a1a',
                'bordercolor': '#2a2a2a',
                'steps': [
                    {'range': [0, 40], 'color': '#0d2d10'},
                    {'range': [40, 70], 'color': '#2d1e08'},
                    {'range': [70, 100], 'color': '#2d1010'},
                ],
                'threshold': {
                    'line': {'color': color, 'width': 3},
                    'thickness': 0.8, 'value': pct
                }
            }
        ))
        fig.update_layout(
            height=280,
            paper_bgcolor='#1a1a1a',
            plot_bgcolor='#1a1a1a',
            font={'color': 'white'},
            margin=dict(t=30, b=10)
        )

        col_g, col_r = st.columns([1, 1])
        with col_g:
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown(f"""
            <div class='customer-card {"high" if prob>=0.7 else "med" if prob>=0.4 else "low"}' style='margin-top:1rem'>
                <div style='font-size:0.7rem;color:#888;margin-bottom:4px'>PREDICTION RESULT</div>
                <div style='font-size:1.5rem;font-weight:700;color:{color}'>{pct}% Churn Risk</div>
                <span class='pill {pill_class}' style='margin-top:6px;display:inline-block'>{risk_label}</span>
                <hr style='border-color:#2a2a2a;margin:12px 0'>
                <div style='font-size:0.75rem;color:#888;margin-bottom:8px'>RISK FACTORS</div>
            """, unsafe_allow_html=True)

            reasons = []
            if contract == "Month-to-month": reasons.append("📋 Month-to-month contract")
            if tenure < 6: reasons.append("⏱️ Very low tenure")
            if monthly_charges > 3000: reasons.append("💸 High charges")
            if internet_service == "Fiber optic": reasons.append("📡 Fiber optic service")
            if online_security == "No": reasons.append("🔓 No online security")
            if tech_support == "No": reasons.append("🛠️ No tech support")

            for r in (reasons if reasons else ["✅ No major risk factors"]):
                st.markdown(f"<div style='font-size:0.8rem;color:#ccc;padding:3px 0'>• {r}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.75rem;color:#888;margin:12px 0 8px'>RECOMMENDED ACTIONS</div>", unsafe_allow_html=True)

            if prob >= 0.7:
                actions = ["🎁 20% discount", "📅 Annual plan", "📞 Priority call", "🔒 Free security"]
            elif prob >= 0.4:
                actions = ["📧 Retention email", "🎯 Plan upgrade"]
            else:
                actions = ["💌 Loyalty reward", "✅ No action needed"]

            chips = " ".join([f"<span class='chip'>{a}</span>" for a in actions])
            st.markdown(chips, unsafe_allow_html=True)

# ══════════════════════════════════════
# TAB 2 — BULK UPLOAD
# ══════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Upload Customer Data</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.dropna(inplace=True)
        ids = df['customerID'].copy()
        df.drop('customerID', axis=1, inplace=True)
        if 'Churn' in df.columns:
            df.drop('Churn', axis=1, inplace=True)

        le = LabelEncoder()
        for col in df.select_dtypes(include='object').columns:
            df[col] = le.fit_transform(df[col])

        probs = model.predict_proba(df)[:, 1]
        preds = model.predict(df)

        # Metric cards
        total = len(df)
        at_risk = int((probs >= 0.7).sum())
        churn_rate = preds.mean() * 100
        rev_at_risk = int(df['MonthlyCharges'].values[probs >= 0.7].sum())

        st.markdown(f"""
        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:1rem 0'>
            <div class='metric-card'>
                <div class='metric-label'>Total Customers</div>
                <div class='metric-val' style='color:white'>{total:,}</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>At Risk</div>
                <div class='metric-val' style='color:#e74c3c'>{at_risk:,}</div>
                <div class='metric-sub'>churn prob &gt;70%</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Churn Rate</div>
                <div class='metric-val' style='color:#f39c12'>{churn_rate:.1f}%</div>
            </div>
            <div class='metric-card'>
                <div class='metric-label'>Revenue at Risk</div>
                <div class='metric-val' style='color:#f39c12'>₹{rev_at_risk:,}</div>
                <div class='metric-sub'>monthly charges</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Results
        result = pd.DataFrame({
            'CustomerID': ids.values,
            'Churn Probability': (probs * 100).round(1),
            'Risk Level': ['🔴 High' if p >= 0.7 else '🟡 Medium' if p >= 0.4 else '🟢 Low' for p in probs],
            'Prediction': ['Churns' if p == 1 else 'Stays' for p in preds]
        }).sort_values('Churn Probability', ascending=False)

        st.markdown("<div class='section-title'>Predictions</div>", unsafe_allow_html=True)
        st.dataframe(result, use_container_width=True, height=300)

        # Chart
        fig = px.histogram(
            x=probs * 100, nbins=20,
            color_discrete_sequence=['#e74c3c'],
            title='Churn Risk Distribution'
        )
        fig.update_layout(
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            font={'color': 'white'},
            xaxis_title='Churn Probability (%)',
            yaxis_title='Customers',
            title_font_size=14
        )
        fig.update_xaxes(gridcolor='#2a2a2a')
        fig.update_yaxes(gridcolor='#2a2a2a')
        st.plotly_chart(fig, use_container_width=True)

        csv = result.to_csv(index=False)
        st.download_button("⬇️ Download Results", csv, "churn_predictions.csv", "text/csv")

# ══════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════
with tab3:
    df_raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    df_raw.dropna(inplace=True)

    def dark_chart(fig):
        fig.update_layout(
            paper_bgcolor='#1a1a1a', plot_bgcolor='#1a1a1a',
            font={'color': 'white'}, title_font_size=13,
            legend=dict(bgcolor='#1a1a1a')
        )
        fig.update_xaxes(gridcolor='#2a2a2a')
        fig.update_yaxes(gridcolor='#2a2a2a')
        return fig

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df_raw, x='Contract', color='Churn', barmode='group',
                           title='Churn by Contract Type',
                           color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
        st.plotly_chart(dark_chart(fig1), use_container_width=True)

        fig3 = px.histogram(df_raw, x='InternetService', color='Churn', barmode='group',
                           title='Churn by Internet Service',
                           color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
        st.plotly_chart(dark_chart(fig3), use_container_width=True)

    with col2:
        churn_counts = df_raw['Churn'].value_counts()
        fig2 = px.pie(values=churn_counts.values, names=churn_counts.index,
                     title='Overall Churn Rate',
                     color_discrete_sequence=['#2ecc71', '#e74c3c'],
                     hole=0.5)
        st.plotly_chart(dark_chart(fig2), use_container_width=True)

        fig4 = px.scatter(df_raw, x='tenure', y='MonthlyCharges', color='Churn',
                         title='Tenure vs Monthly Charges',
                         color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'},
                         opacity=0.6)
        st.plotly_chart(dark_chart(fig4), use_container_width=True)
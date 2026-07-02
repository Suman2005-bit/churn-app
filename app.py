import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
# ─── LOGIN SYSTEM ───
def check_password():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🔒 Churn Predictor — Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "churn123":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Wrong username or password!")
        st.stop()

check_password()

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

tab1, tab2, tab3 = st.tabs(["🔍 Single Customer", "📁 Bulk CSV Upload", "📊 Analytics"])

# ─── TAB 1: Single Customer ───
with tab1:
    st.subheader("Predict churn for one customer")

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
        st.markdown(f"**Total Charges (auto):** ₹{total_charges}")

    if st.button("🔍 Predict Churn", use_container_width=True):
        input_data = {
            'gender': gender,
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        df_input = pd.DataFrame([input_data])
        for col in df_input.select_dtypes(include='object').columns:
            if col in encoders:
                df_input[col] = encoders[col].transform(df_input[col])

        prob = model.predict_proba(df_input)[0][1]

        st.markdown("---")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            title={'text': "Churn Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#e74c3c"},
                'steps': [
                    {'range': [0, 40], 'color': "#2ecc71"},
                    {'range': [40, 70], 'color': "#f39c12"},
                    {'range': [70, 100], 'color': "#e74c3c"},
                ]
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        if prob >= 0.7:
            st.error(f"⚠️ High Churn Risk: {prob*100:.1f}%")
        elif prob >= 0.4:
            st.warning(f"⚠️ Medium Churn Risk: {prob*100:.1f}%")
        else:
            st.success(f"✅ Low Churn Risk: {prob*100:.1f}%")

        st.subheader("📌 Risk Factors")
        reasons = []
        if contract == "Month-to-month":
            reasons.append("📋 Month-to-month contract")
        if tenure < 6:
            reasons.append("⏱️ Very low tenure")
        if monthly_charges > 3000:
            reasons.append("💸 High monthly charges")
        if internet_service == "Fiber optic":
            reasons.append("📡 Fiber optic service")
        if online_security == "No":
            reasons.append("🔓 No online security")
        if tech_support == "No":
            reasons.append("🛠️ No tech support")

        for r in reasons if reasons else ["✅ No major risk factors"]:
            st.markdown(f"- {r}")

        st.subheader("💡 Recommended Actions")
        if prob >= 0.7:
            st.markdown("""
- 🎁 Offer **20% discount** for 3 months
- 📅 Upgrade to **annual contract**
- 📞 **Priority support call**
- 🔒 Free **Online Security** addon
            """)
        elif prob >= 0.4:
            st.markdown("""
- 📧 Send **retention email**
- 🎯 Suggest **plan upgrade**
            """)
        else:
            st.markdown("- ✅ No action needed — send loyalty reward")

# ─── TAB 2: Bulk CSV ───
with tab2:
    st.subheader("Upload customer CSV for bulk predictions")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

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

        df['CustomerID'] = ids.values
        df['Churn Probability'] = (probs * 100).round(1)
        df['Prediction'] = ['🔴 Churns' if p == 1 else '🟢 Stays' for p in preds]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df))
        col2.metric("Likely to Churn", int(preds.sum()))
        col3.metric("Churn Rate", f"{preds.mean()*100:.1f}%")

        result = df[['CustomerID', 'Churn Probability', 'Prediction']].sort_values(
            'Churn Probability', ascending=False)
        st.dataframe(result, use_container_width=True)

        fig = px.histogram(df, x='Churn Probability', nbins=20,
                          color_discrete_sequence=['#e74c3c'],
                          title='Churn Risk Distribution')
        st.plotly_chart(fig, use_container_width=True)

        csv = result.to_csv(index=False)
        st.download_button("⬇️ Download Results", csv, "churn_predictions.csv", "text/csv")

# ─── TAB 3: Analytics ───
with tab3:
    st.subheader("Dataset Analytics")
    df_raw = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    df_raw.dropna(inplace=True)

    col1, col2 = st.columns(2)

    with col1:
        # Churn by contract
        fig1 = px.histogram(df_raw, x='Contract', color='Churn',
                           barmode='group',
                           title='Churn by Contract Type',
                           color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
        st.plotly_chart(fig1, use_container_width=True)

        # Churn by internet service
        fig3 = px.histogram(df_raw, x='InternetService', color='Churn',
                           barmode='group',
                           title='Churn by Internet Service',
                           color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Churn pie
        churn_counts = df_raw['Churn'].value_counts()
        fig2 = px.pie(values=churn_counts.values,
                     names=churn_counts.index,
                     title='Overall Churn Rate',
                     color_discrete_sequence=['#2ecc71', '#e74c3c'])
        st.plotly_chart(fig2, use_container_width=True)

        # Monthly charges vs tenure
        fig4 = px.scatter(df_raw, x='tenure', y='MonthlyCharges',
                         color='Churn',
                         title='Tenure vs Monthly Charges',
                         color_discrete_map={'Yes': '#e74c3c', 'No': '#2ecc71'})
        st.plotly_chart(fig4, use_container_width=True)